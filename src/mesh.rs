//! Drawing a mesh: projection, a depth-buffered target, and triangle fill.
//!
//! With `--pack` the build arrives as triangles — positions, UVs, and per-vertex
//! colours that already carry the game's tint, ambient occlusion and lighting —
//! so drawing it is an ordinary z-buffered rasterization. That is deliberately
//! dull code: the interesting decisions (which model a block draws, how its
//! faces are oriented, what tint applies) are made by the mesher, and this only
//! has to put the triangles on screen.

use nucleation::meshing::{MeshLayer, MeshOutput};
use schematic_mesher::TextureAtlas;

use crate::raster::{Canvas, rgb, tint};
use crate::render::{Bounds, Camera, Eye};

/// How a layer's triangles are drawn.
///
/// The three parts are what the pixel loop needs beyond the geometry itself:
/// where to sample colour from, what to lay over it, and whether to blend with
/// what is already there.
struct Paint<'a> {
    atlas: &'a TextureAtlas,
    /// A colour filtered over every texel drawn, for the changes panel.
    filter: Option<[u8; 3]>,
    /// Whether these triangles blend with what is already drawn.
    blend: bool,
}

/// A point on a triangle, after projection.
#[derive(Clone, Copy)]
struct Screen {
    x: f32,
    y: f32,
    /// View-space depth, used both for the z-test and for fog.
    z: f32,
}

/// The camera, resolved into the form a rasterizer needs.
///
/// Both this and the block-grid raycaster derive from [`Camera`], so the two
/// renderers frame a build identically and swapping between them is not a
/// change of viewpoint.
struct View {
    eye: Eye,
    /// Pixels per unit at unit depth.
    focal: f32,
    width: u32,
    height: u32,
}

impl View {
    fn new(eye: Eye, width: u32, height: u32) -> Self {
        // A 50-degree vertical field of view, the same the raycaster uses.
        let tan_half = (50.0f32.to_radians() / 2.0).tan();
        Self {
            eye,
            focal: (height as f32 / 2.0) / tan_half,
            width,
            height,
        }
    }

    /// Project a world point, or `None` when it is behind the camera.
    fn project(&self, point: [f32; 3]) -> Option<Screen> {
        let rel = [
            point[0] - self.eye.position[0],
            point[1] - self.eye.position[1],
            point[2] - self.eye.position[2],
        ];
        let z = dot(rel, self.eye.forward);
        // Behind the eye, or close enough that dividing by it explodes.
        if z <= 0.05 {
            return None;
        }
        let x = dot(rel, self.eye.right);
        let y = dot(rel, self.eye.up);
        let scale = self.focal / z;
        Some(Screen {
            x: self.width as f32 / 2.0 + x * scale,
            y: self.height as f32 / 2.0 - y * scale,
            z,
        })
    }

    /// Depth fog, matching the block-grid raycaster so the two agree.
    fn fog(&self, z: f32) -> f32 {
        let falloff = 1.0
            - 0.25
                * ((z - self.eye.distance + self.eye.radius) / (self.eye.radius * 2.0))
                    .clamp(0.0, 1.0);
        falloff.max(0.0)
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// An image being drawn into, with a depth buffer.
struct Target {
    width: u32,
    height: u32,
    color: Vec<[u8; 4]>,
    depth: Vec<f32>,
}

impl Target {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            color: vec![[0, 0, 0, 0]; (width as usize) * (height as usize)],
            // Farther than anything can be, so the first write always wins.
            depth: vec![f32::INFINITY; (width as usize) * (height as usize)],
        }
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + x as usize
    }

    /// Draw one triangle, sampling `paint.atlas` through its UVs.
    ///
    /// `colors` are the mesher's per-vertex tints, already carrying the game's
    /// own shading, so they are interpolated and applied rather than replaced.
    fn triangle(
        &mut self,
        view: &View,
        positions: [[f32; 3]; 3],
        uvs: [[f32; 2]; 3],
        colors: [[f32; 4]; 3],
        paint: &Paint<'_>,
    ) {
        let Some(a) = view.project(positions[0]) else {
            return;
        };
        let Some(b) = view.project(positions[1]) else {
            return;
        };
        let Some(c) = view.project(positions[2]) else {
            return;
        };

        // The signed area tells us the winding; either winding draws, so the
        // edge functions are normalised rather than back-face culling. A build
        // is closed geometry and culling saves little, while getting the
        // winding wrong would punch holes in it.
        let area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if area.abs() < 1e-6 {
            return;
        }
        let inv_area = 1.0 / area;

        let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as u32;
        let max_x = (a.x.max(b.x).max(c.x).ceil()).min(self.width as f32 - 1.0) as u32;
        let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as u32;
        let max_y = (a.y.max(b.y).max(c.y).ceil()).min(self.height as f32 - 1.0) as u32;
        if min_x > max_x || min_y > max_y {
            return;
        }

        // Depth is interpolated on `1/z`, which is what makes it linear in
        // screen space. Interpolating `z` directly would let a texture swim
        // across a triangle that is strongly foreshortened.
        let inv_z = [1.0 / a.z, 1.0 / b.z, 1.0 / c.z];

        for y in min_y..=max_y {
            let py = y as f32 + 0.5;
            for x in min_x..=max_x {
                let px = x as f32 + 0.5;
                let w0 = ((b.x - px) * (c.y - py) - (c.x - px) * (b.y - py)) * inv_area;
                let w1 = ((c.x - px) * (a.y - py) - (a.x - px) * (c.y - py)) * inv_area;
                let w2 = 1.0 - w0 - w1;
                // A small negative tolerance keeps the shared edge between two
                // triangles of a quad from being dropped by both.
                if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                    continue;
                }

                let reciprocal = w0 * inv_z[0] + w1 * inv_z[1] + w2 * inv_z[2];
                if reciprocal <= 0.0 {
                    continue;
                }
                let depth = 1.0 / reciprocal;
                let index = self.index(x, y);
                if depth >= self.depth[index] {
                    continue;
                }

                // Perspective-correct UVs, the same weighting as depth.
                let u = (w0 * uvs[0][0] * inv_z[0]
                    + w1 * uvs[1][0] * inv_z[1]
                    + w2 * uvs[2][0] * inv_z[2])
                    * depth;
                let v = (w0 * uvs[0][1] * inv_z[0]
                    + w1 * uvs[1][1] * inv_z[1]
                    + w2 * uvs[2][1] * inv_z[2])
                    * depth;
                let texel = sample_atlas(paint.atlas, u, v);
                // In a mixed layer a see-through texel lets whatever is behind
                // show through; in an alpha-tested one it is either drawn or
                // discarded outright.
                if !paint.blend && texel[3] < 128 {
                    continue;
                }
                if paint.blend && texel[3] == 0 {
                    continue;
                }
                // The mesh layer says what is *meant* to blend; the texel says
                // how much. Glass is mostly clear, water mostly opaque, and both
                // live in the same layer.
                let texel_alpha = f32::from(texel[3]) / 255.0;
                // The mesher's per-vertex colour carries tint, ambient occlusion
                // and lighting; it modulates the texel rather than replacing it.
                let shade = [
                    w0 * colors[0][0] + w1 * colors[1][0] + w2 * colors[2][0],
                    w0 * colors[0][1] + w1 * colors[1][1] + w2 * colors[2][1],
                    w0 * colors[0][2] + w1 * colors[1][2] + w2 * colors[2][2],
                ];
                let fog = view.fog(depth);
                // The transparent layer blends, so it must not occlude what is
                // already drawn at this pixel.
                if !paint.blend {
                    self.depth[index] = depth;
                }
                let lit = lit_texel(&texel, shade, fog, paint.filter);
                if paint.blend {
                    // Nothing drawn behind it yet: blending against the backdrop
                    // would leave a faint square, so the texel's own colour is
                    // used at its own opacity over the background.
                    let under = self.color[index];
                    self.color[index] = [
                        mix(under[0], lit[0], texel_alpha),
                        mix(under[1], lit[1], texel_alpha),
                        mix(under[2], lit[2], texel_alpha),
                        mix(under[3], 255, texel_alpha),
                    ];
                } else {
                    self.color[index] = [lit[0], lit[1], lit[2], 255];
                }
            }
        }
    }

    /// Draw one mesh layer.
    fn layer(&mut self, view: &View, layer: &MeshLayer, paint: &Paint<'_>) {
        for triangle in layer.indices.as_chunks::<3>().0 {
            let [i0, i1, i2] = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            if i0 >= layer.positions.len()
                || i1 >= layer.positions.len()
                || i2 >= layer.positions.len()
            {
                continue;
            }
            self.triangle(
                view,
                [
                    layer.positions[i0],
                    layer.positions[i1],
                    layer.positions[i2],
                ],
                [layer.uvs[i0], layer.uvs[i1], layer.uvs[i2]],
                [
                    color_at(&layer.colors, i0),
                    color_at(&layer.colors, i1),
                    color_at(&layer.colors, i2),
                ],
                paint,
            );
        }
    }

    /// Copy into a canvas, leaving untouched pixels transparent.
    fn into_canvas(self) -> Canvas {
        let mut canvas = Canvas::new(self.width, self.height);
        for (index, texel) in self.color.iter().enumerate() {
            if texel[3] == 0 {
                continue;
            }
            canvas.set(
                (index as u32) % self.width,
                (index as u32) / self.width,
                *texel,
            );
        }
        canvas
    }
}

/// One channel of a blend: `over` laid over `base` at `alpha`.
fn mix(base: u8, over: u8, alpha: f32) -> u8 {
    clamp_byte(f32::from(base) * (1.0 - alpha) + f32::from(over) * alpha)
}

/// The colour a texel ends up as: shaded, set back by fog, and — where the
/// changes panel asks for it — filtered towards the colour of the category the
/// block belongs to.
fn lit_texel(texel: &[u8; 4], shade: [f32; 3], fog: f32, filter: Option<[u8; 3]>) -> [u8; 3] {
    let shaded = [
        clamp_byte(f32::from(texel[0]) * shade[0] * fog),
        clamp_byte(f32::from(texel[1]) * shade[1] * fog),
        clamp_byte(f32::from(texel[2]) * shade[2] * fog),
    ];
    filter.map_or(shaded, |filter| tint(shaded, filter))
}

fn color_at(colors: &[[f32; 4]], index: usize) -> [f32; 4] {
    colors.get(index).copied().unwrap_or([1.0, 1.0, 1.0, 1.0])
}

fn clamp_byte(value: f32) -> u8 {
    value.clamp(0.0, 255.0) as u8
}

/// Nearest-texel sample from the atlas, clamped to its bounds.
///
/// `u` and `v` are atlas coordinates, and the atlas has a margin around each
/// tile, so clamping cannot pull in a neighbouring texture.
fn sample_atlas(atlas: &TextureAtlas, u: f32, v: f32) -> [u8; 4] {
    if atlas.width == 0 || atlas.height == 0 {
        return [255, 255, 255, 255];
    }
    let x = ((u.rem_euclid(1.0) * atlas.width as f32) as u32).min(atlas.width - 1);
    let y = ((v.rem_euclid(1.0) * atlas.height as f32) as u32).min(atlas.height - 1);
    let offset = ((y * atlas.width + x) * 4) as usize;
    match atlas.pixels.get(offset..offset + 4) {
        Some(texel) => [texel[0], texel[1], texel[2], texel[3]],
        None => [255, 255, 255, 255],
    }
}

/// A category of change, meshed: the blocks that changed, and the colour laid
/// over them.
pub struct TintedMesh {
    pub mesh: MeshOutput,
    pub color: u32,
}

/// Draw a meshed build.
///
/// The mesher sorts geometry into opaque, alpha-tested and blended layers, and
/// they are drawn in that order because that is what the sorting is for:
/// opaque fills the picture, cut-outs punch through it, and glass and water
/// tint whatever is already behind them.
pub fn draw(
    output: &MeshOutput,
    frame: Bounds,
    camera: &Camera,
    width: u32,
    height: u32,
) -> Canvas {
    if width == 0 || height == 0 {
        return Canvas::new(width, height);
    }
    let view = View::new(camera.eye_and_basis(frame), width, height);
    let mut target = Target::new(width, height);
    let atlas = &output.atlas;
    target.layer(&view, &output.opaque, &paint(atlas, None, false));
    target.layer(&view, &output.cutout, &paint(atlas, None, false));
    target.layer(&view, &output.transparent, &paint(atlas, None, true));
    target.into_canvas()
}

/// Draw the changed blocks alone, each category's colour filtered over it.
///
/// This is the changes panel. It shows only what changed — a panel about the
/// difference has nothing to say about the blocks that stayed the same — and
/// shows it as the blocks themselves, tinted, rather than as coloured cubes.
pub fn draw_changes(
    categories: &[TintedMesh],
    frame: Bounds,
    camera: &Camera,
    width: u32,
    height: u32,
) -> Canvas {
    if width == 0 || height == 0 {
        return Canvas::new(width, height);
    }
    let view = View::new(camera.eye_and_basis(frame), width, height);
    let mut target = Target::new(width, height);
    // The categories cover disjoint cells, so opaque and cut-out geometry can be
    // drawn a category at a time; the blended layers come after all of them,
    // because they read what is already drawn behind them.
    for category in categories {
        let paint = paint(&category.mesh.atlas, Some(rgb(category.color)), false);
        target.layer(&view, &category.mesh.opaque, &paint);
        target.layer(&view, &category.mesh.cutout, &paint);
    }
    for category in categories {
        let paint = paint(&category.mesh.atlas, Some(rgb(category.color)), true);
        target.layer(&view, &category.mesh.transparent, &paint);
    }
    target.into_canvas()
}

fn paint(atlas: &TextureAtlas, filter: Option<[u8; 3]>, blend: bool) -> Paint<'_> {
    Paint {
        atlas,
        filter,
        blend,
    }
}
