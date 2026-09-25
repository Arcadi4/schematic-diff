//! CPU raycast renderer used when no resource pack is supplied.
//!
//! Each pixel casts a ray through the voxel grid and shades the first occupied
//! cell. Colors come from Nucleation's baked block table, with a stable nonzero
//! fallback for unknown blocks. Packed builds use the mesh renderer instead.

use rayon::prelude::*;
use nucleation::{BlockState, UniversalSchematic};

use crate::error::{Error, Result};
use crate::raster::Canvas;

/// Maximum grid volume in cells permitted before refusing to allocate.
///
/// At four bytes per cell, this permits a 2 GiB grid. The ceiling bounds memory
/// use and raycast time for outsized schematics.
pub const MAX_CELLS: i64 = 512 * 1024 * 1024;

/// Compute a grid's cell count, rejecting non-positive or overflowing extents.
fn grid_cell_count(size: [i32; 3]) -> Result<i64> {
    i64::from(size[0])
        .checked_mul(i64::from(size[1]))
        .and_then(|cells| cells.checked_mul(i64::from(size[2])))
        .filter(|cells| *cells > 0)
        .ok_or_else(|| {
            Error::message(format!(
                "the {}x{}x{} extent is too large to render",
                size[0], size[1], size[2]
            ))
        })
}

fn alloc_cells(cells: i64, size: [i32; 3]) -> Result<Vec<u32>> {
    let mut cells_vec = Vec::new();
    cells_vec.try_reserve(cells as usize).map_err(|_| {
        Error::message(format!(
            "the {}x{}x{} extent ({} cells) is too large to render",
            size[0], size[1], size[2], cells
        ))
    })?;
    cells_vec.resize(cells as usize, 0);
    Ok(cells_vec)
}

/// An axis-aligned inclusive voxel extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub min: [i32; 3],
    pub max: [i32; 3],
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            min: [0; 3],
            max: [0; 3],
        }
    }
}

impl Bounds {
    fn singleton(x: i32, y: i32, z: i32) -> Self {
        Self {
            min: [x, y, z],
            max: [x, y, z],
        }
    }

    fn extend(&mut self, x: i32, y: i32, z: i32) {
        self.min = [self.min[0].min(x), self.min[1].min(y), self.min[2].min(z)];
        self.max = [self.max[0].max(x), self.max[1].max(y), self.max[2].max(z)];
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn size(self) -> [i32; 3] {
        let axis_size = |min: i32, max: i32| -> i32 {
            let diff = i64::from(max) - i64::from(min) + 1;
            if diff <= 0 {
                0
            } else if diff > i64::from(i32::MAX) {
                i32::MAX
            } else {
                diff as i32
            }
        };
        [
            axis_size(self.min[0], self.max[0]),
            axis_size(self.min[1], self.max[1]),
            axis_size(self.min[2], self.max[2]),
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    /// 1.0 fits the framed bounds exactly; larger zooms in.
    pub zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        // A three-quarter view keeps most block shapes legible.
        Self {
            yaw_deg: 45.0,
            pitch_deg: 30.0,
            zoom: 1.0,
        }
    }
}

/// Camera origin and basis derived from [`Camera`].
///
/// Both renderers derive this from the same camera so switching render modes
/// does not change the viewpoint.
pub struct Eye {
    pub position: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub distance: f32,
    pub radius: f32,
}

/// A dense colour grid over the build's extent, indexed from `min`.
///
/// Every cell holds either `0` for empty or its block's colour. No colour the
/// palette produces is zero — the fallback hash is deliberately held away from
/// black — so the sentinel is unambiguous, and without a pack a block is a cube
/// of one colour, so a colour is all a cell needs.
#[derive(Debug)]
pub struct Grid {
    pub min: [i32; 3],
    pub dims: [usize; 3],
    pub cells: Vec<u32>,
}

impl Grid {
    /// Build a dense grid from every non-air block in `schematic`.
    pub fn from_schematic(schematic: &UniversalSchematic) -> Result<Self> {
        let mut bounds: Option<Bounds> = None;
        for (position, block) in schematic.iter_blocks() {
            if is_air(block) {
                continue;
            }
            match &mut bounds {
                Some(extent) => extent.extend(position.x, position.y, position.z),
                None => bounds = Some(Bounds::singleton(position.x, position.y, position.z)),
            }
        }

        let Some(bounds) = bounds else {
            // An empty build is legitimate — a schematic that is all air — and
            // renders as a blank panel rather than an error.
            return Ok(Self {
                min: [0; 3],
                dims: [1, 1, 1],
                cells: vec![0],
            });
        };

        let size = bounds.size();
        let cells = grid_cell_count(size)?;
        if cells > MAX_CELLS {
            return Err(Error::message(format!(
                "the build's {}x{}x{} extent ({} cells) is too large to render",
                size[0], size[1], size[2], cells
            )));
        }
        let mut grid = Self {
            min: bounds.min,
            dims: [size[0] as usize, size[1] as usize, size[2] as usize],
            cells: alloc_cells(cells, size)?,
        };
        for (position, block) in schematic.iter_blocks() {
            if is_air(block) {
                continue;
            }
            grid.set(position.x, position.y, position.z, block_color(block));
        }
        Ok(grid)
    }

    pub fn bounds(&self) -> Bounds {
        Bounds {
            min: self.min,
            max: [
                self.min[0] + self.dims[0] as i32 - 1,
                self.min[1] + self.dims[1] as i32 - 1,
                self.min[2] + self.dims[2] as i32 - 1,
            ],
        }
    }

    pub fn empty_over(bounds: Bounds) -> Result<Self> {
        let size = bounds.size();
        let cells = grid_cell_count(size)?;
        if cells > MAX_CELLS {
            return Err(Error::message(format!(
                "the {}x{}x{} extent is too large to render",
                size[0], size[1], size[2]
            )));
        }
        Ok(Self {
            min: bounds.min,
            dims: [size[0] as usize, size[1] as usize, size[2] as usize],
            cells: alloc_cells(cells, size)?,
        })
    }

    fn index(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        let (dx, dy, dz) = (x - self.min[0], y - self.min[1], z - self.min[2]);
        if dx < 0
            || dy < 0
            || dz < 0
            || dx as usize >= self.dims[0]
            || dy as usize >= self.dims[1]
            || dz as usize >= self.dims[2]
        {
            return None;
        }
        Some((dy as usize * self.dims[2] + dz as usize) * self.dims[0] + dx as usize)
    }

    fn color_at(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        let cell = self.cells[self.index(x, y, z)?];
        (cell != 0).then_some(cell)
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, color: u32) {
        if let Some(index) = self.index(x, y, z) {
            self.cells[index] = color;
        }
    }
}

pub fn is_air(block: &BlockState) -> bool {
    is_air_name(block.get_name())
}

fn is_air_name(name: &str) -> bool {
    let short = name.strip_prefix("minecraft:").unwrap_or(name);
    matches!(short, "air" | "cave_air" | "void_air")
}

/// Return a stable nonzero color when the baked table lacks a block.
pub fn block_color(block: &BlockState) -> u32 {
    let name = block.get_name();
    if let Some(facts) = nucleation::blockpedia::BLOCKS.get(name)
        && let Some(color) = &facts.extras.color
    {
        let [r, g, b] = color.rgb;
        let packed = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
        if packed != 0 {
            return packed;
        }
    }
    fallback_color(name)
}

/// Resolve unknown colors by known-block table, dye family, then stable hash.
fn fallback_color(name: &str) -> u32 {
    let short = name.strip_prefix("minecraft:").unwrap_or(name);
    let table: u32 = match short {
        "stone" | "smooth_stone" | "cobblestone" | "stone_bricks" => 0x8f8f8f,
        "andesite" | "polished_andesite" => 0x888a85,
        "granite" => 0x9f6b58,
        "diorite" => 0xc9c9c6,
        "deepslate" | "cobbled_deepslate" => 0x4d4d51,
        "obsidian" | "crying_obsidian" => 0x1b1226,
        "bedrock" => 0x3a3a3a,
        "dirt" | "rooted_dirt" => 0x8a5a3b,
        "grass_block" | "grass" => 0x5f9e3a,
        "sand" => 0xdbcf9c,
        "gravel" => 0x7f7c78,
        "water" => 0x3b6ecf,
        "lava" => 0xe06a10,
        "ice" | "packed_ice" | "blue_ice" | "frosted_ice" => 0xa5c8f0,
        "glass" | "tinted_glass" => 0xd8f0f2,
        "slime_block" => 0x6fd66a,
        "honey_block" => 0xe8a933,
        "tnt" => 0xc23b2a,
        "redstone_block" => 0xb01e0e,
        "redstone_wire" => 0x8f1010,
        "redstone_torch" | "redstone_wall_torch" => 0xd94a2a,
        "repeater" | "comparator" => 0xb9b0a8,
        "observer" => 0x5f5c58,
        "piston" | "sticky_piston" => 0x9a8054,
        "piston_head" | "moving_piston" => 0xb08d5a,
        "hopper" | "cauldron" => 0x4a4a4a,
        "chest" | "trapped_chest" | "barrel" => 0x9a6b2f,
        "dropper" | "dispenser" | "furnace" => 0x6f6f6f,
        "lever" | "tripwire_hook" => 0x8a7a5a,
        "rail" | "powered_rail" | "activator_rail" | "detector_rail" => 0x99856a,
        "iron_block" => 0xd8d8d8,
        "gold_block" => 0xf0d24a,
        "diamond_block" => 0x62e6d8,
        "emerald_block" => 0x35c65a,
        "quartz_block" | "smooth_quartz" => 0xe8e2d8,
        "glowstone" | "shroomlight" | "sea_lantern" => 0xf0c86a,
        "netherrack" => 0x7a3230,
        "soul_sand" | "soul_soil" => 0x5a4636,
        "snow" | "snow_block" | "powder_snow" => 0xf4f8fa,
        "sculk" => 0x0e2129,
        "stonecutter" | "smithing_table" => 0x6a6258,
        "crafting_table" => 0x8a5a30,
        "furnace_minecart" | "minecart" => 0x6f6f6f,
        _ => 0,
    };
    if table != 0 {
        return table;
    }
    for (suffix, dye) in [
        ("_concrete", true),
        ("_wool", true),
        ("_terracotta", true),
        ("_stained_glass", true),
        ("_stained_glass_pane", true),
        ("_candle", true),
        ("_bed", true),
        ("_shulker_box", true),
    ] {
        if let Some(rest) = short.strip_suffix(suffix)
            && dye
            && let Some(color) = dye_color(rest)
        {
            return color;
        }
    }
    if short.ends_with("_planks") || short.ends_with("_log") || short.ends_with("_wood") {
        return 0xa07a48;
    }
    if short.ends_with("_leaves") {
        return 0x3f7a2a;
    }
    // A stable FNV-1a hash, so the same unknown block is the same colour in
    // every run and on every machine.
    let mut hash: u32 = 0x811c_9dc5;
    for byte in short.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    // Held to the 96..223 range on every channel: never black (which would read
    // as a hole in the build) and never blown out.
    let red = 96 + (hash & 0x7f) as u8;
    let green = 96 + ((hash >> 8) & 0x7f) as u8;
    let blue = 96 + ((hash >> 16) & 0x7f) as u8;
    (u32::from(red) << 16) | (u32::from(green) << 8) | u32::from(blue)
}

fn dye_color(dye: &str) -> Option<u32> {
    Some(match dye {
        "white" => 0xe9ecec,
        "orange" => 0xf07613,
        "magenta" => 0xbd44b3,
        "light_blue" => 0x3aafd9,
        "yellow" => 0xf8c527,
        "lime" => 0x70b919,
        "pink" => 0xed8dac,
        "gray" => 0x3e4447,
        "light_gray" => 0x8e8e86,
        "cyan" => 0x158991,
        "purple" => 0x792aac,
        "blue" => 0x35399d,
        "brown" => 0x724728,
        "green" => 0x546d1b,
        "red" => 0xa12722,
        "black" => 0x141519,
        _ => return None,
    })
}

/// Render `grid` into `width` x `height` pixels, framing `frame`.
///
/// `frame` is passed in rather than read off `grid` so that two builds can be
/// drawn at the same scale and position — without that, a small build and a
/// large one would each fill their panel and be impossible to compare.
pub fn render(grid: &Grid, frame: Bounds, camera: &Camera, width: u32, height: u32) -> Canvas {
    let mut canvas = Canvas::new(width, height);
    if width == 0 || height == 0 {
        return canvas;
    }

    let eye = camera.eye_and_basis(frame);
    let Eye {
        position,
        right,
        up,
        forward,
        distance,
        radius,
    } = eye;

    let tan_half = (Camera::VERTICAL_FOV_DEG.to_radians() / 2.0).tan();
    let aspect = width as f32 / height as f32;
    let max_t = distance + radius * 4.0;
    let scene = Scene {
        grid,
        distance,
        radius,
    };

    let row_bytes = (width as usize) * 4;
    canvas
        .pixels
        .par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(py, row)| {
            let v = (0.5 - (py as f32 + 0.5) / height as f32) * 2.0 * tan_half;
            for px in 0..width {
                let u = ((px as f32 + 0.5) / width as f32 - 0.5) * 2.0 * tan_half * aspect;
                let ray = Ray {
                    origin: position,
                    direction: normalize([
                        forward[0] + right[0] * u + up[0] * v,
                        forward[1] + right[1] * u + up[1] * v,
                        forward[2] + right[2] * u + up[2] * v,
                    ]),
                };
                if let Some(rgba) = trace(&scene, &ray, max_t) {
                    let offset = (px as usize) * 4;
                    row[offset..offset + 4].copy_from_slice(&rgba);
                }
            }
        });

    canvas
}

struct Scene<'a> {
    grid: &'a Grid,
    distance: f32,
    radius: f32,
}

struct Ray {
    origin: [f32; 3],
    direction: [f32; 3],
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length == 0.0 {
        return v;
    }
    [v[0] / length, v[1] / length, v[2] / length]
}

/// Map entry-face indices 0..=5 to directional light levels.
fn face_shade(face: u8) -> f32 {
    match face {
        1 => 1.0,
        4 => 0.55,
        0 | 3 => 0.82,
        _ => 0.68,
    }
}

/// The face a ray enters through when it crosses a box in the negative
/// direction, then the positive one, per axis.
const NEGATIVE_FACE: [u8; 3] = [3, 4, 5];
const POSITIVE_FACE: [u8; 3] = [0, 1, 2];

/// Return the ray's entry and exit distances and its entry-face index.
fn box_span(
    min: [f32; 3],
    max: [f32; 3],
    origin: [f32; 3],
    direction: [f32; 3],
    t0: f32,
    t1: f32,
) -> Option<(f32, f32, u8)> {
    let mut enter = t0;
    let mut exit = t1;
    let mut face = 0u8;
    for axis in 0..3 {
        let (low, high) = (min[axis], max[axis]);
        if direction[axis].abs() < 1e-8 {
            // Parallel to this pair of planes: either always inside them, or
            // never.
            if origin[axis] < low || origin[axis] > high {
                return None;
            }
            continue;
        }
        let inverse = 1.0 / direction[axis];
        let a = (low - origin[axis]) * inverse;
        let b = (high - origin[axis]) * inverse;
        let axis_face = if a < b {
            NEGATIVE_FACE[axis]
        } else {
            POSITIVE_FACE[axis]
        };
        let (near, far) = if a < b { (a, b) } else { (b, a) };
        if near > enter {
            enter = near;
            face = axis_face;
        }
        if far < exit {
            exit = far;
        }
        if enter > exit {
            return None;
        }
    }
    Some((enter, exit, face))
}

/// Trace the first occupied cell and shade it. Every untextured cell is a cube,
/// so the grid bounds are sufficient for intersection tests.
fn trace(scene: &Scene<'_>, ray: &Ray, max_t: f32) -> Option<[u8; 4]> {
    let grid = scene.grid;
    let low = [grid.min[0] as f32, grid.min[1] as f32, grid.min[2] as f32];
    let high = [
        low[0] + grid.dims[0] as f32,
        low[1] + grid.dims[1] as f32,
        low[2] + grid.dims[2] as f32,
    ];
    let (enter, far, initial_face) = box_span(low, high, ray.origin, ray.direction, 0.0, max_t)?;
    if enter >= far {
        return None;
    }

    let p_enter = [
        ray.origin[0] + ray.direction[0] * (enter + 1e-4),
        ray.origin[1] + ray.direction[1] * (enter + 1e-4),
        ray.origin[2] + ray.direction[2] * (enter + 1e-4),
    ];

    let mut cx = p_enter[0].floor() as i32;
    let mut cy = p_enter[1].floor() as i32;
    let mut cz = p_enter[2].floor() as i32;

    let step_x = if ray.direction[0] > 0.0 { 1 } else { -1 };
    let step_y = if ray.direction[1] > 0.0 { 1 } else { -1 };
    let step_z = if ray.direction[2] > 0.0 { 1 } else { -1 };

    let inv_x = if ray.direction[0].abs() > 1e-8 { 1.0 / ray.direction[0] } else { 1e30 };
    let inv_y = if ray.direction[1].abs() > 1e-8 { 1.0 / ray.direction[1] } else { 1e30 };
    let inv_z = if ray.direction[2].abs() > 1e-8 { 1.0 / ray.direction[2] } else { 1e30 };

    let t_delta_x = inv_x.abs();
    let t_delta_y = inv_y.abs();
    let t_delta_z = inv_z.abs();

    let next_x = if step_x > 0 { cx as f32 + 1.0 } else { cx as f32 };
    let next_y = if step_y > 0 { cy as f32 + 1.0 } else { cy as f32 };
    let next_z = if step_z > 0 { cz as f32 + 1.0 } else { cz as f32 };

    let mut t_max_x = (next_x - ray.origin[0]) * inv_x;
    let mut t_max_y = (next_y - ray.origin[1]) * inv_y;
    let mut t_max_z = (next_z - ray.origin[2]) * inv_z;

    let mut hit_t = enter;
    let mut hit_face = initial_face;

    loop {
        if let Some(color) = grid.color_at(cx, cy, cz) {
            return Some(shade(hit_face, hit_t, crate::raster::rgb(color), scene));
        }

        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                if t_max_x >= far {
                    return None;
                }
                hit_t = t_max_x;
                t_max_x += t_delta_x;
                cx += step_x;
                hit_face = if step_x > 0 { 3 } else { 0 };
            } else {
                if t_max_z >= far {
                    return None;
                }
                hit_t = t_max_z;
                t_max_z += t_delta_z;
                cz += step_z;
                hit_face = if step_z > 0 { 5 } else { 2 };
            }
        } else {
            if t_max_y < t_max_z {
                if t_max_y >= far {
                    return None;
                }
                hit_t = t_max_y;
                t_max_y += t_delta_y;
                cy += step_y;
                hit_face = if step_y > 0 { 4 } else { 1 };
            } else {
                if t_max_z >= far {
                    return None;
                }
                hit_t = t_max_z;
                t_max_z += t_delta_z;
                cz += step_z;
                hit_face = if step_z > 0 { 5 } else { 2 };
            }
        }
    }
}

fn shade(face: u8, t: f32, rgb: [u8; 3], scene: &Scene<'_>) -> [u8; 4] {
    let falloff =
        1.0 - 0.25 * ((t - scene.distance + scene.radius) / (scene.radius * 2.0)).clamp(0.0, 1.0);
    let shade = face_shade(face) * falloff.max(0.0);
    [
        (rgb[0] as f32 * shade) as u8,
        (rgb[1] as f32 * shade) as u8,
        (rgb[2] as f32 * shade) as u8,
        255,
    ]
}

impl Camera {
    pub const VERTICAL_FOV_DEG: f32 = 50.0;

    pub fn eye_and_basis(&self, frame: Bounds) -> Eye {
        let dims = [
            (frame.max[0] - frame.min[0] + 1) as f32,
            (frame.max[1] - frame.min[1] + 1) as f32,
            (frame.max[2] - frame.min[2] + 1) as f32,
        ];
        let centre = [
            frame.min[0] as f32 + dims[0] / 2.0,
            frame.min[1] as f32 + dims[1] / 2.0,
            frame.min[2] as f32 + dims[2] / 2.0,
        ];
        let radius =
            ((dims[0] * dims[0] + dims[1] * dims[1] + dims[2] * dims[2]).sqrt() / 2.0).max(0.5);

        let (sin_yaw, cos_yaw) = self.yaw_deg.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch_deg.to_radians().sin_cos();
        // Forward points into the scene; right and up span the image plane.
        let forward = [cos_yaw * cos_pitch, -sin_pitch, sin_yaw * cos_pitch];
        let right = [-sin_yaw, 0.0, cos_yaw];
        let up = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        ];

        let distance = (2.4 * radius / self.zoom.max(0.05)).max(2.0);
        Eye {
            position: [
                centre[0] - forward[0] * distance,
                centre[1] - forward[1] * distance,
                centre[2] - forward[2] * distance,
            ],
            right,
            up,
            forward,
            distance,
            radius,
        }
    }
}
