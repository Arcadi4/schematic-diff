//! The composited image: a small RGBA8 raster, and the panel layout drawn into
//! it.
//!
//! Everything the terminal shows — the two builds, the change overlay, and the
//! captions naming them — is one picture, because the kitty protocol places one
//! image over one rectangle of cells. Nothing is aligned by hand afterwards.

use crate::error::{Error, Result};
use crate::font;
use nucleation::meshing::MeshOutput;

use crate::mesh::{Marker, draw as draw_mesh};
use crate::render::{Bounds, Camera, Grid, render};

/// A packed `0xRRGGBB` as its three channels.
pub fn rgb(color: u32) -> [u8; 3] {
    [
        ((color >> 16) & 0xff) as u8,
        ((color >> 8) & 0xff) as u8,
        (color & 0xff) as u8,
    ]
}

/// A row-major RGBA8 buffer.
///
/// Deliberately not `image::RgbaImage`: this tool only ever produces pixels and
/// encodes them as PNG, so a full image library would be dead weight.
#[derive(Clone)]
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Canvas {
    /// A fully transparent canvas.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width as usize) * (height as usize) * 4],
        }
    }

    /// A canvas filled with an opaque colour.
    fn filled(width: u32, height: u32, color: [u8; 3]) -> Self {
        let mut canvas = Self::new(width, height);
        let opaque = [color[0], color[1], color[2], 255];
        for texel in canvas.pixels.as_chunks_mut::<4>().0 {
            *texel = opaque;
        }
        canvas
    }

    pub fn set(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let offset = ((y * self.width + x) * 4) as usize;
        self.pixels[offset..offset + 4].copy_from_slice(&rgba);
    }

    fn get(&self, x: u32, y: u32) -> [u8; 4] {
        let offset = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[offset],
            self.pixels[offset + 1],
            self.pixels[offset + 2],
            self.pixels[offset + 3],
        ]
    }

    /// Fill an axis-aligned rectangle, clipped to the canvas.
    pub fn fill_rect(&mut self, x: i64, y: i64, w: u32, h: u32, texel: [u8; 4]) {
        let x0 = x.max(0) as u32;
        let y0 = y.max(0) as u32;
        let x1 = (x + i64::from(w)).clamp(0, i64::from(self.width)) as u32;
        let y1 = (y + i64::from(h)).clamp(0, i64::from(self.height)) as u32;
        for py in y0..y1 {
            let row = py as usize * self.width as usize;
            for px in x0..x1 {
                let offset = (row + px as usize) * 4;
                self.pixels[offset..offset + 4].copy_from_slice(&texel);
            }
        }
    }

    /// Copy the opaque pixels of `src` in at `(x, y)`, clipped.
    ///
    /// Transparent pixels are skipped so the background shows through, which is
    /// what leaves the empty space around a build looking like the theme rather
    /// than like a second, darker rectangle.
    fn blit(&mut self, src: &Canvas, x: i64, y: i64) {
        for sy in 0..src.height {
            let py = y + i64::from(sy);
            if py < 0 || py >= i64::from(self.height) {
                continue;
            }
            for sx in 0..src.width {
                let px = x + i64::from(sx);
                if px < 0 || px >= i64::from(self.width) {
                    continue;
                }
                let texel = src.get(sx, sy);
                if texel[3] == 0 {
                    continue;
                }
                self.set(px as u32, py as u32, texel);
            }
        }
    }
}

/// Colours of the composited image.
pub struct Theme {
    /// Fills the whole image; shows through wherever a ray hit nothing.
    pub background: [u8; 3],
    /// The rule drawn between panels.
    pub divider: [u8; 3],
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            background: [18, 20, 24],
            divider: [58, 63, 72],
        }
    }
}

/// A run of caption text drawn in one colour.
///
/// The change legend is built from these so each category is drawn in the same
/// colour as the blocks it names, which is the only thing that tells a reader
/// what a cyan cell means.
pub struct Segment {
    pub text: String,
    pub color: [u8; 3],
}

impl Segment {
    pub fn new(text: impl Into<String>, color: [u8; 3]) -> Self {
        Self {
            text: text.into(),
            color,
        }
    }
}

/// What a panel shows.
pub enum Picture<'a> {
    /// A grid of single-coloured cubes — the no-pack path.
    Flat(&'a Grid),
    /// Meshed geometry, with changed cells marked over it.
    Mesh(&'a MeshOutput, &'a [Marker]),
}

/// One column of the composite: a caption, and the build it names.
pub struct Panel<'a> {
    pub caption: Vec<Segment>,
    pub picture: Picture<'a>,
}

impl<'a> Panel<'a> {
    /// A panel whose caption is a single colour.
    pub fn new(caption: impl Into<String>, color: [u8; 3], picture: Picture<'a>) -> Self {
        Self {
            caption: vec![Segment::new(caption, color)],
            picture,
        }
    }

    /// A panel whose caption is a coloured legend, one segment per category.
    pub fn legend(caption: Vec<Segment>, picture: Picture<'a>) -> Self {
        Self { caption, picture }
    }
}

/// The geometry every panel of one composite shares.
pub struct Layout {
    /// The pixel area the whole composite occupies, which is exactly the cells
    /// the image is placed over.
    pub width: u32,
    pub height: u32,
    /// Pixel size of one terminal cell, which sets the caption text size.
    pub cell_px: (u32, u32),
    /// The extent every panel is framed to. Shared so a build that grew
    /// actually looks bigger; framing each panel to its own bounds would
    /// rescale the two sides independently and hide the size change being
    /// inspected.
    pub frame: Bounds,
}

/// The narrowest a panel may be before the layout drops one.
const MIN_PANEL_PIXELS: u32 = 360;

/// How many panels of at least [`MIN_PANEL_PIXELS`] fit across `width`.
pub fn fits(width: u32, panels: u32) -> bool {
    width >= MIN_PANEL_PIXELS * panels
}

/// Composite `panels` side by side, each framed identically.
pub fn compose(panels: &[Panel<'_>], camera: &Camera, layout: &Layout, theme: &Theme) -> Canvas {
    let mut canvas = Canvas::filled(layout.width, layout.height, theme.background);
    if panels.is_empty() || layout.width == 0 || layout.height == 0 {
        return canvas;
    }

    let count = panels.len() as u32;
    let base = layout.width / count;
    // Spread the division remainder over the leading panels so the composite
    // uses the full width instead of leaving a seam at the right edge.
    let remainder = layout.width % count;

    let mut x = 0u32;
    for (index, panel) in panels.iter().enumerate() {
        let panel_width = base + u32::from((index as u32) < remainder);
        let panel_x = x;
        x += panel_width;

        let caption_scale = caption_scale(&panel.caption, panel_width, layout.cell_px.0);
        let band = (font::HEIGHT * caption_scale + 2 * caption_scale) as i64;

        // Each segment advances by its full character count, so the trailing
        // gap of one segment becomes the space before the next. The last legal
        // pixel is one margin in from the panel's right edge: caption text that
        // ran past it would collide with the next panel's caption, which shares
        // the same band.
        let limit = i64::from(panel_x + panel_width - caption_scale);
        let mut cursor = i64::from(panel_x + caption_scale);
        for segment in &panel.caption {
            let room = limit - cursor;
            if room <= 0 {
                break;
            }
            let text = clip_to_width(&segment.text, room as u32, caption_scale);
            font::draw(
                &mut canvas,
                cursor,
                i64::from(caption_scale),
                &text,
                caption_scale,
                segment.color,
            );
            cursor += i64::from(font::ADVANCE * caption_scale) * text.chars().count() as i64;
        }

        let body_height = layout.height.saturating_sub(band as u32);
        let picture = match &panel.picture {
            Picture::Flat(grid) => render(grid, layout.frame, camera, panel_width, body_height),
            Picture::Mesh(mesh, markers) => draw_mesh(
                mesh,
                markers,
                layout.frame,
                camera,
                panel_width,
                body_height,
            ),
        };
        canvas.blit(&picture, i64::from(panel_x), band);

        if index + 1 < panels.len() {
            canvas.fill_rect(i64::from(x) - 1, 0, 1, band as u32, [0, 0, 0, 0]);
            canvas.fill_rect(
                i64::from(x) - 1,
                band,
                1,
                layout.height - band as u32,
                [theme.divider[0], theme.divider[1], theme.divider[2], 255],
            );
        }
    }

    canvas
}

/// The longest prefix of `text` whose glyphs fit in `width` pixels.
///
/// A file name can be arbitrarily long, so a caption can be wider than the
/// panel it belongs to. Trimming to fit keeps it inside the panel instead of
/// running into the next one.
fn clip_to_width(text: &str, width: u32, scale: u32) -> String {
    if scale == 0 {
        return String::new();
    }
    // `text_width(n, s) == (6n - 1) * s`, so the most glyphs that fit is the
    // largest n with `(6n - 1) * s <= width`.
    let fits = (width / scale + 1) / font::ADVANCE;
    text.chars().take(fits as usize).collect()
}

/// The largest font scale at which `caption` fits inside a panel `width` wide.
///
/// The ceiling comes from the terminal's own cell width, so caption text ends up
/// about the same size as the terminal's text rather than being scaled by a
/// constant that only suits one font size.
fn caption_scale(caption: &[Segment], width: u32, cell_width_px: u32) -> u32 {
    let characters: usize = caption
        .iter()
        .map(|segment| segment.text.chars().count())
        .sum();
    if characters == 0 {
        return 1;
    }
    let available = width.saturating_sub(2);
    let ceiling = (cell_width_px / font::ADVANCE).clamp(1, 4);
    (1..=ceiling)
        .rev()
        .find(|scale| font::text_width(characters, *scale) <= available)
        .unwrap_or(1)
}

/// Encode a canvas as PNG.
pub fn encode_png(canvas: &Canvas) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, canvas.width, canvas.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        // Ask for the smallest file: the whole point of sending PNG rather than
        // raw pixels is the transfer size.
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .map_err(|error| Error::message(format!("PNG header failed: {error}")))?;
        writer
            .write_image_data(&canvas.pixels)
            .map_err(|error| Error::message(format!("PNG encoding failed: {error}")))?;
    }
    Ok(out)
}
