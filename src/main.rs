//! `schematic-diff` — renders Minecraft schematic diffs as images in the
//! terminal, for git to call as an external diff.
//!
//! # How git calls this
//!
//! With `diff.external`, or a `diff.<driver>.command` selected by
//! `.gitattributes`, git invokes this once per changed path as:
//!
//! ```text
//! path old-file old-hex old-mode new-file new-hex new-mode
//! ```
//!
//! where `/dev/null` stands in for a side that does not exist. That is the only
//! contract implemented here: `git difftool` passes filenames instead, and
//! selecting this tool as a difftool would mean carrying two argument protocols
//! to describe one thing.
//!
//! # Why the output stream is chosen the way it is
//!
//! Git gives an external diff a stdout whose kind depends on the invocation,
//! and only a terminal can show an image — see [`terminal`], which is where
//! that decision and its reasoning live.

mod error;
mod font;
mod format;
mod kitty;
mod mesh;
mod pack;
mod raster;
mod render;
mod terminal;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nucleation::diff::{Diff, DiffSpec, diff};
use nucleation::fingerprint::FingerprintSpec;
use nucleation::{BlockState, UniversalSchematic};

use crate::error::{Error, Result};
use crate::format::Loaded;
use crate::mesh::TintedMesh;
use crate::pack::Pack;
use crate::raster::{Layout, Panel, Picture, Segment, Theme, pack, rgb, tint};
use crate::render::{Bounds, Camera, Grid, block_color};
use crate::terminal::Output;

/// Git's stand-in for a side that does not exist.
const DEV_NULL: &str = "/dev/null";

/// Category colours, the single source of truth for the pixels showing a change
/// and the caption text naming it, so the legend cannot drift from what it
/// describes.
const ADDED_COLOR: u32 = 0x70_B9_19;
const REMOVED_COLOR: u32 = 0xA1_27_22;
const CHANGED_COLOR: u32 = 0xF8_C5_27;
const SWAPPED_COLOR: u32 = 0x3A_AF_D9;

/// Caption greys, kept apart from the category colours so a legend stays
/// readable even where one of those colours is dark against the background.
const CAPTION_TEXT: [u8; 3] = [222, 226, 234];
const ADDED_TEXT: [u8; 3] = [146, 208, 80];
const REMOVED_TEXT: [u8; 3] = [228, 138, 133];

const USAGE: &str = "\
schematic-diff — render Minecraft schematic diffs in the terminal

USAGE:
    This is an external diff: git calls it, it is not run directly.
        git config diff.schematic.command schematic-diff
        git diff

    One file across two revisions, or two files on disk:
        git diff <rev1>:<path> <rev2>:<path>
        git diff --no-index <before> <after>

SUPPORTED FORMATS:
    .litematic  .schem  .schematic  .nbt  .snbt  .mcstructure  .nusn

OPTIONS:
    --yaw=<degrees>     Camera horizontal angle            [default: 45]
    --pitch=<degrees>   Camera elevation above the build   [default: 30]
    --zoom=<factor>     Larger zooms in                    [default: 1.0]
    --pack=<file.zip>   Take block colours from a resource pack
    --kitty             Draw the image even if the terminal is unrecognised
    --no-kitty          Never draw the image, print the text summary only
    --output=<file>     Also write the composited image as a PNG
    -h, --help          Show this help
    -V, --version       Show the version

GIT SETUP:
    git config --global diff.schematic.command schematic-diff
    printf '*.litematic diff=schematic\\n' >> ~/.config/git/attributes

    Every extension listed above needs its own line in that attributes file.
    Files without one keep git's normal text diff.

ENVIRONMENT:
    SCHEMATIC_DIFF_KITTY=1|0  Force image output on or off
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("schematic-diff: {error}");
            ExitCode::FAILURE
        }
    }
}

/// A parsed command line.
struct Invocation {
    /// The repo-relative path git reported, used as the heading.
    display_path: String,
    before: Option<PathBuf>,
    after: Option<PathBuf>,
    camera: Camera,
    pack: Option<PathBuf>,
    /// `Some(true)` forces the image on, `Some(false)` off, `None` detects.
    kitty: Option<bool>,
    output: Option<PathBuf>,
}

impl Invocation {
    /// The two sides' file names, as panel captions.
    ///
    /// Both sides come from one git path, so these are equal there; comparing
    /// two files by hand they are each file's own name.
    fn labels(&self) -> (String, String) {
        let name = |path: &Option<PathBuf>| {
            path.as_ref().map(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned())
            })
        };
        let before = name(&self.before);
        let after = name(&self.after);
        (
            before
                .clone()
                .unwrap_or_else(|| after.clone().unwrap_or_default()),
            after.or(before).unwrap_or_default(),
        )
    }
}

fn run() -> Result<()> {
    let Some(invocation) = parse_args(std::env::args().skip(1).collect())? else {
        return Ok(());
    };

    let before = load_side(invocation.before.as_deref())?;
    let after = load_side(invocation.after.as_deref())?;
    if before.is_none() && after.is_none() {
        return Err(Error::message(format!(
            "both sides of {} are missing",
            invocation.display_path
        )));
    }

    // A pack that will not load is a warning, not a failure: the diff is still
    // worth showing in the built-in colours.
    let pack = match &invocation.pack {
        Some(path) => match Pack::open(path) {
            Ok(pack) => Some(pack),
            Err(error) => {
                report(format_args!(
                    "--pack {} could not be used ({error}); using the built-in colours",
                    path.display()
                ));
                None
            }
        },
        None => None,
    };

    // With a pack the build is meshed — real geometry, real textures; without
    // one it is a grid of coloured cubes. Two paths, but they share a camera
    // and a framing, so swapping between them is not a change of viewpoint.
    let before_mesh = match (&pack, &before) {
        (Some(pack), Some(loaded)) => Some(pack.mesh(&loaded.schematic)?),
        _ => None,
    };
    let after_mesh = match (&pack, &after) {
        (Some(pack), Some(loaded)) => Some(pack.mesh(&loaded.schematic)?),
        _ => None,
    };
    let meshed = before_mesh.is_some() || after_mesh.is_some();

    let (before_grid, after_grid) = if meshed {
        (None, None)
    } else {
        (grid_of(&before)?, grid_of(&after)?)
    };

    // One frame for every panel: framing each side to its own extent would
    // rescale the two independently and hide the size change being inspected.
    let frame = if meshed {
        [before_mesh.as_ref(), after_mesh.as_ref()]
            .into_iter()
            .flatten()
            .map(mesh_bounds)
            .reduce(Bounds::union)
    } else {
        [before_grid.as_ref(), after_grid.as_ref()]
            .into_iter()
            .flatten()
            .map(Grid::bounds)
            .reduce(Bounds::union)
    }
    .unwrap_or_default();

    let changes = match (&before, &after) {
        (Some(before), Some(after)) => Some(diff(
            &before.schematic,
            &after.schematic,
            &DiffSpec::from_preset(FingerprintSpec::exact()),
        )),
        _ => None,
    };
    let categories = change_categories(changes.as_ref());

    // The changes panel, in the form this run's renderer draws it: a grid of
    // coloured blocks without a pack, one mesh per category with one.
    let flat_changes = (pack.is_none() && !categories.is_empty())
        .then(|| changes_grid(frame, &categories))
        .transpose()?;
    let meshed_changes = match &pack {
        Some(pack) => categories
            .iter()
            .map(|category| {
                Ok(TintedMesh {
                    mesh: pack.mesh(&subset(&category.cells))?,
                    color: category.color,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };

    let mut output = Output::detect();
    let draw_image = output.is_terminal() && kitty::supported(invocation.kitty);

    // The image occupies everything above the text, so that is the area it is
    // laid out for. One spare row keeps the last text line from scrolling the
    // image, since a placed image scrolls with the text it sits over.
    let text_lines = summary_lines(&before, &after, pack.is_some());
    let image_cells = output.screen.rows.saturating_sub(text_lines + 1).max(8);
    let (width, height) = output.image_pixels(image_cells);
    let columns = output.screen.columns;

    // One `--pack` decides the renderer for the whole run, and a side that does
    // not exist simply has no mesh — so the choice keys off whether the run is
    // meshed at all, not off whether both sides produced one.
    let scene = if meshed {
        Scene::Meshed {
            before: before_mesh.as_ref(),
            after: after_mesh.as_ref(),
            changes: &meshed_changes,
        }
    } else {
        Scene::Flat {
            before: before_grid.as_ref(),
            after: after_grid.as_ref(),
            changes: flat_changes.as_ref(),
        }
    };
    let image = compose_image(
        &invocation,
        &scene,
        Layout {
            width,
            height,
            cell_px: output.screen.cell_px,
            frame,
        },
    );

    let png = raster::encode_png(&image)?;
    if let Some(path) = &invocation.output {
        std::fs::write(path, &png)?;
    }

    // The heading names the file the picture belongs to, so it goes above it.
    write_heading(&mut output, &invocation, &before, &after)?;

    if draw_image {
        // The image is placed over `image_cells` rows without moving the
        // cursor, so the text below starts after those rows.
        kitty::write_png(&mut output, &png, columns, image_cells)?;
        output.write_all(b"\n".repeat(image_cells as usize).as_slice())?;
    }

    write_details(
        &mut output,
        invocation.pack.as_ref(),
        &before,
        &after,
        &changes,
        pack.as_ref(),
    )?;
    output.flush()?;

    if !draw_image {
        report_once(format_args!("{}", no_image_reason(&invocation, &output)));
    }
    Ok(())
}

/// Report a problem with this run's own arguments.
///
/// Emitted for every file git asks about. A flag the user wrote is their
/// instruction, and a run that quietly ignored it would be the worst outcome —
/// worse than the same line repeated once per changed file. Reporting per file
/// is also what a multi-file tool normally does, and the degradation is
/// genuinely per file: every one of them renders without the pack.
fn report(message: std::fmt::Arguments<'_>) {
    eprintln!("schematic-diff: {message}");
}

/// Report a condition that is the same for every file of a `git diff`.
///
/// Git runs an external diff once per changed path, so an environment note
/// repeated per file adds nothing that the first one did not already say.
fn report_once(message: std::fmt::Arguments<'_>) {
    if std::env::var("GIT_DIFF_PATH_COUNTER").is_ok_and(|counter| counter != "1") {
        return;
    }
    report(message);
}

/// The diff, in the form it will be drawn.
///
/// One `--pack` decides the renderer for the whole run, so a diff is either all
/// flat or all meshed and never a mixture. Splitting the two representations
/// into separate variants says that in the type rather than in a comment.
///
/// The changed cells are not a variant of this: both renderers put them in the
/// same three-panel layout, so they are carried alongside the scene instead.
enum Scene<'a> {
    Flat {
        before: Option<&'a Grid>,
        after: Option<&'a Grid>,
        /// The changed blocks as a grid, for the changes panel.
        changes: Option<&'a Grid>,
    },
    Meshed {
        before: Option<&'a nucleation::meshing::MeshOutput>,
        after: Option<&'a nucleation::meshing::MeshOutput>,
        /// The changed blocks, one mesh per category, for the changes panel.
        changes: &'a [TintedMesh],
    },
}

/// The extent of a meshed build, rounded out to whole cells.
fn mesh_bounds(output: &nucleation::meshing::MeshOutput) -> Bounds {
    let min = output.bounds.min;
    let max = output.bounds.max;
    // The mesh's extent is in world units; the camera frames whole cells, so
    // the bounds are rounded out to the cells the geometry occupies.
    Bounds {
        min: [
            min[0].floor() as i32,
            min[1].floor() as i32,
            min[2].floor() as i32,
        ],
        max: [
            max[0].ceil() as i32 - 1,
            max[1].ceil() as i32 - 1,
            max[2].ceil() as i32 - 1,
        ],
    }
}

/// One category of change: the changed cells, and the colour they are shown in.
struct Category<'a> {
    color: u32,
    /// Each changed cell with the block to draw in it. For a removal that is the
    /// block that used to be there; every other category shows the block that is
    /// there now, which is what the change produced.
    ///
    /// Taken from the diff rather than looked up in either build, because the
    /// diff reports every cell in the after build's frame — the same frame the
    /// camera and the two panels beside this one are working in.
    cells: Vec<((i32, i32, i32), &'a BlockState)>,
}

/// The changes, grouped into the categories they are drawn in.
///
/// Grouping is what lets each category be drawn in its own colour: a category is
/// meshed on its own, and the colour is applied as the panel is rasterized, so
/// the renderer never has to work out which category a triangle belongs to.
fn change_categories(changes: Option<&Diff>) -> Vec<Category<'_>> {
    let Some(changes) = changes else {
        // A change is defined against both sides, so a diff of an added or a
        // deleted file has none to show.
        return Vec::new();
    };
    let category = |color, cells| Category { color, cells };
    [
        category(
            ADDED_COLOR,
            changes
                .added
                .iter()
                .map(|(position, block)| (*position, block))
                .collect(),
        ),
        category(
            REMOVED_COLOR,
            changes
                .removed
                .iter()
                .map(|(position, block)| (*position, block))
                .collect(),
        ),
        category(
            CHANGED_COLOR,
            changes
                .changed
                .iter()
                .map(|(position, _, now)| (*position, now))
                .collect(),
        ),
        category(
            SWAPPED_COLOR,
            changes
                .swapped
                .iter()
                .map(|(position, _, now)| (*position, now))
                .collect(),
        ),
    ]
    .into_iter()
    .filter(|category| !category.cells.is_empty())
    .collect()
}

/// The changed blocks as a grid, for the flat renderer's changes panel.
///
/// Each cell holds the colour of the block that changed, with the category's
/// colour laid over it — the same treatment the meshed panels give their
/// textures, and what keeps the two renderers showing the same thing.
///
/// The grid spans the union extent, because a removal can sit outside the after
/// build's own bounds: that is precisely what a shrunken build looks like, and
/// it has to stay visible.
fn changes_grid(frame: Bounds, categories: &[Category<'_>]) -> Result<Grid> {
    let mut grid = Grid::empty_over(frame)?;
    for category in categories {
        for ((x, y, z), block) in &category.cells {
            let block = rgb(block_color(block));
            grid.set(*x, *y, *z, pack(tint(block, rgb(category.color))));
        }
    }
    Ok(grid)
}

/// One category's cells as a schematic of their own, ready to mesh.
///
/// The mesher draws exactly what it is given, so handing it a category's cells
/// and nothing else is what keeps the changes panel to the blocks that changed.
fn subset(cells: &[((i32, i32, i32), &BlockState)]) -> UniversalSchematic {
    let mut subset = UniversalSchematic::new(String::new());
    for ((x, y, z), block) in cells {
        subset.set_block(*x, *y, *z, block);
    }
    subset
}

/// Rasterize one side, if it exists.
fn grid_of(side: &Option<Loaded>) -> Result<Option<Grid>> {
    match side {
        Some(loaded) => Ok(Some(Grid::from_schematic(&loaded.schematic)?)),
        None => Ok(None),
    }
}

fn load_side(path: Option<&Path>) -> Result<Option<Loaded>> {
    match path {
        Some(path) => Ok(Some(format::load(path)?)),
        None => Ok(None),
    }
}

/// Choose the panels and composite them.
///
/// The layout is the same in both renderers — before, after, then the changed
/// blocks — so only what each panel is drawn from differs, which is exactly the
/// difference [`Scene`] carries.
fn compose_image(invocation: &Invocation, scene: &Scene<'_>, layout: Layout) -> raster::Canvas {
    let (before_name, after_name) = invocation.labels();
    let (before, after, changes) = match *scene {
        Scene::Flat {
            before,
            after,
            changes,
        } => (
            before.map(Picture::Flat),
            after.map(Picture::Flat),
            changes.map(Picture::Flat),
        ),
        Scene::Meshed {
            before,
            after,
            changes,
        } => (
            before.map(Picture::Mesh),
            after.map(Picture::Mesh),
            (!changes.is_empty()).then_some(Picture::Changes(changes)),
        ),
    };

    let mut panels: Vec<Panel<'_>> = Vec::new();
    match (before, after) {
        (Some(before), Some(after)) => {
            panels.push(Panel::new(
                format!("BEFORE  {before_name}"),
                CAPTION_TEXT,
                before,
            ));
            // The after panel carries the note: it is the side the diff is read
            // against, and the one a reader looks at when nothing is tinted.
            let note = if changes.is_some() { "" } else { "  (no changes)" };
            panels.push(Panel::new(
                format!("AFTER  {after_name}{note}"),
                CAPTION_TEXT,
                after,
            ));
            // Two panels are always drawn, so the comparison survives a narrow
            // window; the changes panel is the one that can go.
            if let Some(changes) = changes
                && raster::fits(layout.width, 3)
            {
                panels.push(Panel::legend(change_legend(), changes));
            }
        }
        (None, Some(after)) => panels.push(Panel::new(
            format!("ADDED  {after_name}"),
            ADDED_TEXT,
            after,
        )),
        (Some(before), None) => panels.push(Panel::new(
            format!("DELETED  {before_name}"),
            REMOVED_TEXT,
            before,
        )),
        (None, None) => {}
    }

    raster::compose(&panels, &invocation.camera, &layout, &Theme::default())
}

/// The caption of the changes panel, one segment per category, each drawn in
/// the colour of the blocks it names.
fn change_legend() -> Vec<Segment> {
    vec![
        Segment::new("CHANGES  ", CAPTION_TEXT),
        Segment::new("+added  ", rgb(ADDED_COLOR)),
        Segment::new("-removed  ", rgb(REMOVED_COLOR)),
        Segment::new("*changed  ", rgb(CHANGED_COLOR)),
        Segment::new("#re-paletted", rgb(SWAPPED_COLOR)),
    ]
}

/// How many lines the summary prints: the heading, one line per side that
/// loaded, the change tally, and the pack line when one was given.
fn summary_lines(before: &Option<Loaded>, after: &Option<Loaded>, pack: bool) -> u32 {
    2 + u32::from(before.is_some()) + u32::from(after.is_some()) + u32::from(pack)
}

/// The line naming the path and what happened to it.
fn write_heading(
    out: &mut impl Write,
    invocation: &Invocation,
    before: &Option<Loaded>,
    after: &Option<Loaded>,
) -> std::io::Result<()> {
    let status = match (before.is_some(), after.is_some()) {
        (false, true) => "added",
        (true, false) => "deleted",
        _ => "modified",
    };
    writeln!(out, "{}  [{status}]", invocation.display_path)
}

/// What each side held, what changed between them, and what the pack contained.
fn write_details(
    out: &mut impl Write,
    pack_path: Option<&PathBuf>,
    before: &Option<Loaded>,
    after: &Option<Loaded>,
    changes: &Option<Diff>,
    pack: Option<&Pack>,
) -> std::io::Result<()> {
    if let Some(loaded) = before {
        writeln!(out, "  before  {}", describe(loaded))?;
    }
    if let Some(loaded) = after {
        writeln!(out, "  after   {}", describe(loaded))?;
    }

    match changes {
        Some(changes) => writeln!(
            out,
            "  changes +{} added, -{} removed, ~{} changed, {} re-paletted \
             ({}% of cells aligned)",
            changes.added.len(),
            changes.removed.len(),
            changes.changed.len(),
            changes.swapped.len(),
            (changes.support * 100.0).round() as i32,
        )?,
        None => writeln!(out, "  changes not computed (only one side exists)")?,
    }

    // Reported because `--pack` is the one argument whose failure is silent: a
    // pack that loads but covers nothing renders every block in the fallback
    // colour, which is indistinguishable from the flag being ignored. These
    // counts say the pack was read and has content.
    if let Some(pack) = pack {
        let stats = pack.stats();
        let path = pack_path
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        writeln!(
            out,
            "  pack    {} blockstates, {} models, {} textures  [{path}]",
            stats.blockstate_count, stats.model_count, stats.texture_count,
        )?;
    }
    Ok(())
}

/// One side's size, block count and provenance, for the summary.
fn describe(loaded: &Loaded) -> String {
    let schematic = &loaded.schematic;
    let (x, y, z) = schematic.get_tight_dimensions();
    let regions = schematic.get_region_names().len();
    let blocks = schematic
        .iter_blocks()
        .filter(|(_, block)| !render::is_air(block))
        .count();
    format!(
        "{x}x{y}x{z}  {blocks} blocks  {regions} region{}{}  [{}]",
        if regions == 1 { "" } else { "s" },
        match loaded.entities {
            0 => String::new(),
            1 => "  1 entity".to_string(),
            count => format!("  {count} entities"),
        },
        loaded.format,
    )
}

/// Why no image was drawn, phrased as what to do about it.
fn no_image_reason(invocation: &Invocation, output: &Output) -> String {
    if let Some(path) = &invocation.output {
        return format!("no image drawn; it was written to {}", path.display());
    }
    if !output.is_terminal() {
        return "no terminal to draw in; pass --output=<file> to save the image, or run this \
                under kitty, Ghostty or Konsole"
            .to_string();
    }
    "the terminal was not recognised as speaking the kitty graphics protocol; pass --kitty \
     to draw anyway"
        .to_string()
}

/// Parse the command line, returning `None` for `--help`/`--version`.
fn parse_args(args: Vec<String>) -> Result<Option<Invocation>> {
    let mut positional: Vec<String> = Vec::new();
    let mut camera = Camera::default();
    let mut pack = None;
    let mut kitty = None;
    let mut output = None;

    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("schematic-diff {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--kitty" => {
                kitty = Some(true);
                continue;
            }
            "--no-kitty" => {
                kitty = Some(false);
                continue;
            }
            _ => {}
        }
        if let Some(value) = arg.strip_prefix("--yaw=") {
            camera.yaw_deg = number(value, "--yaw")?;
        } else if let Some(value) = arg.strip_prefix("--pitch=") {
            camera.pitch_deg = number(value, "--pitch")?;
        } else if let Some(value) = arg.strip_prefix("--zoom=") {
            let zoom = number(value, "--zoom")?;
            if zoom <= 0.0 {
                return Err(Error::message("--zoom must be greater than zero"));
            }
            camera.zoom = zoom;
        } else if let Some(value) = arg.strip_prefix("--pack=") {
            pack = Some(expand_tilde(value));
        } else if let Some(value) = arg.strip_prefix("--output=") {
            output = Some(expand_tilde(value));
        } else if arg.starts_with('-') && arg.len() > 1 && !Path::new(&arg).exists() {
            return Err(Error::message(format!("unknown option {arg:?}")));
        } else {
            positional.push(arg);
        }
    }

    let (display_path, before, after) = match positional.len() {
        // Git's seven arguments. `git diff --no-index` appends two more — the
        // second path and the index line — and the leading seven keep their
        // meaning, so they are read the same way.
        7 | 9 => {
            let mut values = positional.into_iter();
            let path = values.next().unwrap();
            let old_file = values.next().unwrap();
            // Skip `<old-hex>` and `<old-mode>`.
            let new_file = values.nth(2).unwrap();
            (path, side_path(&old_file), side_path(&new_file))
        }
        count => {
            return Err(Error::message(format!(
                "expected 7 arguments from git, got {count}\n\n\
                 This tool is an external diff: git calls it, it is not called \
                 directly. To compare one file across two revisions:\n    \
                 git diff <rev1>:<path> <rev2>:<path>\n\
                 To compare two files on disk:\n    \
                 git diff --no-index <before> <after>"
            )));
        }
    };

    for side in [&before, &after].into_iter().flatten() {
        if side.is_dir() {
            return Err(Error::message(format!(
                "{} is a directory; comparing two directories is not supported",
                side.display()
            )));
        }
    }

    Ok(Some(Invocation {
        display_path,
        before,
        after,
        camera,
        pack,
        kitty,
        output,
    }))
}

/// Expand a leading `~` in a user-supplied path.
///
/// The command line usually reaches this tool from a git config string, where
/// `--pack=~/pack.zip` looks right but arrives literally: a shell only expands
/// `~` at the start of a word, and here it follows the `=`. Expanding it here is
/// what lets a config keep the home-relative path a person would write.
fn expand_tilde(raw: &str) -> PathBuf {
    let rest = raw.strip_prefix("~/").or_else(|| raw.strip_prefix('~'));
    match (rest, std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => {
            let mut path = PathBuf::from(home);
            if !rest.is_empty() {
                path.push(rest);
            }
            path
        }
        _ => PathBuf::from(raw),
    }
}

fn number(value: &str, flag: &str) -> Result<f32> {
    value
        .trim()
        .parse::<f32>()
        .map_err(|_| Error::message(format!("{flag} expects a number, got {value:?}")))
}

/// Treat git's `/dev/null` placeholder as "this side does not exist".
fn side_path(raw: &str) -> Option<PathBuf> {
    (raw != DEV_NULL).then(|| PathBuf::from(raw))
}
