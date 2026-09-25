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

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rayon::prelude::*;
use bpaf::{Bpaf, Parser, construct, long};
use nucleation::diff::{Diff, DiffSpec, diff};
use nucleation::fingerprint::FingerprintSpec;
use nucleation::{BlockState, Region, UniversalSchematic};

use crate::error::{Error, Result};
use crate::format::Loaded;
use crate::mesh::TintedMesh;
use crate::pack::Pack;
use crate::raster::{Layout, Panel, Picture, Segment, Theme, pack, rgb, tint};
use crate::render::{Bounds, Camera, Grid, block_color};
use crate::terminal::Output;

const DEV_NULL: &str = "/dev/null";

fn parse_angle(flag: &'static str) -> impl Fn(String) -> Result<f32, String> + Clone {
    move |raw: String| {
        raw.trim()
            .parse::<f32>()
            .map_err(|_| format!("{flag} expects a number, got {raw:?}"))
    }
}

fn parse_zoom(raw: String) -> Result<f32, String> {
    parse_angle("--zoom")(raw)
        .and_then(|zoom| {
            (zoom > 0.0)
                .then_some(zoom)
                .ok_or_else(|| "--zoom must be greater than zero".to_string())
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatMode {
    /// Print all blocks without truncation.
    Uncapped,
    /// Print at most `count` entries, noting the remainder.
    Capped(usize),
}

fn stat_parser() -> impl Parser<Option<StatMode>> {
    let with_val = long("stat")
        .help("print a block-level breakdown of materials or changes (optional COUNT limit)")
        .argument::<usize>("COUNT")
        .map(StatMode::Capped)
        .map(Some);

    let just_flag = long("stat")
        .help("print a block-level breakdown of materials or changes")
        .req_flag(Some(StatMode::Uncapped));

    construct!([with_val, just_flag]).fallback(None)
}

#[derive(Clone, Debug, Bpaf)]
#[bpaf(options, version(env!("CARGO_PKG_VERSION")))]
/// render Minecraft schematic diffs in the terminal
struct Cli {
    /// camera horizontal angle, in degrees
    #[bpaf(
        long("yaw"),
        argument::<String>("DEGREES"),
        parse(parse_angle("--yaw")),
        fallback(Camera::default().yaw_deg),
        display_fallback
    )]
    yaw: f32,
    /// camera elevation above the build, in degrees
    #[bpaf(
        long("pitch"),
        argument::<String>("DEGREES"),
        parse(parse_angle("--pitch")),
        fallback(Camera::default().pitch_deg),
        display_fallback
    )]
    pitch: f32,
    /// camera zoom; larger zooms in
    #[bpaf(
        long("zoom"),
        argument::<String>("FACTOR"),
        parse(parse_zoom),
        fallback(Camera::default().zoom),
        display_fallback
    )]
    zoom: f32,
    /// take block colours from a resource pack
    #[bpaf(long("pack"), argument("PACK"))]
    pack: Option<PathBuf>,
    /// draw the image even if the terminal is unrecognised
    #[bpaf(switch)]
    kitty: bool,
    /// never draw the image, print the text summary only
    #[bpaf(long("no-kitty"), switch)]
    no_kitty: bool,
    /// also write the composited image as a PNG
    #[bpaf(long("output"), argument("FILE"))]
    output: Option<PathBuf>,
    /// print a block-level breakdown of materials or changes
    #[bpaf(external(stat_parser))]
    stat: Option<StatMode>,
    /// show the version and exit
    #[bpaf(long("version"), short('V'), switch, hide)]
    version: bool,
    /// schematic files to inspect or diff, or git's seven per-path arguments
    #[bpaf(positional("PATH"))]
    rest: Vec<OsString>,
}

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

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(Error::HelpExit(failure)) => {
            failure.print_message(100);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("schematic-diff: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Invocation {
    display_path: String,
    before: Option<PathBuf>,
    after: Option<PathBuf>,
    camera: Camera,
    pack: Option<PathBuf>,
    /// `Some(true)` forces the image on, `Some(false)` off, `None` detects.
    kitty: Option<bool>,
    output: Option<PathBuf>,
    stat: Option<StatMode>,
    standalone: bool,
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
    let Some(invocation) = parse_args()? else {
        return Ok(());
    };

    let ((before, after), pack) = rayon::join(
        || {
            rayon::join(
                || load_side(invocation.before.as_deref()),
                || load_side(invocation.after.as_deref()),
            )
        },
        || {
            match &invocation.pack {
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
            }
        },
    );
    let before = before?;
    let after = after?;
    if before.is_none() && after.is_none() {
        return Err(Error::message(format!(
            "both sides of {} are missing",
            invocation.display_path
        )));
    }

    let changes = match (&before, &after) {
        (Some(before), Some(after)) => Some(diff(
            &before.schematic,
            &after.schematic,
            &DiffSpec::from_preset(FingerprintSpec::exact()),
        )),
        _ => None,
    };
    let categories = change_categories(changes.as_ref());

    let mut output = Output::detect();
    let draw_image = output.is_terminal() && kitty::supported(invocation.kitty);

    let stat = match invocation.stat {
        Some(mode) => format_stat(&before, &after, changes.as_ref(), invocation.standalone, mode),
        None => Vec::new(),
    };

    let rendered = if invocation.kitty == Some(false) {
        None
    } else {
        match render_scene(
            &invocation,
            &before,
            &after,
            pack.as_ref(),
            &categories,
            &output,
            stat.len(),
        ) {
            Ok(rendered) => Some(rendered),
            Err(error) => {
                report(format_args!("{error}"));
                None
            }
        }
    };

    if let Some(rendered) = &rendered
        && let Some(path) = &invocation.output
    {
        std::fs::write(path, &rendered.png)?;
    }

    write_heading(&mut output, &invocation, &before, &after)?;

    if let Some(rendered) = &rendered
        && draw_image
    {
        let columns = output.screen.columns;
        // The image is placed over `image_cells` rows without moving the
        // cursor, so the text below starts after those rows.
        kitty::write_png(&mut output, &rendered.png, columns, rendered.image_cells)?;
        output.write_all(b"\n".repeat(rendered.image_cells as usize).as_slice())?;
    }

    write_details(
        &mut output,
        &invocation,
        &before,
        &after,
        &changes,
        pack.as_ref(),
        &stat,
    )?;
    output.flush()?;

    if rendered.is_some() && !draw_image {
        report_once(format_args!("{}", no_image_reason(&invocation, &output)));
    }
    Ok(())
}

#[derive(Debug)]
struct Rendered {
    png: Vec<u8>,
    image_cells: u32,
}

fn render_scene(
    invocation: &Invocation,
    before: &Option<Loaded>,
    after: &Option<Loaded>,
    pack: Option<&Pack>,
    categories: &[Category<'_>],
    output: &Output,
    stat_len: usize,
) -> Result<Rendered> {
    let (before_mesh, after_mesh) = rayon::join(
        || match (pack, before) {
            (Some(pack), Some(loaded)) => pack.mesh(&loaded.schematic).map(Some),
            _ => Ok(None),
        },
        || match (pack, after) {
            (Some(pack), Some(loaded)) => pack.mesh(&loaded.schematic).map(Some),
            _ => Ok(None),
        },
    );
    let before_mesh = before_mesh?;
    let after_mesh = after_mesh?;
    let meshed = before_mesh.is_some() || after_mesh.is_some();

    let (before_grid, after_grid) = if meshed {
        (None, None)
    } else {
        let (bg, ag) = rayon::join(|| grid_of(before), || grid_of(after));
        (bg?, ag?)
    };

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

    let flat_changes = (pack.is_none() && !categories.is_empty())
        .then(|| changes_grid(frame, categories))
        .transpose()?;
    let meshed_changes = match pack {
        Some(pack) => categories
            .par_iter()
            .map(|category| {
                Ok(TintedMesh {
                    mesh: pack.mesh(&schematic_from_cells(&category.cells))?,
                    color: category.color,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };

    let text_lines = summary_lines(
        before,
        after,
        pack.is_some(),
        stat_len as u32,
        invocation.standalone,
    );
    // Reserve one spare row so trailing text does not scroll the placed image.
    let fit_height = output.screen.rows.saturating_sub(text_lines + 1);
    let default_height = (output.screen.rows.saturating_sub(5)).max(18);
    let image_cells = if fit_height >= 12 {
        fit_height
    } else {
        default_height
    };
    let (width, height) = output.image_pixels(image_cells);

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
        invocation,
        &scene,
        Layout {
            width,
            height,
            cell_px: output.screen.cell_px,
            frame,
        },
    );

    let png = raster::encode_png(&image)?;
    Ok(Rendered { png, image_cells })
}

/// Report a run-specific argument error. Git invokes this once per changed path.
fn report(message: std::fmt::Arguments<'_>) {
    eprintln!("schematic-diff: {message}");
}

/// Report a run-wide condition once, even when Git invokes the tool per path.
fn report_once(message: std::fmt::Arguments<'_>) {
    if std::env::var("GIT_DIFF_PATH_COUNTER").is_ok_and(|counter| counter != "1") {
        return;
    }
    report(message);
}

/// A run uses one renderer for every panel: flat or meshed.
///
/// Both representations keep the same before, after, and optional changes
/// layout; the variants differ only in their panel sources.
enum Scene<'a> {
    Flat {
        before: Option<&'a Grid>,
        after: Option<&'a Grid>,
        changes: Option<&'a Grid>,
    },
    Meshed {
        before: Option<&'a nucleation::meshing::MeshOutput>,
        after: Option<&'a nucleation::meshing::MeshOutput>,
        changes: &'a [TintedMesh],
    },
}

/// Bounding box in integer cell coordinates, rounded out from mesh bounds.
fn mesh_bounds(output: &nucleation::meshing::MeshOutput) -> Bounds {
    let min = output.bounds.min;
    let max = output.bounds.max;
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

struct Category<'a> {
    color: u32,
    /// Each changed cell with the block to draw in it. Removals use the block
    /// from the before build; other categories use the block from the after build.
    ///
    /// Taken from the diff rather than looked up in either build, because the
    /// diff reports every cell in the after build's frame — the same frame the
    /// camera and the two panels beside this one are working in.
    cells: Vec<((i32, i32, i32), &'a BlockState)>,
}

/// Group changes by category so each category can carry its own color through
/// the renderer.
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

/// Build the changes grid over the union extent so removals outside the after
/// build remain visible.
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

fn schematic_from_cells(cells: &[((i32, i32, i32), &BlockState)]) -> UniversalSchematic {
    let mut schematic = UniversalSchematic::new(String::new());
    for ((x, y, z), block) in cells {
        schematic.set_block(*x, *y, *z, block);
    }
    schematic
}

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

/// Convert the scene representation into before, after, and optional changes
/// panels.
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
        (None, Some(after)) => {
            let (label, color) = if invocation.standalone {
                (after_name, CAPTION_TEXT)
            } else {
                (format!("ADDED  {after_name}"), ADDED_TEXT)
            };
            panels.push(Panel::new(label, color, after));
        }
        (Some(before), None) => {
            let (label, color) = if invocation.standalone {
                (before_name, CAPTION_TEXT)
            } else {
                (format!("DELETED  {before_name}"), REMOVED_TEXT)
            };
            panels.push(Panel::new(label, color, before));
        }
        (None, None) => {}
    }

    raster::compose(&panels, &invocation.camera, &layout, &Theme::default())
}

fn change_legend() -> Vec<Segment> {
    vec![
        Segment::new("CHANGES  ", CAPTION_TEXT),
        Segment::new("+added  ", rgb(ADDED_COLOR)),
        Segment::new("-removed  ", rgb(REMOVED_COLOR)),
        Segment::new("*changed  ", rgb(CHANGED_COLOR)),
        Segment::new("#re-paletted", rgb(SWAPPED_COLOR)),
    ]
}

/// Count summary lines to reserve the terminal rows occupied by text.
fn summary_lines(
    before: &Option<Loaded>,
    after: &Option<Loaded>,
    pack: bool,
    stat_lines: u32,
    standalone: bool,
) -> u32 {
    let base = if standalone && (before.is_none() || after.is_none()) {
        2
    } else {
        2 + u32::from(before.is_some()) + u32::from(after.is_some())
    };
    base + u32::from(pack) + stat_lines
}

fn write_heading(
    out: &mut impl Write,
    invocation: &Invocation,
    before: &Option<Loaded>,
    after: &Option<Loaded>,
) -> std::io::Result<()> {
    if invocation.standalone && (before.is_none() || after.is_none()) {
        return writeln!(out, "{}", invocation.display_path);
    }
    let status = match (before.is_some(), after.is_some()) {
        (false, true) => "added",
        (true, false) => "deleted",
        _ => "modified",
    };
    writeln!(out, "{}  [{status}]", invocation.display_path)
}

fn write_details(
    out: &mut impl Write,
    invocation: &Invocation,
    before: &Option<Loaded>,
    after: &Option<Loaded>,
    changes: &Option<Diff>,
    pack: Option<&Pack>,
    stat: &[String],
) -> std::io::Result<()> {
    if invocation.standalone && (before.is_none() || after.is_none()) {
        let loaded = after.as_ref().or(before.as_ref()).unwrap();
        writeln!(out, "  build   {}", describe(loaded))?;
    } else {
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
    }

    if let Some(pack) = pack {
        let stats = pack.stats();
        let path = invocation
            .pack
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        writeln!(
            out,
            "  pack    {} blockstates, {} models, {} textures  [{path}]",
            stats.blockstate_count, stats.model_count, stats.texture_count,
        )?;
    }

    for line in stat {
        writeln!(out, "{line}")?;
    }
    Ok(())
}

fn format_stat(
    before: &Option<Loaded>,
    after: &Option<Loaded>,
    changes: Option<&Diff>,
    standalone: bool,
    mode: StatMode,
) -> Vec<String> {
    let mut lines = Vec::new();
    match (before, after) {
        (Some(_), Some(_)) => {
            let Some(changes) = changes else {
                lines.push("  stat".to_string());
                lines.push("    (no changes)".to_string());
                return lines;
            };

            let (added_counts, (removed_counts, changed_counts)) = rayon::join(
                || {
                    let mut counts: HashMap<&str, usize> = HashMap::new();
                    for (_, block) in &changes.added {
                        *counts.entry(block.get_name()).or_default() += 1;
                    }
                    counts
                },
                || {
                    rayon::join(
                        || {
                            let mut counts: HashMap<&str, usize> = HashMap::new();
                            for (_, block) in &changes.removed {
                                *counts.entry(block.get_name()).or_default() += 1;
                            }
                            counts
                        },
                        || {
                            let mut counts: HashMap<(&str, Option<&str>), usize> = HashMap::new();
                            for (_, was, now) in changes.changed.iter().chain(changes.swapped.iter()) {
                                let was_name = was.get_name();
                                let now_name = now.get_name();
                                let key = if was_name == now_name {
                                    (was_name, None)
                                } else {
                                    (was_name, Some(now_name))
                                };
                                *counts.entry(key).or_default() += 1;
                            }
                            counts
                        },
                    )
                },
            );

            enum StatEntry<'a> {
                Added { count: usize, name: &'a str },
                Removed { count: usize, name: &'a str },
                Modified {
                    count: usize,
                    was: &'a str,
                    now: Option<&'a str>,
                },
            }

            let mut items: Vec<StatEntry<'_>> = Vec::new();
            for (name, count) in added_counts {
                items.push(StatEntry::Added { count, name });
            }
            for (name, count) in removed_counts {
                items.push(StatEntry::Removed { count, name });
            }
            for ((was, now), count) in changed_counts {
                items.push(StatEntry::Modified { count, was, now });
            }

            if items.is_empty() {
                lines.push("  stat".to_string());
                lines.push("    (no block differences)".to_string());
                return lines;
            }

            items.sort_unstable_by(|a, b| {
                let (count_a, label_a) = match a {
                    StatEntry::Added { count, name } => (*count, *name),
                    StatEntry::Removed { count, name } => (*count, *name),
                    StatEntry::Modified { count, was, .. } => (*count, *was),
                };
                let (count_b, label_b) = match b {
                    StatEntry::Added { count, name } => (*count, *name),
                    StatEntry::Removed { count, name } => (*count, *name),
                    StatEntry::Modified { count, was, .. } => (*count, *was),
                };
                count_b.cmp(&count_a).then_with(|| label_a.cmp(label_b))
            });

            lines.push("  stat".to_string());
            let max_count_len = items
                .iter()
                .map(|item| match item {
                    StatEntry::Added { count, .. }
                    | StatEntry::Removed { count, .. }
                    | StatEntry::Modified { count, .. } => count.to_string().len(),
                })
                .max()
                .unwrap_or(1);

            let (visible, remaining) = match mode {
                StatMode::Uncapped => (&items[..], 0),
                StatMode::Capped(cap) => {
                    if items.len() <= cap {
                        (&items[..], 0)
                    } else {
                        (&items[..cap], items.len() - cap)
                    }
                }
            };

            for item in visible {
                let line = match item {
                    StatEntry::Added { count, name } => {
                        format!("    + {:>width$}  {name}", count, width = max_count_len)
                    }
                    StatEntry::Removed { count, name } => {
                        format!("    - {:>width$}  {name}", count, width = max_count_len)
                    }
                    StatEntry::Modified {
                        count,
                        was,
                        now: None,
                    } => {
                        format!("    ~ {:>width$}  {was} (state)", count, width = max_count_len)
                    }
                    StatEntry::Modified {
                        count,
                        was,
                        now: Some(now),
                    } => {
                        format!("    ~ {:>width$}  {was} -> {now}", count, width = max_count_len)
                    }
                };
                lines.push(line);
            }

            if remaining > 0 {
                lines.push(format!("    ... and {remaining} more block changes"));
            }
        }
        (None, Some(after)) => {
            let prefix = if standalone { "" } else { "+ " };
            format_palette_stat(&mut lines, after, prefix, mode);
        }
        (Some(before), None) => {
            let prefix = if standalone { "" } else { "- " };
            format_palette_stat(&mut lines, before, prefix, mode);
        }
        (None, None) => {}
    }
    lines
}

fn format_palette_stat(lines: &mut Vec<String>, loaded: &Loaded, prefix: &str, mode: StatMode) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let regions: Vec<&Region> = std::iter::once(&loaded.schematic.default_region)
        .chain(loaded.schematic.other_regions.values())
        .collect();

    let region_counts: Vec<HashMap<String, usize>> = regions
        .par_iter()
        .map(|region| {
            let palette = region.get_palette();
            let names: Vec<(&str, bool)> = palette
                .iter()
                .map(|b| (b.get_name(), render::is_air(b)))
                .collect();

            let chunk_counts = region
                .blocks
                .par_chunks(65536)
                .fold(
                    || vec![0usize; palette.len()],
                    |mut acc, chunk| {
                        for &idx in chunk {
                            if idx < acc.len() {
                                acc[idx] += 1;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0usize; palette.len()],
                    |mut a, b| {
                        for (acc, count) in a.iter_mut().zip(b) {
                            *acc += count;
                        }
                        a
                    },
                );

            let mut local_map: HashMap<String, usize> = HashMap::new();
            for (idx, &count) in chunk_counts.iter().enumerate() {
                if idx < names.len() {
                    let (name, is_air) = names[idx];
                    if !is_air && count > 0 {
                        *local_map.entry(name.to_string()).or_default() += count;
                    }
                }
            }
            local_map
        })
        .collect();

    for r_map in region_counts {
        for (name, count) in r_map {
            *counts.entry(name).or_default() += count;
        }
    }

    lines.push("  stat".to_string());
    if counts.is_empty() {
        lines.push("    (empty schematic)".to_string());
        return;
    }

    let mut items: Vec<(String, usize)> = counts.into_iter().collect();
    items.sort_unstable_by(|(name_a, count_a), (name_b, count_b)| {
        count_b.cmp(count_a).then_with(|| name_a.cmp(name_b))
    });

    let max_count_len = items
        .iter()
        .map(|(_, count)| count.to_string().len())
        .max()
        .unwrap_or(1);

    let (visible, remaining) = match mode {
        StatMode::Uncapped => (&items[..], 0),
        StatMode::Capped(cap) => {
            if items.len() <= cap {
                (&items[..], 0)
            } else {
                (&items[..cap], items.len() - cap)
            }
        }
    };

    for (name, count) in visible {
        lines.push(format!(
            "    {prefix}{:>width$}  {name}",
            count,
            width = max_count_len
        ));
    }

    if remaining > 0 {
        lines.push(format!("    ... and {remaining} more block types"));
    }
}
fn count_non_air_blocks_parallel(schematic: &UniversalSchematic) -> usize {
    let regions: Vec<&Region> = std::iter::once(&schematic.default_region)
        .chain(schematic.other_regions.values())
        .collect();

    regions
        .par_iter()
        .map(|region| {
            let palette = region.get_palette();
            let is_air_flags: Vec<bool> = palette.iter().map(render::is_air).collect();

            let chunk_counts = region
                .blocks
                .par_chunks(65536)
                .fold(
                    || vec![0usize; palette.len()],
                    |mut acc, chunk| {
                        for &idx in chunk {
                            if idx < acc.len() {
                                acc[idx] += 1;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0usize; palette.len()],
                    |mut a, b| {
                        for (acc, count) in a.iter_mut().zip(b) {
                            *acc += count;
                        }
                        a
                    },
                );

            chunk_counts
                .iter()
                .enumerate()
                .filter(|(idx, _)| !is_air_flags.get(*idx).copied().unwrap_or(true))
                .map(|(_, count)| count)
                .sum::<usize>()
        })
        .sum()
}

fn describe(loaded: &Loaded) -> String {
    let schematic = &loaded.schematic;
    let (x, y, z) = schematic.get_tight_dimensions();
    let regions = schematic.get_region_names().len();
    let blocks = count_non_air_blocks_parallel(schematic);
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

/// Parse the command line, returning `None` for `--version`.
///
/// Help and parse failures return through [`Error`] so `main` owns output and
/// exit status. The derive handles ordinary options; the positional protocol
/// below handles Git's external-diff arguments.
fn parse_args() -> Result<Option<Invocation>> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let refs: Vec<&str> = args
        .iter()
        .map(|arg| {
            arg.to_str().ok_or_else(|| {
                Error::message(format!("argument is not valid unicode: {}", arg.to_string_lossy()))
            })
        })
        .collect::<Result<_>>()?;
    let cli: Cli = cli().run_inner(&refs[..])?;
    if cli.version {
        println!("schematic-diff {}", env!("CARGO_PKG_VERSION"));
        return Ok(None);
    }
    let camera = Camera {
        yaw_deg: cli.yaw,
        pitch_deg: cli.pitch,
        zoom: cli.zoom,
    };
    let pack = cli.pack.as_deref().map(expand_tilde_path);
    let output = cli.output.as_deref().map(expand_tilde_path);
    // `--kitty --kitty` is rejected by the parser, so both set means one of
    // each: last flag wins, resolved by re-scanning the raw arguments.
    let kitty = match (cli.kitty, cli.no_kitty) {
        (true, true) => last_kitty_flag(&args),
        (true, false) => Some(true),
        (false, true) => Some(false),
        (false, false) => None,
    };

    let (display_path, before, after, standalone) = match cli.rest.len() {
        1 => {
            let path = PathBuf::from(cli.rest.into_iter().next().unwrap());
            let display_path = path.to_string_lossy().into_owned();
            (display_path, None, Some(path), true)
        }
        2 => {
            let mut values = cli.rest.into_iter();
            let old_file = values.next().unwrap();
            let new_file = values.next().unwrap();
            let before = side_path(&old_file);
            let after = side_path(&new_file);
            let display_path = match (&before, &after) {
                (Some(b), Some(a)) => format!("{} -> {}", b.display(), a.display()),
                (Some(b), None) => b.display().to_string(),
                (None, Some(a)) => a.display().to_string(),
                (None, None) => format!("{DEV_NULL} -> {DEV_NULL}"),
            };
            (display_path, before, after, true)
        }
        // Git's seven arguments. `git diff --no-index` appends two more — the
        // second path and the index line — and the leading seven keep their
        // meaning, so they are read the same way.
        7 | 9 => {
            let mut values = cli.rest.into_iter();
            let path = values.next().unwrap();
            let display_path = path.to_string_lossy().into_owned();
            let old_file = values.next().unwrap();
            // Skip `<old-hex>` and `<old-mode>`.
            let new_file = values.nth(2).unwrap();
            (display_path, side_path(&old_file), side_path(&new_file), false)
        }
        count => {
            return Err(Error::message(format!(
                "expected 1 or 2 file paths, or 7 arguments from git, got {count}\n\n\
                 Usage:\n    \
                 schematic-diff <file>                  Inspect a single schematic\n    \
                 schematic-diff <before> <after>        Diff two schematics directly\n    \
                 git diff <rev1>:<path> <rev2>:<path>   Diff revisions via git\n    \
                 git diff --no-index <before> <after>   Diff files via git"
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
        stat: cli.stat,
        standalone,
    }))
}

fn last_kitty_flag(args: &[OsString]) -> Option<bool> {
    args.iter().rev().find_map(|arg| match arg.to_str()? {
        "--kitty" => Some(true),
        "--no-kitty" => Some(false),
        _ => None,
    })
}

fn side_path(raw: &OsStr) -> Option<PathBuf> {
    (raw != OsStr::new(DEV_NULL)).then(|| PathBuf::from(raw))
}

/// Expand a leading `~` in a path supplied through Git's command string.
fn expand_tilde_path(path: &Path) -> PathBuf {
    let raw = path.as_os_str();
    let rest = raw
        .to_str()
        .and_then(|s| s.strip_prefix("~/").or_else(|| s.strip_prefix('~')));
    match (rest, std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => {
            let mut expanded = PathBuf::from(home);
            if !rest.is_empty() {
                expanded.push(rest);
            }
            expanded
        }
        _ => PathBuf::from(raw),
    }
}

