//! Blockscope-backed litematic path.
//!
//! Litematic files are parsed, diffed, meshed, and rasterized by Blockscope's
//! crates. Other formats keep the Nucleation path in `format`/`main`, because
//! Blockscope only reads litematic containers.
//!
//! The two paths meet at [`crate::raster::Picture::Blockscope`]: Blockscope
//! renders each panel to RGBA at the panel's final pixel size, and the terminal
//! compositor places it exactly like a flat grid or a Nucleation mesh.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use litematic_core::{DecodeLimits, LitematicFile, LitematicLimits};
use litematic_material::{AssetSource, MaterialSet, MaterialSnapshot};
use litematic_mesh::{
    BlockCatalog, ColorBlockCatalog, Mesh, MeshBuildOptions, MeshLimits, ResolvedBlock,
    build_mesh_parallel_with_options,
};
use litematic_scene::{
    BlockStateKey, Bounds as SceneBounds, PackedIndices, Region, Scene, SceneMetadata,
};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::kitty;
use crate::raster::{BlockscopePicture, Layout, Panel, Picture, Segment, Theme, encode_png, fits};
use crate::render::Bounds;
use crate::terminal::Output;
use crate::{
    ADDED_COLOR, ADDED_TEXT, CAPTION_TEXT, CHANGED_COLOR, Invocation, REMOVED_COLOR, REMOVED_TEXT,
    Rendered, SWAPPED_COLOR, StatMode, no_image_reason, report, report_once,
};

/// One changed cell. States are interned so a large diff keeps one copy of
/// each distinct block state rather than one per cell.
type Cell = ((i32, i32, i32), Arc<BlockStateKey>);
type Pair = ((i32, i32, i32), Arc<BlockStateKey>, Arc<BlockStateKey>);

#[derive(Default)]
struct Changes {
    added: Vec<Cell>,
    removed: Vec<Cell>,
    changed: Vec<Pair>,
    swapped: Vec<Pair>,
    aligned: u64,
    non_air: u64,
}

impl Changes {
    fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.changed.is_empty()
            && self.swapped.is_empty()
    }

    fn merge(&mut self, mut other: Self) {
        self.added.append(&mut other.added);
        self.removed.append(&mut other.removed);
        self.changed.append(&mut other.changed);
        self.swapped.append(&mut other.swapped);
        self.aligned += other.aligned;
        self.non_air += other.non_air;
    }

    fn support(&self) -> f64 {
        if self.non_air == 0 {
            1.0
        } else {
            (self.aligned + self.changed.len() as u64 + self.swapped.len() as u64) as f64
                / self.non_air as f64
        }
    }
}

/// True when both supplied sides are litematic files, so Blockscope can read
/// and render them without falling back to Nucleation.
pub fn handles(invocation: &Invocation) -> bool {
    let litematic = |path: &Option<std::path::PathBuf>| {
        path.as_ref().is_none_or(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("litematic"))
        })
    };
    (invocation.before.is_some() || invocation.after.is_some())
        && litematic(&invocation.before)
        && litematic(&invocation.after)
}

pub fn run(invocation: &Invocation, output: &mut Output, draw_image: bool) -> Result<()> {
    let before = invocation.before.as_deref().map(load).transpose()?;
    let after = invocation.after.as_deref().map(load).transpose()?;
    if before.is_none() && after.is_none() {
        return Err(Error::message(format!(
            "both sides of {} are missing",
            invocation.display_path
        )));
    }

    let before_desc = before.as_ref().map(describe);
    let after_desc = after.as_ref().map(describe);
    let changes = match (&before, &after) {
        (Some(before), Some(after)) => Some(diff(before, after)),
        _ => None,
    };
    let stat = invocation
        .stat
        .map(|mode| {
            format_stat(
                &before,
                &after,
                changes.as_ref(),
                invocation.standalone,
                mode,
            )
        })
        .unwrap_or_default();

    let pack = match &invocation.pack {
        Some(path) => match open_pack(path, &before, &after) {
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

    let rendered = if invocation.kitty == Some(false) {
        None
    } else {
        match render_scene(
            invocation,
            before,
            after,
            changes.as_ref(),
            pack.as_ref(),
            output,
            stat.len() as u32,
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

    write_heading(output, invocation)?;
    write_details(
        output,
        invocation,
        before_desc.as_deref(),
        after_desc.as_deref(),
        changes.as_ref(),
        pack.as_ref(),
        &stat,
    )?;
    output.flush()?;

    if let Some(rendered) = &rendered
        && draw_image
    {
        let columns = output.screen.columns;
        kitty::write_png(output, &rendered.png, columns, rendered.image_cells)?;
        output.write_all(b"\n".repeat(rendered.image_cells as usize).as_slice())?;
    }

    if rendered.is_some() && !draw_image {
        report_once(format_args!("{}", no_image_reason(invocation, output)));
    }
    Ok(())
}

fn decode_limits() -> DecodeLimits {
    DecodeLimits {
        max_input_bytes: 1024 * 1024 * 1024,
        max_decompressed_bytes: 1024 * 1024 * 1024,
        max_array_len: 512 * 1024 * 1024,
        max_total_bytes: 1024 * 1024 * 1024,
        ..DecodeLimits::default()
    }
}

fn scene_limits() -> LitematicLimits {
    LitematicLimits {
        max_volume_per_region: 512 * 1024 * 1024,
        max_total_volume: 512 * 1024 * 1024,
        ..LitematicLimits::default()
    }
}

fn load(path: &Path) -> Result<Scene> {
    let bytes = std::fs::read(path)?;
    let file =
        LitematicFile::parse_bytes(&bytes, &decode_limits(), &scene_limits()).map_err(|error| {
            Error::message(format!("{} could not be read: {error}", path.display()))
        })?;
    file.into_scene()
        .map_err(|error| Error::message(format!("{} could not be read: {error}", path.display())))
}

/// Palette indices are stored y-major, then z, then x, matching litematic.
fn block_at(scene: &Scene, position: [i64; 3]) -> Option<&BlockStateKey> {
    for region in scene.regions.iter().rev() {
        let local = [
            position[0] - region.origin[0],
            position[1] - region.origin[1],
            position[2] - region.origin[2],
        ];
        if local
            .iter()
            .zip(region.size)
            .any(|(value, size)| *value < 0 || *value >= i64::from(size))
        {
            continue;
        }
        let size_x = u64::from(region.size[0]);
        let size_z = u64::from(region.size[2]);
        let index = local[1] as u64 * size_x * size_z + local[2] as u64 * size_x + local[0] as u64;
        let Some(state) = region.palette.get(region.states.get(index) as usize) else {
            continue;
        };
        // A later region wins even when it stores air.
        return (!state.is_air()).then_some(state);
    }
    None
}

/// Litematic saves a build relative to its region origin, so a moved save
/// differs only by the origin delta. Align on the first region when the two
/// files have the same region layout; otherwise compare in absolute
/// coordinates.
fn translation(before: &Scene, after: &Scene) -> [i64; 3] {
    let (Some(before_region), Some(after_region)) = (before.regions.first(), after.regions.first())
    else {
        return [0; 3];
    };
    if before.regions.len() != after.regions.len() || before_region.name != after_region.name {
        return [0; 3];
    }
    [
        after_region.origin[0] - before_region.origin[0],
        after_region.origin[1] - before_region.origin[1],
        after_region.origin[2] - before_region.origin[2],
    ]
}

fn diff(before: &Scene, after: &Scene) -> Changes {
    let offset = translation(before, after);
    let min = [
        before.bounds.min[0]
            .saturating_add(offset[0])
            .min(after.bounds.min[0]),
        before.bounds.min[1]
            .saturating_add(offset[1])
            .min(after.bounds.min[1]),
        before.bounds.min[2]
            .saturating_add(offset[2])
            .min(after.bounds.min[2]),
    ];
    let max = [
        before.bounds.max_exclusive[0]
            .saturating_add(offset[0])
            .max(after.bounds.max_exclusive[0]),
        before.bounds.max_exclusive[1]
            .saturating_add(offset[1])
            .max(after.bounds.max_exclusive[1]),
        before.bounds.max_exclusive[2]
            .saturating_add(offset[2])
            .max(after.bounds.max_exclusive[2]),
    ];

    let slices: Vec<Changes> = (min[2]..max[2])
        .into_par_iter()
        .map(|z| {
            let mut changes = Changes::default();
            let mut before_states = HashMap::new();
            let mut after_states = HashMap::new();
            for y in min[1]..max[1] {
                for x in min[0]..max[0] {
                    let position = [x, y, z];
                    let old = block_at(
                        before,
                        [
                            position[0] - offset[0],
                            position[1] - offset[1],
                            position[2] - offset[2],
                        ],
                    )
                    .map(|state| intern(&mut before_states, state));
                    let new =
                        block_at(after, position).map(|state| intern(&mut after_states, state));
                    match (old, new) {
                        (None, None) => {}
                        (None, Some(new)) => {
                            changes.non_air += 1;
                            changes
                                .added
                                .push(((x as i32, y as i32, z as i32), new.clone()));
                        }
                        (Some(old), None) => {
                            changes.non_air += 1;
                            changes
                                .removed
                                .push(((x as i32, y as i32, z as i32), old.clone()));
                        }
                        (Some(old), Some(new)) => {
                            changes.non_air += 1;
                            if old == new {
                                changes.aligned += 1;
                            } else {
                                let position = (x as i32, y as i32, z as i32);
                                if old.name == new.name {
                                    changes.swapped.push((position, old.clone(), new.clone()));
                                } else {
                                    changes.changed.push((position, old.clone(), new.clone()));
                                }
                            }
                        }
                    }
                }
            }
            changes
        })
        .collect();

    let mut merged = Changes::default();
    for slice in slices {
        merged.merge(slice);
    }
    collapse_swaps(&mut merged);
    merged
}

/// A block state that consistently becomes another state across the build is a
/// palette swap, not thousands of unrelated edits. Keep the same 80% dominance
/// rule the Nucleation diff used so the category colors retain their meaning.
fn collapse_swaps(changes: &mut Changes) {
    let mut totals: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for (_, old, new) in &changes.changed {
        *totals
            .entry(old.canonical())
            .or_default()
            .entry(new.canonical())
            .or_default() += 1;
    }

    let mut swaps = HashSet::new();
    for (old, targets) in totals {
        let total: usize = targets.values().sum();
        if let Some((new, count)) = targets
            .into_iter()
            .max_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
            && count * 100 >= total * 80
        {
            swaps.insert((old, new));
        }
    }

    let mut changed = Vec::new();
    let mut swapped = Vec::new();
    for (position, old, new) in std::mem::take(&mut changes.changed) {
        if swaps.contains(&(old.canonical(), new.canonical())) {
            swapped.push((position, old, new));
        } else {
            changed.push((position, old, new));
        }
    }
    changes.changed = changed;
    changes.swapped = swapped;
}

fn intern(
    cache: &mut HashMap<BlockStateKey, Arc<BlockStateKey>>,
    state: &BlockStateKey,
) -> Arc<BlockStateKey> {
    if let Some(interned) = cache.get(state) {
        return Arc::clone(interned);
    }
    let interned = Arc::new(state.clone());
    cache.insert(state.clone(), Arc::clone(&interned));
    interned
}

fn open_pack(
    path: &Path,
    before: &Option<Scene>,
    after: &Option<Scene>,
) -> Result<Arc<MaterialSnapshot>> {
    let mut source = AssetSource::new();
    if path.is_dir() {
        source.add_dir(path).map_err(|error| {
            Error::message(format!("{} could not be read: {error}", path.display()))
        })?;
    } else {
        source.add_jar(path).map_err(|error| {
            Error::message(format!("{} could not be read: {error}", path.display()))
        })?;
    }
    let materials = MaterialSet::open(source).map_err(|error| {
        Error::message(format!("{} could not be read: {error}", path.display()))
    })?;
    let states = before
        .iter()
        .chain(after.iter())
        .flat_map(|scene| scene.regions.iter())
        .flat_map(|region| region.palette.iter())
        .cloned();
    let snapshot = materials.snapshot(states);
    if snapshot.is_empty() {
        return Err(Error::message(format!(
            "{} holds no blockstates, so it is not a resource pack",
            path.display()
        )));
    }
    Ok(Arc::new(snapshot))
}

#[derive(Clone)]
enum BaseCatalog {
    Flat(Arc<StaticCatalog>),
    Textured(Arc<MaterialSnapshot>),
}

impl BlockCatalog for BaseCatalog {
    fn resolve(&self, state: &BlockStateKey) -> ResolvedBlock {
        match self {
            Self::Flat(catalog) => catalog.resolve(state),
            Self::Textured(catalog) => catalog.resolve(state),
        }
    }
}

struct StaticCatalog {
    entries: HashMap<BlockStateKey, ResolvedBlock>,
    missing: ResolvedBlock,
}

impl BlockCatalog for StaticCatalog {
    fn resolve(&self, state: &BlockStateKey) -> ResolvedBlock {
        self.entries
            .get(state)
            .cloned()
            .unwrap_or_else(|| self.missing.clone())
    }
}

fn flat_catalog(before: &Option<Scene>, after: &Option<Scene>) -> Result<Arc<StaticCatalog>> {
    let colors = ColorBlockCatalog::new().map_err(|error| Error::message(error.to_string()))?;
    let mut entries = HashMap::new();
    for scene in before.iter().chain(after.iter()) {
        for region in &scene.regions {
            for state in &region.palette {
                entries.entry(state.clone()).or_insert_with(|| {
                    let mut resolved = colors.resolve(state);
                    if !resolved.known {
                        resolved.color = legacy_color(state);
                    }
                    resolved
                });
            }
        }
    }
    let missing = ResolvedBlock {
        color: litematic_mesh::Rgba::rgb(0x80, 0x80, 0x80),
        layer: litematic_mesh::RenderLayer::Opaque,
        model: litematic_mesh::full_cube_model(),
        known: false,
    };
    Ok(Arc::new(StaticCatalog { entries, missing }))
}

/// Blockscope's embedded palette paints unknown states magenta. Keep the old
/// renderer's stable lookup/table fallback instead, so an unrecognised state
/// stays a readable colour rather than dominating the frame.
fn legacy_color(state: &BlockStateKey) -> litematic_mesh::Rgba {
    let packed = nucleation::BlockState::from_block_string(&state.canonical())
        .map(|state| crate::render::block_color(&state))
        .unwrap_or(0x80_80_80);
    litematic_mesh::Rgba::rgb(
        ((packed >> 16) & 0xff) as u8,
        ((packed >> 8) & 0xff) as u8,
        (packed & 0xff) as u8,
    )
}

struct TintCatalog {
    base: BaseCatalog,
    overrides: HashMap<BlockStateKey, ResolvedBlock>,
}

impl BlockCatalog for TintCatalog {
    fn resolve(&self, state: &BlockStateKey) -> ResolvedBlock {
        self.overrides
            .get(state)
            .cloned()
            .unwrap_or_else(|| self.base.resolve(state))
    }
}

fn tint(color: litematic_mesh::Rgba, category: [u8; 3]) -> litematic_mesh::Rgba {
    const TINT: f32 = 0.55;
    let channel = |base: u8, over: u8| {
        (f32::from(base) * (1.0 - TINT) + f32::from(over) * TINT).clamp(0.0, 255.0) as u8
    };
    litematic_mesh::Rgba::new(
        channel(color.r, category[0]),
        channel(color.g, category[1]),
        channel(color.b, category[2]),
        color.a,
    )
}

fn synthetic_key(category: u8, state: &BlockStateKey) -> BlockStateKey {
    BlockStateKey::simple(format!("__schematic_diff/{category}/{}", state.canonical()))
}

fn mesh_limits() -> MeshLimits {
    MeshLimits {
        max_pages: 1 << 20,
        max_quads: 1 << 30,
        max_vertices: 1 << 30,
    }
}

fn build_mesh(scene: &Scene, catalog: &(impl BlockCatalog + Sync)) -> Result<Mesh> {
    build_mesh_parallel_with_options(
        scene,
        catalog,
        &mesh_limits(),
        0,
        MeshBuildOptions::default(),
    )
    .map_err(|error| Error::message(format!("the build could not be meshed: {error}")))
}

fn air_region(name: &str, origin: [i64; 3]) -> Region {
    Region {
        name: name.to_string(),
        origin,
        size: [1, 1, 1],
        data_version: 0,
        palette: vec![BlockStateKey::simple("minecraft:air")],
        states: PackedIndices::from_values(PackedIndices::MIN_BITS, &[0])
            .expect("one air cell is valid"),
        block_entities: Vec::new(),
        entity_count: 0,
        pending_block_tick_count: 0,
        pending_fluid_tick_count: 0,
    }
}

/// Add air corners so the camera frames the shared extent even when a side
/// leaves an edge empty. Air regions go first so real regions win overlaps.
///
/// The parsed scene is already validated and its statistics are unchanged by
/// two air cells, so this avoids rescanning every cell in the region to rebuild
/// them.
fn framed(scene: Scene, frame: SceneBounds) -> Scene {
    let mut regions = vec![
        air_region("__schematic_diff_frame_min", frame.min),
        air_region(
            "__schematic_diff_frame_max",
            [
                frame.max_exclusive[0] - 1,
                frame.max_exclusive[1] - 1,
                frame.max_exclusive[2] - 1,
            ],
        ),
    ];
    regions.extend(scene.regions);
    Scene {
        metadata: scene.metadata,
        bounds: frame,
        regions,
        statistics: scene.statistics,
        diagnostics: scene.diagnostics,
    }
}

struct Chunk {
    origin: [i64; 3],
    size: [u32; 3],
    palette: Vec<BlockStateKey>,
    index: HashMap<BlockStateKey, u32>,
    values: Vec<u32>,
}

impl Chunk {
    fn new(origin: [i64; 3], size: [u32; 3]) -> Self {
        let volume = u64::from(size[0]) * u64::from(size[1]) * u64::from(size[2]);
        Self {
            origin,
            size,
            palette: vec![BlockStateKey::simple("minecraft:air")],
            index: HashMap::new(),
            values: vec![0; volume as usize],
        }
    }

    fn set(&mut self, position: [i64; 3], state: BlockStateKey) {
        let local = [
            position[0] - self.origin[0],
            position[1] - self.origin[1],
            position[2] - self.origin[2],
        ];
        let palette = match self.index.get(&state) {
            Some(&palette) => palette,
            None => {
                let palette = self.palette.len() as u32;
                self.palette.push(state.clone());
                self.index.insert(state, palette);
                palette
            }
        };
        let size_x = u64::from(self.size[0]);
        let size_z = u64::from(self.size[2]);
        let index = local[1] as u64 * size_x * size_z + local[2] as u64 * size_x + local[0] as u64;
        self.values[index as usize] = palette;
    }

    fn into_region(self) -> Result<Region> {
        let bits = PackedIndices::bits_for_palette(self.palette.len());
        let states = PackedIndices::from_values(bits, &self.values)
            .ok_or_else(|| Error::message("a change chunk could not be packed"))?;
        Ok(Region {
            name: format!("__schematic_diff_changes_{:?}", self.origin),
            origin: self.origin,
            size: self.size,
            data_version: 0,
            palette: self.palette,
            states,
            block_entities: Vec::new(),
            entity_count: 0,
            pending_block_tick_count: 0,
            pending_fluid_tick_count: 0,
        })
    }
}

fn change_scene(
    changes: &Changes,
    frame: SceneBounds,
    catalog: &mut TintCatalog,
) -> Result<Option<Scene>> {
    if changes.is_empty() {
        return Ok(None);
    }

    let mut chunks: BTreeMap<[i64; 3], Chunk> = BTreeMap::new();
    let mut add = |position: [i64; 3], state: &BlockStateKey, category: u8| {
        let synthetic = synthetic_key(category, state);
        if !catalog.overrides.contains_key(&synthetic) {
            let mut resolved = catalog.base.resolve(state);
            resolved.color = tint(resolved.color, category_color(category));
            catalog.overrides.insert(synthetic.clone(), resolved);
        }
        let key = [
            position[0].div_euclid(16),
            position[1].div_euclid(16),
            position[2].div_euclid(16),
        ];
        let chunk = chunks.entry(key).or_insert_with(|| {
            let origin = [
                (key[0] * 16).max(frame.min[0]),
                (key[1] * 16).max(frame.min[1]),
                (key[2] * 16).max(frame.min[2]),
            ];
            let end = [
                (key[0] * 16 + 16).min(frame.max_exclusive[0]),
                (key[1] * 16 + 16).min(frame.max_exclusive[1]),
                (key[2] * 16 + 16).min(frame.max_exclusive[2]),
            ];
            Chunk::new(
                origin,
                [
                    (end[0] - origin[0]) as u32,
                    (end[1] - origin[1]) as u32,
                    (end[2] - origin[2]) as u32,
                ],
            )
        });
        chunk.set(position, synthetic);
    };

    for (position, state) in &changes.added {
        add(
            [position.0 as i64, position.1 as i64, position.2 as i64],
            state.as_ref(),
            0,
        );
    }
    for (position, state) in &changes.removed {
        add(
            [position.0 as i64, position.1 as i64, position.2 as i64],
            state.as_ref(),
            1,
        );
    }
    for (position, _, state) in &changes.changed {
        add(
            [position.0 as i64, position.1 as i64, position.2 as i64],
            state.as_ref(),
            2,
        );
    }
    for (position, _, state) in &changes.swapped {
        add(
            [position.0 as i64, position.1 as i64, position.2 as i64],
            state.as_ref(),
            3,
        );
    }

    let mut regions = vec![
        air_region("__schematic_diff_frame_min", frame.min),
        air_region(
            "__schematic_diff_frame_max",
            [
                frame.max_exclusive[0] - 1,
                frame.max_exclusive[1] - 1,
                frame.max_exclusive[2] - 1,
            ],
        ),
    ];
    for chunk in chunks.into_values() {
        regions.push(chunk.into_region()?);
    }
    let metadata = SceneMetadata::default();
    Scene::new(metadata, regions, Vec::new())
        .map(Some)
        .map_err(|error| Error::message(format!("changes could not be framed: {error}")))
}

fn category_color(category: u8) -> [u8; 3] {
    let color = match category {
        0 => ADDED_COLOR,
        1 => REMOVED_COLOR,
        2 => CHANGED_COLOR,
        _ => SWAPPED_COLOR,
    };
    [
        ((color >> 16) & 0xff) as u8,
        ((color >> 8) & 0xff) as u8,
        (color & 0xff) as u8,
    ]
}

fn project_bounds(frame: SceneBounds) -> Result<Bounds> {
    let axis = |value: i64| {
        i32::try_from(value)
            .map_err(|_| Error::message("the build extends beyond the renderable coordinate range"))
    };
    Ok(Bounds {
        min: [
            axis(frame.min[0])?,
            axis(frame.min[1])?,
            axis(frame.min[2])?,
        ],
        max: [
            axis(frame.max_exclusive[0] - 1)?,
            axis(frame.max_exclusive[1] - 1)?,
            axis(frame.max_exclusive[2] - 1)?,
        ],
    })
}

/// `--zoom` narrows the framed extent around its center, which is the same
/// effect the flat renderer's camera zoom has. Blockscope fits the camera to the
/// scene bounds, so shrinking those bounds zooms the build.
fn zoomed(frame: SceneBounds, zoom: f32) -> SceneBounds {
    if (zoom - 1.0).abs() < 1e-3 {
        return frame;
    }
    let zoom = f64::from(zoom.max(0.01));
    let mut min = [0i64; 3];
    let mut max_exclusive = [0i64; 3];
    for axis in 0..3 {
        let center = (frame.min[axis] as f64 + frame.max_exclusive[axis] as f64) * 0.5;
        let half = (frame.max_exclusive[axis] - frame.min[axis]) as f64 * 0.5 / zoom;
        let low = (center - half).floor() as i64;
        let mut high = (center + half).ceil() as i64;
        if high <= low {
            high = low + 1;
        }
        min[axis] = low;
        max_exclusive[axis] = high;
    }
    SceneBounds { min, max_exclusive }
}

fn union(left: SceneBounds, right: SceneBounds) -> SceneBounds {
    SceneBounds {
        min: [
            left.min[0].min(right.min[0]),
            left.min[1].min(right.min[1]),
            left.min[2].min(right.min[2]),
        ],
        max_exclusive: [
            left.max_exclusive[0].max(right.max_exclusive[0]),
            left.max_exclusive[1].max(right.max_exclusive[1]),
            left.max_exclusive[2].max(right.max_exclusive[2]),
        ],
    }
}

fn render_scene(
    invocation: &Invocation,
    before: Option<Scene>,
    after: Option<Scene>,
    changes: Option<&Changes>,
    pack: Option<&Arc<MaterialSnapshot>>,
    output: &Output,
    stat_len: u32,
) -> Result<Rendered> {
    let frame = match (&before, &after) {
        (Some(before), Some(after)) => union(before.bounds, after.bounds),
        (Some(before), None) => before.bounds,
        (None, Some(after)) => after.bounds,
        (None, None) => return Err(Error::message("no scene to render")),
    };
    let frame = zoomed(frame, invocation.camera.zoom);
    let bounds = project_bounds(frame)?;

    let catalog = match pack {
        Some(pack) => BaseCatalog::Textured(Arc::clone(pack)),
        None => BaseCatalog::Flat(flat_catalog(&before, &after)?),
    };

    let before_picture = before
        .map(|scene| -> Result<BlockscopePicture> {
            let scene = framed(scene, frame);
            let mesh = build_mesh(&scene, &catalog)?;
            Ok(BlockscopePicture {
                scene,
                mesh,
                textures: pack.map(|pack| pack.textures().clone()),
            })
        })
        .transpose()?;
    let after_picture = after
        .map(|scene| -> Result<BlockscopePicture> {
            let scene = framed(scene, frame);
            let mesh = build_mesh(&scene, &catalog)?;
            Ok(BlockscopePicture {
                scene,
                mesh,
                textures: pack.map(|pack| pack.textures().clone()),
            })
        })
        .transpose()?;

    let changes_picture = match changes {
        Some(changes) if !changes.is_empty() => {
            let mut tinted = TintCatalog {
                base: catalog,
                overrides: HashMap::new(),
            };
            change_scene(changes, frame, &mut tinted)?
                .map(|scene| -> Result<BlockscopePicture> {
                    let mesh = build_mesh(&scene, &tinted)?;
                    Ok(BlockscopePicture {
                        scene,
                        mesh,
                        textures: pack.map(|pack| pack.textures().clone()),
                    })
                })
                .transpose()?
        }
        _ => None,
    };

    let (before_name, after_name) = invocation.labels();
    let mut panels = Vec::new();
    match (&before_picture, &after_picture) {
        (Some(before), Some(after)) => {
            panels.push(Panel::new(
                format!("BEFORE  {before_name}"),
                CAPTION_TEXT,
                Picture::Blockscope(before),
            ));
            let note = if changes_picture.is_some() {
                ""
            } else {
                "  (no changes)"
            };
            panels.push(Panel::new(
                format!("AFTER  {after_name}{note}"),
                CAPTION_TEXT,
                Picture::Blockscope(after),
            ));
            if let Some(changes) = &changes_picture
                && fits(output.screen.columns * output.screen.cell_px.0, 3)
            {
                panels.push(Panel::legend(change_legend(), Picture::Blockscope(changes)));
            }
        }
        (None, Some(after)) => {
            let (label, color) = if invocation.standalone {
                (after_name, CAPTION_TEXT)
            } else {
                (format!("ADDED  {after_name}"), ADDED_TEXT)
            };
            panels.push(Panel::new(label, color, Picture::Blockscope(after)));
        }
        (Some(before), None) => {
            let (label, color) = if invocation.standalone {
                (before_name, CAPTION_TEXT)
            } else {
                (format!("DELETED  {before_name}"), REMOVED_TEXT)
            };
            panels.push(Panel::new(label, color, Picture::Blockscope(before)));
        }
        (None, None) => {}
    }

    let text_lines = summary_lines(
        invocation,
        &before_picture,
        &after_picture,
        pack.is_some(),
        stat_len,
    );
    let fit_height = output.screen.rows.saturating_sub(text_lines + 1);
    let default_height = (output.screen.rows.saturating_sub(5)).max(18);
    let image_cells = if fit_height >= 12 {
        fit_height
    } else {
        default_height
    };
    let (width, height) = output.image_pixels(image_cells);

    let image = crate::raster::compose(
        &panels,
        &invocation.camera,
        &Layout {
            width,
            height,
            cell_px: output.screen.cell_px,
            frame: bounds,
        },
        &Theme::default(),
    )?;
    let png = encode_png(&image)?;
    Ok(Rendered { png, image_cells })
}

fn summary_lines(
    invocation: &Invocation,
    before: &Option<BlockscopePicture>,
    after: &Option<BlockscopePicture>,
    pack: bool,
    stat_len: u32,
) -> u32 {
    let base = if invocation.standalone && (before.is_none() || after.is_none()) {
        2
    } else {
        2 + u32::from(before.is_some()) + u32::from(after.is_some())
    };
    base + u32::from(pack) + stat_len
}

fn change_legend() -> Vec<Segment> {
    vec![
        Segment::new("CHANGES  ", CAPTION_TEXT),
        Segment::new("+added  ", category_color(0)),
        Segment::new("-removed  ", category_color(1)),
        Segment::new("*changed  ", category_color(2)),
        Segment::new("#re-paletted", category_color(3)),
    ]
}

fn write_heading(out: &mut impl Write, invocation: &Invocation) -> std::io::Result<()> {
    if invocation.standalone && (invocation.before.is_none() || invocation.after.is_none()) {
        return writeln!(out, "{}", invocation.display_path);
    }
    let status = match (invocation.before.is_some(), invocation.after.is_some()) {
        (false, true) => "added",
        (true, false) => "deleted",
        _ => "modified",
    };
    writeln!(out, "{}  [{status}]", invocation.display_path)
}

fn write_details(
    out: &mut impl Write,
    invocation: &Invocation,
    before_desc: Option<&str>,
    after_desc: Option<&str>,
    changes: Option<&Changes>,
    pack: Option<&Arc<MaterialSnapshot>>,
    stat: &[String],
) -> std::io::Result<()> {
    if invocation.standalone && (invocation.before.is_none() || invocation.after.is_none()) {
        if let Some(description) = after_desc.or(before_desc) {
            writeln!(out, "  build   {description}")?;
        }
    } else {
        if let Some(description) = before_desc {
            writeln!(out, "  before  {description}")?;
        }
        if let Some(description) = after_desc {
            writeln!(out, "  after   {description}")?;
        }
        if let Some(changes) = changes {
            writeln!(
                out,
                "  changes +{} added, -{} removed, ~{} changed, {} re-paletted \
                 ({}% of cells aligned)",
                changes.added.len(),
                changes.removed.len(),
                changes.changed.len(),
                changes.swapped.len(),
                (changes.support() * 100.0).round() as i32,
            )?;
        } else {
            writeln!(out, "  changes not computed (only one side exists)")?;
        }
    }

    if let Some(pack) = pack {
        let diagnostics = pack.diagnostics();
        let path = invocation
            .pack
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        writeln!(
            out,
            "  pack    {} states, {} textures, {} fallback, {} missing models, {} missing textures  [{path}]",
            pack.len(),
            pack.textures().len(),
            diagnostics.fallback_shapes,
            diagnostics.missing_models,
            diagnostics.missing_textures,
        )?;
    }

    for line in stat {
        writeln!(out, "{line}")?;
    }
    Ok(())
}

fn describe(scene: &Scene) -> String {
    let size = scene.bounds.size();
    let regions = scene.regions.len();
    format!(
        "{}x{}x{}  {} blocks  {} region{}{}  [litematic]",
        size[0],
        size[1],
        size[2],
        scene.statistics.non_air,
        regions,
        if regions == 1 { "" } else { "s" },
        match scene.statistics.entities {
            0 => String::new(),
            1 => "  1 entity".to_string(),
            count => format!("  {count} entities"),
        },
    )
}

fn format_stat(
    before: &Option<Scene>,
    after: &Option<Scene>,
    changes: Option<&Changes>,
    standalone: bool,
    mode: StatMode,
) -> Vec<String> {
    let mut lines = Vec::new();
    match (before, after) {
        (Some(_), Some(_)) => {
            let Some(changes) = changes else {
                lines.push("  stat".to_string());
                lines.push("    (no block differences)".to_string());
                return lines;
            };

            let mut added: HashMap<&str, usize> = HashMap::new();
            let mut removed: HashMap<&str, usize> = HashMap::new();
            let mut modified: HashMap<(&str, Option<&str>), usize> = HashMap::new();
            for (_, state) in &changes.added {
                *added.entry(state.name.as_str()).or_default() += 1;
            }
            for (_, state) in &changes.removed {
                *removed.entry(state.name.as_str()).or_default() += 1;
            }
            for (_, was, now) in changes.changed.iter().chain(&changes.swapped) {
                let was_name = was.name.as_str();
                let now_name = now.name.as_str();
                let key = if was_name == now_name {
                    (was_name, None)
                } else {
                    (was_name, Some(now_name))
                };
                *modified.entry(key).or_default() += 1;
            }

            enum Entry<'a> {
                Added {
                    count: usize,
                    name: &'a str,
                },
                Removed {
                    count: usize,
                    name: &'a str,
                },
                Modified {
                    count: usize,
                    was: &'a str,
                    now: Option<&'a str>,
                },
            }

            let mut items = Vec::new();
            for (name, count) in added {
                items.push(Entry::Added { count, name });
            }
            for (name, count) in removed {
                items.push(Entry::Removed { count, name });
            }
            for ((was, now), count) in modified {
                items.push(Entry::Modified { count, was, now });
            }

            if items.is_empty() {
                lines.push("  stat".to_string());
                lines.push("    (no block differences)".to_string());
                return lines;
            }

            fn entry_key<'a>(item: &'a Entry<'a>) -> (usize, &'a str) {
                match item {
                    Entry::Added { count, name } => (*count, *name),
                    Entry::Removed { count, name } => (*count, *name),
                    Entry::Modified { count, was, .. } => (*count, *was),
                }
            }
            items.sort_unstable_by(|left, right| {
                let (count_left, name_left) = entry_key(left);
                let (count_right, name_right) = entry_key(right);
                count_right
                    .cmp(&count_left)
                    .then_with(|| name_left.cmp(name_right))
            });

            lines.push("  stat".to_string());
            let width = items
                .iter()
                .map(|item| match item {
                    Entry::Added { count, .. }
                    | Entry::Removed { count, .. }
                    | Entry::Modified { count, .. } => count.to_string().len(),
                })
                .max()
                .unwrap_or(1);
            let (visible, remaining) = match mode {
                StatMode::Uncapped => (&items[..], 0),
                StatMode::Capped(cap) if items.len() > cap => (&items[..cap], items.len() - cap),
                StatMode::Capped(_) => (&items[..], 0),
            };
            for item in visible {
                let line = match item {
                    Entry::Added { count, name } => {
                        format!("    + {count:>width$}  {name}")
                    }
                    Entry::Removed { count, name } => {
                        format!("    - {count:>width$}  {name}")
                    }
                    Entry::Modified {
                        count,
                        was,
                        now: None,
                    } => format!("    ~ {count:>width$}  {was} (state)"),
                    Entry::Modified {
                        count,
                        was,
                        now: Some(now),
                    } => format!("    ~ {count:>width$}  {was} -> {now}"),
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

fn format_palette_stat(lines: &mut Vec<String>, scene: &Scene, prefix: &str, mode: StatMode) {
    let mut counts: HashMap<&str, u64> = HashMap::new();
    for (state, count) in &scene.statistics.blocks {
        if !state.is_air() {
            *counts.entry(state.name.as_str()).or_default() += count;
        }
    }
    let mut items: Vec<(&str, u64)> = counts.into_iter().collect();
    if items.is_empty() {
        lines.push("  stat".to_string());
        lines.push("    (no blocks)".to_string());
        return;
    }
    items.sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    lines.push("  stat".to_string());
    let width = items
        .iter()
        .map(|(_, count)| count.to_string().len())
        .max()
        .unwrap_or(1);
    let (visible, remaining) = match mode {
        StatMode::Uncapped => (&items[..], 0),
        StatMode::Capped(cap) if items.len() > cap => (&items[..cap], items.len() - cap),
        StatMode::Capped(_) => (&items[..], 0),
    };
    for (name, count) in visible {
        lines.push(format!("    {prefix}{count:>width$}  {name}"));
    }
    if remaining > 0 {
        lines.push(format!("    ... and {remaining} more block types"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(cells: &[([i64; 3], &str)]) -> Scene {
        let mut min = [i64::MAX; 3];
        let mut max = [i64::MIN; 3];
        for (position, _) in cells {
            for axis in 0..3 {
                min[axis] = min[axis].min(position[axis]);
                max[axis] = max[axis].max(position[axis]);
            }
        }
        let size = [
            (max[0] - min[0] + 1) as u32,
            (max[1] - min[1] + 1) as u32,
            (max[2] - min[2] + 1) as u32,
        ];
        let mut palette = vec![BlockStateKey::simple("minecraft:air")];
        let mut index = HashMap::new();
        let mut values = vec![0u32; (size[0] * size[1] * size[2]) as usize];
        for (position, name) in cells {
            let next = palette.len() as u32;
            let id = *index.entry(*name).or_insert_with(|| {
                palette.push(BlockStateKey::simple(*name));
                next
            });
            let local = [
                (position[0] - min[0]) as u32,
                (position[1] - min[1]) as u32,
                (position[2] - min[2]) as u32,
            ];
            let offset = (local[1] * size[0] * size[2] + local[2] * size[0] + local[0]) as usize;
            values[offset] = id;
        }
        let region = Region {
            name: "test".into(),
            origin: min,
            size,
            data_version: 0,
            palette: palette.clone(),
            states: PackedIndices::from_values(
                PackedIndices::bits_for_palette(palette.len()),
                &values,
            )
            .unwrap(),
            block_entities: Vec::new(),
            entity_count: 0,
            pending_block_tick_count: 0,
            pending_fluid_tick_count: 0,
        };
        Scene::new(SceneMetadata::default(), vec![region], Vec::new()).unwrap()
    }

    #[test]
    fn diff_separates_added_removed_changed_and_swapped() {
        let before = scene(&[
            ([0, 0, 0], "minecraft:stone"),
            ([1, 0, 0], "minecraft:dirt"),
            ([2, 0, 0], "minecraft:grass_block"),
        ]);
        let after = scene(&[
            ([0, 0, 0], "minecraft:stone"),
            ([2, 0, 0], "minecraft:gold_block"),
            ([0, 1, 0], "minecraft:sand"),
        ]);

        let changes = diff(&before, &after);
        assert_eq!(changes.added.len(), 1);
        assert_eq!(changes.removed.len(), 1);
        assert_eq!(changes.aligned, 1);
        assert_eq!(changes.changed.len() + changes.swapped.len(), 1);
        assert!((changes.support() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn dominant_palette_mapping_becomes_a_swap() {
        let mut before_cells = Vec::new();
        let mut after_cells = Vec::new();
        for x in 0..5 {
            before_cells.push(([x, 0, 0], "minecraft:grass_block"));
            after_cells.push(([x, 0, 0], "minecraft:gold_block"));
        }
        before_cells.push(([5, 0, 0], "minecraft:grass_block"));
        after_cells.push(([5, 0, 0], "minecraft:stone"));

        let changes = diff(&scene(&before_cells), &scene(&after_cells));
        assert_eq!(changes.swapped.len(), 5);
        assert_eq!(changes.changed.len(), 1);
    }

    #[test]
    fn region_origin_translation_aligns_a_moved_build() {
        let before = scene(&[([0, 0, 0], "minecraft:stone")]);
        let after = scene(&[([10, 0, -4], "minecraft:stone")]);
        assert_eq!(translation(&before, &after), [10, 0, -4]);
        assert!(block_at(&before, [0, 0, 0]).is_some());
        assert!(block_at(&after, [10, 0, -4]).is_some());
        let changes = diff(&before, &after);
        println!(
            "added={:?} removed={:?} changed={:?} swapped={:?} aligned={} non_air={}",
            changes.added,
            changes.removed,
            changes.changed,
            changes.swapped,
            changes.aligned,
            changes.non_air
        );
        assert_eq!(changes.added.len(), 0);
        assert_eq!(changes.removed.len(), 0);
        assert_eq!(changes.aligned, 1);
        assert_eq!(changes.support(), 1.0);
    }

    #[test]
    fn a_later_air_region_shadows_an_earlier_block() {
        let stone = Region {
            name: "stone".into(),
            origin: [0, 0, 0],
            size: [1, 1, 1],
            data_version: 0,
            palette: vec![BlockStateKey::simple("minecraft:stone")],
            states: PackedIndices::from_values(2, &[0]).unwrap(),
            block_entities: Vec::new(),
            entity_count: 0,
            pending_block_tick_count: 0,
            pending_fluid_tick_count: 0,
        };
        let air = Region {
            name: "air".into(),
            origin: [0, 0, 0],
            size: [1, 1, 1],
            data_version: 0,
            palette: vec![BlockStateKey::simple("minecraft:air")],
            states: PackedIndices::from_values(2, &[0]).unwrap(),
            block_entities: Vec::new(),
            entity_count: 0,
            pending_block_tick_count: 0,
            pending_fluid_tick_count: 0,
        };
        let scene = Scene::new(SceneMetadata::default(), vec![stone, air], Vec::new()).unwrap();
        assert!(block_at(&scene, [0, 0, 0]).is_none());
    }
}
