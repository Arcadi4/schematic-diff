//! Schematic format loading.
//!
//! Nucleation's `FormatManager` detects most containers from their bytes. Binary
//! vanilla structure NBT is decoded here because Nucleation's SNBT path caps
//! structures at 256 blocks per axis and 262,144 cells.

use std::io::{Cursor, Read};
use std::path::Path;

use flate2::read::GzDecoder;
use nucleation::UniversalSchematic;
use nucleation::block_entity::BlockEntity;
use nucleation::block_position::BlockPosition;
use nucleation::formats::limits::DecodeLimits;
use nucleation::formats::manager::get_manager;
use nucleation::nbt::NbtMap;
use nucleation::{BlockState, Entity, Region};
use quartz_nbt::io::Flavor;
use quartz_nbt::{NbtCompound, NbtList, NbtTag};

use crate::error::{Error, Result};

/// Cell-count ceiling for bounded Nucleation decoders and the binary NBT path.
/// Prevents corrupt dimensions and decompression bombs from triggering huge
/// allocations.
const MAX_VOLUME: i64 = 512 * 1024 * 1024;
const MAX_DECOMPRESSED_BYTES: u32 = 1024 * 1024 * 1024;
/// Input budget for Nucleation decoders, overriding their 256 MiB default.
const MAX_NUCLEATION_INPUT: usize = 1024 * 1024 * 1024;

fn nucleation_limits() -> DecodeLimits {
    DecodeLimits {
        max_input_bytes: MAX_NUCLEATION_INPUT,
        ..DecodeLimits::default()
    }
}

pub struct Loaded {
    pub schematic: UniversalSchematic,
    /// The format that actually parsed the file, which may differ from the
    /// extension when a file is misnamed.
    pub format: String,
    pub entities: usize,
}

pub fn load(path: &Path) -> Result<Loaded> {
    let bytes = std::fs::read(path)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("schematic")
        .to_string();
    load_bytes(&bytes, &name)
}

fn load_bytes(bytes: &[u8], name: &str) -> Result<Loaded> {
    if bytes.is_empty() {
        return Err(Error::message(format!("{name} is empty")));
    }

    let manager = get_manager();
    let guard = manager
        .lock()
        .map_err(|_| Error::message("the format manager is poisoned".to_string()))?;
    match guard.read_bounded_with_format(bytes, &nucleation_limits()) {
        Ok((format, schematic)) => {
            drop(guard);
            Ok(Loaded {
                entities: schematic.get_entities_as_list().len(),
                schematic,
                format,
            })
        }
        Err(error) => {
            drop(guard);
            if is_binary_structure(bytes) {
                load_structure_nbt(bytes, name)
            } else {
                Err(Error::message(format!("{name} could not be read: {error}")))
            }
        }
    }
}

fn is_binary_structure(bytes: &[u8]) -> bool {
    let raw = match maybe_gunzip(bytes, u32::MAX) {
        Ok(raw) => raw,
        Err(_) => return false,
    };
    let Ok((root, _)) = quartz_nbt::io::read_nbt(&mut Cursor::new(&raw), Flavor::Uncompressed)
    else {
        return false;
    };
    // `blocks` and `palette` distinguish structures from other NBT roots such
    // as level data, chunks, and items.
    root.contains_key("blocks") && root.contains_key("palette")
}

fn load_structure_nbt(bytes: &[u8], name: &str) -> Result<Loaded> {
    let raw = maybe_gunzip(bytes, MAX_DECOMPRESSED_BYTES)?;
    let (root, _) = quartz_nbt::io::read_nbt(&mut Cursor::new(&raw), Flavor::Uncompressed)
        .map_err(|error| Error::message(format!("{name} is not readable NBT: {error}")))?;

    let unreadable = || Error::message(format!("{name} is not a Java structure"));

    let size = triple(&root, "size").ok_or_else(unreadable)?;
    if size.iter().any(|axis| *axis <= 0) {
        return Err(unreadable());
    }
    let volume = i64::from(size[0]) * i64::from(size[1]) * i64::from(size[2]);
    if volume > MAX_VOLUME {
        return Err(Error::message(format!(
            "{name} is {}x{}x{} ({} cells), beyond the {MAX_VOLUME}-cell limit",
            size[0], size[1], size[2], volume
        )));
    }

    let data_version = match root.inner().get("DataVersion") {
        Some(NbtTag::Int(value)) => Some(*value),
        _ => None,
    };

    let mut schematic = UniversalSchematic::new(name.to_string());
    schematic.metadata.name = Some(name.to_string());
    schematic.metadata.mc_version = data_version;
    schematic.metadata.source_data_version = data_version;
    schematic.default_region = Region::try_new(
        schematic.default_region_name.clone(),
        (0, 0, 0),
        (size[0], size[1], size[2]),
    )
    .map_err(|error| Error::message(format!("{name}: {error}")))?;

    let palette = read_palette(&root).ok_or_else(unreadable)?;

    let Some(NbtTag::List(blocks)) = root.inner().get("blocks") else {
        return Err(unreadable());
    };
    if blocks.len() as i64 > volume {
        return Err(unreadable());
    }

    for entry in blocks.iter() {
        let NbtTag::Compound(entry) = entry else {
            return Err(unreadable());
        };
        let Some(position) = triple(entry, "pos") else {
            return Err(unreadable());
        };
        // Reject coordinates outside the declared grid even when the dimensions
        // themselves are well formed.
        if position
            .iter()
            .enumerate()
            .any(|(axis, value)| *value < 0 || *value >= size[axis])
        {
            return Err(Error::message(format!(
                "{name} has a block at {position:?}, outside its {size:?} size"
            )));
        }
        let Some(NbtTag::Int(index)) = entry.inner().get("state") else {
            return Err(unreadable());
        };
        let Some(state) = palette.get(*index as usize) else {
            return Err(unreadable());
        };

        let block = BlockState::from_block_string(state)
            .map_err(|error| Error::message(format!("{name}: {error}")))?;
        let (x, y, z) = (position[0], position[1], position[2]);
        schematic.set_block(x, y, z, &block);

        if let Some(NbtTag::Compound(nbt)) = entry.inner().get("nbt") {
            let map = NbtMap::from_quartz_nbt(nbt);
            let id = nbt
                .inner()
                .get("id")
                .or_else(|| nbt.inner().get("Id"))
                .and_then(|tag| match tag {
                    NbtTag::String(value) => Some(value.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| block.get_name().to_string());
            let mut block_entity = BlockEntity::new(id, (x, y, z));
            block_entity.set_nbt(map);
            schematic.set_block_entity(BlockPosition { x, y, z }, block_entity);
        }
    }

    let mut entities = 0;
    if let Some(NbtTag::List(list)) = root.inner().get("entities") {
        for entry in list.iter() {
            let NbtTag::Compound(entry) = entry else {
                continue;
            };
            let Some(NbtTag::Compound(nbt)) = entry.inner().get("nbt") else {
                continue;
            };
            // Match vanilla by ignoring entities without a type id instead of
            // failing the whole load.
            if !nbt.contains_key("id") && !nbt.contains_key("Id") {
                continue;
            }
            let mut nbt = nbt.clone();
            if let Some(position) = double_triple(entry, "pos") {
                let coordinates = position.map(NbtTag::Double);
                nbt.insert("Pos", NbtList::clone_from(&coordinates));
            }
            if let Ok(entity) = Entity::from_nbt(&nbt) {
                schematic.add_entity(entity);
                entities += 1;
            }
        }
    }

    Ok(Loaded {
        schematic,
        format: "nbt (structure)".to_string(),
        entities,
    })
}

fn read_palette(root: &NbtCompound) -> Option<Vec<String>> {
    let NbtTag::List(palette) = root.inner().get("palette")? else {
        return None;
    };
    let mut states = Vec::with_capacity(palette.len());
    for entry in palette.iter() {
        let NbtTag::Compound(entry) = entry else {
            return None;
        };
        let name = match entry.inner().get("Name") {
            Some(NbtTag::String(name)) => name.clone(),
            _ => return None,
        };
        let mut properties: Vec<(String, String)> = Vec::new();
        if let Some(NbtTag::Compound(properties_tag)) = entry.inner().get("Properties") {
            for (key, value) in properties_tag.inner().iter() {
                let NbtTag::String(value) = value else {
                    return None;
                };
                properties.push((key.clone(), value.clone()));
            }
        }
        // Property order is not meaningful; sorting avoids false diffs caused
        // by serialization order alone.
        properties.sort();
        states.push(if properties.is_empty() {
            name
        } else {
            let body = properties
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(",");
            format!("{name}[{body}]")
        });
    }
    Some(states)
}

/// Read a three-element integer vector, accepting both the list encoding the
/// game writes and the int-array encoding other tools emit.
fn triple(compound: &NbtCompound, key: &str) -> Option<[i32; 3]> {
    let tag = compound.inner().get(key)?;
    let values: Vec<i32> = match tag {
        NbtTag::IntArray(values) => values.clone(),
        NbtTag::List(list) => {
            let mut out = Vec::with_capacity(list.len());
            for entry in list.iter() {
                out.push(match entry {
                    NbtTag::Byte(v) => i32::from(*v),
                    NbtTag::Short(v) => i32::from(*v),
                    NbtTag::Int(v) => *v,
                    _ => return None,
                });
            }
            out
        }
        _ => return None,
    };
    let [x, y, z] = values.as_slice() else {
        return None;
    };
    Some([*x, *y, *z])
}

/// Read a three-element f64 NBT list, as used for entity positions.
fn double_triple(compound: &NbtCompound, key: &str) -> Option<[f64; 3]> {
    let NbtTag::List(list) = compound.inner().get(key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(list.len());
    for entry in list.iter() {
        out.push(match entry {
            NbtTag::Float(v) => f64::from(*v),
            NbtTag::Double(v) => *v,
            _ => return None,
        });
    }
    let [x, y, z] = out.as_slice() else {
        return None;
    };
    Some([*x, *y, *z])
}

/// Decompress gzip data when present, rejecting declared output over `limit`.
///
/// The gzip trailer's ISIZE sizes the inflater output buffer, so reading it up
/// front bounds both allocation and decode work.
fn maybe_gunzip(bytes: &[u8], limit: u32) -> Result<Vec<u8>> {
    if bytes.len() < 2 || bytes[0] != 0x1f || bytes[1] != 0x8b {
        return Ok(bytes.to_vec());
    }
    if bytes.len() < 4 {
        return Err(Error::message("truncated gzip stream".to_string()));
    }
    if limit != u32::MAX {
        let declared = u32::from_le_bytes([
            bytes[bytes.len() - 4],
            bytes[bytes.len() - 3],
            bytes[bytes.len() - 2],
            bytes[bytes.len() - 1],
        ]);
        if declared > limit {
            return Err(Error::message(format!(
                "compressed payload expands to {} MiB, beyond the {} MiB limit",
                declared / (1024 * 1024),
                limit / (1024 * 1024)
            )));
        }
    }
    let mut out = Vec::new();
    GzDecoder::new(Cursor::new(bytes))
        .read_to_end(&mut out)
        .map_err(|error| Error::message(format!("gzip decompression failed: {error}")))?;
    Ok(out)
}
