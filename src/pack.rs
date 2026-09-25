//! Resource pack loading and schematic meshing.
//!
//! `schematic_mesher`, the mesher used by Nucleation, owns Minecraft block model
//! and texture interpretation; this module adapts it to CLI errors.

use std::path::Path;

use nucleation::UniversalSchematic;
use nucleation::meshing::{MeshConfig, MeshOutput, ResourcePackSource, ResourcePackStats};

use crate::error::{Error, Result};

pub struct Pack {
    source: ResourcePackSource,
}

impl Pack {
    pub fn open(path: &Path) -> Result<Self> {
        let source = ResourcePackSource::from_file(path).map_err(|error| {
            Error::message(format!("{} could not be read: {error}", path.display()))
        })?;
        // A pack with no usable blockstates would silently render every block
        // in the fallback colour.
        if source.stats().blockstate_count == 0 {
            return Err(Error::message(format!(
                "{} holds no blockstates, so it is not a resource pack",
                path.display()
            )));
        }
        Ok(Self { source })
    }

    pub fn mesh(&self, schematic: &UniversalSchematic) -> Result<MeshOutput> {
        schematic
            .to_mesh(&self.source, &MeshConfig::default())
            .map_err(|error| Error::message(format!("the build could not be meshed: {error}")))
    }

    pub fn stats(&self) -> ResourcePackStats {
        self.source.stats()
    }
}
