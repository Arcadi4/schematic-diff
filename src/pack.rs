//! `--pack`: mesh a build with the game's own block model format.
//!
//! A resource pack is the only source of real block textures — Nucleation
//! ships none — and interpreting one correctly means implementing the whole
//! model format: blockstate `variants` and `multipart` selection, `parent`
//! chains, `elements`, per-element rotation, UV projection, and the runtime
//! tints the game multiplies in. That is a large specification, and
//! `schematic_mesher` implements it: it is the crate Nucleation's own `meshing`
//! feature is built on.
//!
//! So this module is deliberately thin. It loads the pack, hands the mesher the
//! build, and returns triangles. Everything that decides *what* is drawn
//! happens inside the mesher.

use std::path::Path;

use nucleation::UniversalSchematic;
use nucleation::meshing::{MeshConfig, MeshOutput, ResourcePackSource, ResourcePackStats};

use crate::error::{Error, Result};

/// A loaded resource pack.
pub struct Pack {
    source: ResourcePackSource,
}

impl Pack {
    /// Read a pack from a `.zip` on disk.
    pub fn open(path: &Path) -> Result<Self> {
        let source = ResourcePackSource::from_file(path).map_err(|error| {
            Error::message(format!("{} could not be read: {error}", path.display()))
        })?;
        // A pack that parses but defines nothing usable would silently render
        // every block in the fallback colour, which reads as "my pack was
        // ignored". Reporting it here names the real problem.
        if source.stats().blockstate_count == 0 {
            return Err(Error::message(format!(
                "{} holds no blockstates, so it is not a resource pack",
                path.display()
            )));
        }
        Ok(Self { source })
    }

    /// Mesh a build with this pack.
    pub fn mesh(&self, schematic: &UniversalSchematic) -> Result<MeshOutput> {
        schematic
            .to_mesh(&self.source, &MeshConfig::default())
            .map_err(|error| Error::message(format!("the build could not be meshed: {error}")))
    }

    /// How much the pack contains, for the summary.
    ///
    /// Reported because `--pack` is the one argument whose failure is silent: a
    /// pack that loads but covers nothing renders exactly like the flag being
    /// ignored.
    pub fn stats(&self) -> ResourcePackStats {
        self.source.stats()
    }
}
