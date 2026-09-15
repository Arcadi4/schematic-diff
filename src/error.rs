//! The one error type this tool reports.

use std::fmt;

/// Every way a run can fail.
///
/// The variants exist only to be printed: each carries text already written
/// for a person, so nothing downstream matches on which one it is.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Message(String),
}

impl Error {
    pub fn message(text: impl Into<String>) -> Self {
        Self::Message(text.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Message(text) => f.write_str(text),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Message(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
