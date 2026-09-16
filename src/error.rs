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
    /// `--help`/`--version`: print the held output to stdout, exit 0.
    HelpExit(bpaf::ParseFailure),
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
            // Never printed with the `schematic-diff: ` prefix: `main` prints
            // the held output to stdout and exits 0 instead.
            Self::HelpExit(_) => Ok(()),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Message(_) | Self::HelpExit(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<bpaf::ParseFailure> for Error {
    fn from(failure: bpaf::ParseFailure) -> Self {
        use bpaf::ParseFailure as Failure;
        match failure {
            // `--help`/`--version` print to stdout, exit 0; `run_inner` keeps
            // them inside `Result` instead of exiting, so carry the output up
            // for `main` to print rather than failing here.
            Failure::Stdout(..) | Failure::Completion(_) => Self::HelpExit(failure),
            Failure::Stderr(_) => Self::message(failure.unwrap_stderr()),
        }
    }
}
