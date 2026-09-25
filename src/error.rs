use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Message(String),
    /// Parser-generated help or completion output; print to stdout and exit successfully.
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
            Failure::Stdout(..) | Failure::Completion(_) => Self::HelpExit(failure),
            Failure::Stderr(_) => Self::message(failure.unwrap_stderr()),
        }
    }
}
