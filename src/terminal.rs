//! Terminal output routing and geometry.
//!
//! Git can send an external diff's stdout to a terminal, pipe, or regular file.
//! Image escapes are safe only on a terminal. For pipes, [`Output::detect`] tries
//! `/dev/tty`; for regular files it leaves stdout untouched.

use std::fs::File;
use std::io::{self, IsTerminal, Write};
use std::os::fd::{AsRawFd, RawFd};

/// Fallback cell dimensions in pixels when `TIOCGWINSZ` reports zeroes. The 1:2
/// ratio preserves terminal text proportions.
const ASSUMED_CELL: (u32, u32) = (12, 24);

#[derive(Clone, Copy, Debug)]
pub struct Screen {
    pub columns: u32,
    pub rows: u32,
    pub cell_px: (u32, u32),
}

impl Screen {
    /// Read cell counts and pixel dimensions from `fd`. Zero pixel dimensions
    /// fall back to [`ASSUMED_CELL`].
    pub fn measure(fd: RawFd) -> Self {
        let Some(size) = winsize(fd) else {
            return Self::fallback();
        };
        if size.ws_col == 0 || size.ws_row == 0 {
            return Self::fallback();
        }
        let columns = u32::from(size.ws_col);
        let rows = u32::from(size.ws_row);
        let cell_px = match (size.ws_xpixel, size.ws_ypixel) {
            (w, h) if w > 0 && h > 0 => {
                let cell_w = (u32::from(w) / columns).max(4);
                let cell_h = (u32::from(h) / rows).max(8);
                (cell_w, cell_h)
            }
            _ => ASSUMED_CELL,
        };
        Self {
            columns,
            rows,
            cell_px,
        }
    }

    pub fn fallback() -> Self {
        Self {
            columns: 100,
            rows: 30,
            cell_px: ASSUMED_CELL,
        }
    }
}

fn winsize(fd: RawFd) -> Option<libc::winsize> {
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: `size` is a live `winsize` and `fd` is an open descriptor; the
    // kernel writes exactly one `winsize` through the pointer.
    let result = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut size) };
    (result == 0).then_some(size)
}

fn is_pipe_or_socket(fd: RawFd) -> bool {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `stat` is a live `stat` and `fd` is an open descriptor.
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return false;
    }
    let kind = stat.st_mode & libc::S_IFMT;
    kind == libc::S_IFIFO || kind == libc::S_IFSOCK
}

pub struct Output {
    stream: Box<dyn Write>,
    terminal: bool,
    pub screen: Screen,
}

impl Output {
    pub fn detect() -> Self {
        let stdout = io::stdout();
        if stdout.is_terminal() {
            return Self {
                screen: Screen::measure(stdout.as_raw_fd()),
                stream: Box::new(stdout),
                terminal: true,
            };
        }

        // A pipe's reader cannot render the image, so use `/dev/tty`. A file is
        // an explicit redirect and must stay untouched.
        if is_pipe_or_socket(stdout.as_raw_fd())
            && let Ok(tty) = File::options().write(true).open("/dev/tty")
            && tty.is_terminal()
        {
            return Self {
                screen: Screen::measure(tty.as_raw_fd()),
                stream: Box::new(tty),
                terminal: true,
            };
        }

        Self {
            screen: Screen::fallback(),
            stream: Box::new(stdout),
            terminal: false,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// The pixel area available for an image `rows` cells tall, spanning the
    /// full width.
    pub fn image_pixels(&self, rows: u32) -> (u32, u32) {
        (
            self.screen.columns * self.screen.cell_px.0,
            rows * self.screen.cell_px.1,
        )
    }
}

impl Write for Output {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.write(buf)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.stream.write_all(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}
