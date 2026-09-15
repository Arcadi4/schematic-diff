//! Where a run's output goes, and the terminal geometry the image is sized for.
//!
//! Git hands an external diff a stream whose kind depends on how it was
//! invoked, and only one of those kinds can show an image:
//!
//! | invocation             | stdout | an image can be drawn  |
//! |------------------------|--------|------------------------|
//! | `git diff`             | a tty  | yes, right where it is |
//! | `git diff \| less`     | a pipe | not into the pipe      |
//! | `git diff > patch.txt` | a file | no                     |
//!
//! A pipe means something else is reading the output — git's pager, or another
//! program — and pixels written into it are lost or corrupt the reader. A
//! regular file is a deliberate redirect, so escapes must never be written to
//! it. In both cases the terminal the user is sitting at is still reachable
//! through `/dev/tty`, which is where a pipe-directed run sends its image.

use std::fs::File;
use std::io::{self, IsTerminal, Write};
use std::os::fd::{AsRawFd, RawFd};

/// Cell size assumed when the terminal does not report its pixel dimensions.
///
/// The usual cell is about twice as tall as it is wide, which is the ratio
/// that keeps a build from being stretched when the image is scaled into its
/// cell box. The absolute values only set how much detail is available.
const ASSUMED_CELL: (u32, u32) = (12, 24);

/// Terminal size in cells, plus the pixel size of one cell.
#[derive(Clone, Copy, Debug)]
pub struct Screen {
    pub columns: u32,
    pub rows: u32,
    pub cell_px: (u32, u32),
}

impl Screen {
    /// Read the geometry of the terminal behind `fd`.
    ///
    /// The pixel dimensions come from the same `TIOCGWINSZ` the cell counts do,
    /// so the image is sized from what the terminal actually reports rather
    /// than from an assumed aspect ratio. Terminals that leave the pixel fields
    /// zero fall back to [`ASSUMED_CELL`], which costs nothing but the exact
    /// aspect ratio.
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

    /// A usable size for a run with no terminal anywhere.
    pub fn fallback() -> Self {
        Self {
            columns: 100,
            rows: 30,
            cell_px: ASSUMED_CELL,
        }
    }
}

/// One `TIOCGWINSZ` on `fd`.
fn winsize(fd: RawFd) -> Option<libc::winsize> {
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: `size` is a live `winsize` and `fd` is an open descriptor; the
    // kernel writes exactly one `winsize` through the pointer.
    let result = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut size) };
    (result == 0).then_some(size)
}

/// True when `fd` is a pipe or a socket rather than a file or a terminal.
///
/// This is the difference between output being read as it is produced — by a
/// pager or another program — and being collected into a file.
fn is_stream(fd: RawFd) -> bool {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `stat` is a live `stat` and `fd` is an open descriptor.
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return false;
    }
    let kind = stat.st_mode & libc::S_IFMT;
    kind == libc::S_IFIFO || kind == libc::S_IFSOCK
}

/// The stream a run writes to.
pub struct Output {
    stream: Box<dyn Write>,
    /// Whether `stream` is a terminal, and so can show an image.
    terminal: bool,
    pub screen: Screen,
}

impl Output {
    /// Choose the stream for this run.
    pub fn detect() -> Self {
        let stdout = io::stdout();
        if stdout.is_terminal() {
            return Self {
                screen: Screen::measure(stdout.as_raw_fd()),
                stream: Box::new(stdout),
                terminal: true,
            };
        }

        // stdout goes somewhere else. Divert to the terminal only when
        // something is *reading* that somewhere — a pager or another program,
        // which will not see the image either way — so the picture still
        // reaches the user. A regular file is a redirect the user asked for,
        // and rewriting it to the terminal would silently empty their file.
        if is_stream(stdout.as_raw_fd())
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

    /// Whether image escapes can be written to this stream.
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
