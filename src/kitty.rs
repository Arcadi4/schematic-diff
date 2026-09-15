//! The kitty graphics protocol: capability detection and PNG display.
//!
//! Reference: <https://sw.kovidgoyal.net/kitty/graphics-protocol/>.
//!
//! Only the part a print-and-exit diff tool needs is implemented: transmit a
//! PNG and place it over a rectangle of cells. Images are sent with `f=100`
//! (PNG), so the payload is compressed before it is base64'd rather than after.

use std::io::{self, Write};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

/// The largest number of base64 characters one escape sequence may carry; the
/// spec fixes this at 4096.
const CHUNK_CHARS: usize = 4096;

/// Raw bytes that encode to exactly one whole chunk.
///
/// 3072 is a multiple of 3, so base64 emits no padding and every chunk but the
/// last is exactly [`CHUNK_CHARS`] characters — which is what the spec requires
/// of all but the final chunk.
const CHUNK_BYTES: usize = CHUNK_CHARS / 4 * 3;

/// Introducer for an APC (Application Programming Command) sequence.
const APC: &[u8] = b"\x1b_G";
/// String terminator.
const ST: &[u8] = b"\x1b\\";

/// Whether the attached terminal can be asked to draw an image.
pub fn supported(mode: Option<bool>) -> bool {
    // Detection is by environment variable, which is a heuristic: the
    // authoritative method is to query the terminal and wait for a reply, but
    // that needs the tty in raw mode and a timeout to race, and losing that race
    // hangs the tool. `--kitty` / `--no-kitty` cover anything this misses.
    let Some(forced) = mode else {
        return std::env::var_os("KITTY_WINDOW_ID").is_some()
            || std::env::var_os("KONSOLE_VERSION").is_some()
            || std::env::var("TERM").is_ok_and(|term| term.contains("kitty"))
            // Ghostty and Konsole announce themselves under their own names.
            || std::env::var("TERM_PROGRAM").is_ok_and(|program| {
                let program = program.to_ascii_lowercase();
                program == "ghostty" || program == "konsole"
            });
    };
    forced
}

/// Wrap one escape sequence for transport through tmux, if tmux is in the way.
///
/// tmux needs `set -g allow-passthrough on` for the wrapped sequence to reach
/// the terminal underneath.
fn framed(sequence: &[u8]) -> Vec<u8> {
    if std::env::var_os("TMUX").is_none() {
        return sequence.to_vec();
    }
    let mut wrapped = Vec::with_capacity(sequence.len() + 16);
    wrapped.extend_from_slice(b"\x1bPtmux;");
    for byte in sequence {
        // A literal ESC inside the passthrough payload must be doubled.
        if *byte == 0x1b {
            wrapped.push(0x1b);
        }
        wrapped.push(*byte);
    }
    wrapped.extend_from_slice(ST);
    wrapped
}

/// Write the escape sequences that place `png` over `columns` x `rows` cells.
///
/// `C=1` leaves the cursor where it was, so the caller decides what follows an
/// image instead of depending on where the terminal would have moved it.
///
/// The PNG is encoded a chunk at a time rather than in one pass, so the base64
/// of a whole screenshot is never held alongside the screenshot itself.
pub fn write_png(out: &mut impl Write, png: &[u8], columns: u32, rows: u32) -> io::Result<()> {
    let mut chunks = png.chunks(CHUNK_BYTES).peekable();
    let mut first = true;
    while let Some(chunk) = chunks.next() {
        let more = u8::from(chunks.peek().is_some());
        let mut sequence = Vec::with_capacity(chunk.len() / 3 * 4 + 64);
        sequence.extend_from_slice(APC);
        if first {
            // `a=T` transmit and display, `f=100` PNG, and `q=2` suppresses
            // both the OK and the failure reply so nothing is left in the
            // input queue for whatever runs next.
            write!(sequence, "a=T,f=100,c={columns},r={rows},C=1,q=2,m={more};")?;
            first = false;
        } else {
            write!(sequence, "m={more};")?;
        }
        sequence.extend_from_slice(BASE64.encode(chunk).as_bytes());
        sequence.extend_from_slice(ST);
        out.write_all(&framed(&sequence))?;
    }
    Ok(())
}
