//! Kitty graphics protocol support for transmitting a PNG over terminal cells.
//!
//! Reference: <https://sw.kovidgoyal.net/kitty/graphics-protocol/>. PNG payloads
//! use `f=100` and are compressed before base64 encoding.

use std::io::{self, Write};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

/// Protocol maximum number of base64 characters in one escape sequence.
const CHUNK_CHARS: usize = 4096;

/// Raw bytes that encode to one full chunk without base64 padding.
const CHUNK_BYTES: usize = CHUNK_CHARS / 4 * 3;

const APC: &[u8] = b"\x1b_G";
const ST: &[u8] = b"\x1b\\";

pub fn supported(mode: Option<bool>) -> bool {
    // Active terminal queries need raw mode and a timeout race, which can hang
    // an external diff. Environment variables are the safer heuristic.
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

/// Wrap Kitty escape sequences for tmux passthrough. tmux requires
/// `set -g allow-passthrough on`.
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

/// Transmits a PNG in protocol-sized chunks and positions it over the cell
/// rectangle. `C=1` leaves the cursor position unchanged.
pub fn write_png(out: &mut impl Write, png: &[u8], columns: u32, rows: u32) -> io::Result<()> {
    let mut chunks = png.chunks(CHUNK_BYTES).peekable();
    let mut first = true;
    while let Some(chunk) = chunks.next() {
        let more = u8::from(chunks.peek().is_some());
        let mut sequence = Vec::with_capacity(chunk.len() / 3 * 4 + 64);
        sequence.extend_from_slice(APC);
        if first {
            // `a=T` transmits and displays, `f=100` selects PNG, and `q=2`
            // suppresses terminal replies that would remain in the input queue.
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
