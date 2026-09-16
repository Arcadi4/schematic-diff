# schematic-diff

`git diff` renderer for Minecraft schematics in the terminal.

![demo](/img/demo.png)

## Supported Formats

| Extension | File format |
| --- | --- |
| `.litematic` | Litematica |
| `.schem` | Sponge schematic |
| `.schematic` | MCEdit |
| `.nbt` | Java structure block |
| `.snbt` | Structure SNBT |
| `.mcstructure` | Bedrock structure |
| `.nusn` | Nucleation snapshot |


## Install

```sh
cargo install schematic-diff              # rust natively
brew install arcadi4/tap/schematic-diff   # homebrew
```

Image output requires the kitty image protocol. This is known to be implemented by the following terminal emulators:

- Kitty
- Ghostty
- iTerm2
- Warp
- WezTerm

There are more. Check your terminal's docs.

## Set up git

For this to work with the native `git diff` command, set `schematic-diff` as an external diff program.

YOLO setup script:

```sh
git config --global diff.schematic.command schematic-diff
mkdir -p ~/.config/git
cat >> ~/.config/git/attributes <<'EOF'
*.litematic   diff=schematic
*.schem       diff=schematic
*.schematic   diff=schematic
*.nbt         diff=schematic
*.snbt        diff=schematic
*.mcstructure diff=schematic
*.nusn        diff=schematic
EOF
```

Or write the config files manually in `~/.gitconfig` and define the renderer:

```ini
[diff "schematic"]
    command = schematic-diff
```

Then, in `.gitattributes` (per repo) or `~/.config/git/attributes` (global), mark the following file types to use `schematic-diff`:

```text
*.litematic   diff=schematic
*.schem       diff=schematic
*.schematic   diff=schematic
*.nbt         diff=schematic
*.snbt        diff=schematic
*.mcstructure diff=schematic
*.nusn        diff=schematic
```

Verify:

```sh
git config --get diff.schematic.command
git check-attr diff -- build.litematic
```

Should print `schematic-diff` and `build.litematic: diff: schematic`.

## Flags

```text
--yaw=<degrees>     Camera horizontal angle            [default: 45]
--pitch=<degrees>   Camera elevation above the build   [default: 30]
--zoom=<factor>     Larger zooms in                    [default: 1.0]
--pack=<file.zip>   Draw with a resource pack's models and textures
--kitty             Enforce image rendering even if the terminal is unrecognised
--no-kitty          Never render images, text summary only
--output=<file>     Also write the composited image as a PNG
-h, --help          Show this help
-V, --version       Show the version
```

## Textures

Without `--pack`, every block renders as a flat cube with pure color. For a better experience, please bring your own vanilla textures and pass them with `--pack`. We cannot redistribute Mojang's assets in the binary.

![no resource pack](/img/no-pack.png)

> Just like this

Once you've got the resource pack, configure your `~/.gitconfig` as follows instead:

```ini
[diff "schematic"]
    command = schematic-diff --pack path/to/pack
```

## Acknowledgements

Thanks to [Nucleation](https://github.com/Schem-at/Nucleation) for the format parsers and mesher.

The demo uses a [tree farm](https://github.com/DuskScorpio/Scorpio-File-Downloads/tree/main/Files/Gemini) by @DuskScorpio.
