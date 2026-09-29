# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file parametric OpenSCAD model (`kria_tower.scad`) of a screw-free, magnet-free stacking tower: one TP-Link TL-SG108 switch tray at the bottom, four Kria KV260 trays, and a honeycomb top cap. Every level also has a snap-on honeycomb fascia on the front. `README.md` covers the print list, cable routing, the sources for the verified dimensions, and the numbered **Assumptions** (each one names the parameter to change). `stl/` and `previews/` are generated outputs that are committed. The repo has no build system, tests, or preview-rendering script.

The STLs were exported with OpenSCAD 2021.01, which is the installed version.

## Commands

Quick check (well under a second, no CGAL render). This evaluates every `assert` and prints the dimension summary `ECHO`:

```sh
openscad -o /tmp/check.echo kria_tower.scad 2>&1 | grep -E 'WARNING|ERROR'; cat /tmp/check.echo
```

- In 2021.01 a failed `assert` **still exits 0**. Errors only show up on stderr, so grep for `ERROR`/`WARNING` and don't rely on the exit code.
- Use `-D 'part="kv_tray"'` to pick a part. Use `-D` for parameters too, e.g. `-D bay_d=50`. Without `-D part=…`, OpenSCAD uses whatever `part =` is set to at the top of the file, and the user changes that while previewing in the GUI.
- `part` values: `assembly`, `print_layout`, `kv_tray`, `switch_tray`, `top_cap`, `kv_fascia`, `sw_fascia`.

Preview check: this is what the GUI's F5 does. The `.echo` check above does **not** catch preview failures. Exporting a PNG without `--render` goes through the same OpenCSG preview path (it needs a display):

```sh
openscad -D 'part="assembly"' -o /tmp/preview.png kria_tower.scad 2>&1 | grep -E 'WARNING|ERROR|Normalized CSG'
```

A healthy run prints `Normalized CSG tree has N elements`. That's a few hundred for `assembly` and under 100 for each tray. `Aborting normalization` / `resulted in an empty tree` means F5 shows nothing (see "Preview budget" below).

Export STLs (a full CGAL render, several seconds per part):

```sh
for p in kv_tray switch_tray top_cap kv_fascia sw_fascia; do
  openscad -D "part=\"$p\"" -o stl/$p.stl kria_tower.scad
done
```

Exports are not byte-reproducible: facet order and some lines change between runs even when the geometry doesn't. Re-export only the parts whose geometry actually changed, or the git diffs get huge. To check whether geometry changed, compare facet count, volume and bounding box rather than bytes.

## Architecture

The file runs top to bottom: `PARAMS` blocks → `DERIVED` values → `echo` + `assert`s → primitives → `tray_shell` → level modules → `fascia` → `top_cap` → ghost models → `assembly`/`print_layout` → the `part` dispatch chain at the end.

- **Parameters vs derived values.** Change only the `PARAMS` values. Tray size (`W_out`, `D_out`), overall fascia width (`fascia_w`), level pitches (`kv_pitch`, `sw_pitch`), `cap_h`, `total_h`, post positions, board origins (`kv_x0/kv_y0`, `sw_x0/sw_y0`) and shaft positions are all computed in `DERIVED`. The asserts check the 220×220×250 mm envelope, the collision constraints and the clip geometry. The README's dimension table repeats the echoed numbers, so update it whenever the defaults change.
- **Coordinate convention (all modules).** The origin is the outer front-left-bottom corner of the tray body: +X right (viewed from the front), +Y toward the rear, +Z up. The cable bay covers `0..bay_d` in Y, and the board stop sits at `stop_y = bay_d`. Fascias sit at negative Y. `fascia()` is modelled upright in its own local frame (front face at y=0, bottom edge at z=0, bosses ending at the tower's front face), and `place_fascia()` moves it into position. The fascia's X matches the trays, so its clip wings sit at x < 0 and x > `W_out`.
- **`tray_shell(h, …)` is shared** by `kv_tray` and `switch_tray`. `bottom_level=true` changes three things: exit windows replace cable drop shafts, the bay part of each side wall gets no vents, and the switch tray is the only one with windows. The whole shell is intersected with `outline_prism` to round the corners. Parts added in the `kv_tray`/`switch_tray` unions (standoffs, switch guides, rear stop) sit outside that intersection and don't get clipped.
- **Stacking interface.** A tray of height `h` has pegs on its post tops at `z=h` and sockets cut upward from `z=0` in the same posts, so the next level sits at `z=h`. Socket cutters get a 45° pyramid roof so trays print floor-down. The cap prints upside-down, so it uses flat sockets (`cap_flat_socket`).
- **Fascia ↔ tray snap clips** (no magnets):
  - `fascia_plan()` is the fascia's plan-view outline: the plate plus a wing at each end that wraps the tower's front corner.
  - A 45° ridge on each wing's inner face clicks into the V-groove that `clip_grooves(h)` cuts into the front posts' outer side faces.
  - Both halves get their positions from `clip_y`, `clip_catch`, `clip_preload` and `clip_margin_z`, so change the clip only through those parameters.
  - With `clip_preload > 2*fit_tol` the ridge rests on the groove's front slope and pulls the fascia back onto its bosses (heights from `fascia_boss_z`).
  - The long plate provides the spring by bowing, which is why flex runs along the print layers and the wings themselves stay stiff.
- **Floor vents** skip keep-out discs `[x, y, r]`, which `kv_tray` passes so the slots avoid the standoff pads.
- **Ghost models** (`kv260_ghost`, `switch_ghost`) and the translucent envelope box appear only in `assembly()`, controlled by `show_boards`/`show_fascias`. They are never exported. The port positions in them come from UG1089 and are used for visual checks only.
- **Adding a part** touches four places: the `part` enum comment, the dispatch chain at the end of the file, `print_layout()`, and the export loop (in the file's header comment and in the README).

## Design constraints to preserve

- **Preview budget.** Build repeated cutouts in 2D and extrude each one once: `side_vents_2d`, `floor_vents_2d`, `cable_shafts_2d` and `hex_grid` already work this way. Never subtract many 3D primitives, or any 3D `intersection()`, inside `tray_shell`/`top_cap`/`fascia`. Before drawing a preview, OpenSCAD rewrites the model as a flat list of shapes to add and subtract. `A − (B ∩ C)` turns into two copies, and every copy lists every subtracted shape, so the list multiplies quickly. The preview aborts at 100k elements from the CLI (200k by default in the GUI) and renders empty, while the `.echo` check and STL export still succeed.
- **Printability.** Nothing may overhang more than 45° in its print orientation: trays floor-down, cap upside-down, fascias face-down. That's why honeycombs use pointy-top hexagons (every edge within 30° of vertical), why vents are 45° chevrons or pointed vertical slots, why exit windows have a 45° house roof, why pegs have 45° chamfers, and why the clip ridges and grooves are 45° V's with grooves tapering at 45° at their ends.
- **Layer direction.** Don't put flexing features on the fascia's back that bend across the print layers (split pegs, thin hooks). Face-down, anything projecting backward is built from stacked layers and snaps off along them.
- **Fit tolerance.** Every mating clearance (pegs, board locating pegs, clip ridge/groove) goes through `fit_tol`, so printer tuning is a single parameter.
- **README sync.** Parameter names cited in the README's Assumptions section must keep matching the `.scad` file.
