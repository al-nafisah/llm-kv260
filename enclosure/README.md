# Kria KV260 x4 + TL-SG108 tower enclosure (OpenSCAD)

`kria_tower.scad` is a fully parametric, screw-free stacking tower:
bottom = TP-Link TL-SG108 tray, then four Kria KV260 trays, then a honeycomb cap.
Every level is a separate print that locates on square pegs/sockets at the four corner posts.
Each level has a front cable bay hidden behind a snap-on honeycomb fascia.

| Item | Value (defaults) |
|---|---|
| Tray body (W x D) | 187 x 171 mm (193.5 x 176.5 mm with fascia) |
| Cable bay depth | 40 mm in front of the port edge |
| KV260 level pitch | 51.5 mm (2.5 floor + 6 standoff + 36 kit + 7 fan clearance) |
| Switch level pitch | 32.5 mm (2.5 floor + 25 switch + 5 clearance) |
| Top cap | 7.5 mm |
| Total height | 246 mm (limit 250, asserted in the file) |

## Files

- `kria_tower.scad` — params block, `kv_tray()`, `switch_tray()`, `top_cap()`, `fascia()`, `assembly()`, `print_layout()`.
- `stl/` — `kv_tray`, `switch_tray`, `top_cap`, `kv_fascia`, `sw_fascia` (OpenSCAD 2021.01).
- `previews/` — software-rendered views of the STLs.

## Print list

| Part | Qty | Orientation | Notes |
|---|---|---|---|
| `kv_tray` | 4 | floor down | 187 x 171 mm footprint |
| `switch_tray` | 1 | floor down | has the cable exit windows |
| `top_cap` | 1 | upside-down | honeycomb top |
| `kv_fascia` | 4 | face down | textured bed sheet gives a nice front finish |
| `sw_fascia` | 1 | face down | |

Hardware: none. No screws, magnets or glue; the fascias snap on.

## Export / preview

```sh
for p in kv_tray switch_tray top_cap kv_fascia sw_fascia; do
  openscad -D "part=\"$p\"" -o stl/$p.stl kria_tower.scad
done
openscad -D 'part="print_layout"' kria_tower.scad   # GUI: all parts in print orientation
```

## Cable management

- **Bay**: every tray extends 40 mm in front of the port edge. Plugs (RJ45, USB, HDMI/DP, 12 V barrel)
  live in this bay behind the fascia.
- **Shafts**: two 30 mm wide openings in each bay floor, left (Ethernet, next to the KV260 RJ45) and right
  (12 V DC, next to the KV260 jack), let cables drop straight to the level below. Zip-tie slots sit beside each shaft.
- **Exit**: the bottom (switch) tray has a house-shaped window in each side wall of its bay, so the four 12 V leads
  leave on the right and the uplink leaves on the left without lifting the tower. Switch power goes out the open rear.
- **Fascia**: floats 2.5 mm off the posts on four bosses and leaves a 1.5 mm shadow gap between levels.
  A wing at each end wraps the tower's front corner; a 45° ridge inside it clicks into a vertical groove in the
  front post's side face. The honeycomb keeps the switch and KV260 LEDs visible and keeps the bay ventilated.
  Pull one off to reach a port.

## Style touches (all within a 45° overhang)

- 3 mm rounded vertical corners on every tray and the cap.
- Chevron side vents: 45° slashes leaning toward the middle of each wall.
- Honeycomb (pointy-top hexagons, 7 mm across flats) on fascias and cap. Pointy-top keeps every hole edge <= 30° from vertical.
- Two-tone suggestion: trays and cap in one colour, fascias in an accent colour.

## Orientation in the tray

Front bay: KV260 RJ45, 4x USB 3.0, HDMI, DisplayPort and the 12 V DC jack; switch 8x RJ45.
Rear (open): KV260 microSD (J11), micro-USB UART/JTAG (J4), Pmod; switch 9 V DC socket (rear-right).
Side walls carry the chevron vents; every floor is slotted and the cap is honeycombed so each fansink draws
air through the floor above it.

## Verified dimensions

- KV260 (AMD DS986, Mechanical): kit 123 x 140 x 36 mm; carrier card 123 x 140 x 23 mm; SOM + fansink 60 x 77 x 27 mm.
- KV260 (AMD UG1089 Fig. 2, scaled drawing): PCB 140 x 119 mm; four corner holes ~Ø3.5 mm, 5 mm in from each edge;
  connector edges as listed above. 12 V jack barrel 5.5 mm OD / 2.5 mm ID (Table 3).
- Allied Vision 22563 adapter guide Fig. 3: its Ø3.4 holes sit at 5 and 114 mm on a 119 mm edge and mate with the
  KV260's right-edge holes, which independently confirms the 119 mm depth and 5 mm hole inset.
- TL-SG108 (TP-Link TL-SG105/108 datasheet): 158 x 101 x 25 mm, external 9 V / 0.6 A adapter;
  user guide: RJ45 ports on the front panel, power socket on the rear panel.

## Assumptions (change the matching parameter if your hardware differs)

1. **KV260 hole positions** were scaled from the UG1089 line drawing, not read from a dimensioned drawing
   (AMD does not publish one outside the login-gated STEP model). Expect +/-0.5 mm. `kv_holes`, `kv_hole_d`.
   Locating pegs are Ø3.0 (hole minus 2x `fit_tol`); switch `kv_standoff_mode="screw"` for Ø2.5 pilots + M3 screws.
2. **36 mm kit height is applied from the PCB underside** (`kv_kit_h`). DS986 measures it from the lowest point of
   the kit, so the real fansink top is ~4 mm lower and the true clearance is ~11 mm, not 7.
3. **Underside components <= 4.5 mm** tall, so 6 mm standoffs clear them (`kv_standoff_h`).
4. **PCB thickness 1.6 mm** (`kv_pcb_t`); locating pegs stand 0.9 mm proud of the PCB.
5. **Front connector overhang 4 mm** (123 - 119) (`kv_front_ovh`).
6. **40 mm bay depth** suits standard RJ45 plugs with a short or snagless boot turning sideways into the shaft;
   long moulded boots or stiff Cat6a may need `bay_d = 50` (still inside the 220 mm envelope) or 90° plugs.
7. **TL-SG108 25 mm height includes its rubber feet**, and its RJ45 openings start >= 4 mm above the chassis bottom,
   so a 2 mm board stop does not block plugs (`sw_front_lip_h`). The rear stop bar is 40 mm wide and centred, so the
   rear-right DC socket stays clear (`sw_rear_stop_w`).
8. **Fascia snap clips** grip `clip_catch - fit_tol` = 0.55 mm deep on each side. While the wings spread, the fascia
   plate bows toward the tower (estimated ~2 mm at mid-span, inside the 2.5 mm boss gap). For a firmer hold raise
   `clip_catch` to ~0.9; beyond that the plate is estimated to bottom out on the tray's floor edge while snapping on.
   For an easier pull lower it to 0.6. `clip_preload` (0.7 mm) pulls the fascia snug onto its bosses; it must exceed
   2 x `fit_tol` to do so. The hold was sized by calculation, not measured: check it on the first test print.
9. **Peg/socket fit 0.25 mm per side** (`fit_tol`) suits a well-tuned FDM printer; use 0.3-0.35 for a loose fit.
10. **The switch sits at the bottom** (heaviest, fanless); trays are symmetric front/back only in outline, so stack
    them with the bays all facing the same way.
11. Boards are held by gravity plus locating pegs; nothing stops a board from being lifted straight up.
12. Fansinks draw air from above through the slotted floor of the next level (or the honeycomb cap) and exhaust
    through the chevron wall slots. No extra fans assumed.
