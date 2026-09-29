// =============================================================================
//  kria_tower.scad  —  Stacked tower enclosure: 4x AMD Kria KV260 + 1x TP-Link TL-SG108
// =============================================================================
//  Layout (bottom -> top):  switch tray, KV260 tray x4, top cap.
//  Every level is a separate print.  Levels locate on square pegs/sockets at the
//  four corner posts — no screws anywhere.
//
//  Each tray has a front CABLE BAY (bay_d deep) in front of the port edge.  Plugs
//  live in the bay; cables drop to the level below through shafts at the left
//  (Ethernet) and right (12 V DC) ends of the bay floor, and leave the tower through
//  house-shaped exit windows in the bottom (switch) tray's side walls.  A per-level
//  honeycomb FASCIA (snap-clipped over the tower's front corners, floating 2.5 mm off
//  the posts with a shadow gap between levels) hides everything while keeping LEDs
//  visible and air flowing.  Pull a fascia off to reach any port.  No magnets, no glue.
//
//    front bay : KV260 RJ45 / 4x USB3 / HDMI / DP / 12 V DC jack,   switch 8x RJ45
//    rear (open): KV260 microSD (J11) + micro-USB UART/JTAG (J4) + Pmod, switch 9 V DC jack
//
//  Coordinate convention (every module): origin = outer front-left-bottom corner
//  of the tray body, +X to the right (viewed from the front), +Y toward the rear, +Z up.
//  Fascias sit at negative Y.
//
//  Print orientation: trays as modelled (floor on the bed); top cap upside-down;
//  fascias face-down (print_layout() shows all of this).  No feature exceeds 45°.
//
//  CLI export:
//    for p in kv_tray switch_tray top_cap kv_fascia sw_fascia; do
//      openscad -D "part=\"$p\"" -o stl/$p.stl kria_tower.scad; done
//
//  Dimension sources (verified 2026-09-14):
//    * AMD DS986 "Kria KV260 Vision AI Starter Kit Data Sheet", Mechanical:
//        kit 123 x 140 x 36 mm, carrier card 123 x 140 x 23 mm, SOM+fansink 60 x 77 x 27 mm
//    * AMD UG1089 "KV260 Starter Kit User Guide", Fig. 2 (scaled drawing X24750):
//        PCB outline 140 x 119 mm (123 mm depth = PCB + ~4 mm front connector overhang),
//        4 corner mounting holes Ø3.5 mm, 5 mm in from each edge; connector edges.
//        12 V jack = 5.5 mm OD / 2.5 mm ID barrel (Table 3).
//    * Allied Vision 22563 adapter user guide Fig. 3: mates to the KV260's two
//        right-edge holes at 5 mm and 114 mm along a 119 mm edge (Ø3.4) — confirms above.
//    * TP-Link TL-SG105/108 datasheet: TL-SG108 = 158 x 101 x 25 mm (W x D x H),
//        external adapter 9 V / 0.6 A; user guide: 8x RJ45 on front, power socket on
//        the rear panel (right side).
// =============================================================================

// ----------------------------- PARAMS: what to show ------------------------
part = "assembly";   // ["assembly","print_layout","kv_tray","switch_tray","top_cap","kv_fascia","sw_fascia"]
show_boards  = true; // ghost KV260 / switch models in assembly()
show_fascias = true; // fascias in assembly()
$fn = 48;

// ----------------------------- PARAMS: printing ---------------------------
fit_tol       = 0.25;  // per-side clearance for peg->socket and board->guide fits [mm]
wall_t        = 3.0;   // side wall / lip thickness (2.5–3 mm print-friendly)
floor_t       = 2.5;   // tray floor thickness
post_min_wall = 2.5;   // min. material around a socket inside a corner post
corner_r      = 3;     // plan-view rounding of the tower's vertical corners (0 = square)

// ----------------------------- PARAMS: stacking pegs -----------------------
peg_s         = 7.0;   // square peg side
peg_h         = 5.0;   // peg height above the post top
peg_chamfer   = 1.5;   // 45° chamfer on the peg top
socket_extra  = 0.5;   // socket depth beyond peg_h (so the tray seats on the walls, not the pegs)

// ----------------------------- PARAMS: envelope ---------------------------
env_w = 220;  env_d = 220;  env_h = 250;   // must fit inside this box

// ----------------------------- PARAMS: cable bay & fascia -----------------
bay_d          = 40;    // depth of the plug/cable bay in front of the port edge
drop_w         = 30;    // cable drop shaft width (X) at each side of the bay floor
drop_margin    = 3;     // shaft distance from the corner post / front edge
tie_slots      = true;  // 4 x 1.6 mm zip-tie anchor slots beside each shaft
sw_exit_w      = 24;    // cable exit window width in the bottom tray's bay side walls
sw_exit_h      = 12;    // straight height of that window (+ a 45° pointed top)

fascia_t       = 3;     // fascia plate thickness
fascia_reveal  = 1.5;   // shadow gap between the fascias of adjacent levels
fascia_boss_h  = 2.5;   // stand-off bosses on the fascia back => plate floats off the posts
fascia_boss_d  = 10;    // boss Ø; the bosses bear on the front faces of the two front posts
fascia_boss_z  = [14, -6]; // boss heights above the level's floor plane; negative = from the level top
fascia_end_r   = 1.2;   // rounding of the fascia's outer vertical edges (< fascia_t/2 and < clip_t/2)
// Snap clips: a wing at each fascia end wraps the tower's front corner, and a 45° ridge on its
// inner face clicks into a vertical V-groove in the front post's outer side face.  The long plate
// bows a little to let the wings spread, so the flex runs along the print layers.
clip_t         = 3;     // wing thickness
clip_y         = 9;     // groove centre, measured back from the tower's front face
clip_catch     = 0.8;   // ridge height; grip depth = clip_catch - fit_tol (0.6 = easier pull ... 0.9 = firmer)
clip_preload   = 0.7;   // ridge sits this far ahead of the groove centre; > 2*fit_tol pulls the fascia onto its bosses
clip_margin_z  = 8;     // ridge and groove stop this far short of the fascia's top and bottom edges

hex_af         = 7;     // honeycomb hole across-flats
hex_wall       = 2;     // rib between honeycomb holes
fascia_margin_x = 16;   // solid margin at the fascia ends (covers the posts / bosses)
fascia_margin_z = 5;    // solid margin top & bottom
cap_hex        = true;  // honeycomb on the cap instead of slots

// ----------------------------- PARAMS: KV260 -----------------------------
kv_pcb_w     = 140;    // PCB width  (port edge, X)               [DS986 / UG1089]
kv_pcb_d     = 119;    // PCB depth  (Y) — 123 mm in DS986 includes connector overhang
kv_pcb_t     = 1.6;    // PCB thickness (assumed, standard)
kv_front_ovh = 4;      // front connectors overhang beyond the PCB edge (123-119)
kv_kit_h     = 36;     // top of fansink above the PCB *underside* (DS986: 36 mm kit height,
                       // measured conservatively from the PCB bottom — see README)
kv_hole_d    = 3.5;    // mounting hole diameter (scaled from UG1089 Fig. 2; AV adapter uses 3.4)
kv_holes     = [[5,5],[kv_pcb_w-5,5],[5,kv_pcb_d-5],[kv_pcb_w-5,kv_pcb_d-5]]; // from PCB front-left
kv_standoff_d = 7.0;   // standoff outer diameter
kv_standoff_h = 6.0;   // standoff height (clears ~4.5 mm of underside components)
kv_standoff_mode = "peg"; // ["peg","screw"]: locating peg into the hole, or Ø2.5 pilot for M3 screws
kv_peg_h      = 2.5;   // locating peg height above the standoff (PCB 1.6 + 0.9 proud)
kv_fan_clear  = 7;     // free air above the fansink to the (slotted) floor of the next level
kv_front_gap  = 2;     // PCB front edge set back from the inner face of the board stop
kv_rear_gap   = 4;     // PCB rear edge to inner face of the rear lip
kv_lip_h      = 4;     // board-stop / rear lip height (must stay below kv_standoff_h)

// KV260 ghost-model geometry (from UG1089 Fig. 2, mm from PCB front-left corner)
kv_som_xy   = [5.5, 32];  kv_som_wd = [77, 60];  kv_som_h = 27; // SOM + fansink footprint
kv_fan_d    = 40;
kv_front_ports = [ // [x0, x1, height, label]  y: -kv_front_ovh .. ~15
  [13.3, 32.7, 13.5, "RJ45"], [37.2, 51.6, 15.5, "USB3 x2"], [56.6, 71.0, 15.5, "USB3 x2"],
  [75.6, 90.6,  6.0, "HDMI"], [96.9,114.7,  6.0, "DP"],       [118.9,127.8, 11.0, "DC 12V"] ];
kv_rear_ports  = [ // [x0, x1, height, label]  at the rear (Y = kv_pcb_d) edge
  [22, 35, 2.5, "microSD"], [81.6, 89.5, 3.0, "microUSB UART/JTAG"], [97, 111, 8.5, "Pmod"] ];

// ----------------------------- PARAMS: TL-SG108 ---------------------------
sw_w = 158;  sw_d = 101;  sw_h = 25;   // W x D x H incl. rubber feet [TP-Link datasheet]
sw_side_tol   = 0.5;   // per-side clearance to the side guides
sw_front_gap  = 1;     // switch front face set back from inner face of the board stop
sw_front_lip_h = 2;    // low stop so the RJ45 openings (≈4 mm above the chassis bottom) stay clear
sw_guide_t    = 3;     // side guide block thickness
sw_guide_h    = 8;     // side guide block height
sw_guide_len  = 30;    // side guide block length
sw_rear_stop_w = 40;   // rear stop bar width (centred; DC jack is at the rear-right, stays clear)
sw_rear_stop_h = 5;
sw_top_clear  = 5;     // air gap above the switch to the next floor

// ----------------------------- PARAMS: ventilation ------------------------
vent_style       = "chevron"; // ["chevron","vertical"] side-wall slot style
side_slot_w      = 3;    // side-wall slot width
side_slot_pitch  = 7;    // slot pitch along Y
side_slot_margin = 3;    // margin from floor / wall top / posts
floor_slot_w     = 3;    // floor slot width (X)
floor_slot_len   = 18;   // floor slot length (Y)
floor_slot_gap   = 3.5;  // rib between slots
floor_margin     = 3;    // solid border around the floor cavity

// ----------------------------- PARAMS: side gaps --------------------------
side_gap = 2;          // clearance between the widest board and a corner post

// ----------------------------- PARAMS: top cap ----------------------------
cap_top_t = 2;         // material above the sockets in the cap
cap_flat_socket = true; // cap prints upside-down, so its sockets need no 45° roof

// =============================================================================
//  DERIVED
// =============================================================================
eps      = 0.01;
post_xy  = peg_s + 2*fit_tol + 2*post_min_wall;        // corner post side
post_in  = post_xy - wall_t;                            // how far a post intrudes into the cavity
cav_w    = max(kv_pcb_w, sw_w) + 2*(post_in + side_gap);
W_out    = cav_w + 2*wall_t;
stop_y   = bay_d;                                       // Y of the front board stop (lip)
cav_d    = kv_pcb_d + kv_front_gap + kv_rear_gap;
D_out    = bay_d + wall_t + cav_d + wall_t;             // tray body depth incl. bay
kv_pitch = floor_t + kv_standoff_h + kv_kit_h + kv_fan_clear;
sw_pitch = floor_t + sw_h + sw_top_clear;
sock_s   = peg_s + 2*fit_tol;
sock_d   = peg_h + socket_extra;                        // straight part of the socket
sock_roof = sock_s/2;                                   // 45° pyramid roof height
cap_h    = sock_d + (cap_flat_socket ? 0 : sock_roof) + cap_top_t;
total_h  = sw_pitch + 4*kv_pitch + cap_h;
total_d  = D_out + fascia_boss_h + fascia_t;
kv_x0    = (W_out - kv_pcb_w)/2;                        // KV260 PCB origin inside the tray
kv_y0    = stop_y + wall_t + kv_front_gap;
sw_x0    = (W_out - sw_w)/2;
sw_y0    = stop_y + wall_t + sw_front_gap;
post_pos = [[0,0],[W_out-post_xy,0],[0,D_out-post_xy],[W_out-post_xy,D_out-post_xy]];
front_posts = [post_pos[0], post_pos[1]];
drop_x   = [wall_t + post_in + drop_margin, W_out - wall_t - post_in - drop_margin - drop_w];
drop_y0  = post_xy + drop_margin;
drop_len = bay_d - drop_y0 - drop_margin;
fascia_w = W_out + 2*(fit_tol + clip_t);                 // overall fascia width incl. clip wings
clip_wing_y = clip_y + clip_catch + 2;                   // how far behind the tower front the clip wings reach
hex_r    = hex_af/sqrt(3);                               // circumradius of the hex hole
hex_px   = hex_af + hex_wall;                            // honeycomb pitch (columns)
hex_py   = hex_px*sqrt(3)/2;                             // honeycomb pitch (rows)
function boss_z(h) = [for (z = fascia_boss_z) z < 0 ? h + z : z];

echo(str("Tray body W x D = ", W_out, " x ", D_out, " mm (", fascia_w, " x ", total_d, " mm with fascia) ; KV260 pitch = ",
         kv_pitch, " ; switch pitch = ", sw_pitch, " ; cap = ", cap_h, " ; total height = ", total_h,
         " mm (limit ", env_h, ")"));
assert(fascia_w <= env_w && total_d <= env_d, "Footprint exceeds envelope");
assert(clip_catch > fit_tol, "clip_catch must exceed fit_tol or the fascia clips cannot engage");
assert(clip_catch + fit_tol <= post_min_wall - 1, "Clip groove would leave < 1 mm between it and the corner-post socket");
assert(clip_y - clip_catch - fit_tol > corner_r && clip_y + clip_catch + fit_tol < post_xy,
       "Clip groove must sit on the flat outer face of the front post");
assert(clip_wing_y < post_xy + side_slot_margin && clip_wing_y < post_xy + (bay_d - post_xy - sw_exit_w)/2,
       "Fascia clip wings would cover the side vents / exit windows");
assert(fascia_end_r < min(fascia_t, clip_t)/2, "fascia_end_r must stay below half of fascia_t and clip_t");
assert(total_h <= env_h, "Tower height exceeds envelope — reduce kv_fan_clear / sw_top_clear");
assert(kv_lip_h < kv_standoff_h, "Board stop would block the KV260 ports");
assert(kv_x0 - post_in - wall_t >= side_gap - eps, "KV260 collides with corner posts");
assert(sw_x0 - post_in - wall_t >= side_gap - eps, "Switch collides with corner posts");
assert(drop_len > 8, "bay_d too small for the cable drop shafts");

// =============================================================================
//  PRIMITIVES
// =============================================================================
module rounded_rect(w, d, r) { offset(r = r) offset(delta = -r) square([w, d]); }
module outline_prism(h) { linear_extrude(h) rounded_rect(W_out, D_out, corner_r); }

// Square peg with 45° chamfered top, centred on XY, base at z=0
module peg() {
    translate([-peg_s/2, -peg_s/2, 0]) cube([peg_s, peg_s, peg_h - peg_chamfer]);
    translate([0, 0, peg_h - peg_chamfer])
        linear_extrude(height = peg_chamfer, scale = (peg_s - 2*peg_chamfer)/peg_s)
            square(peg_s, center = true);
}

// Socket cutter: square pocket from z=0 upward with a 45° pyramid roof (printable floor-down)
module socket(flat = false) {
    translate([0, 0, -eps]) linear_extrude(sock_d + eps) square(sock_s, center = true);
    if (!flat)
        translate([0, 0, sock_d]) linear_extrude(height = sock_roof, scale = 0.001) square(sock_s, center = true);
}

// 2D profile in (y,z) -> prism along +X of length t
module yz_prism(t) { rotate([90, 0, 90]) linear_extrude(t) children(); }

module kv_standoff() {
    cylinder(d = kv_standoff_d, h = kv_standoff_h);
    if (kv_standoff_mode == "peg") {
        pd = kv_hole_d - 2*fit_tol;
        translate([0, 0, kv_standoff_h]) {
            cylinder(d = pd, h = kv_peg_h - 1);
            translate([0, 0, kv_peg_h - 1]) cylinder(d1 = pd, d2 = pd - 2, h = 1);
        }
    }
}

// Pointy-top hexagon hole grid filling [x0..x1] x [y0..y1] (2D), whole cells only
module hex_grid(x0, x1, y0, y1) {
    nx = floor((x1 - x0 - hex_af)/hex_px);  ny = floor((y1 - y0 - 2*hex_r)/hex_py);
    ox = (x1 - x0 - nx*hex_px - hex_af)/2;   oy = (y1 - y0 - ny*hex_py - 2*hex_r)/2;
    for (j = [0:ny], i = [0:nx]) {
        cx = x0 + ox + hex_af/2 + i*hex_px + (j % 2 == 1 ? hex_px/2 : 0);
        if (cx + hex_af/2 <= x1 - ox + eps)
            translate([cx, y0 + oy + hex_r + j*hex_py]) rotate(90) circle(r = hex_r, $fn = 6);
    }
}

// =============================================================================
//  GENERIC TRAY SHELL
//  floor + 2 side walls + 4 corner posts + board stop + rear lip + pegs
//  − sockets − side vents − floor vents − cable shafts (or exit windows) − fascia clip grooves
// =============================================================================
module tray_shell(h, stop_h, lip_rear_h, floor_keepouts = [], bottom_level = false) {
    intersection() {
        outline_prism(h + peg_h + 1);
        difference() {
            union() {
                cube([W_out, D_out, floor_t]);                                            // floor
                for (x = [0, W_out - wall_t]) translate([x, 0, 0]) cube([wall_t, D_out, h]); // side walls
                for (p = post_pos) translate([p[0], p[1], 0]) cube([post_xy, post_xy, h]);  // posts
                translate([0, stop_y, 0]) cube([W_out, wall_t, stop_h]);                  // board stop
                translate([0, D_out - wall_t, 0]) cube([W_out, wall_t, lip_rear_h]);      // rear lip
                for (p = post_pos) translate([p[0] + post_xy/2, p[1] + post_xy/2, h]) peg();
            }
            for (p = post_pos) translate([p[0] + post_xy/2, p[1] + post_xy/2, 0]) socket();
            // Vents and shafts are built in 2D and extruded once, so each is a single CSG leaf.
            // Hundreds of 3D cutters (or 3D intersections) overflow the F5 preview's CSG normalization.
            for (x = [-eps, W_out - wall_t - eps])
                translate([x, 0, 0]) yz_prism(wall_t + 2*eps) side_vents_2d(h, bottom_level);
            translate([0, 0, -eps]) linear_extrude(floor_t + 2*eps) {
                floor_vents_2d(floor_keepouts);
                if (!bottom_level) cable_shafts_2d();
            }
            if (bottom_level) exit_windows();
            clip_grooves(h);
        }
    }
}

// ---- side-wall ventilation: slot profile in (y,z), extruded through each wall ----
module side_vents_2d(h, bottom_level) {
    y0 = post_xy + side_slot_margin;  y1 = D_out - post_xy - side_slot_margin;
    z0 = floor_t + side_slot_margin;  sh = h - z0 - side_slot_margin;
    // on the bottom level keep the bay region of the walls solid (exit windows live there)
    ya = bottom_level ? bay_d + side_slot_margin : y0;
    intersection() {
        translate([ya, z0]) square([y1 - ya, sh]);
        if (vent_style == "vertical") {
            n = floor((y1 - ya - side_slot_w)/side_slot_pitch);
            for (i = [0:n]) translate([ya + i*side_slot_pitch + side_slot_w/2, z0])
                polygon([[0,0],[side_slot_w/2,side_slot_w/2],[side_slot_w/2,sh-side_slot_w/2],
                         [0,sh],[-side_slot_w/2,sh-side_slot_w/2],[-side_slot_w/2,side_slot_w/2]]);
        } else {   // chevron: 45° slashes, each half leaning toward the middle  ///  \\\
            ym = (ya + y1)/2;
            for (half = [0, 1]) {
                s  = half == 0 ? 1 : -1;
                hy0 = half == 0 ? ya : ym + side_slot_w/2;
                hy1 = half == 0 ? ym - side_slot_w/2 : y1;
                n  = ceil((hy1 - hy0 + 2*sh)/side_slot_pitch);
                intersection() {
                    translate([hy0, z0]) square([hy1 - hy0, sh]);
                    for (i = [0:n]) translate([hy0 - sh + i*side_slot_pitch, z0])
                        polygon([[0,0],[side_slot_w,0],[side_slot_w + s*sh, sh],[s*sh, sh]]);
                }
            }
        }
    }
}

// ---- floor slots (2D): rectangular slots, skipping keep-out discs [x,y,r] ----------
module floor_vents_2d(keepouts) {
    x0 = wall_t + post_in + floor_margin;  x1 = W_out - x0;
    y0 = post_xy + floor_margin;           y1 = D_out - wall_t - floor_margin;
    px = floor_slot_w + floor_slot_gap;    py = floor_slot_len + floor_slot_gap;
    nx = floor((x1 - x0 - floor_slot_w)/px);
    ny = floor((y1 - y0 - floor_slot_len)/py);
    ox = (x1 - x0 - (nx*px + floor_slot_w))/2;
    oy = (y1 - y0 - (ny*py + floor_slot_len))/2;
    for (i = [0:nx], j = [0:ny]) {
        sx = x0 + ox + i*px;  sy = y0 + oy + j*py;
        cx = sx + floor_slot_w/2;  cy = sy + floor_slot_len/2;
        clear = [for (k = keepouts) if (norm([cx - k[0], cy - k[1]]) < k[2] + floor_slot_len/2) 1];
        if (len(clear) == 0)
            translate([sx, sy]) square([floor_slot_w, floor_slot_len]);
    }
}

// ---- cable drop shafts in the bay floor (+ zip-tie anchor slots), 2D ---------------
module cable_shafts_2d() {
    for (x = drop_x) {
        translate([x, drop_y0]) square([drop_w, drop_len]);
        if (tie_slots)
            for (dx = [-4, drop_w + 2.4], dy = [4, drop_len - 8])
                translate([x + dx, drop_y0 + dy]) square([1.6, 4]);
    }
}

// ---- house-shaped cable exit windows in the bay side walls (bottom level) --------
module exit_windows() {
    yw = post_xy + (bay_d - post_xy - sw_exit_w)/2;
    for (x = [-eps, W_out - wall_t - eps])
        translate([x, yw, floor_t]) yz_prism(wall_t + 2*eps)
            polygon([[0,0],[sw_exit_w,0],[sw_exit_w,sw_exit_h],[sw_exit_w/2,sw_exit_h + sw_exit_w/2],[0,sw_exit_h]]);
}

// ---- fascia snap-clip catches: vertical V-grooves in the front posts' outer side faces ----
module clip_grooves(h) {
    d  = clip_catch + fit_tol;
    z0 = fascia_reveal/2 + clip_margin_z - fit_tol;          // full-depth span covers the ridge
    z1 = h - fascia_reveal/2 - clip_margin_z + fit_tol;
    for (m = [0, 1])
        translate([m*W_out, clip_y, 0]) mirror([m, 0, 0])
            hull() {                                           // ends taper at 45° so the groove roof prints
                translate([0, 0, z0]) linear_extrude(z1 - z0) polygon([[-eps, -d - eps], [d, 0], [-eps, d + eps]]);
                for (z = [z0 - d, z1 + d]) translate([-eps, 0, z]) cube(2*eps, center = true);
            }
}

// =============================================================================
//  LEVEL 1..4 : KV260 TRAY
// =============================================================================
module kv_tray() {
    keep = [for (hp = kv_holes) [kv_x0 + hp[0], kv_y0 + hp[1], kv_standoff_d/2 + 2]];
    union() {
        tray_shell(kv_pitch, kv_lip_h, kv_lip_h, keep, false);
        for (hp = kv_holes)
            translate([kv_x0 + hp[0], kv_y0 + hp[1], 0])
                difference() {
                    union() {
                        cylinder(d = kv_standoff_d + 4, h = floor_t);       // solid pad in the floor
                        translate([0, 0, floor_t - eps]) kv_standoff();
                    }
                    if (kv_standoff_mode == "screw")
                        translate([0, 0, floor_t + 1]) cylinder(d = 2.5, h = kv_standoff_h + 1);
                }
    }
}

// =============================================================================
//  LEVEL 0 : SWITCH TRAY  (bottom level: cable exit windows instead of drop shafts)
// =============================================================================
module switch_tray() {
    union() {
        tray_shell(sw_pitch, sw_front_lip_h, sw_front_lip_h, [], true);
        for (s = [0, 1]) {                                        // side guide blocks
            gx = s == 0 ? sw_x0 - sw_side_tol - sw_guide_t : sw_x0 + sw_w + sw_side_tol;
            for (gy = [sw_y0 + 8, sw_y0 + sw_d - sw_guide_len - 4])
                translate([gx, gy, floor_t - eps]) cube([sw_guide_t, sw_guide_len, sw_guide_h]);
        }
        translate([(W_out - sw_rear_stop_w)/2, sw_y0 + sw_d + sw_side_tol, floor_t - eps])   // rear stop
            cube([sw_rear_stop_w, wall_t, sw_rear_stop_h]);
    }
}

// =============================================================================
//  FASCIA — honeycomb front panel, one per level.  Modelled upright: front face at
//  y=0, plate to y=fascia_t, bosses to y=fascia_t+fascia_boss_h (the tower's front face);
//  z=0 at its bottom edge.  X matches the trays, so the clip wings sit at x<0 and x>W_out.
// =============================================================================
// Plan view (x,y): plate + a clip wing at each end; outer corners rounded, wing roots filleted
module fascia_plan() {
    x0 = -(fit_tol + clip_t);  x1 = W_out + fit_tol + clip_t;
    offset(r = fascia_end_r) offset(delta = -fascia_end_r)
        offset(r = -1) offset(delta = 1) {
            translate([x0, 0]) square([x1 - x0, fascia_t]);
            for (x = [x0, x1 - clip_t]) translate([x, 0]) square([clip_t, fascia_t + fascia_boss_h + clip_wing_y]);
        }
}

module fascia(level_h) {
    fh = level_h - fascia_reveal;
    zs = [for (z = boss_z(level_h)) z - fascia_reveal/2];
    yr = fascia_t + fascia_boss_h + clip_y - clip_preload;         // snap ridge apex (local Y)
    difference() {
        union() {
            linear_extrude(fh) fascia_plan();
            for (p = front_posts, z = zs)                                  // stand-off bosses
                translate([p[0] + post_xy/2, fascia_t - eps, z]) rotate([-90, 0, 0])
                    cylinder(d = fascia_boss_d, h = fascia_boss_h + eps);
            for (m = [0, 1])                                               // snap ridges, 45° both faces
                translate([m*W_out, 0, clip_margin_z]) mirror([m, 0, 0]) linear_extrude(fh - 2*clip_margin_z)
                    polygon([[-fit_tol - eps, yr - clip_catch], [clip_catch - fit_tol, yr], [-fit_tol - eps, yr + clip_catch]]);
        }
        // honeycomb (2D grid in (x,z) -> prism through Y)
        translate([0, -eps, 0]) rotate([90, 0, 0]) mirror([0, 0, 1]) linear_extrude(fascia_t + 2*eps)
            hex_grid(fascia_margin_x, W_out - fascia_margin_x, fascia_margin_z, fh - fascia_margin_z);
    }
}
module kv_fascia() { fascia(kv_pitch); }
module sw_fascia() { fascia(sw_pitch); }

// =============================================================================
//  TOP CAP — honeycomb (or slotted) plate with corner sockets; print upside-down
// =============================================================================
module top_cap() {
    intersection() {
        outline_prism(cap_h + 1);
        difference() {
            union() {
                cube([W_out, D_out, floor_t]);
                for (p = post_pos) translate([p[0], p[1], 0]) cube([post_xy, post_xy, cap_h]);
                for (x = [0, W_out - wall_t]) translate([x, 0, 0]) cube([wall_t, D_out, cap_h]);
            }
            for (p = post_pos) translate([p[0] + post_xy/2, p[1] + post_xy/2, 0]) socket(cap_flat_socket);
            if (cap_hex)
                translate([0, 0, -eps]) linear_extrude(floor_t + 2*eps)
                    hex_grid(wall_t + post_in + floor_margin, W_out - wall_t - post_in - floor_margin,
                             post_xy + floor_margin, D_out - post_xy - floor_margin);
            else
                translate([0, 0, -eps]) linear_extrude(floor_t + 2*eps) floor_vents_2d([]);
        }
    }
}

// =============================================================================
//  GHOST MODELS
// =============================================================================
module kv260_ghost() {   // origin at PCB front-left-bottom, +Y rearward
    color("darkgreen", 0.5) cube([kv_pcb_w, kv_pcb_d, kv_pcb_t]);
    color("silver", 0.5) translate([kv_som_xy[0], kv_som_xy[1], kv_pcb_t]) {
        cube([kv_som_wd[0], kv_som_wd[1], kv_som_h - 3]);
        translate([kv_som_wd[0]/2, kv_som_wd[1]/2, kv_som_h - 3]) cylinder(d = kv_fan_d, h = 3);
    }
    for (p = kv_front_ports)
        color("dimgray", 0.6) translate([p[0], -kv_front_ovh, kv_pcb_t]) cube([p[1]-p[0], 15 + kv_front_ovh, p[2]]);
    for (p = kv_rear_ports)
        color("dimgray", 0.6) translate([p[0], kv_pcb_d - 12, kv_pcb_t]) cube([p[1]-p[0], 13, p[2]]);
    for (hp = kv_holes) color("black") translate([hp[0], hp[1], -eps]) cylinder(d = kv_hole_d, h = kv_pcb_t + 2*eps);
}

module switch_ghost() {
    color("steelblue", 0.5) cube([sw_w, sw_d, sw_h]);
    for (i = [0:7]) color("dimgray", 0.7) translate([14 + i*16.5, -1, 4]) cube([13, 1, 14]);  // 8x RJ45
    color("black", 0.7) translate([sw_w - 20, sw_d, sw_h/2 - 4]) cube([8, 1, 8]);            // rear DC socket
}

// =============================================================================
//  ASSEMBLY & PRINT LAYOUT
// =============================================================================
module place_fascia(level_h) {
    translate([0, -(fascia_t + fascia_boss_h), fascia_reveal/2]) color("white", 0.9) fascia(level_h);
}

module assembly() {
    switch_tray();
    if (show_fascias) place_fascia(sw_pitch);
    if (show_boards) translate([sw_x0, sw_y0, floor_t]) switch_ghost();
    for (i = [0:3]) translate([0, 0, sw_pitch + i*kv_pitch]) {
        kv_tray();
        if (show_fascias) place_fascia(kv_pitch);
        if (show_boards) translate([kv_x0, kv_y0, floor_t + kv_standoff_h]) kv260_ghost();
    }
    translate([0, 0, sw_pitch + 4*kv_pitch]) top_cap();
    if (show_boards) %translate([(W_out - env_w)/2, -(fascia_t + fascia_boss_h), 0]) cube([env_w, env_d, env_h]);   // envelope
}

// Every part in its print orientation: trays floor-down, cap upside-down, fascias face-down
module print_layout() {
    gap = 15;
    switch_tray();
    for (i = [0:3]) translate([(i+1)*(W_out + gap), 0, 0]) kv_tray();
    translate([0, D_out + gap, cap_h]) rotate([180, 0, 0]) translate([0, -D_out, 0]) top_cap();
    translate([W_out + gap, D_out + gap + sw_pitch - fascia_reveal, 0]) rotate([90, 0, 0]) sw_fascia();
    for (i = [0:3]) translate([W_out + gap, D_out + gap + sw_pitch + 5 + (i+1)*(kv_pitch + 5) - fascia_reveal, 0])
        rotate([90, 0, 0]) kv_fascia();
}

// =============================================================================
if      (part == "assembly")     assembly();
else if (part == "print_layout") print_layout();
else if (part == "kv_tray")      kv_tray();
else if (part == "switch_tray")  switch_tray();
else if (part == "top_cap")      top_cap();
else if (part == "kv_fascia")    kv_fascia();
else if (part == "sw_fascia")    sw_fascia();
