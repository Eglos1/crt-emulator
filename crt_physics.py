#!/usr/bin/env python3
"""
RF TV Signal Emulator — Physics-Faithful Analog Receiver Chain
==============================================================

This program emulates the full analog television signal chain through a
realistic RF receiver path. It models how a real RF signal degrades as
it propagates through an environment with multipath, noise, and
interference — then gets decoded by an analog TV receiver.

Signal Chain Modeled:
  1. Source RGB → NTSC YIQ encoding (ITU-R BT.470 / SMPTE C phosphors)
  2. YIQ → Composite video (luminance + modulated chrominance subcarrier)
  3. Composite → RF propagation (multipath reflections, AWGN, path loss)
  4. RF → Receiver demodulation (IF bandwidth filter, Y/C separation)
  5. Decoded video (cross-color, cross-luma, phase error, differential gain)
  6. Noise, snow, hum, dropout, AGC artifacts from weak signals

3D Propagation Environment:
  - Buildings (concrete, glass, brick, metal, wood)
  - Hills and terrain slopes
  - Trees (scattering/diffusion)
  - Vehicles (parked cars)
  - Power line towers
  - Water towers (strong specular)
  - Billboards
  - Fences
  - Ground reflection (always present)
  - Second-order (double-bounce) reflections

Physics References:
  - NTSC: ITU-R BT.470-6, SMPTE 170M (composite video timing)
  - RF:   FCC Part 73, Friis path loss, 2-ray ground model
  - Multipath: image-source method, Fresnel coefficients

Controls:
  F1         Toggle settings overlay
  F3         Toggle SMPTE color bars test pattern
  F4         Cycle physical propagation scenarios (Rooftop/Indoor/Suburban/Valley)
  F5         Pick source image file
  F6         Toggle live screen capture
  F11        Toggle fullscreen
  1/2        Move receiver closer/farther (distance)
  3/4        Move receiver laterally (cross buildings)
  5/6        Raise/lower transmitter antenna height
  7/8        Adjust broadcast frequency (MHz)
  9/0        Adjust screen curvature radius
  -/=        Adjust bloom / electron beam spot size
  [ ]        Adjust horizontal focus (electron gun astigmatism)
  ; '        Adjust vertical focus
  Tab        Toggle interlace simulation
  Backspace  Reset to "good signal" defaults
  Esc        Quit

  Drag-and-drop: Drop an image file onto the window to use as source

Usage:
  python3 crt_physics.py [--width W] [--height H] [--fullscreen]
  python3 crt_physics.py --source image.png
  python3 crt_physics.py --screen --fullscreen   # CRT-filter your live desktop
"""

import sys, os, time, math, argparse, struct
import numpy as np

try:
    import moderngl as mgl
except ImportError:
    sys.exit("pip3 install moderngl")
try:
    import glfw
except ImportError:
    sys.exit("pip3 install glfw")
from PIL import Image

try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

# ═══════════════════════════════════════════════════════════════════════════════
#  NTSC-BT.470 TIMING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# These values come from ITU-R BT.470-6 and SMPTE 170M.
# They define the precise temporal structure of an NTSC-M composite signal.

# Line/frequency constants
H_LINE_FREQ     = 15734.264       # Hz — horizontal scan frequency (NTSC-M)
H_PERIOD_US     = 63.5556         # µs — one complete horizontal line
V_FIELD_RATE    = 59.94           # Hz — field rate (interlaced)
V_FRAME_RATE    = 29.97           # Hz — frame rate

# Timing within one line (all in µs)
FRONT_PORCH_US  = 1.50            # front porch before H-sync
H_SYNC_US       = 4.70            # horizontal sync pulse width
BACK_PORCH_US   = 4.50            # back porch (includes color burst)
ACTIVE_VIDEO_US = H_PERIOD_US - FRONT_PORCH_US - H_SYNC_US - BACK_PORCH_US  # ≈52.86 µs

# Color subcarrier
COLOR_SUBCARRIER_MHZ = 3.579545   # MHz exactly (315/88 × 10⁶ Hz)
COLOR_BURST_CYCLES   = 10         # nominal number of burst cycles
BURST_START_US       = 0.60       # µs after H-sync trailing edge

# IRE levels (1 IRE = 1/140 of video range, or 7.14 mV into 75Ω)
IRE_SYNC_TIP   = -40.0           # sync tip level
IRE_BLANKING   =   0.0           # blanking/pedestal level
IRE_WHITE      = 100.0           # peak white
IRE_BURST_AMP  =  20.0           # color burst amplitude (±20 IRE = 40 IRE p-p)

# RF channel constants (FCC Part 73)
RF_CHANNEL_BW_MHZ    = 6.0       # total channel bandwidth
RF_PICTURE_CARRIER   = 1.25      # MHz above lower channel edge
RF_SOUND_CARRIER     = 4.50      # MHz above picture carrier
RF_VSB_CUTOFF        = 0.75      # MHz — vestigial sideband rolloff region

# CRT phosphor chromaticity (CIE 1931, SMPTE P22)
# These are the 1931 CIE xy coordinates for each phosphor color
PHOSPHOR_R_X, PHOSPHOR_R_Y = 0.670, 0.330  # P22 red
PHOSPHOR_G_X, PHOSPHOR_G_Y = 0.210, 0.710  # P22 green
PHOSPHOR_B_X, PHOSPHOR_B_Y = 0.140, 0.080  # P22 blue
# White point (D65 illuminant)
WHITE_X, WHITE_Y = 0.3127, 0.3290

# CRT electron gun physics
CRT_GAMMA             = 2.2   # typical CRT voltage→luminance exponent
CRT_PHOSPHOR_TAU_MS   = 2.0   # P22 medium-persistence decay time constant (ms)
CRT_SPOT_BASE_UM      = 80.0  # base electron spot diameter (µm) at center, low current
CRT_SPOT_BLOOM_FACTOR = 3.0   # spot size multiplier at peak white vs blanking


# ═══════════════════════════════════════════════════════════════════════════════
#  RF PROPAGATION ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════
#
# Models a 3D physical environment with a transmitter, receiver, and
# reflecting/absorbing surfaces. Ray-traces the signal paths and computes
# the multipath impulse response at the receiver.
#
# Each reflecting surface has a reflection coefficient that depends on
# the material and angle of incidence (Fresnel equations, simplified).
#
# The output is a set of ray paths, each with:
#   - delay (seconds): path length / speed of light
#   - amplitude (linear): path loss × reflection coefficient
#   - phase (radians): 2π × frequency × path_length / c

class RFPropagationEnvironment:
    """3D RF multipath propagation environment."""

    C = 299792458.0  # m/s

    MATERIALS = {
        'concrete':  0.6,
        'glass':     0.3,
        'metal':     0.95,
        'wood':      0.2,
        'terrain':   0.4,
        'water':     0.8,
        'brick':     0.5,
        'asphalt':   0.35,
        'foliage':   0.15,   # leaves scatter/diffuse, weak specular
        'car_body':  0.7,    # painted steel
        'chainlink': 0.3,    # mesh fence — partial transmission + reflection
    }

    def __init__(self):
        # TX and RX positions in meters [x, y, z]
        self.tx_pos = [0.0, 0.0, 30.0]       # transmitter on a 30m tower
        self.rx_pos = [15000.0, 0.0, 3.0]    # receiver at 15km, rooftop 3m

        # Reflecting surfaces: list of dicts
        # Each: {'type': 'rect', 'center': (x,y,z), 'size': (w,h),
        #         'normal': (nx,ny,nz), 'material': 'concrete'}
        self.surfaces = []

        # Ground reflection (always present)
        self.ground_enabled = True
        self.ground_z = 0.0  # ground height

        # RF parameters
        self.frequency_hz = 200e6  # carrier frequency

        # Maximum rays to compute
        self.max_rays = 8

        # Enable second-order reflections (double bounces)
        self.second_order = True
        self.max_second_order = 15  # limit double-bounce surface pairs

    def add_building(self, x, y, w, d, h, material='concrete'):
        """Add a rectangular building footprint to the environment."""
        # Four vertical walls + roof
        hw, hd = w / 2, d / 2
        walls = [
            # front face (y + d/2)
            {'center': (x, y + hd, h / 2), 'size': (w, h),
             'normal': (0, 1, 0), 'material': material},
            # back face (y - d/2)
            {'center': (x, y - hd, h / 2), 'size': (w, h),
             'normal': (0, -1, 0), 'material': material},
            # right face (x + w/2)
            {'center': (x + hw, y, h / 2), 'size': (d, h),
             'normal': (1, 0, 0), 'material': material},
            # left face (x - w/2)
            {'center': (x - hw, y, h / 2), 'size': (d, h),
             'normal': (-1, 0, 0), 'material': material},
            # roof
            {'center': (x, y, h), 'size': (w, d),
             'normal': (0, 0, 1), 'material': material},
        ]
        self.surfaces.extend(walls)

    def add_hill(self, x, y, slope_len, slope_h, width, material='terrain'):
        """Add a sloped hillside/embankment facing along +x.
        slope_len = horizontal run, slope_h = height gain, width = cross-track."""
        # A tilted rectangle: the slope face from ground to top
        # Normal of the slope: perpendicular to the surface
        # Surface goes from (x, y, 0) to (x+slope_len, y, slope_h)
        # Normal = normalize(cross(slope_dir, cross_track))
        slope_len = max(slope_len, 1.0)
        import math as _m
        slen = _m.sqrt(slope_len**2 + slope_h**2)
        nx = -slope_h / slen  # points back toward TX (uphill normal)
        nz = slope_len / slen
        # Center of slope face
        cx = x + slope_len / 2
        cz = slope_h / 2
        self.surfaces.append({
            'center': (cx, y, cz), 'size': (slen, width),
            'normal': (nx, 0, nz), 'material': material,
        })
        # Vertical back wall behind the hill top
        if slope_h > 5:
            self.surfaces.append({
                'center': (x + slope_len, y, slope_h / 2), 'size': (width, slope_h),
                'normal': (1, 0, 0), 'material': material,
            })
        # Top surface
        if slope_len > 10:
            self.surfaces.append({
                'center': (x + slope_len, y, slope_h), 'size': (min(slope_len, 50.0), width),
                'normal': (0, 0, 1), 'material': material,
            })

    def add_water_tower(self, x, y, tank_radius=3.0, tank_h=5.0, leg_h=10.0):
        """Add a water tower — strong specular reflector (metal cylinder).
        Modeled as 4 vertical metal panels approximating a cylinder."""
        import math as _m
        n_panels = 6
        for i in range(n_panels):
            angle = 2 * _m.pi * i / n_panels
            nx = _m.cos(angle)
            ny = _m.sin(angle)
            cx = x + tank_radius * nx
            cy = y + tank_radius * ny
            # Tank body panel
            self.surfaces.append({
                'center': (cx, cy, leg_h + tank_h / 2), 'size': (2 * _m.pi * tank_radius / n_panels, tank_h),
                'normal': (nx, ny, 0), 'material': 'metal',
            })
            # Support leg (thin metal, lower reflection)
            self.surfaces.append({
                'center': (cx * 0.8, cy * 0.8, leg_h / 2), 'size': (0.3, leg_h),
                'normal': (nx, ny, 0), 'material': 'metal',
            })
        # Bottom dish of tank (slightly concave — reflector toward ground)
        self.surfaces.append({
            'center': (x, y, leg_h), 'size': (tank_radius * 2, tank_radius * 2),
            'normal': (0, 0, -1), 'material': 'metal',
        })

    def add_billboard(self, x, y, w, h, face_angle_deg=0, pole_h=6.0):
        """Add a billboard — flat metal/wood panel on a pole.
        face_angle_deg: rotation of the face normal from +x axis."""
        import math as _m
        rad = face_angle_deg * _m.pi / 180.0
        nx = _m.cos(rad)
        ny = _m.sin(rad)
        # The sign face
        self.surfaces.append({
            'center': (x, y, pole_h + h / 2), 'size': (w, h),
            'normal': (nx, ny, 0), 'material': 'metal',
        })
        # Back face
        self.surfaces.append({
            'center': (x, y, pole_h + h / 2), 'size': (w, h),
            'normal': (-nx, -ny, 0), 'material': 'wood',
        })

    def add_vehicle(self, x, y, length=4.5, width=1.8, height=1.5):
        """Add a parked vehicle (car/truck) — painted metal body."""
        hw, hd = length / 2, width / 2
        # Four sides + roof
        self.surfaces.extend([
            {'center': (x, y + hd, height / 2), 'size': (length, height),
             'normal': (0, 1, 0), 'material': 'car_body'},
            {'center': (x, y - hd, height / 2), 'size': (length, height),
             'normal': (0, -1, 0), 'material': 'car_body'},
            {'center': (x + hw, y, height / 2), 'size': (width, height),
             'normal': (1, 0, 0), 'material': 'car_body'},
            {'center': (x - hw, y, height / 2), 'size': (width, height),
             'normal': (-1, 0, 0), 'material': 'car_body'},
            {'center': (x, y, height), 'size': (length, width),
             'normal': (0, 0, 1), 'material': 'car_body'},
        ])

    def add_tree_cluster(self, x, y, count=5, spread=20.0, trunk_h=8.0,
                         canopy_r=4.0, canopy_h=5.0):
        """Add a cluster of trees. Trees scatter RF rather than specularly
        reflecting — modeled as multiple tilted foliage panels that diffuse."""
        import math as _m
        import random as _r
        _r.seed(hash((x, y)))
        for _ in range(count):
            tx = x + _r.uniform(-spread, spread)
            ty = y + _r.uniform(-spread, spread)
            # Trunk (wood, narrow vertical surface)
            self.surfaces.append({
                'center': (tx, ty, trunk_h / 2), 'size': (0.4, trunk_h),
                'normal': (1, 0, 0), 'material': 'wood',
            })
            # Canopy — 4 panels arranged as a rough octahedron
            for angle in [0, 90]:
                rad = angle * _m.pi / 180.0
                nx = _m.cos(rad)
                ny = _m.sin(rad)
                self.surfaces.append({
                    'center': (tx + nx * 0.5, ty + ny * 0.5, trunk_h + canopy_h / 2),
                    'size': (canopy_r * 2, canopy_h),
                    'normal': (nx, ny, 0), 'material': 'foliage',
                })

    def add_power_line_tower(self, x, y, height=30.0, arm_width=8.0):
        """Add a power line transmission tower — metal lattice structure.
        Strong reflector, especially for vertical polarization."""
        # Main vertical faces (4 sides of a narrow tower)
        hw = 1.5  # tower half-width at base
        self.surfaces.extend([
            {'center': (x + hw, y, height / 2), 'size': (3.0, height),
             'normal': (1, 0, 0), 'material': 'metal'},
            {'center': (x - hw, y, height / 2), 'size': (3.0, height),
             'normal': (-1, 0, 0), 'material': 'metal'},
            {'center': (x, y + hw, height / 2), 'size': (3.0, height),
             'normal': (0, 1, 0), 'material': 'metal'},
            {'center': (x, y - hw, height / 2), 'size': (3.0, height),
             'normal': (0, -1, 0), 'material': 'metal'},
        ])
        # Cross arms near the top (horizontal metal bars holding wires)
        for arm_y_off in [-arm_width / 2, 0, arm_width / 2]:
            self.surfaces.append({
                'center': (x, y + arm_y_off, height * 0.9), 'size': (arm_width, 1.0),
                'normal': (1, 0, 0), 'material': 'metal',
            })

    def add_fence(self, x1, y1, x2, y2, height=2.0):
        """Add a chain-link or metal fence between two points.
        Modeled as a single tall narrow rectangle along the line."""
        import math as _m
        dx = x2 - x1
        dy = y2 - y1
        length = _m.sqrt(dx * dx + dy * dy)
        if length < 0.1:
            return
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # Normal is perpendicular to fence line, horizontal
        nx = -dy / length
        ny = dx / length
        self.surfaces.append({
            'center': (cx, cy, height / 2), 'size': (length, height),
            'normal': (nx, ny, 0), 'material': 'chainlink',
        })

    def add_terrain_slope(self, x_start, x_end, y_center, y_width,
                          z_start=0.0, z_end=50.0, segments=5):
        """Add a terrain contour as a series of slope panels.
        Models a hillside, riverbank, or gradual elevation change."""
        import math as _m
        dx = (x_end - x_start) / segments
        for i in range(segments):
            x0 = x_start + i * dx
            x1 = x_start + (i + 1) * dx
            z0 = z_start + (z_end - z_start) * i / segments
            z1 = z_start + (z_end - z_start) * (i + 1) / segments
            seg_len = _m.sqrt(dx**2 + (z1 - z0)**2)
            if seg_len < 0.1:
                continue
            nx = -(z1 - z0) / seg_len
            nz = dx / seg_len
            cx = (x0 + x1) / 2
            cz = (z0 + z1) / 2
            self.surfaces.append({
                'center': (cx, y_center, cz), 'size': (seg_len, y_width),
                'normal': (nx, 0, nz), 'material': 'terrain',
            })

    def _reflect_point(self, P, center, normal):
        """Reflect point P across a plane defined by center and normal."""
        # Vector from center to P
        d = [P[i] - center[i] for i in range(3)]
        # Distance along normal
        dn = sum(d[i] * normal[i] for i in range(3))
        # Reflected point
        return [P[i] - 2 * dn * normal[i] for i in range(3)]

    def _path_length(self, A, B):
        """Distance between two 3D points."""
        return math.sqrt(sum((B[i] - A[i]) ** 2 for i in range(3)))

    def _fresnel_attenuation(self, material, incident_angle):
        """Simplified Fresnel reflection coefficient.
        At grazing incidence (near 90°), reflection is stronger.
        At normal incidence (0°), some energy transmits through."""
        gamma = self.MATERIALS.get(material, 0.5)
        # Angle from surface normal: 0° = head-on, 90° = grazing
        # Reflection increases toward grazing
        sin_a = math.sin(incident_angle)
        return gamma * (0.5 + 0.5 * sin_a)

    def compute_rays(self):
        """Trace all significant ray paths from TX to RX.

        Returns list of dicts: {'delay_s', 'amplitude', 'phase_rad', 'path_length'}
        Sorted by amplitude (strongest first).
        """
        TX = self.tx_pos
        RX = self.rx_pos
        f = self.frequency_hz
        c = self.C
        rays = []

        # ── 1. Direct path (LOS) ──
        d_direct = self._path_length(TX, RX)
        # Free-space path loss (Friis): amplitude ∝ 1/d
        # Normalized to 1.0 at 1m reference distance
        amp_direct = 1.0 / max(d_direct, 1.0)
        phase_direct = 2.0 * math.pi * f * d_direct / c
        rays.append({
            'delay_s': d_direct / c,
            'amplitude': amp_direct,
            'phase_rad': phase_direct,
            'path_length': d_direct,
            'label': 'direct',
        })

        # ── 2. Ground reflection ──
        if self.ground_enabled:
            # Image method: mirror TX across ground plane
            TX_ground = [TX[0], TX[1], 2 * self.ground_z - TX[2]]
            d_ground = self._path_length(TX_ground, RX)
            # Ground reflection coefficient (vertical polarization, grazing angle)
            elev_angle = math.atan2(abs(TX[2] - self.ground_z) + abs(RX[2] - self.ground_z),
                                    d_direct) if d_direct > 0 else 0
            gamma_ground = 0.7 * (0.5 + 0.5 * math.sin(elev_angle))
            amp_ground = gamma_ground / max(d_ground, 1.0)
            phase_ground = 2.0 * math.pi * f * d_ground / c
            rays.append({
                'delay_s': d_ground / c,
                'amplitude': amp_ground,
                'phase_rad': phase_ground,
                'path_length': d_ground,
                'label': 'ground',
            })

        # ── 3. First-order surface reflections ──
        for surf in self.surfaces:
            center = surf['center']
            normal = surf['normal']
            material = surf['material']
            sw, sh = surf['size']

            # Reflect TX across this surface to get image source
            TX_image = self._reflect_point(TX, center, normal)

            # Check if the reflected path actually hits the surface
            # (the image-to-RX line must intersect the surface rectangle)
            d_reflected = self._path_length(TX_image, RX)

            # Compute intersection point with the surface plane
            # Parametric line: P = TX_image + t * (RX - TX_image)
            # Surface plane: (P - center) · normal = 0
            diff = [RX[i] - TX_image[i] for i in range(3)]
            denom = sum(diff[i] * normal[i] for i in range(3))
            if abs(denom) < 1e-10:
                continue  # parallel, no intersection
            t = sum((center[i] - TX_image[i]) * normal[i] for i in range(3)) / denom
            # Allow wide range — TX may be very far away, making t close to 1.0.
            # We only reject intersections far behind TX or far past RX.
            if t < -0.01 or t > 1.01:
                continue

            hit = [TX_image[i] + t * diff[i] for i in range(3)]

            # Check if hit point is within the surface rectangle.
            # Find the two axes perpendicular to the surface normal and check
            # the hit point is within the half-extents on those axes.
            perp_axes = [i for i in range(3) if abs(normal[i]) < 0.5]
            local = [hit[i] - center[i] for i in range(3)]

            in_surface = True
            if len(perp_axes) == 2:
                # Map surface size to the perpendicular axes in order
                for k, pi in enumerate(perp_axes):
                    half_size = surf['size'][k] / 2.0
                    if abs(local[pi]) > half_size:
                        in_surface = False
                        break

            if not in_surface:
                continue

            # Incident angle at the surface
            incident_dir = [TX[i] - hit[i] for i in range(3)]
            inc_len = math.sqrt(sum(d ** 2 for d in incident_dir))
            if inc_len < 1e-10:
                continue
            cos_angle = abs(sum(incident_dir[i] * normal[i] for i in range(3)) / inc_len)
            incident_angle = math.acos(min(max(cos_angle, 0), 1))

            gamma = self._fresnel_attenuation(material, incident_angle)
            amp = gamma / max(d_reflected, 1.0)
            phase = 2.0 * math.pi * f * d_reflected / c

            rays.append({
                'delay_s': d_reflected / c,
                'amplitude': amp,
                'phase_rad': phase,
                'path_length': d_reflected,
                'label': f'reflect_{material}',
            })

        # Sort by amplitude (strongest first), keep top N
        rays.sort(key=lambda r: -r['amplitude'])

        # ── 4. Second-order (double-bounce) reflections ──
        second_order_rays = self.compute_rays_second_order()
        rays.extend(second_order_rays)

        # Merge all rays, sort, keep strongest
        rays.sort(key=lambda r: -r['amplitude'])
        return rays[:self.max_rays]

    def _ray_hits_surface(self, A, B, surf):
        """Check if the ray from A to B hits a surface, returning hit point or None."""
        center = surf['center']
        normal = surf['normal']
        diff = [B[i] - A[i] for i in range(3)]
        denom = sum(diff[i] * normal[i] for i in range(3))
        if abs(denom) < 1e-10:
            return None
        t = sum((center[i] - A[i]) * normal[i] for i in range(3)) / denom
        if t < 0.01 or t > 0.99:
            return None
        hit = [A[i] + t * diff[i] for i in range(3)]
        # Check bounds
        perp_axes = [i for i in range(3) if abs(normal[i]) < 0.5]
        local = [hit[i] - center[i] for i in range(3)]
        if len(perp_axes) == 2:
            for k, pi in enumerate(perp_axes):
                if abs(local[pi]) > surf['size'][k] / 2.0:
                    return None
        return hit

    def compute_rays_second_order(self):
        """Compute second-order (double-bounce) multipath rays.

        For each pair of surfaces (A, B), the image-source method gives:
        TX → bounce on A → bounce on B → RX

        This creates additional multipath copies with longer delays
        and weaker amplitudes (product of two reflection coefficients).
        """
        TX = self.tx_pos
        RX = self.rx_pos
        f = self.frequency_hz
        c = self.C
        rays = []

        n = len(self.surfaces)
        if n < 2 or not self.second_order:
            return rays

        count = 0
        for i in range(n):
            if count >= self.max_second_order:
                break
            surf_a = self.surfaces[i]
            center_a = surf_a['center']
            normal_a = surf_a['normal']

            # Image source for surface A: mirror TX across A
            TX_a = self._reflect_point(TX, center_a, normal_a)

            for j in range(n):
                if i == j:
                    continue
                if count >= self.max_second_order:
                    break

                surf_b = self.surfaces[j]
                center_b = surf_b['center']
                normal_b = surf_b['normal']

                # Image of TX_a across surface B (double image)
                TX_ab = self._reflect_point(TX_a, center_b, normal_b)

                # Path: TX → surface A → surface B → RX
                # Check if ray from TX_ab to RX hits surface B
                hit_b = self._ray_hits_surface(TX_ab, RX, surf_b)
                if hit_b is None:
                    continue

                # Check if ray from TX_a to hit_b hits surface A
                hit_a = self._ray_hits_surface(TX_a, hit_b, surf_a)
                if hit_a is None:
                    continue

                # Total path length
                d1 = self._path_length(TX, hit_a)
                d2 = self._path_length(hit_a, hit_b)
                d3 = self._path_length(hit_b, RX)
                d_total = d1 + d2 + d3

                # Reflection coefficients at each bounce
                # Bounce A
                inc_a = [TX[i] - hit_a[i] for i in range(3)]
                inc_a_len = math.sqrt(sum(d ** 2 for d in inc_a))
                if inc_a_len < 1e-10:
                    continue
                cos_a = abs(sum(inc_a[i] * normal_a[i] for i in range(3)) / inc_a_len)
                angle_a = math.acos(min(max(cos_a, 0), 1))
                gamma_a = self._fresnel_attenuation(surf_a['material'], angle_a)

                # Bounce B
                inc_b = [hit_a[i] - hit_b[i] for i in range(3)]
                inc_b_len = math.sqrt(sum(d ** 2 for d in inc_b))
                if inc_b_len < 1e-10:
                    continue
                cos_b = abs(sum(inc_b[i] * normal_b[i] for i in range(3)) / inc_b_len)
                angle_b = math.acos(min(max(cos_b, 0), 1))
                gamma_b = self._fresnel_attenuation(surf_b['material'], angle_b)

                # Amplitude: product of two reflection coefficients / path length
                amp = (gamma_a * gamma_b) / max(d_total, 1.0)
                phase = 2.0 * math.pi * f * d_total / c

                # Only keep rays with meaningful amplitude
                if amp < 1e-12:
                    continue

                rays.append({
                    'delay_s': d_total / c,
                    'amplitude': amp,
                    'phase_rad': phase,
                    'path_length': d_total,
                    'label': f'dbounce_{surf_a["material"]}_{surf_b["material"]}',
                })
                count += 1

        return rays

    def get_environment_map(self, width, height, margin=0.1):
        """Return a 2D top-down view of the environment for visualization.

        Returns: list of drawable elements:
          {'type': 'rect', 'x', 'y', 'w', 'h', 'color'}
          {'type': 'circle', 'x', 'y', 'r', 'color'}
          {'type': 'line', 'x1', 'y1', 'x2', 'y2', 'color'}
        """
        all_pts = [self.tx_pos[:2], self.rx_pos[:2]]
        for s in self.surfaces:
            all_pts.append([s['center'][0], s['center'][1]])

        if not all_pts:
            return [], 0, 0, 1, 1

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max_x - min_x or 1.0
        dy = max_y - min_y or 1.0

        # Add margin
        min_x -= dx * margin
        min_y -= dy * margin
        dx *= 1 + 2 * margin
        dy *= 1 + 2 * margin

        def to_uv(px, py):
            return ((px - min_x) / dx, (py - min_y) / dy)

        elements = []
        # TX
        tx_u, tx_v = to_uv(self.tx_pos[0], self.tx_pos[1])
        elements.append({'type': 'circle', 'x': tx_u, 'y': 1 - tx_v, 'r': 0.02, 'color': (1, 0.2, 0.2)})

        # RX
        rx_u, rx_v = to_uv(self.rx_pos[0], self.rx_pos[1])
        elements.append({'type': 'circle', 'x': rx_u, 'y': 1 - rx_v, 'r': 0.02, 'color': (0.2, 0.6, 1.0)})

        # Buildings (simplified footprint from surfaces)
        seen = set()
        for s in self.surfaces:
            cx, cy = s['center'][0], s['center'][1]
            key = (round(cx, 1), round(cy, 1))
            if key not in seen:
                seen.add(key)
                bu, bv = to_uv(cx, cy)
                bw = s['size'][0] / dx
                bh = s['size'][1] / dy
                elements.append({
                    'type': 'rect', 'x': bu - bw/2, 'y': 1 - bv - bh/2,
                    'w': bw, 'h': bh, 'color': (0.3, 0.3, 0.35)
                })

        return elements, 0, 0, 1, 1

    def compute_rays_for_shader(self):
        """Compute rays and format them as shader-ready arrays.

        Returns ray data normalized so:
          - Ray 0 is the direct path (delay=0, amp=1.0, phase=0) — reference
          - Subsequent rays have differential delay/phase relative to direct
          - Amplitudes are relative to the direct path
        """
        rays = self.compute_rays()
        if not rays:
            return 0, [0.0]*8, [0.0]*8, [0.0]*8

        # Use the first (strongest) ray as the reference
        d0 = rays[0]['delay_s']
        a0 = rays[0]['amplitude']
        p0 = rays[0]['phase_rad']

        delays = []
        amps = []
        phases = []
        for r in rays:
            # Differential delay (excess over direct path)
            delta_delay_us = (r['delay_s'] - d0) * 1e6
            # Relative amplitude (reflection strength vs direct)
            rel_amp = r['amplitude'] / a0 if a0 > 1e-20 else 0.0
            # Differential phase (excess over direct path)
            delta_phase = (r['phase_rad'] - p0) % (2 * math.pi)

            delays.append(max(0.0, delta_delay_us))
            amps.append(min(rel_amp, 1.0))
            phases.append(delta_phase)

        # Pad to 8 rays
        while len(delays) < 8:
            delays.append(0.0)
            amps.append(0.0)
            phases.append(0.0)

        return len(rays), delays[:8], amps[:8], phases[:8]


# ═══════════════════════════════════════════════════════════════════════════════
#  GLSL SHADER SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

VERT_FULLSCREEN = """
#version 460 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec2 a_uv;
out vec2 v_uv;
void main(){
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"""

# ─── Pass 1: Source image / Test pattern ─────────────────────────────────────
# Generates SMPTE RP 219 color bars and zone plate test signals.
# Reference: SMPTE RP 219-2007 (color bar specification)
FRAG_SOURCE = """
#version 460 core
in vec2 v_uv;
out vec4 fragColor;
uniform float u_time;
uniform vec2  u_res;

/*
 * SMPTE RP 219 color bars (75% amplitude, 100% saturation)
 * These are the standard "EBU bars" used for broadcast calibration.
 * The 75% version is used because 100% bars would exceed legal signal levels
 * after RF modulation (they'd clip in the VSB filter).
 */
vec3 smpte_75(float x) {
    // Top bar sequence: White(75%) Yellow Cyan Green Magenta Red Blue Black
    const vec3 cols[8] = vec3[8](
        vec3(0.75),                           // white  @ 75%
        vec3(0.75, 0.75, 0.0),               // yellow
        vec3(0.0,  0.75, 0.75),              // cyan
        vec3(0.0,  0.75, 0.0),               // green
        vec3(0.75, 0.0,  0.75),              // magenta
        vec3(0.75, 0.0,  0.0),               // red
        vec3(0.0,  0.0,  0.75),              // blue
        vec3(0.0)                             // black (0 IRE)
    );
    int i = clamp(int(floor(x * 8.0)), 0, 7);
    return cols[i];
}

/* Zone plate: spatial frequency sweep for testing resolution & aliasing
 * f(x,y) = f0 * ((x-x0)² + (y-y0)²)
 * This generates circular rings that increase in spatial frequency outward,
 * which is ideal for measuring the MTF (modulation transfer function) of
 * the signal chain and CRT display. */
float zone_plate(vec2 p) {
    vec2 c = p - 0.5;
    float r2 = dot(c, c);
    float freq = 180.0 * r2;       // spatial frequency in cycles/pixel
    return 0.5 + 0.5 * cos(freq * 6.28318);
}

/* Grayscale staircase (16 steps) — for measuring differential gain/phase
 * and gamma transfer function accuracy. Each step should be evenly spaced
 * in the final displayed luminance if the system is linear. */
float staircase(vec2 p) {
    return floor(p.x * 16.0) / 16.0;
}

/* Multiburst pattern — discrete spatial frequency bursts at 0.5, 1.0, 2.0,
 * 3.0, 3.58, and 4.2 MHz equivalent. Each burst is a sinusoidal grating
 * at a specific frequency. After passing through the signal chain, the
 * modulation depth of each burst reveals the frequency response. */
float multiburst(vec2 p) {
    float freqs[6] = float[6](0.5, 1.0, 2.0, 3.0, 3.58, 4.2);
    float x = p.x * 6.0;
    int band = int(floor(x));
    band = clamp(band, 0, 5);
    float local_x = fract(x);
    float freq = freqs[band] * 40.0;  // scale to visible range
    return 0.5 + 0.5 * cos(local_x * freq * 6.28318);
}

/* Crosshatch — for measuring CRT convergence and geometry.
 * Convergence errors show up as color fringing on the white lines. */
vec3 crosshatch(vec2 p) {
    vec2 g = fract(p * 24.0);
    float hline = 1.0 - smoothstep(0.0, 0.04, g.y);  // horizontal
    float vline = 1.0 - smoothstep(0.0, 0.04, g.x);  // vertical
    float line = max(hline, vline);
    return vec3(line) * vec3(0.9, 0.95, 1.0);  // slightly blue-white (like actual phosphor)
}

void main() {
    vec2 p = v_uv;

    // Layout: top 67% = bars, bottom 33% split into quadrants
    vec3 col;
    if (p.y > 0.67) {
        // SMPTE bars
        col = smpte_75(p.x);
        // Narrow sub-bars (I, Q, -Q signals)
        float by = (p.y - 0.67) / 0.33;
        if (by < 0.33) {
            // I and Q signal reference bars
            int si = int(floor(p.x * 8.0));
            si = clamp(si, 0, 7);
            // -I, +Q, white, -4%, +4%, -Q, black, +I
            const vec3 sub[8] = vec3[8](
                vec3(0.0, 0.0, 0.75),  // blue (-I)
                vec3(0.0),              // black
                vec3(0.75, 0.0, 0.75), // magenta (+Q)
                vec3(0.0),              // black
                vec3(0.75, 0.75, 0.0), // yellow (-Q)
                vec3(0.0),              // black
                vec3(0.75),             // white
                vec3(0.0625, 0.0625, 0.0625)  // -4% black (for PLUGE)
            );
            col = sub[si];
        }
    } else if (p.y > 0.335) {
        // Middle: staircase (left) + zone plate (right)
        vec2 mp = vec2(p.x, (p.y - 0.335) / 0.335);
        if (mp.x < 0.5) {
            float g = staircase(vec2(mp.x * 2.0, mp.y));
            col = vec3(g);
        } else {
            float z = zone_plate(vec2((mp.x - 0.5) * 2.0, mp.y));
            col = vec3(z);
        }
    } else {
        // Bottom: multiburst (left) + crosshatch (right)
        vec2 bp = vec2(p.x, p.y / 0.335);
        if (bp.x < 0.5) {
            float m = multiburst(vec2(bp.x * 2.0, bp.y));
            col = vec3(m);
        } else {
            col = crosshatch(vec2((bp.x - 0.5) * 2.0, bp.y));
        }
    }

    // Subtle time-varying dither to prevent banding in gradient regions
    float dither = (fract(sin(dot(p * u_res + u_time, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
    col += dither;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# ─── Pass 2: NTSC Signal Chain ───────────────────────────────────────────────
#
# Rather than encoding→decoding through composite (which is error-prone),
# this pass directly models the physical effects of the NTSC signal chain
# on the RGB image. Each effect is derived from the actual RF/composite
# signal physics.
#
# Effects modeled:
#   1. Horizontal bandwidth limiting (4.2 MHz luminance, 1.5/0.5 MHz I/Q)
#   2. Y/C separation artifacts:
#      - Cross-color: luminance harmonics near 3.58 MHz → false color
#      - Cross-luma: chrominance → false brightness (dot crawl)
#   3. Composite subcarrier phase artifacts (line-alternating pattern)
#   4. AWGN (thermal noise in receiver front-end)
#   5. Multipath ghosting (reflected signal copies)
#   6. Differential gain/phase (nonlinear amplifier distortion)
#   7. Power supply hum (60/120 Hz brightness modulation)

FRAG_SIGNAL = """
#version 460 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_source;
uniform float     u_time;
uniform vec2      u_res;
uniform float     u_snr_db;
uniform float     u_yc_separation;     // 0=bandpass (worst),1=comb filter (best)
uniform float     u_phase_error_deg;
uniform float     u_differential_gain;
uniform float     u_hum_amplitude;
uniform float     u_composite_artifacts;

// Ray-traced multipath (up to 8 rays from the 3D environment)
uniform int   u_num_rays;
uniform float u_ray_delay[8];    // delay in µs
uniform float u_ray_amp[8];      // amplitude (linear, 0-1)
uniform float u_ray_phase[8];    // carrier phase shift (radians)

const float PI  = 3.14159265358979;
const float TAU = 6.28318530717959;

/* ── Noise Generation ──
 * High-quality hash with per-pixel decorrelation.
 * Uses a 2D simplex-style hash that avoids the spatial correlation
 * artifacts of the naive sin(dot) approach. */
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

/* Box-Muller transform for Gaussian noise from uniform hashes.
 * Uses two independent uniform samples to produce one Gaussian sample. */
float gaussian_noise(vec2 p, float sigma) {
    float u1 = hash(p) + 1e-6;
    float u2 = hash(p + vec2(52.9871, 11.1379));
    return sigma * sqrt(-2.0 * log(u1)) * cos(TAU * u2);
}

/* Temporal noise: shifts seed each frame so noise animates smoothly
 * but doesn't have inter-frame correlation artifacts. */
float temporal_noise(vec2 pixel, float seed, float sigma) {
    return gaussian_noise(pixel + vec2(seed, seed * 1.7) + u_time * vec2(173.1, 91.7), sigma);
}

/* ── NTSC Composite Signal Model ──
 *
 * The NTSC composite signal at horizontal position x on line n is:
 *   V(x,n) = Y(x) + I(x)·cos(ωc·x + n·π) + Q(x)·sin(ωc·x + n·π)
 * where:
 *   ωc = 2π × 3.579545 MHz (color subcarrier)
 *   n·π = line-alternating phase (NTSC "line lock" trick to reduce
 *         visible dot patterns — subcarrier inverts each line)
 *
 * The subcarrier frequency is exactly 227.5 × horizontal scan frequency:
 *   fc = 227.5 × 15734.264 Hz = 3579545 Hz
 * This half-integer relationship is what makes the interlaced dot pattern
 * look like a checkerboard rather than vertical stripes.
 *
 * At the pixel level, the subcarrier phase advances by:
 *   Δφ = 2π × fc / f_pixel  per pixel
 * For a 1280-pixel active region spanning 52.86 µs:
 *   f_pixel = 1280 / 52.86µs ≈ 24.21 MHz
 *   Δφ = 2π × 3.579545 / 24.21 ≈ 0.928 rad/pixel ≈ 53.2°/pixel
 *
 * ── Y/C Separation Artifacts ──
 *
 * The receiver must separate Y (luminance) from C (chrominance) from the
 * composite signal. This is done with a Y/C separator filter.
 *
 * Bandpass method (poor separation, cheap sets):
 *   - Low-pass filter extracts Y, bandpass extracts C around 3.58 MHz
 *   - Problem: Y has significant energy near 3.58 MHz (fine detail),
 *     which leaks into C → CROSS-COLOR (rainbow fringing on edges)
 *   - C has low-frequency content that leaks into Y → CROSS-LUMA (dot crawl)
 *
 * Comb filter method (better separation, mid-range sets):
 *   - Delay composite by one line (63.56 µs) and add/subtract
 *   - Since subcarrier inverts each line, adding adjacent lines cancels C
 *   - H(f) = 1 + z^(-N) where N = samples per line
 *   - Nulls at odd multiples of f_H/2 → rejects chrominance harmonics
 *   - BUT: nulls are not exactly at 3.58 MHz (it's 227.5 × f_H),
 *     so some residual leakage remains → artifacts are reduced but present
 *
 * 1D comb filter (common in practice):
 *   Y(n) = [V(n) + V(n-1)] / 2    (average adjacent pixels)
 *   C(n) = [V(n) - V(n-1)] / 2    (difference of adjacent pixels)
 *   This puts a null at Nyquist/2 but NOT at the subcarrier → imperfect
 *
 * We model the quality with u_yc_separation:
 *   0.0 = bandpass only (heavy cross-color)
 *   0.5 = 1D comb (moderate artifacts)
 *   1.0 = ideal comb (minimal artifacts) */

void main() {
    vec2 pos = v_uv;
    vec3 src = texture(u_source, pos).rgb;

    if (u_composite_artifacts < 0.5) {
        fragColor = vec4(src, 1.0);
        return;
    }

    // ── 1b. Coherent Multipath Propagation ──
    // Applied to the raw source BEFORE Y/C separation and bandwidth limiting.
    //
    // Real RF multipath operates in the COMPOSITE signal domain, not RGB.
    // The NTSC composite signal at each pixel is:
    //   V(t) = Y(t) + I(t)*cos(ωc·t) + Q(t)*sin(ωc·t)
    //
    // A reflected signal arrives delayed by τ, attenuated by α, phase-shifted by φ:
    //   V_rx(t) = V(t) + Σ α_i · V(t - τ_i) · cos(ωc·τ_i + φ_i)
    //
    // This creates interference patterns that depend on the subcarrier
    // phase relationship between direct and reflected paths at EACH PIXEL.
    // The result is frequency-selective fading (vertical/diagonal banding),
    // cross-color patterns that vary spatially, and color fringing.

    if (u_num_rays > 1) {
        // Pixel position within active video in microseconds
        float pixel_us = pos.x * 52.86;
        float line_num = floor(pos.y * u_res.y);

        // Direct path: encode to composite signal
        float Y0 = dot(src, vec3(0.299, 0.587, 0.114));
        float I0 = dot(src, vec3(0.596, -0.274, -0.322));
        float Q0 = dot(src, vec3(0.211, -0.523, 0.312));
        float subcarrier = TAU * 3.579545;  // MHz
        float omega_t = subcarrier * pixel_us + line_num * PI;

        // Composite = Y + I*cos(ωt) + Q*sin(ωt)
        float composite = Y0 + I0 * cos(omega_t) + Q0 * sin(omega_t);

        // Add coherent multipath copies of the composite signal
        float rx_composite = composite;

        for (int i = 1; i < u_num_rays && i < 8; i++) {
            float amp = u_ray_amp[i];
            if (amp < 0.001) continue;

            // Delay converts to a horizontal UV shift (time delay → pixel offset)
            float delay_us = u_ray_delay[i];
            float delayed_uv_x = clamp(pos.x + delay_us / 52.86, 0.001, 0.999);
            vec2 delayed_uv = vec2(delayed_uv_x, pos.y);

            // Sample the source at the delayed position
            vec3 delayed_rgb = texture(u_source, delayed_uv).rgb;

            // Encode the delayed copy to composite domain
            float Y_d = dot(delayed_rgb, vec3(0.299, 0.587, 0.114));
            float I_d = dot(delayed_rgb, vec3(0.596, -0.274, -0.322));
            float Q_d = dot(delayed_rgb, vec3(0.211, -0.523, 0.312));

            // The delayed signal's subcarrier phase is offset by ωc·τ + φ
            float omega_tau_phi = subcarrier * delay_us + u_ray_phase[i];
            float delayed_composite = Y_d + I_d * cos(omega_t + omega_tau_phi)
                                        + Q_d * sin(omega_t + omega_tau_phi);

            // Coherent sum in composite domain
            rx_composite += amp * delayed_composite;
        }

        // Decode the received composite back to YIQ.
        // The carrier phase at the receiver is the direct path's phase (omega_t).
        // The multipath has shifted the effective carrier phase, creating
        // cross-color/cross-luma interference patterns.
        float Y_rx = rx_composite - 0.0;  // DC component is luminance
        // To recover I and Q, we multiply by cos(ωt) and sin(ωt) and lowpass:
        //   I_rx = LPF[composite · cos(ωt)] * 2
        //   Q_rx = LPF[composite · sin(ωt)] * 2
        // But since we already have Y, I, Q of each sample, we can approximate
        // the demodulation by looking at how the multipath shifts the effective
        // phase. The key insight: the multipath-modified composite has:
        //   Y_rx ≈ Y0 + Σ α·Y_d·cos(ωτ+φ-0)  (luminance interference)
        //   I_rx ≈ I0·cos(0) + Σ α·I_d·cos(ωτ+φ)  (chrominance interference)
        //   Q_rx ≈ Q0·sin(0) + Σ α·Q_d·sin(ωτ+φ)

        // Recompose using the modified composite + subcarrier phase
        // This is an approximation of coherent demodulation
        float total_weight = 1.0;
        float I_mix = I0;
        float Q_mix = Q0;
        float Y_mod = Y0;

        for (int i = 1; i < u_num_rays && i < 8; i++) {
            float amp = u_ray_amp[i];
            if (amp < 0.001) continue;

            float delay_us = u_ray_delay[i];
            float delayed_uv_x = clamp(pos.x + delay_us / 52.86, 0.001, 0.999);
            vec2 delayed_uv = vec2(delayed_uv_x, pos.y);
            vec3 delayed_rgb = texture(u_source, delayed_uv).rgb;

            float Y_d = dot(delayed_rgb, vec3(0.299, 0.587, 0.114));
            float I_d = dot(delayed_rgb, vec3(0.596, -0.274, -0.322));
            float Q_d = dot(delayed_rgb, vec3(0.211, -0.523, 0.312));

            float omega_tau_phi = subcarrier * delay_us + u_ray_phase[i];
            float cos_tph = cos(omega_tau_phi);
            float sin_tph = sin(omega_tau_phi);

            // Coherent demodulation: multiply by cos/sin and filter
            // This mixes the delayed signal's I/Q with the direct carrier phase
            Y_mod += amp * Y_d * cos_tph;   // luminance cross-talk
            I_mix += amp * I_d * cos_tph;   // I cross-talk
            Q_mix += amp * Q_d * sin_tph;   // Q cross-talk
            total_weight += amp;
        }

        // Convert YIQ back to RGB
        Y_mod = clamp(Y_mod / total_weight, -0.5, 1.5);
        I_mix = I_mix / total_weight;
        Q_mix = Q_mix / total_weight;
        src = vec3(
            Y_mod + 0.956 * I_mix + 0.621 * Q_mix,
            Y_mod - 0.272 * I_mix - 0.647 * Q_mix,
            Y_mod - 1.106 * I_mix + 1.703 * Q_mix
        );
    }

    /* ── 2. Horizontal Bandwidth Limiting ──
     * NTSC luminance bandwidth: 4.2 MHz
     * Pixel clock ≈ 24.21 MHz (1280 pixels / 52.86 µs active)
     * Cutoff frequency: 4.2 / 24.21 ≈ 0.173 cycles/pixel
     * Gaussian kernel σ ≈ 0.21 / cutoff ≈ 1.2 pixels
     *
     * On weak signals, the receiver's IF bandwidth is reduced to reject noise,
     * which loses horizontal detail. At 60 dB SNR: 4.2 MHz (full NTSC).
     * At 10 dB SNR: ~2.5 MHz (soft, smeared). At 0 dB: ~1.5 MHz (blurry). */

    float snr_norm = clamp(u_snr_db / 60.0, 0.0, 1.0);
    float bw_mhz = mix(1.5, 4.2, snr_norm);  // IF bandwidth adapts to signal
    float bw_cutoff = bw_mhz / 24.21;
    float bw_sigma = 0.21 / bw_cutoff;

    vec3 bw_blur = vec3(0.0);
    float bw_total = 0.0;
    // 11-tap Gaussian kernel covers ±3σ
    for (int i = -5; i <= 5; i++) {
        float w = exp(-float(i*i) / (2.0 * bw_sigma * bw_sigma));
        vec2 sp = clamp(pos + vec2(float(i) / u_res.x, 0.0), vec2(0.001), vec2(0.999));
        bw_blur += texture(u_source, sp).rgb * w;
        bw_total += w;
    }
    vec3 color = bw_blur / bw_total;

    /* ── 2. Y/C Separation Artifacts ──
     *
     * Cross-color: luminance high frequencies near the subcarrier are
     * misinterpreted as chrominance. On screen, this appears as rainbow
     * fringing on sharp horizontal edges and shimmering on fine patterns.
     *
     * The artifact is proportional to the horizontal luminance gradient
     * multiplied by the subcarrier phase at each pixel. With a perfect
     * comb filter, adjacent lines' chrominance cancels; with a bandpass,
     * it doesn't.
     *
     * Cross-luma (dot crawl): chrominance leaks into luminance, creating
     * a subtle dot pattern that crawls along edges. The dots follow the
     * subcarrier phase and are most visible on saturated color boundaries. */

    // Subcarrier phase at this pixel
    // Position within active video in µs
    float pixel_us = pos.x * 52.86;  // active video duration
    float line_num = floor(pos.y * u_res.y);
    float subcarrier_phase = TAU * 3.579545 * pixel_us + line_num * PI;

    // Horizontal luminance gradient (high-pass) — this is what leaks into chroma
    vec2 dx = vec2(1.0 / u_res.x, 0.0);
    vec3 left  = texture(u_source, clamp(pos - dx, vec2(0.001), vec2(0.999))).rgb;
    vec3 right = texture(u_source, clamp(pos + dx, vec2(0.001), vec2(0.999))).rgb;
    vec3 grad = right - left;  // horizontal gradient
    float lum_grad = dot(grad, vec3(0.299, 0.587, 0.114));

    // Cross-color artifact: gradient energy at subcarrier frequency
    // appears as false color modulation
    float cross_color_strength = 1.0 - u_yc_separation;  // worse with bandpass
    float cc_phase = sin(subcarrier_phase);  // subcarrier oscillation
    vec3 cross_color = vec3(
        lum_grad * cc_phase * 0.35,
        lum_grad * cc_phase * -0.15,
        lum_grad * cc_phase * 0.25
    ) * cross_color_strength;

    // Cross-luma (dot crawl): chrominance leaks into luminance
    // Most visible at color edges (high chroma, low luma gradient)
    float chroma_amplitude = length(src - vec3(dot(src, vec3(0.299, 0.587, 0.114))));
    float dot_crawl = chroma_amplitude * sin(subcarrier_phase * 2.0) * 0.15
                    * cross_color_strength;

    color += cross_color;
    color += vec3(dot_crawl);

    // Phase error: color burst PLL instability shifts all hues
    // This rotates the I-Q plane, shifting colors
    float phase_rad = u_phase_error_deg * PI / 180.0;
    if (abs(phase_rad) > 0.001) {
        // Convert to YIQ, rotate I-Q, convert back
        float Y = dot(color, vec3(0.299, 0.587, 0.114));
        float I = dot(color, vec3(0.596, -0.274, -0.322));
        float Q = dot(color, vec3(0.211, -0.523, 0.312));
        float new_I = I * cos(phase_rad) - Q * sin(phase_rad);
        float new_Q = I * sin(phase_rad) + Q * cos(phase_rad);
        // Reconstruct RGB from rotated YIQ
        color = vec3(
            Y + 0.956*new_I + 0.621*new_Q,
            Y - 0.272*new_I - 0.647*new_Q,
            Y - 1.106*new_I + 1.703*new_Q
        );
    }

    /* ── 3. RF Noise / Weak Signal Degradation ──
     * When a signal is weak but present, the receiver's AGC amplifies
     * both the signal and the noise floor. The result is a visible image
     * with a layer of snow overlaid — not the overwhelming static of a
     * dead channel, but a "fuzzy" look where detail is softened and the
     * image shimmers with grain.
     *
     * The noise is luminance-dominant (grayscale snow) because it's
     * thermal noise modulating the beam intensity, not the color. */

    float noise_linear = pow(10.0, -u_snr_db / 20.0);

    // ── 4a. Base snow ──
    // Grayscale snow overlaid on the image. At moderate SNR (20-30 dB),
    // the image is clearly visible but has a "static grain" look.
    // At low SNR (<15 dB), the snow starts to wash out detail.
    float snow_sigma = noise_linear * 1.5;
    float snow_luma = temporal_noise(pos * u_res, 0.0, snow_sigma);
    color += vec3(snow_luma);

    // ── 4b. Per-line dropout ──
    // On weak signals, horizontal sync occasionally fails on individual
    // lines. The line goes haywire — filled with heavy random snow or
    // displaced. These "dropouts" flicker in and out randomly.
    float line_y = floor(pos.y * u_res.y);
    float dropout_frame = floor(u_time * 59.94);
    float line_dropout_rand = hash(vec2(line_y, dropout_frame));
    float dropout_density = smoothstep(45.0, 10.0, u_snr_db);

    if (line_dropout_rand < dropout_density * 0.08) {
        // This line dissolves into pure snow
        float heavy_luma = temporal_noise(pos * u_res, 10.0, snow_sigma * 5.0);
        color = mix(color, vec3(heavy_luma), 0.8);
    }

    // ── 4c. AGC wobble ──
    // The receiver's automatic gain control hunts on a weak signal,
    // causing the whole image brightness to drift and pulse.
    // This is the "breathing" effect you see when a signal fades in
    // and out — the snow gets brighter and dimmer along with the image.
    float agc_speed = smoothstep(45.0, 15.0, u_snr_db);  // more wobble on weak signals
    float wobble1 = sin(u_time * 1.3) * 0.03 * agc_speed;
    float wobble2 = sin(u_time * 3.7 + 1.5) * 0.015 * agc_speed;
    float wobble3 = (hash(vec2(floor(u_time * 3.0), 99.0)) - 0.5) * 0.04 * agc_speed;
    color += vec3(wobble1 + wobble2 + wobble3);

    /* ── 5. Power Supply Hum ──
     * 120 Hz modulation of the blanking level (full-wave rectified 60 Hz).
     * Appears as a horizontal brightness band scrolling vertically.
     * Higher harmonics (240, 360 Hz) create narrower bands. */

    float hum = sin(v_uv.y * u_res.y * 0.3 + u_time * 3.77) * u_hum_amplitude;
    hum += sin(v_uv.y * u_res.y * 0.6 + u_time * 7.54) * u_hum_amplitude * 0.3;
    color += hum;

    /* ── 6. Differential Gain / Phase ──
     * Nonlinearities in the video amplifier cause:
     *   - Differential gain: chroma amplitude varies with luminance
     *     (colors appear more/less saturated at different brightness)
     *   - Differential phase: chroma hue rotates with luminance
     *     (colors shift hue between dark and bright regions)
     *
     * Measured in % (gain) and degrees (phase). Broadcast spec:
     *   DG < 10%, DP < 5° for acceptable quality */

    float lum = dot(color, vec3(0.299, 0.587, 0.114));

    // Differential gain: scale chroma with brightness
    float chroma_scale = 1.0 + u_differential_gain * (lum - 0.5) * 0.5;
    color = vec3(lum) + (color - vec3(lum)) * chroma_scale;

    // Differential phase: rotate hue with brightness
    float dp_angle = u_differential_gain * (lum - 0.5) * 0.2;  // radians
    if (abs(dp_angle) > 0.001) {
        float I_ch = dot(color, vec3(0.596, -0.274, -0.322));
        float Q_ch = dot(color, vec3(0.211, -0.523, 0.312));
        float new_I = I_ch * cos(dp_angle) - Q_ch * sin(dp_angle);
        float new_Q = I_ch * sin(dp_angle) + Q_ch * cos(dp_angle);
        color = vec3(
            lum + 0.956*new_I + 0.621*new_Q,
            lum - 0.272*new_I - 0.647*new_Q,
            lum - 1.106*new_I + 1.703*new_Q
        );
    }

    /* ── Black level / signal clipping ──
     * Below blanking (0 IRE), the signal is clipped.
     * Overmodulated signals clip above 100 IRE. */

    color = clamp(color, 0.0, 1.0);

    // Soft clip at broadcast limit
    float peak = max(max(color.r, color.g), color.b);
    if (peak > 1.05) {
        color = color / peak * 1.05;
    }

    fragColor = vec4(color, 1.0);
}
"""

# ─── Pass 3: CRT Display Physics ─────────────────────────────────────────────
#
# This pass models the actual CRT display hardware. The key physical processes:
#
# 3a. Electron gun:
#     - Thermionic cathode emits electrons (Child-Langmuir space-charge law)
#     - Control grid (G1) modulates beam current: I_beam ∝ (V_G1 - V_cutoff)^1.5
#     - Focusing anodes (G3-G4) converge beam to a spot
#     - Deflection yoke (H and V coils) steers beam across screen
#
# 3b. Beam spot profile:
#     - The electron beam has a roughly Gaussian intensity profile
#     - Spot diameter depends on beam current (blooming), focus voltage,
#       and deflection angle (worse at edges due to astigmatism)
#     - Typical spot: 80-150 µm at center, 200-400 µm at corners
#
# 3c. Phosphor screen:
#     - Each pixel is a triad of R, G, B phosphor dots (shadow mask)
#     - or vertical stripes (aperture grille / Trinitron)
#     - Electrons excite phosphor → photon emission
#     - Phosphor response: P22 medium-persistence, τ ≈ 1-5 ms
#     - Luminance ∝ beam_current × phosphor_efficiency (with saturation)
#     - The CRT transfer function is a power law: L = V^γ, γ ≈ 2.2
#
# 3d. Shadow mask / Aperture grille:
#     - Shadow mask: perforated metal sheet with holes in a hexagonal pattern
#       Each hole allows electrons to reach only one phosphor color.
#       Typical pitch: 0.6-0.9 mm (consumer), 0.25-0.35 mm (Trinitron)
#     - Aperture grille: vertical tensioned wires, one per color
#       Better light transmission (~30% vs ~20% for shadow mask)
#       Horizontal support wires may be visible (1-2 per screen)
#     - Trinitron: hybrid — uses aperture grille but with cylindrical faceplate
#
# 3e. Screen geometry:
#     - CRT faceplate is curved (radius ≈ 2-3× screen diagonal)
#     - Electron beam deflection causes pincushion distortion
#     - Horizontal and vertical overscan (~5% typical in consumer sets)
#     - Focus degrades toward edges (electronic astigmatism)

FRAG_CRT = """
#version 460 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_signal;
uniform sampler2D u_prev_frame;   // previous frame for phosphor persistence
uniform float     u_time;
uniform vec2      u_res;
uniform float     u_curvature;         //0=flat,1=maximum curvature
uniform float     u_beam_width;        // electron beam spot σ (0-1)
uniform float     u_bloom;             // beam blooming factor (0-1)
uniform float     u_scanline_density;  // lines per screen height (0=auto)
uniform float     u_mask_type;         // 0=shadow mask,1=aperture grille, 2=trinitron
uniform float     u_mask_pitch;        // mask pitch in screen-relative units
uniform float     u_brightness;        // display brightness (cd/m² relative)
uniform float     u_contrast;          // display contrast ratio
uniform float     u_gamma;             // CRT gamma exponent
uniform float     u_focus_h;           // horizontal focus quality (0=poor,1=perfect)
uniform float     u_focus_v;           // vertical focus quality
uniform float     u_convergence;       // convergence error magnitude (0=perfect)
uniform float     u_overscan_h;        // horizontal overscan (0=none,1=5%)
uniform float     u_overscan_v;        // vertical overscan
uniform float     u_interlace;         //0=progressive,1=interlaced
uniform float     u_persistence;       // phosphor persistence (0=instant,1=long)

const float PI = 3.14159265358979;

/* ── CRT Screen Curvature ──
 * The CRT faceplate is curved to withstand atmospheric pressure and to
 * maintain uniform electron beam focus. The curvature is typically specified
 * as a radius in terms of the screen diagonal.
 *
 * For a "flat" CRT (like Sony WEGA), the radius is ~3× diagonal.
 * For a traditional CRT, the radius is ~1.5-2× diagonal.
 *
 * The barrel distortion maps screen coordinates (x,y) to faceplate
 * coordinates (x',y') using the formula:
 *   x' = x + x × (r² / R²) × k
 * where r = distance from center, R = curvature radius, k = distortion coeff.
 *
 * We model this with a polynomial approximation that produces the
 * characteristic barrel distortion of a curved CRT. */

vec2 crt_curvature(vec2 p) {
    p = p * 2.0 - 1.0;  // map to -1..1

    // Overscan: zoom in to hide edge artifacts (standard CRT behavior)
    // This INVERSELY scales the UV, effectively zooming into the center.
    p *= vec2(1.0 / (1.0 + u_overscan_h * 0.06), 1.0 / (1.0 + u_overscan_v * 0.06));

    // Apply barrel distortion only if curvature > 0
    if (u_curvature > 0.001) {
        float r2 = dot(p, p);
        // Real CRT barrel distortion: 2-12% at corners depending on faceplate radius.
        // Pincushion of sampling coords = barrel of displayed image.
        float k1 = u_curvature * 0.12;
        float k2 = u_curvature * 0.02;
        float distort = 1.0 + k1 * r2 + k2 * r2 * r2;
        // Invert: pull sampling coords inward so the displayed image bulges outward
        p /= distort;
    }

    return p * 0.5 + 0.5;  // back to 0..1
}

/* ── Electron Beam Spot Profile ──
 * The electron beam has a roughly Gaussian intensity distribution:
 *   I(r) = I₀ × exp(-r² / (2σ²))
 * where σ is the beam waist (1/e² radius).
 *
 * The spot size depends on:
 *   1. Beam current (higher current → larger spot due to space-charge repulsion)
 *   2. Focus voltage (controls the electrostatic lens focal length)
 *   3. Deflection angle (astigmatism increases toward edges)
 *
 * At center screen with low current, σ ≈ 40 µm (spot diameter ≈ 80 µm).
 * At peak white, σ can increase by 2-3× due to blooming.
 * At corners, σ increases by another 50-100% due to deflection astigmatism.
 *
 * For a 27" CRT with 1920×1080 addressable, each pixel is about 89 µm,
 * so the beam spot covers roughly 1-4 pixels depending on conditions. */

float beam_profile(vec2 pos, vec2 center, float sigma) {
    vec2 d = pos - center;
    float r2 = dot(d, d);
    return exp(-r2 / (2.0 * sigma * sigma));
}

/* ── Phosphor Mask Transmission Function ──
 *
 * Shadow Mask (standard aperture mask):
 *   - Perforated metal sheet with holes in a "delta" pattern
 *   - Each hole is positioned so electrons can only hit one phosphor dot
 *   - Hole spacing (pitch): typically 0.6-0.9 mm for consumer CRTs
 *   - The mask is slightly behind the phosphor screen (finite distance)
 *   - As beam current increases, the spot grows and illuminates adjacent
 *     dots through neighboring holes → reduced color purity
 *
 *   Transmission function: array of circular apertures
 *   T(x,y) = Σ_holes circle(x-xh, y-yh, r_hole)
 *
 * Aperture Grille (Sony Trinitron, Mitsubishi Diamondtron):
 *   - Vertical tensioned wires (one per color: R, G, B)
 *   - Better light transmission (~30% vs ~20% for shadow mask)
 *   - Horizontal support wires visible (1-2 per screen, called "damper wires")
 *   - No hexagonal pattern — pure vertical stripes
 *
 *   Transmission function: vertical stripe pattern
 *   T(x,y) = stripe(x mod pitch)
 *
 * Trinitron (Sony's hybrid):
 *   - Uses aperture grille with cylindrical faceplate
 *   - No shadow mask — better brightness but potentially less geometry
 *   - The cylindrical curvature is only horizontal (vertical is flat or nearly so)
 *
 * All three are modeled here with their correct geometric patterns. */

vec3 mask_shadow(vec2 p, vec2 res) {
    // Shadow mask: RGB triads in a cell
    // Each cell contains 3 phosphor dots (R, G, B) in a horizontal row.
    // This models the standard delta-triad shadow mask used in most color CRTs.
    // The dots are circular with a Gaussian falloff to model the finite
    // electron beam spot size and phosphor bloom.
    float pitch = u_mask_pitch;

    // Map to cell coordinates (3 dots per cell in x, periodic in y)
    float x = p.x * res.x;
    float cell_x = mod(x, pitch * 3.0) / (pitch * 3.0);  // 0-1 across one RGB triad

    // Three dot centers within the cell (evenly spaced in x)
    // R at 1/6, G at 3/6, B at 5/6 of cell width
    float dr = abs(cell_x - 1.0/6.0);
    float dg = abs(cell_x - 3.0/6.0);
    float db = abs(cell_x - 5.0/6.0);

    // Gaussian falloff (more physically accurate than hard circles)
    // σ ≈ 0.15 of cell width — dots fill ~60% of their allocated space,
    // with overlap creating the characteristic CRT subpixel blending
    float sigma = 0.15;
    float dot_r_mask = exp(-dr*dr / (2.0*sigma*sigma));
    float dot_g_mask = exp(-dg*dg / (2.0*sigma*sigma));
    float dot_b_mask = exp(-db*db / (2.0*sigma*sigma));

    // Gap between dots (mask metal absorbs electrons)
    // The minimum distance to any dot center determines the gap attenuation
    float min_d = min(min(dr, dg), db);
    float gap = 1.0 - smoothstep(0.10, 0.14, min_d) * 0.25;

    return vec3(dot_r_mask, dot_g_mask, dot_b_mask) * gap;
}

vec3 mask_aperture_grille(vec2 p, vec2 res) {
    // Aperture grille: pure vertical stripes (R, G, B)
    // Used in Sony Trinitron and Mitsubishi Diamondtron displays.
    // Each stripe spans the full height of the screen.
    float pitch = u_mask_pitch;

    // Three stripes per pitch period
    float x = p.x * res.x;
    float cell = mod(x, pitch * 3.0) / (pitch * 3.0);  // 0-1 across R-G-B

    // Gaussian falloff for each stripe
    float sigma = 0.15;
    float r_mask = exp(-pow(cell - 1.0/6.0, 2.0) / (2.0*sigma*sigma));
    float g_mask = exp(-pow(cell - 3.0/6.0, 2.0) / (2.0*sigma*sigma));
    float b_mask = exp(-pow(cell - 5.0/6.0, 2.0) / (2.0*sigma*sigma));

    // Horizontal damper wire (visible in Trinitron displays)
    // These are thin support wires that hold the aperture grille tension
    float y = p.y * res.y;
    float wire1 = 1.0 - smoothstep(0.0, 1.5, abs(y - res.y * 0.33));
    float wire2 = 1.0 - smoothstep(0.0, 1.5, abs(y - res.y * 0.67));
    float wire_mask = 1.0 - max(wire1, wire2) * 0.4;

    return vec3(r_mask, g_mask, b_mask) * wire_mask;
}

vec3 mask_trinitron(vec2 p, vec2 res) {
    // Trinitron: aperture grille + cylindrical horizontal curvature
    // The cylindrical faceplate means horizontal curvature but flat vertically
    vec3 grille = mask_aperture_grille(p, res);

    // Slight vertical modulation due to cylindrical faceplate
    float vert_mod = 1.0 - 0.05 * sin(p.y * PI);
    return grille * vert_mod;
}

/* ── Phosphor Persistence / Decay ──
 * CRT phosphors have a characteristic decay time after excitation.
 * P22 (standard color CRT phosphor) has:
 *   - Fast component: τ₁ ≈ 0.5-1 ms (dominant green)
 *   - Slow component: τ₂ ≈ 5-10 ms (red phosphor, causes red trailing)
 *
 * The persistence is modeled as an exponential decay:
 *   L(t) = L₀ × exp(-t / τ)
 *
 * In an interlaced display, each phosphor dot is refreshed at the field rate
 * (59.94 Hz = 16.69 ms period). With τ = 2 ms, the phosphor retains about
 * 30% of its brightness between refreshes, which is what gives CRTs their
 * characteristic smooth motion (temporal blending).
 *
 * For progressive scan, the refresh rate is 29.97 Hz (33.38 ms), and the
 * persistence becomes much more noticeable. */

/* ── Scanline Beam Profile ──
 * As the electron beam scans horizontally, it has a Gaussian profile
 * in the vertical direction. The beam illuminates the current scanline
 * and partially illuminates adjacent lines (inter-line glow).
 *
 * For a well-focused CRT:
 *   - Beam σ ≈ 0.3-0.5 × line pitch
 *   - Adjacent line illumination: ~5-15%
 *
 * For interlaced display, adjacent lines are drawn in different fields
 * (16.69 ms apart). With phosphor persistence, both are visible but
 * at slightly different brightnesses. This creates the characteristic
 * "interlaced look" — a subtle line structure that is smoother than
 * a progressive display at the same line count. */

void main() {
    vec2 p = v_uv;

    // Apply CRT curvature
    vec2 crt_uv = crt_curvature(p);

    // Check if outside visible area (CRT mask/border)
    if (crt_uv.x < 0.0 || crt_uv.x > 1.0 || crt_uv.y < 0.0 || crt_uv.y > 1.0) {
        // Dark border (CRT bezel / overscan region)
        float edge_dist = min(min(crt_uv.x, 1.0 - crt_uv.x),
                              min(crt_uv.y, 1.0 - crt_uv.y));
        float bezel = smoothstep(-0.02, 0.01, edge_dist);
        fragColor = vec4(vec3(0.002) * bezel, 1.0);
        return;
    }

    // ── Electron Beam Scanning ──
    // The beam scans horizontally at 15.734 kHz and vertically at 59.94 Hz.
    // The beam current is modulated by the video signal (after gamma correction).
    //
    // During horizontal and vertical blanking intervals, the beam is blanked
    // (cutoff) to hide the retrace. This creates the characteristic black
    // borders on a CRT screen.

    float line_num = crt_uv.y * u_res.y;
    float field_phase = u_time * 59.94 * 0.5;  // field timing

    // Interlace: odd/even lines drawn in alternating fields
    float field = floor(u_time * 30.0);  // 30 fields per second
    float line_parity = mod(line_num + field, 2.0);

    // ── Convergence Errors ──
    // In a color CRT, three electron guns (R, G, B) must converge at the
    // same point on the shadow mask. Perfect convergence is impossible
    // because:
    //   1. The three guns are physically offset (they can't all be on-axis)
    //   2. Deflection causes different geometric distortion for each gun
    //   3. The shadow mask introduces parallax errors
    //
    // Static convergence (center): adjusted by magnets on the neck
    // Dynamic convergence (edges): adjusted by yoke shape and additional coils
    //
    // Typical convergence errors:
    //   Center: < 0.3 mm (nearly invisible)
    //   Edges: 0.5-2.0 mm (visible as color fringing on sharp transitions)
    //
    // The error is largest at the corners and smallest at center.

    vec2 center_offset = crt_uv - 0.5;
    float dist_from_center = length(center_offset);

    // Convergence shift (increases toward edges, different for each gun)
    float conv_err = u_convergence * dist_from_center * 0.08;
    vec2 shift_r = center_offset * conv_err * vec2(1.0, 0.5);   // red shifts outward
    vec2 shift_b = center_offset * conv_err * vec2(-0.7, -0.7); // blue shifts inward
    // Green is reference (on-axis gun)

    // ── Beam Spot Size (with blooming and defocus) ──
    // The electron beam spot size depends on:
    //   1. Base focus: set by focus voltage (G3/G4 anode)
    //   2. Blooming: spot grows with beam current (space-charge repulsion)
    //   3. Defocus at edges: astigmatism from deflection
    //
    // The Gaussian σ of the beam profile is:
    //   σ = σ_base × (1 + bloom × brightness) × (1 + edge_defocus × r²)

    float base_sigma = 0.5 + u_beam_width * 2.0;  // in pixel units
    float bloom_factor = 1.0 + u_bloom * 2.0;
    float edge_defocus = (1.0 - mix(u_focus_h, u_focus_v, 0.5)) * 3.0;

    // Sample with convergence offsets
    vec2 sample_r = crt_uv + shift_r / u_res;
    vec2 sample_g = crt_uv;
    vec2 sample_b = crt_uv + shift_b / u_res;

    // Clamp to valid range
    sample_r = clamp(sample_r, vec2(0.001), vec2(0.999));
    sample_g = clamp(sample_g, vec2(0.001), vec2(0.999));
    sample_b = clamp(sample_b, vec2(0.001), vec2(0.999));

    // Sample source with bloom-adjusted filtering, using convergence offsets
    float sigma_bloomed = base_sigma * bloom_factor * (1.0 + edge_defocus * dist_from_center * dist_from_center);

    // Multi-sample bloom (Gaussian weighted) — each gun samples from its own
    // convergence-shifted center. This produces color fringing at edges where
    // the three beams don't perfectly overlap, exactly like a real CRT with
    // convergence error.
    vec3 bloom_color = vec3(0.0);
    float bloom_total = 0.0;
    const int BLOOM_SAMPLES = 3;  // 7×7 kernel — enough for the spot profile
    for (int y = -BLOOM_SAMPLES; y <= BLOOM_SAMPLES; y++) {
        for (int x = -BLOOM_SAMPLES; x <= BLOOM_SAMPLES; x++) {
            vec2 offset = vec2(float(x), float(y)) / u_res * sigma_bloomed;
            float w = exp(-float(x*x + y*y) / (2.0 * sigma_bloomed * sigma_bloomed));
            // Each gun (R, G, B) has its own convergence center
            float r = texture(u_signal, sample_r + offset).r;
            float g = texture(u_signal, sample_g + offset).g;
            float b = texture(u_signal, sample_b + offset).b;
            bloom_color += vec3(r, g, b) * w;
            bloom_total += w;
        }
    }
    vec3 color = bloom_color / bloom_total;

    // ── Gamma Correction (CRT Transfer Function) ──
    // The CRT electron gun produces a current that follows a power law:
    //   I_beam = k × (V_grid - V_cutoff)^γ
    // where γ ≈ 2.2 for a typical color CRT.
    //
    // IMPORTANT: Broadcast video is gamma-ENCODED at the source (sRGB, γ≈2.2)
    // precisely to pre-compensate for this CRT gamma. So if the source is
    // already sRGB-encoded, the CRT's gamma simply decodes it back to linear
    // light, and the displayed image matches the original scene.
    //
    // We apply a mild gamma to model the slight nonlinearity that remains
    // after the source encoding doesn't perfectly match the CRT's actual γ.
    // A value of 1.0 = no additional gamma (source already encoded).
    // Values >1.0 add a slight darkening of midtones (real CRT behavior).

    color = pow(max(color, vec3(0.0)), vec3(u_gamma * 0.15 + 0.85));  // maps 1.0→1.0, 2.2→1.18

    // ── Scanline Structure ──
    // The electron beam illuminates the screen in discrete horizontal lines.
    // Between lines, there's a dip in brightness (scanline valley).
    // The depth of this dip depends on:
    //   1. Beam focus (tighter beam → deeper valleys)
    //   2. Phosphor persistence (longer persistence → shallower valleys)
    //   3. Line pitch vs. beam width ratio
    //
    // For interlaced displays, the effective line pitch is halved because
    // adjacent lines are drawn in different fields. The phosphor persistence
    // blends them, but at slightly different brightnesses.

    float scan_phase = line_num * PI;
    // Tighter beam = sharper scanlines (higher exponent).
    // 0.5 base + beam_width gives a gentle modulation that doesn't destroy small text.
    float scanline_profile = exp(-pow(mod(line_num, 1.0) - 0.5, 2.0) * (4.0 + 8.0 * u_beam_width));

    // Interlace blending
    if (u_interlace > 0.5) {
        if (mod(line_parity, 2.0) > 0.5) {
            scanline_profile *= 0.85;  // non-active field lines are dimmer
        }
    }

    color *= mix(1.0, scanline_profile, 0.5);

    // ── Phosphor Mask ──
    // The shadow mask / aperture grille filters the electron beam so each
    // phosphor dot is excited only by its corresponding color channel.
    //
    // The mask attenuates each color channel according to the geometric
    // arrangement of the apertures. This is the primary mechanism that
    // creates the characteristic "screen door" look of CRT displays.

    vec3 mask = vec3(1.0);
    if (u_mask_type < 0.5) {
        mask = mask_shadow(crt_uv, u_res);
    } else if (u_mask_type < 1.5) {
        mask = mask_aperture_grille(crt_uv, u_res);
    } else {
        mask = mask_trinitron(crt_uv, u_res);
    }

    // The mask modulates each phosphor independently.
    // The mask peak is1.0 (Gaussian), so at each subpixel position only
    // one channel passes through at full brightness. The eye spatially
    // integrates R+G+B across the mask pattern to perceive the correct
    // color. A multiplier of ~1.5 accounts for the average mask
    // transmission (~60-70% for shadow mask, ~80% for aperture grille).
    color *= mask * 1.5;

    // ── Brightness / Contrast ──
    // CRT brightness is set by the ABL (Automatic Brightness Limiter)
    // and the video amplifier gain. Contrast is the ratio of peak white
    // to black level.
    color = color * u_brightness;
    color = (color - 0.5) * u_contrast + 0.5;

    // ── Vignette (CRT edge rolloff) ──
    // The CRT electron beam loses intensity at large deflection angles
    // because:
    //   1. The beam path is longer (inverse-square law)
    //   2. The beam cross-section increases (deflection astigmatism)
    //   3. The phosphor screen is not perpendicular to the beam at edges
    //
    // This creates a natural vignetting effect where the corners are
    // noticeably dimmer than the center (~20-40% for a typical CRT).

    float vignette = 1.0 - dist_from_center * dist_from_center * 0.5;
    color *= vignette;

    // ── Phosphor Persistence ──
    // Blend with previous frame to simulate phosphor decay.
    // The previous frame contribution follows an exponential decay:
    //   L_persist = L_prev × exp(-Δt / τ)
    // where Δt is the frame period and τ is the phosphor decay constant.

    if (u_persistence > 0.01) {
        vec3 prev = texture(u_prev_frame, v_uv).rgb;
        float decay = exp(-16.69 / (u_persistence * 20.0 + 0.1));  // ~16.69ms frame period
        color = max(color, prev * decay);  // phosphor can only add light
    }

    // ── Flicker ──
    // CRTs exhibit subtle brightness fluctuations from:
    //   1. Power supply ripple (60/120 Hz)
    //   2. Deflection coil current ripple
    //   3. Beam current regulation variations
    float flicker = 1.0 + 0.003 * sin(u_time * 377.0);  // 60 Hz component
    flicker *= 1.0 + 0.001 * sin(u_time * 7.54);  // slow drift
    color *= flicker;

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""

# ─── Pass 4: UI Overlay ──────────────────────────────────────────────────────
FRAG_UI = """
#version 460 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_screen;
uniform float u_show_ui;
uniform float u_snr;
uniform float u_num_rays;
uniform float u_yc_sep;

void main() {
    vec3 col = texture(u_screen, v_uv).rgb;

    if (u_show_ui > 0.5) {
        vec2 p = v_uv;
        col *= 0.55;

        if (p.x > 0.02 && p.x < 0.50 && p.y > 0.03 && p.y < 0.72) {
            vec2 np = (p - vec2(0.02, 0.03)) / vec2(0.48, 0.69);

            // Border
            if (np.x < 0.01 || np.x > 0.99 || np.y < 0.01 || np.y > 0.99) {
                col = vec3(0.2, 0.35, 0.55);
            } else if (np.x > 0.02 && np.x < 0.98 && np.y > 0.02 && np.y < 0.98) {
                col = vec3(0.04, 0.06, 0.1);

                // Settings bars: SNR, Rays, Y/C Sep
                float bar_idx = floor(np.y * 3.0);
                float bar_frac = fract(np.y * 3.0);

                if (bar_frac > 0.15 && bar_frac < 0.85 && np.x > 0.35) {
                    float val = 0.0;
                    vec3 bar_col = vec3(0.2, 0.5, 0.8);
                    if (bar_idx == 0.0) { val = u_snr / 60.0; bar_col = vec3(0.1, 0.7, 0.3); }
                    else if (bar_idx == 1.0) { val = u_num_rays / 8.0; bar_col = vec3(0.8, 0.3, 0.2); }
                    else if (bar_idx == 2.0) { val = u_yc_sep; bar_col = vec3(0.8, 0.6, 0.2); }

                    float track_x = (np.x - 0.35) / 0.60;
                    float track_bg = 0.03;

                    if (track_x < val) {
                        col = mix(vec3(track_bg), bar_col, 0.8);
                    }

                    if (abs(track_x - val) < 0.008) {
                        col = vec3(1.0);
                    }
                }
            }
        }
    }

    fragColor = vec4(col, 1.0);
}
"""

# ─── Pass 5: Copy ────────────────────────────────────────────────────────────
FRAG_COPY = """
#version 460 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTHON APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class CRTEphysicsSimulator:
    """
    Faithful RF TV signal chain emulator.
    Models the full analog receiver path: RF propagation, multipath,
    noise, demodulation, and video artifacts.

    The rendering pipeline has 5 stages:
      1. Source    — generates or loads the test image
      2. Signal    — NTSC composite encode → RF channel → decode
      3. CRT       — electron beam → phosphor display
      4. UI        — settings overlay
      5. Copy      — blit to screen

    RF signal quality is derived from a physical propagation model:
    free-space path loss, atmospheric absorption, multipath reflections,
    and receiver noise figure.
    """

    # Speed of light (m/s)
    C = 299792458.0

    # Physical environment presets — each creates a real 3D scene.
    # TX power is in dBm (0 dBm = 1 mW; real TV stations: 50-80 dBm = 100W-100kW)
    # Ghost delay ∝ extra path length: 1 µs ≈ 300m extra path
    # Buildings must be wide enough that the reflection geometry hits them.
    PRESETS = {
        'rooftop_antenna': {
            'desc':        'Rooftop antenna, urban multipath',
            'tx_pos':      [0.0, 0.0, 30.0],
            'rx_pos':      [15000.0, 200.0, 3.0],
            'buildings':   [
                # Near TX — industrial/commercial
                (200.0,   100.0, 60.0, 40.0, 25.0, 'concrete'),
                (200.0,  -100.0, 50.0, 35.0, 20.0, 'glass'),
                # South side reflectors (normal_y=-1), geometrically valid:
                # y=-100 → x_hit=3750, delay≈0.01µs
                (3750.0, -120.0, 300.0, 40.0, 20.0, 'concrete'),
                # y=-200 → x_hit=5000, delay≈0.04µs
                (5000.0, -220.0, 400.0, 40.0, 25.0, 'brick'),
                # y=-400 → x_hit=6000, delay≈0.11µs
                (6000.0, -420.0, 500.0, 50.0, 30.0, 'concrete'),
                # y=-600 → x_hit=6429, delay≈0.21µs
                (6429.0, -620.0, 400.0, 40.0, 22.0, 'metal'),
                # y=-800 → x_hit=6667, delay≈0.35µs
                (6667.0, -820.0, 500.0, 60.0, 35.0, 'glass'),
                # y=-1000 → x_hit=6818, delay≈0.53µs
                (6818.0, -1020.0, 400.0, 50.0, 28.0, 'concrete'),
                # y=-1500 → x_hit=7125, delay≈1.26µs
                (7125.0, -1520.0, 500.0, 60.0, 30.0, 'brick'),
                # North side reflectors (normal_y=+1), geometrically valid:
                # y=+300 → x_hit=11250, delay≈0.01µs
                (11250.0, 280.0, 400.0, 40.0, 22.0, 'concrete'),
                # y=+500 → x_hit=9375, delay≈0.07µs
                (9375.0, 480.0, 500.0, 50.0, 28.0, 'metal'),
                # y=+700 → x_hit=8438, delay≈0.16µs
                (8438.0, 680.0, 400.0, 40.0, 25.0, 'brick'),
                # y=+1000 → x_hit=8182, delay≈0.53µs
                (8182.0, 980.0, 500.0, 50.0, 30.0, 'concrete'),
                # y=+1500 → x_hit=7969, delay≈1.12µs
                (7969.0, 1480.0, 400.0, 50.0, 28.0, 'glass'),
                # Near RX — large building face
                (14500.0, -100.0, 60.0, 40.0, 20.0, 'concrete'),
                (15200.0,  150.0, 80.0, 50.0, 30.0, 'glass'),
            ],
            'trees':       [(3000.0, 250.0, 8), (6000.0, -200.0, 6),
                            (10000.0, 300.0, 10), (14000.0, -250.0, 7)],
            'vehicles':    [(4500.0, 50.0), (7000.0, -80.0), (12000.0, 120.0),
                            (14800.0, -50.0)],
            'power_towers': [(3000.0, -300.0), (9000.0, 400.0)],
            'bills':       [(6000.0, 100.0, 6.0, 3.0, 90.0),
                            (11000.0, -150.0, 8.0, 4.0, 90.0)],
            'freq_mhz':    200.0,
            'tx_power_dbm': 60.0,
            'noise_figure_db': 4.0,
            'yc_separation': 0.95,
            'phase_error_deg': 1.0,
        },
        'indoor_rabbit_ears': {
            'desc':        'Indoor rabbit ears, urban canyon',
            'tx_pos':      [0.0, 0.0, 30.0],
            'rx_pos':      [30000.0, 500.0, 3.0],
            'buildings':   [
                # TX-side block
                (100.0,   100.0, 80.0, 50.0, 30.0, 'concrete'),
                (100.0,  -100.0, 70.0, 45.0, 25.0, 'glass'),
                # South side reflectors:
                # y=-100→x=4286, y=-200→x=6667, y=-300→x=8182
                (4286.0,  -120.0, 400.0, 40.0, 25.0, 'concrete'),
                (6667.0,  -220.0, 500.0, 50.0, 30.0, 'brick'),
                (8182.0,  -320.0, 400.0, 40.0, 28.0, 'metal'),
                # y=-500→x=10000, y=-700→x=11053
                (10000.0, -520.0, 500.0, 50.0, 32.0, 'concrete'),
                (11053.0, -720.0, 400.0, 40.0, 25.0, 'brick'),
                # y=-1000→x=12000, y=-1300→x=12581
                (12000.0, -1020.0, 500.0, 50.0, 30.0, 'glass'),
                (12581.0, -1320.0, 400.0, 40.0, 28.0, 'concrete'),
                # y=-1600→x=12973, y=-2000→x=13333
                (12973.0, -1620.0, 400.0, 40.0, 25.0, 'metal'),
                (13333.0, -2020.0, 500.0, 50.0, 30.0, 'brick'),
                # North side reflectors:
                # y=+600→x=25714, y=+700→x=23333, y=+800→x=21818
                (25714.0, 580.0, 400.0, 40.0, 25.0, 'concrete'),
                (23333.0, 680.0, 500.0, 50.0, 30.0, 'metal'),
                (21818.0, 780.0, 400.0, 40.0, 28.0, 'glass'),
                # y=+1000→x=20000, y=+1300→x=18947
                (20000.0, 980.0, 500.0, 50.0, 32.0, 'brick'),
                (18947.0, 1280.0, 400.0, 40.0, 25.0, 'concrete'),
                # Near RX
                (29500.0, 600.0, 100.0, 60.0, 30.0, 'concrete'),
                (30500.0, 400.0, 60.0, 40.0, 20.0, 'brick'),
                (31000.0, 500.0, 70.0, 45.0, 28.0, 'glass'),
            ],
            'trees':       [(5000.0, 300.0, 12), (10000.0, -250.0, 8),
                            (20000.0, 200.0, 10), (27000.0, -150.0, 6)],
            'vehicles':    [(7000.0, 50.0), (14000.0, -80.0), (20000.0, 120.0),
                            (28000.0, 200.0), (29500.0, 350.0)],
            'power_towers': [(6000.0, -400.0), (16000.0, 350.0), (24000.0, -300.0)],
            'water_towers': [(10000.0, -100.0), (22000.0, 250.0)],
            'bills':       [(9000.0, 200.0, 6.0, 3.0, 90.0),
                            (18000.0, -100.0, 8.0, 4.0, 90.0),
                            (26000.0, 150.0, 7.0, 3.5, 90.0)],
            'freq_mhz':    200.0,
            'tx_power_dbm': 60.0,
            'noise_figure_db': 5.0,
            'building_loss_db': 6.0,
            'yc_separation': 0.5,
            'phase_error_deg': 5.0,
        },
        'distant_suburban': {
            'desc':        'Distant station, suburban sprawl',
            'tx_pos':      [0.0, 0.0, 50.0],
            'rx_pos':      [60000.0, 0.0, 3.0],
            'buildings':   [
                # TX area
                (200.0,   0.0, 120.0, 80.0, 20.0, 'metal'),
                (400.0,  200.0, 60.0, 40.0, 15.0, 'concrete'),
                # Mid-path: long buildings perpendicular at calculated reflection pts
                # Symmetric path (y=0): all x_hit = 30000
                (30000.0,  100.0, 400.0, 40.0, 25.0, 'concrete'),
                (30000.0, -100.0, 400.0, 40.0, 25.0, 'brick'),
                (30000.0,  200.0, 300.0, 40.0, 20.0, 'metal'),
                (30000.0, -200.0, 300.0, 40.0, 20.0, 'concrete'),
                (30000.0,  350.0, 500.0, 50.0, 30.0, 'glass'),
                (30000.0, -350.0, 500.0, 50.0, 30.0, 'brick'),
                (30000.0,  500.0, 400.0, 40.0, 25.0, 'concrete'),
                (30000.0, -500.0, 400.0, 40.0, 25.0, 'metal'),
                (30000.0,  700.0, 300.0, 40.0, 22.0, 'brick'),
                (30000.0, -700.0, 300.0, 40.0, 22.0, 'concrete'),
                (30000.0, 1000.0, 400.0, 50.0, 28.0, 'glass'),
                (30000.0,-1000.0, 400.0, 50.0, 28.0, 'metal'),
                (30000.0, 1300.0, 300.0, 40.0, 20.0, 'concrete'),
                (30000.0,-1300.0, 300.0, 40.0, 20.0, 'brick'),
                # RX area
                (58000.0, 100.0, 20.0, 15.0, 8.0, 'wood'),
                (59000.0, -80.0, 25.0, 20.0, 10.0, 'brick'),
                (60500.0, 100.0, 35.0, 28.0, 11.0, 'concrete'),
                (61000.0, -50.0, 28.0, 22.0, 9.0, 'brick'),
            ],
            'hills':       [(35000.0, 0.0, 400.0, 60.0, 600.0),
                            (55000.0, -400.0, 200.0, 40.0, 400.0)],
            'trees':       [(10000.0, 200.0, 15), (20000.0, -200.0, 12),
                            (30000.0, 300.0, 20), (40000.0, -150.0, 18),
                            (50000.0, 250.0, 10)],
            'vehicles':    [(15000.0, 50.0), (30000.0, -30.0), (45000.0, 80.0),
                            (55000.0, -60.0)],
            'power_towers': [(8000.0, -300.0), (25000.0, 400.0),
                            (40000.0, -350.0), (55000.0, 300.0)],
            'freq_mhz':    500.0,
            'tx_power_dbm': 57.0,
            'noise_figure_db': 5.0,
            'yc_separation': 0.3,
            'phase_error_deg': 10.0,
        },
        'valley_multipath': {
            'desc':        'Mountain valley, heavy multipath',
            'tx_pos':      [0.0, 0.0, 100.0],
            'rx_pos':      [80000.0, 0.0, 3.0],
            'buildings':   [
                # TX area
                (1000.0, 100.0, 30.0, 25.0, 10.0, 'wood'),
                (1500.0, -80.0, 25.0, 20.0, 8.0, 'brick'),
                # Valley walls — long buildings at reflection midpoints
                # All x_hit ≈ 40000 for symmetric path
                (40000.0,  200.0, 500.0, 50.0, 30.0, 'concrete'),
                (40000.0, -200.0, 500.0, 50.0, 30.0, 'brick'),
                (40000.0,  400.0, 400.0, 40.0, 25.0, 'metal'),
                (40000.0, -400.0, 400.0, 40.0, 25.0, 'concrete'),
                (40000.0,  600.0, 500.0, 60.0, 35.0, 'glass'),
                (40000.0, -600.0, 500.0, 60.0, 35.0, 'metal'),
                (40000.0,  800.0, 400.0, 50.0, 28.0, 'concrete'),
                (40000.0, -800.0, 400.0, 50.0, 28.0, 'brick'),
                (40000.0, 1000.0, 500.0, 50.0, 30.0, 'metal'),
                (40000.0,-1000.0, 500.0, 50.0, 30.0, 'concrete'),
                (40000.0, 1300.0, 400.0, 40.0, 25.0, 'brick'),
                (40000.0,-1300.0, 400.0, 40.0, 25.0, 'metal'),
                (40000.0, 1700.0, 500.0, 60.0, 32.0, 'concrete'),
                (40000.0,-1700.0, 500.0, 60.0, 32.0, 'glass'),
                # Metal structures — strong specular
                (25000.0, -200.0, 15.0, 15.0, 8.0, 'metal'),
                (45000.0, 250.0, 20.0, 20.0, 10.0, 'metal'),
                # RX area
                (78000.0, -40.0, 25.0, 20.0, 9.0, 'wood'),
                (79500.0, 60.0, 30.0, 25.0, 11.0, 'brick'),
            ],
            'hills':       [
                # Left valley wall (negative y)
                (5000.0, -800.0, 600.0, 200.0, 400.0),
                (20000.0, -600.0, 400.0, 150.0, 300.0),
                (40000.0, -700.0, 500.0, 180.0, 350.0),
                (60000.0, -800.0, 600.0, 220.0, 400.0),
                # Right valley wall (positive y)
                (10000.0, 800.0, 500.0, 180.0, 380.0),
                (30000.0, 600.0, 450.0, 160.0, 320.0),
                (50000.0, 900.0, 550.0, 200.0, 420.0),
                (70000.0, 700.0, 500.0, 170.0, 360.0),
            ],
            'trees':       [(5000.0, 500.0, 20), (15000.0, -400.0, 15),
                            (25000.0, 600.0, 18), (35000.0, -500.0, 22),
                            (45000.0, 400.0, 16), (55000.0, -300.0, 20),
                            (65000.0, 500.0, 14), (75000.0, -400.0, 12)],
            'vehicles':    [(10000.0, 50.0), (25000.0, -30.0), (40000.0, 80.0),
                            (55000.0, -60.0), (70000.0, 40.0)],
            'power_towers': [(10000.0, 500.0), (30000.0, -500.0),
                            (50000.0, 600.0), (70000.0, -500.0)],
            'water_towers': [(20000.0, 100.0), (40000.0, -80.0),
                            (60000.0, 120.0)],
            'freq_mhz':    500.0,
            'tx_power_dbm': 47.0,
            'noise_figure_db': 6.0,
            'building_loss_db': 6.0,
            'yc_separation': 0.15,
            'phase_error_deg': 18.0,
        },
    }
    PRESET_NAMES = list(PRESETS.keys())

    def _compute_signal(self):
        """Derive shader uniforms from the 3D propagation environment."""
        e = self.env

        # Build the 3D environment
        self.prop_env = RFPropagationEnvironment()
        self.prop_env.tx_pos = list(e['tx_pos'])
        self.prop_env.rx_pos = list(e['rx_pos'])
        self.prop_env.frequency_hz = e['freq_mhz'] * 1e6

        # Buildings
        for bld in e.get('buildings', []):
            self.prop_env.add_building(*bld)

        # Trees
        for tree in e.get('trees', []):
            self.prop_env.add_tree_cluster(*tree)

        # Vehicles
        for veh in e.get('vehicles', []):
            self.prop_env.add_vehicle(*veh)

        # Power line towers
        for pt in e.get('power_towers', []):
            self.prop_env.add_power_line_tower(*pt)

        # Water towers
        for wt in e.get('water_towers', []):
            self.prop_env.add_water_tower(*wt)

        # Billboards
        for bb in e.get('bills', []):
            self.prop_env.add_billboard(*bb)

        # Hills / terrain slopes
        for hill in e.get('hills', []):
            self.prop_env.add_hill(*hill)

        # Fences
        for fence in e.get('fences', []):
            self.prop_env.add_fence(*fence)

        # Terrain slopes (gradual elevation changes)
        for ts in e.get('terrain_slopes', []):
            self.prop_env.add_terrain_slope(*ts)

        # Trace rays through the environment
        rays = self.prop_env.compute_rays()

        # Compute received power and SNR from the direct path
        d_direct = self.prop_env._path_length(self.prop_env.tx_pos, self.prop_env.rx_pos)
        d_km = d_direct / 1000.0
        f_mhz = e['freq_mhz']
        fspl_db = 20.0 * math.log10(max(d_km, 0.01)) + 20.0 * math.log10(f_mhz) + 32.45

        # Real TX power in dBm
        tx_power_dbm = e.get('tx_power_dbm', 60.0)
        building_loss_db = e.get('building_loss_db', 0.0)

        # Received power = TX power - path loss - building penetration loss
        rx_power_dbm = tx_power_dbm - fspl_db - building_loss_db

        # Receiver noise floor: kTB + NF (NTSC = 6 MHz bandwidth)
        noise_floor_dbm = -174.0 + 10.0 * math.log10(6e6) + e['noise_figure_db']
        snr_db = max(0.0, min(60.0, rx_power_dbm - noise_floor_dbm))

        # Format ray data for shader
        num_rays, delays, amps, phases = self.prop_env.compute_rays_for_shader()

        # Composite quality degrades with noise
        yc_sep = e['yc_separation'] * max(0.3, min(1.0, (snr_db - 15.0) / 25.0))
        phase_err = e['phase_error_deg'] * max(1.0, 45.0 / max(snr_db, 10.0))

        self.rf = {
            'snr_db':           snr_db,
            'num_rays':         num_rays,
            'ray_delays':       delays,
            'ray_amps':         amps,
            'ray_phases':       phases,
            'yc_separation':    yc_sep,
            'phase_error_deg':  phase_err,
            'differential_gain': max(0.02, (45.0 - snr_db) / 80.0),
            'hum_amplitude':    max(0.002, (45.0 - snr_db) / 500.0),
        }

    def __init__(self, width=1280, height=720, fullscreen=False, source_path=None, screen_capture=False):
        self.W = width
        self.H = height
        self.fullscreen = fullscreen
        self.source_path = source_path
        self.screen_capture = screen_capture
        self.show_ui = False
        self.show_test = not (screen_capture or source_path)
        self.running = False
        self.t0 = time.time()
        self.frame_count = 0
        self.sct = None  # mss screen capture context

        # Physical environment → derived RF signal quality
        self.env = dict(self.PRESETS['rooftop_antenna'])
        self.env['buildings'] = list(self.PRESETS['rooftop_antenna']['buildings'])
        self.env['tx_pos'] = list(self.PRESETS['rooftop_antenna']['tx_pos'])
        self.env['rx_pos'] = list(self.PRESETS['rooftop_antenna']['rx_pos'])
        self.rf = {}
        self._compute_signal()
        self.preset_idx = 0

        # CRT display settings
        self.crt = {
            'curvature':      0.0,
            'beam_width':     0.15,   # tight focus for high-SNR (was 0.35)
            'bloom':          0.10,   # minimal bloom at 60dB SNR (was 0.25)
            'mask_type':      0.0,     # 0=shadow mask,1=aperture grille, 2=trinitron
            'mask_pitch':     3.0,     # pixels per mask period (higher = less visible on small text)
            'brightness':     1.0,
            'contrast':       1.0,
            'gamma':          2.2,
            'focus_h':        0.92,    # sharper focus (was 0.85)
            'focus_v':        0.95,    # sharper focus (was 0.90)
            'convergence':    0.08,    # tighter convergence (was 0.15)
            'overscan_h':     0.0,
            'overscan_v':     0.0,
            'interlace':      1.0,
            'persistence':    0.4,
        }

        self._init_glfw()
        self.ctx = mgl.create_context()
        print(f"GPU: {self.ctx.info['GL_RENDERER']}")
        print(f"GL:  {self.ctx.info['GL_VERSION']}")

        self._compile_shaders()
        self._make_quad()
        self._make_fbos()
        self._load_source()
        self._init_screen_capture()

    # ── GLFW setup ────────────────────────────────────────────────────────
    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        if self.fullscreen:
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon)
            self.W, self.H = mode.size.width, mode.size.height
        else:
            mon = None

        self.win = glfw.create_window(self.W, self.H, "CRT/RF Physics Emulator", mon, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("Window creation failed")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.win, self._key_cb)
        glfw.set_framebuffer_size_callback(self.win, self._resize_cb)
        glfw.set_drop_callback(self.win, self._drop_cb)

    def _compile_shaders(self):
        print("Compiling shaders...")
        self.p_source = self.ctx.program(vertex_shader=VERT_FULLSCREEN, fragment_shader=FRAG_SOURCE)
        self.p_signal = self.ctx.program(vertex_shader=VERT_FULLSCREEN, fragment_shader=FRAG_SIGNAL)
        self.p_crt    = self.ctx.program(vertex_shader=VERT_FULLSCREEN, fragment_shader=FRAG_CRT)
        self.p_ui     = self.ctx.program(vertex_shader=VERT_FULLSCREEN, fragment_shader=FRAG_UI)
        self.p_copy   = self.ctx.program(vertex_shader=VERT_FULLSCREEN, fragment_shader=FRAG_COPY)
        print("  All 5 shader programs compiled.")

    def _make_quad(self):
        verts = np.array([
            -1,-1, 0,0,   1,-1, 1,0,   1,1, 1,1,
            -1,-1, 0,0,   1,1, 1,1,   -1,1, 0,1,
        ], dtype='f4')
        buf = self.ctx.buffer(verts)
        # moderngl requires a separate VAO per program — create one for each
        self.vaos = {}
        for name, prog in [('source', self.p_source), ('signal', self.p_signal),
                           ('crt', self.p_crt), ('ui', self.p_ui), ('copy', self.p_copy)]:
            self.vaos[name] = self.ctx.vertex_array(prog, [(buf, '2f 2f', 'a_pos', 'a_uv')])

    def _make_fbo(self, w, h):
        tex = self.ctx.texture((w, h), 4, dtype='f1')
        tex.filter = (mgl.LINEAR, mgl.LINEAR)
        return self.ctx.framebuffer(color_attachments=[tex]), tex

    def _make_fbos(self):
        self.fbo_src,  self.tex_src  = self._make_fbo(self.W, self.H)
        self.fbo_sig,  self.tex_sig  = self._make_fbo(self.W, self.H)
        self.fbo_crt,  self.tex_crt  = self._make_fbo(self.W, self.H)
        self.fbo_prev, self.tex_prev = self._make_fbo(self.W, self.H)
        self.fbo_ui,   self.tex_ui   = self._make_fbo(self.W, self.H)

    def _load_source(self):
        if self.source_path and os.path.exists(self.source_path):
            try:
                img = Image.open(self.source_path).convert('RGB')
                img = img.resize((self.W, self.H), Image.Resampling.LANCZOS)
                data = np.array(img, dtype='f4') / 255.0
                self.tex_src = self.ctx.texture((self.W, self.H), 3, data=data.tobytes())
                self.tex_src.filter = (mgl.LINEAR, mgl.LINEAR)
                self.show_test = False
                print(f"Loaded: {self.source_path}")
            except Exception as e:
                print(f"Load failed: {e} — using test pattern")

    def _init_screen_capture(self):
        """Initialize mss screen capture."""
        if not HAS_MSS:
            if self.screen_capture:
                print("  mss not installed — pip install mss")
                self.screen_capture = False
            return
        if self.screen_capture or not self.source_path:
            # Pre-create mss capture of primary monitor
            self.sct = mss.MSS()
            mon = self.sct.monitors[1]  # primary monitor
            self.sct_mon = mon
            print(f"  Screen capture: {mon['width']}×{mon['height']}")
            if self.screen_capture:
                self.show_test = False

    def _capture_screen(self):
        """Grab one frame of the screen, resize to window, and upload to source texture."""
        if not self.sct or not self.screen_capture:
            return
        # Grab screen as RGB raw bytes
        shot = self.sct.grab(self.sct_mon)
        # shot.rgb gives raw RGB bytes, shot.size gives (w, h)
        img = Image.frombytes('RGB', shot.size, shot.rgb)
        # Flip vertically — mss captures top-down, OpenGL expects bottom-up
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # Resize to window dimensions
        img = img.resize((self.W, self.H), Image.Resampling.LANCZOS)
        # Convert to RGBA uint8 (texture is 4-component)
        data = np.array(img, dtype=np.uint8)
        alpha = np.full((self.H, self.W, 1), 255, dtype=np.uint8)
        data_rgba = np.concatenate([data, alpha], axis=2)
        self.tex_src.write(data_rgba.tobytes())

    def _use_quad(self, name):
        self.vaos[name].render(mgl.TRIANGLES)

    # ── Render passes ─────────────────────────────────────────────────────
    def _pass_source(self, t):
        self.fbo_src.use()
        self.ctx.clear(0, 0, 0, 1)
        self.p_source['u_time'].value = t
        self.p_source['u_res'].value = (float(self.W), float(self.H))
        self._use_quad('source')

    def _pass_signal(self, t):
        self.fbo_sig.use()
        self.ctx.clear(0, 0, 0, 1)
        tex = self.tex_src if self.show_test else self.tex_src
        tex.use(location=0)
        self.p_signal['u_source'].value = 0
        self.p_signal['u_time'].value = t
        self.p_signal['u_res'].value = (float(self.W), float(self.H))
        self.p_signal['u_snr_db'].value = self.rf['snr_db']
        self.p_signal['u_num_rays'].value = self.rf['num_rays']
        self.p_signal['u_ray_delay'].value = tuple(self.rf['ray_delays'])
        self.p_signal['u_ray_amp'].value = tuple(self.rf['ray_amps'])
        self.p_signal['u_ray_phase'].value = tuple(self.rf['ray_phases'])
        self.p_signal['u_yc_separation'].value = self.rf['yc_separation']
        self.p_signal['u_phase_error_deg'].value = self.rf['phase_error_deg']
        self.p_signal['u_differential_gain'].value = self.rf['differential_gain']
        self.p_signal['u_hum_amplitude'].value = self.rf['hum_amplitude']
        self.p_signal['u_composite_artifacts'].value = 1.0
        self._use_quad('signal')

    def _pass_crt(self, t):
        self.fbo_crt.use()
        self.ctx.clear(0, 0, 0, 1)
        self.tex_sig.use(location=0)
        self.tex_prev.use(location=1)
        self.p_crt['u_signal'].value = 0
        self.p_crt['u_prev_frame'].value = 1
        self.p_crt['u_time'].value = t
        self.p_crt['u_res'].value = (float(self.W), float(self.H))
        for k in ['curvature', 'beam_width', 'bloom', 'mask_type', 'mask_pitch',
                   'brightness', 'contrast', 'gamma', 'focus_h', 'focus_v',
                   'convergence', 'overscan_h', 'overscan_v', 'interlace', 'persistence']:
            self.p_crt[f'u_{k}'].value = self.crt[k]
        self._use_quad('crt')

        # Copy CRT output to prev_frame for persistence
        self.fbo_prev.use()
        self.ctx.clear(0, 0, 0, 1)
        self.tex_crt.use(location=0)
        self.p_copy['u_tex'].value = 0
        self._use_quad('copy')

    def _pass_ui(self, t):
        self.fbo_ui.use()
        self.ctx.clear(0, 0, 0, 1)
        self.tex_sig.use(location=0)   # read from signal pass, skip CRT
        self.p_ui['u_screen'].value = 0
        self.p_ui['u_show_ui'].value = 1.0 if self.show_ui else 0.0
        self.p_ui['u_snr'].value = self.rf['snr_db']
        self.p_ui['u_num_rays'].value = float(self.rf['num_rays'])
        self.p_ui['u_yc_sep'].value = self.rf['yc_separation']
        self._use_quad('ui')

    def _pass_copy_screen(self):
        self.ctx.screen.use()
        self.ctx.clear(0, 0, 0, 1)
        self.tex_ui.use(location=0)
        self.p_copy['u_tex'].value = 0
        self._use_quad('copy')

    # ── Input handling ────────────────────────────────────────────────────
    def _key_cb(self, win, key, sc, action, mods):
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            self.running = False

        elif key == glfw.KEY_F1:
            self.show_ui = not self.show_ui

        elif key == glfw.KEY_F3:
            self.show_test = not self.show_test
            print(f"  Test pattern: {'ON' if self.show_test else 'OFF'}")

        elif key == glfw.KEY_F5:
            self._pick_file()

        elif key == glfw.KEY_F6:
            if HAS_MSS:
                self.screen_capture = not self.screen_capture
                if self.screen_capture:
                    self._init_screen_capture()
                    self.show_test = False
                    print(f"  Screen capture: ON")
                else:
                    print(f"  Screen capture: OFF")
            else:
                print("  mss not installed — pip install mss")

        elif key == glfw.KEY_F11:
            self._toggle_fullscreen()

        elif key == glfw.KEY_F4:
            self.preset_idx = (self.preset_idx + 1) % len(self.PRESET_NAMES)
            name = self.PRESET_NAMES[self.preset_idx]
            self.env = dict(self.PRESETS[name])
            self.env['buildings'] = list(self.PRESETS[name]['buildings'])
            self.env['tx_pos'] = list(self.PRESETS[name]['tx_pos'])
            self.env['rx_pos'] = list(self.PRESETS[name]['rx_pos'])
            self._compute_signal()
            print(f"  {self.env['desc']}")
            d = self.prop_env._path_length(self.prop_env.tx_pos, self.prop_env.rx_pos)
            print(f"    TX→RX: {d/1000:.1f}km  SNR={self.rf['snr_db']:.1f}dB  rays={self.rf['num_rays']}")

        # Move RX closer/farther from TX
        elif key == glfw.KEY_1:
            dx = self.env['rx_pos'][0] - self.env['tx_pos'][0]
            self.env['rx_pos'][0] = self.env['tx_pos'][0] + max(1000, dx - 5000)
            self._compute_signal()
            d = self.prop_env._path_length(self.prop_env.tx_pos, self.prop_env.rx_pos)
            print(f"  RX distance: {d/1000:.1f} km → SNR {self.rf['snr_db']:.1f} dB, {self.rf['num_rays']} rays")
        elif key == glfw.KEY_2:
            dx = self.env['rx_pos'][0] - self.env['tx_pos'][0]
            self.env['rx_pos'][0] = self.env['tx_pos'][0] + min(200000, dx + 5000)
            self._compute_signal()
            d = self.prop_env._path_length(self.prop_env.tx_pos, self.env['rx_pos'])
            print(f"  RX distance: {d/1000:.1f} km → SNR {self.rf['snr_db']:.1f} dB, {self.rf['num_rays']} rays")

        # Move RX sideways (into/out of building reflections)
        elif key == glfw.KEY_3:
            self.env['rx_pos'][1] -= 500
            self._compute_signal()
            print(f"  RX lateral: {self.env['rx_pos'][1]:.0f}m → {self.rf['num_rays']} rays")
        elif key == glfw.KEY_4:
            self.env['rx_pos'][1] += 500
            self._compute_signal()
            print(f"  RX lateral: {self.env['rx_pos'][1]:.0f}m → {self.rf['num_rays']} rays")

        # Raise/lower TX antenna
        elif key == glfw.KEY_5:
            self.env['tx_pos'][2] = max(5, self.env['tx_pos'][2] - 5)
            self._compute_signal()
            print(f"  TX height: {self.env['tx_pos'][2]:.0f}m → SNR {self.rf['snr_db']:.1f} dB")
        elif key == glfw.KEY_6:
            self.env['tx_pos'][2] = min(500, self.env['tx_pos'][2] + 5)
            self._compute_signal()
            print(f"  TX height: {self.env['tx_pos'][2]:.0f}m → SNR {self.rf['snr_db']:.1f} dB")

        # Frequency: lower = longer wavelength, better diffraction; higher = more noise
        elif key == glfw.KEY_7:
            self.env['freq_mhz'] = max(50.0, self.env['freq_mhz'] - 50.0)
            self._compute_signal()
            print(f"  Frequency: {self.env['freq_mhz']:.0f} MHz → SNR {self.rf['snr_db']:.1f} dB")
        elif key == glfw.KEY_8:
            self.env['freq_mhz'] = min(1000.0, self.env['freq_mhz'] + 50.0)
            self._compute_signal()
            print(f"  Frequency: {self.env['freq_mhz']:.0f} MHz → SNR {self.rf['snr_db']:.1f} dB")

        elif key == glfw.KEY_9:
            self.crt['curvature'] = max(0.0, self.crt['curvature'] - 0.05)
            print(f"  Curvature: {self.crt['curvature']:.2f}")
        elif key == glfw.KEY_0:
            self.crt['curvature'] = min(1.0, self.crt['curvature'] + 0.05)
            print(f"  Curvature: {self.crt['curvature']:.2f}")

        elif key == glfw.KEY_MINUS:
            self.crt['bloom'] = max(0.0, self.crt['bloom'] - 0.05)
            print(f"  Bloom: {self.crt['bloom']:.2f}")
        elif key == glfw.KEY_EQUAL:
            self.crt['bloom'] = min(1.0, self.crt['bloom'] + 0.05)
            print(f"  Bloom: {self.crt['bloom']:.2f}")

        elif key == glfw.KEY_TAB:
            self.crt['interlace'] = 1.0 - self.crt['interlace']
            print(f"  Interlace: {'ON' if self.crt['interlace'] > 0.5 else 'OFF'}")

        elif key == glfw.KEY_BACKSPACE:
            name = self.PRESET_NAMES[0]
            self.env = dict(self.PRESETS[name])
            self.env['buildings'] = list(self.PRESETS[name]['buildings'])
            self.env['tx_pos'] = list(self.PRESETS[name]['tx_pos'])
            self.env['rx_pos'] = list(self.PRESETS[name]['rx_pos'])
            self._compute_signal()
            self.crt.update({
                'curvature': 0.0, 'beam_width': 0.35, 'bloom': 0.25,
                'mask_type': 0.0, 'brightness': 1.0, 'contrast': 1.0,
                'gamma': 2.2, 'focus_h': 0.85, 'focus_v': 0.90,
                'convergence': 0.15, 'interlace': 1.0, 'persistence': 0.4,
            })
            self.preset_idx = 0
            print(f"  Reset: {self.env['desc']}")

    def _resize_cb(self, win, w, h):
        if w <= 0 or h <= 0:
            return
        self.W, self.H = w, h
        self.ctx.viewport = (0, 0, w, h)
        self._make_fbos()

    def _drop_cb(self, win, paths):
        """Handle files dragged onto the window."""
        if paths:
            self._load_source_file(paths[0])

    def _load_source_file(self, path):
        """Load an image file as the CRT source content."""
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((self.W, self.H), Image.Resampling.LANCZOS)
            data = np.array(img, dtype='f4') / 255.0
            self.tex_src = self.ctx.texture((self.W, self.H), 3, data=data.tobytes())
            self.tex_src.filter = (mgl.LINEAR, mgl.LINEAR)
            self.show_test = False
            self.source_path = path
            print(f"  Source: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Load failed: {e}")

    def _toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon)
            glfw.set_window_monitor(self.win, mon, 0, 0,
                                     mode.size.width, mode.size.height,
                                     mode.refresh_rate)
        else:
            glfw.set_window_monitor(self.win, None, 100, 100, self.W, self.H, 0)
        # FBOs need to be rebuilt at the new size
        glfw.poll_events()
        w, h = glfw.get_framebuffer_size(self.win)
        if w > 0 and h > 0:
            self.W, self.H = w, h
            self.ctx.viewport = (0, 0, w, h)
            self._make_fbos()
        print(f"  {'Fullscreen' if self.fullscreen else 'Windowed'} ({self.W}×{self.H})")

    def _pick_file(self):
        """Open a file picker dialog (uses tkinter)."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Select source image",
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
                    ("All files", "*.*"),
                ]
            )
            root.destroy()
            if path:
                self._load_source_file(path)
        except ImportError:
            print("  tkinter not available — use --source flag or drag-and-drop")
        except Exception as e:
            print(f"  File picker error: {e}")

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        self.running = True
        print()
        print("═" * 62)
        print("  RF TV Signal Physics Emulator")
        print("═" * 62)
        print(f"  {self.W}×{self.H}  |  {self.env['desc']}")
        print(f"  F1=UI  F3=pattern  F4=scenarios  F5=file  F6=screen  F11=fullscreen  ESC=quit")
        print(f"  1/2=RX distance  3/4=RX lateral  5/6=TX height  7/8=frequency")
        print("═" * 62)
        print()

        while self.running and not glfw.window_should_close(self.win):
            glfw.poll_events()
            t = time.time() - self.t0
            self.frame_count += 1

            # Screen capture: grab desktop and upload to source texture
            if self.screen_capture and self.sct:
                self._capture_screen()
                # Skip source shader pass — tex_src already has the screen
                self._pass_signal(t)
            else:
                self._pass_source(t)
                self._pass_signal(t)

            # Skip CRT pass — output raw RF receiver video directly
            self._pass_ui(t)
            self._pass_copy_screen()

            glfw.swap_buffers(self.win)

        glfw.terminate()


def main():
    ap = argparse.ArgumentParser(
        description="RF TV Signal Physics Emulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard controls:
  F1         Toggle settings overlay
  F3         Toggle test pattern
  F4         Cycle physical scenarios (Rooftop/Indoor/Suburban/Valley)
  F5         Pick source image file
  F6         Toggle live screen capture
  F11        Toggle fullscreen
  1/2        Move receiver closer/farther
  3/4        Move receiver laterally
  5/6        Raise/lower TX antenna height
  7/8        Adjust broadcast frequency (MHz)
  Backspace  Reset all to defaults
  Esc        Quit

  Drag-and-drop: Drop an image file onto the window
""")
    ap.add_argument('-W', '--width',  type=int, default=1280)
    ap.add_argument('-H', '--height', type=int, default=720)
    ap.add_argument('-f', '--fullscreen', action='store_true')
    ap.add_argument('-s', '--source', type=str, help='Source image path')
    ap.add_argument('--screen', action='store_true', help='Capture live screen as source')
    args = ap.parse_args()

    try:
        app = CRTEphysicsSimulator(args.width, args.height, args.fullscreen, args.source,
                                   screen_capture=args.screen)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
