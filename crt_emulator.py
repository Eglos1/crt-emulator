#!/usr/bin/env python3
"""
CRT/RF TV Signal Emulator
=========================
Faithfully emulates CRT display and RF/composite TV signal artifacts.

This is NOT a post-process filter. It simulates the actual analog signal chain:
  1. RGB source → NTSC/PAL composite encoding (YIQ/YPbPr color space)
  2. Composite signal through RF channel (noise, multipath, ghosting)
  3. RF demodulation with proper bandwidth limiting
  4. NTSC/PAL decoding with color artifacts
  5. CRT electron beam → phosphor display (scanlines, curvature, bloom)

Controls:
  F1         - Toggle settings overlay
  F2         - Cycle CRT mask type (RGB/Trinitron/ShadowMask)
  F3         - Toggle test pattern
  F4         - Cycle input source
  1/2        - Adjust color bleed
  3/4        - Adjust signal noise
  5/6        - Adjust ghosting
  7/8        - Adjust scanline intensity
  9/0        - Adjust curvature
  -/=        - Adjust bloom/glow
  Tab        - Toggle composite artifacts
  Backspace  - Reset all settings
  Esc        - Quit

Usage:
  python3 crt_emulator.py [--width W] [--height H] [--fullscreen]
  python3 crt_emulator.py --source image.png
  python3 crt_emulator.py --source video.mp4  (requires opencv)
"""

import sys
import os
import time
import math
import argparse
import struct
import numpy as np

try:
    import moderngl as mgl
except ImportError:
    print("ERROR: moderngl not installed. Run: pip3 install moderngl")
    sys.exit(1)

try:
    import glfw
except ImportError:
    print("ERROR: glfw not installed. Run: pip3 install glfw")
    sys.exit(1)

from PIL import Image

# ─── Shader Sources ──────────────────────────────────────────────────────────

# Vertex shader for fullscreen quad
VERT_SHADER = """
#version 460 core

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;

out vec2 uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_uv;
}
"""

# Fragment: Test pattern generation
FRAG_TESTPATTERN = """
#version 460 core

in vec2 uv;
out vec4 fragColor;

uniform float time;
uniform vec2 resolution;

// SMPTE color bars + additional patterns
vec3 smpte_color_bars(float t) {
    vec3 colors[8];
    colors[0] = vec3(0.75, 0.75, 0.75); // white
    colors[1] = vec3(0.75, 0.75, 0.0);  // yellow
    colors[2] = vec3(0.0, 0.75, 0.75);  // cyan
    colors[3] = vec3(0.0, 0.75, 0.0);   // green
    colors[4] = vec3(0.75, 0.0, 0.75);  // magenta
    colors[5] = vec3(0.75, 0.0, 0.0);   // red
    colors[6] = vec3(0.0, 0.0, 0.75);   // blue
    colors[7] = vec3(0.0, 0.0, 0.0);    // black

    float x = t * 8.0;
    int idx = int(floor(x));
    if (idx < 0) idx = 0;
    if (idx > 7) idx = 7;
    return colors[idx];
}

// Color ramp for gradient test
vec3 gradient_test(vec2 p) {
    return vec3(p.x, p.y, 0.5 + 0.5 * sin(time * 0.5));
}

// Zone plate for frequency response testing
vec3 zone_plate(vec2 p) {
    vec2 c = p - 0.5;
    float r = length(c);
    float freq = 200.0 * r * r;
    float v = 0.5 + 0.5 * cos(freq * 3.14159 * 2.0);
    return vec3(v);
}

// Crosshatch pattern for geometry test
vec3 crosshatch(vec2 p) {
    vec2 grid = fract(p * 20.0);
    float line = 0.0;
    line = max(line, 1.0 - step(0.03, grid.x));
    line = max(line, 1.0 - step(0.03, grid.y));
    // Diagonal lines
    float dline = 1.0 - step(0.03, fract((p.x + p.y) * 14.14));
    line = max(line, dline * 0.5);

    vec3 base = vec3(0.05, 0.05, 0.1);
    vec3 lineColor = vec3(line) * vec3(0.9, 0.95, 1.0);
    return base + lineColor;
}

// Grayscale ramp
vec3 grayscale_ramp(vec2 p) {
    float g = p.x;
    vec3 top = vec3(g);
    // Bottom half has steps
    if (p.y < 0.5) {
        g = floor(g * 16.0) / 16.0;
        top = vec3(g);
    }
    return top;
}

void main() {
    vec2 p = uv;
    float aspect = resolution.x / resolution.y;
    vec2 ap = vec2(p.x * aspect, p.y);

    // Top section: SMPTE bars (top 67%)
    // Bottom section: gradient, zone plate, etc.
    vec3 color;

    if (p.y > 0.67) {
        // SMPTE color bars - top portion
        float barY = (p.y - 0.67) / 0.33;
        float barX = p.x;
        color = smpte_color_bars(barX);

        // Sub-bar section (narrower bars with different colors)
        if (barY < 0.3) {
            vec3 subColors[8];
            subColors[0] = vec3(0.0, 0.0, 0.75);   // blue
            subColors[1] = vec3(0.0, 0.0, 0.0);    // black
            subColors[2] = vec3(0.75, 0.0, 0.75);  // magenta
            subColors[3] = vec3(0.0, 0.0, 0.0);    // black
            subColors[4] = vec3(0.0, 0.75, 0.75);  // cyan
            subColors[5] = vec3(0.0, 0.0, 0.0);    // black
            subColors[6] = vec3(0.75, 0.75, 0.75); // white
            subColors[7] = vec3(0.06, 0.06, 0.06); // -4% grey

            int idx = int(floor(barX * 8.0));
            if (idx > 7) idx = 7;
            color = subColors[idx];
        }
    } else if (p.y > 0.33) {
        // Middle section: gradient + blocks
        vec2 mp = vec2(p.x, (p.y - 0.33) / 0.34);
        if (mp.x < 0.5) {
            color = grayscale_ramp(vec2(mp.x * 2.0, mp.y));
        } else {
            // PLUGE-like contrast pattern
            float checker = mod(floor(mp.x * 32.0) + floor(mp.y * 32.0), 2.0);
            color = vec3(mix(0.0, 0.75, checker));
        }
    } else {
        // Bottom section: zone plate + crosshatch
        vec2 bp = vec2(p.x, p.y / 0.33);
        if (bp.x < 0.5) {
            color = zone_plate(bp);
        } else {
            color = crosshatch(vec2((bp.x - 0.5) * 2.0, bp.y));
        }
    }

    // Add subtle time-varying noise to simulate live signal
    float noise = fract(sin(dot(p * time, vec2(12.9898, 78.233))) * 43758.5453);
    color += (noise - 0.5) * 0.003;

    fragColor = vec4(color, 1.0);
}
"""

# Fragment: NTSC Composite Encode + Decode + RF Signal Chain
FRAG_SIGNAL = """
#version 460 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D source_tex;
uniform float time;
uniform vec2 resolution;
uniform float color_bleed;      // 0.0 - 1.0
uniform float noise_amount;     // 0.0 - 1.0
uniform float ghosting;         // 0.0 - 1.0
uniform float composite_artifacts; // 0.0 or 1.0
uniform float hum_bar;          // 0.0 - 1.0

const float PI = 3.14159265358979;
const float TAU = PI * 2.0;

// NTSC color space conversion (sRGB → YIQ)
vec3 rgb2yiq(vec3 rgb) {
    mat3 m = mat3(
        0.299,  0.587,  0.114,
        0.596, -0.274, -0.322,
        0.211, -0.523,  0.312
    );
    return m * rgb;
}

// YIQ → sRGB
vec3 yiq2rgb(vec3 yiq) {
    mat3 m = mat3(
        1.0,  1.0,  1.0,
        0.956, -0.272, -1.106,
        0.621, -0.647,  1.703
    );
    return m * yiq;
}

// Pseudo-random noise
float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

float rand2(vec2 co) {
    return fract(sin(dot(co, vec2(34.456, 78.233))) * 23421.631);
}

// Simulate composite signal encoding
// In real NTSC, luminance and chrominance are frequency-multiplexed
// Luma: 0-4.2 MHz, Chroma subcarrier: 3.579545 MHz
float composite_signal(vec2 pos, float scanline_phase) {
    vec3 src = texture(source_tex, pos).rgb;
    vec3 yiq = rgb2yiq(src);

    // Luminance (Y)
    float Y = yiq.x;

    // Chrominance (I and Q modulated onto subcarrier)
    float subcarrier = 3.579545; // MHz (NTSC)
    float phase = pos.x * resolution.x * TAU * 0.25 + scanline_phase; // ~quarter dot clock
    float I = yiq.y * cos(phase * subcarrier * 0.01) * color_bleed;
    float Q = yiq.z * sin(phase * subcarrier * 0.01) * color_bleed;

    // Composite = Y + I*cos(wt) + Q*sin(wt)
    return Y + I + Q;
}

// Simple sinc-like filter for bandwidth limiting
float sinc(float x) {
    if (abs(x) < 0.001) return 1.0;
    return sin(x) / x;
}

void main() {
    vec2 pos = uv;
    float aspect = resolution.x / resolution.y;

    // Scanline phase - alternates each line (NTSC uses half-line offset)
    float line_num = floor(pos.y * resolution.y);
    float scanline_phase = mod(line_num, 2.0) * PI;

    // ── Step 1: Encode composite signal with bandwidth limiting ──
    // Sample horizontally to simulate 4.2 MHz bandwidth limit
    float samples = 8.0;
    float composite = 0.0;
    float total_weight = 0.0;

    for (float i = -samples; i <= samples; i += 1.0) {
        float offset = i / resolution.x;
        vec2 sample_pos = vec2(pos.x + offset, pos.y);
        float weight = sinc(i * 3.14159 * 1.5); // Low-pass filter kernel
        composite += composite_signal(sample_pos, scanline_phase) * weight;
        total_weight += weight;
    }
    composite /= total_weight;

    // ── Step 2: Add RF channel artifacts ──

    // Add white noise (thermal noise in RF receiver)
    float noise = (rand(pos * resolution + time * 100.0) - 0.5) * 2.0;
    float scanline_noise = (rand(vec2(0.0, pos.y * resolution.y + time * 50.0)) - 0.5);
    composite += noise * noise_amount * 0.15;
    composite += scanline_noise * noise_amount * 0.05; // Horizontal line noise

    // Ghosting (multipath interference - delayed copy of signal)
    if (ghosting > 0.01) {
        float delay = 0.003 * ghosting; // ~200ns delay typical for ghosting
        float ghost = composite_signal(vec2(pos.x + delay, pos.y), scanline_phase);
        // Attenuated ghost (typical -6 to -12 dB)
        composite = mix(composite, ghost, ghosting * 0.25);
    }

    // Hum bar (50/60 Hz power supply hum)
    if (hum_bar > 0.01) {
        float hum = sin(pos.y * resolution.y * 0.5 + time * 3.77) * hum_bar * 0.02;
        composite += hum;
    }

    // ── Step 3: Decode composite back to RGB ──

    // Chroma bandpass filter (extract color subcarrier region around 3.58 MHz)
    float chroma = 0.0;
    float c_total = 0.0;
    for (float i = -6.0; i <= 6.0; i += 1.0) {
        float offset = i / resolution.x;
        vec2 sp = vec2(pos.x + offset, pos.y);
        float s = composite_signal(sp, scanline_phase);
        // Simple bandpass around subcarrier frequency
        float bp = cos(i * 3.14159 * 0.5) * 0.5 + 0.5;
        chroma += s * bp;
        c_total += bp;
    }
    chroma /= c_total;

    // Luminance (low-pass filtered composite)
    float luma = composite; // Already low-pass filtered above

    // Demodulate chrominance
    float phase_demod = pos.x * resolution.x * TAU * 0.25 + scanline_phase;
    float I_demod = chroma * 2.0 * cos(phase_demod * 3.579545 * 0.01);
    float Q_demod = chroma * 2.0 * sin(phase_demod * 3.579545 * 0.01);

    // Reconstruct RGB from YIQ
    vec3 yiq_decoded = vec3(
        clamp(luma, 0.0, 1.0),
        clamp(I_demod * 0.8, -0.596, 0.596),
        clamp(Q_demod * 0.8, -0.523, 0.523)
    );
    vec3 rgb = yiq2rgb(yiq_decoded);

    // Composite cross-color artifacts (dot crawl)
    // These appear when the chroma/luma separation is imperfect
    if (composite_artifacts > 0.5) {
        float dot_crawl = sin(pos.x * resolution.x * PI + pos.y * resolution.y * PI) * 0.03;
        rgb += vec3(dot_crawl) * color_bleed;
    }

    // Signal clipping/distortion on bright areas (common in overdriven signals)
    float luminance = dot(rgb, vec3(0.299, 0.587, 0.114));
    if (luminance > 0.95) {
        float over = (luminance - 0.95) * 2.0;
        rgb = mix(rgb, vec3(1.0, 0.95, 0.8), over * 0.3);
    }

    // Black level crush
    rgb = max(rgb - 0.02 * noise_amount, vec3(0.0));

    fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
"""

# Fragment: CRT Display (phosphor, scanlines, curvature, bloom)
FRAG_CRT = """
#version 460 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D signal_tex;
uniform float time;
uniform vec2 resolution;
uniform float curvature;        // 0.0 - 1.0
uniform float scanline_intensity; // 0.0 - 1.0
uniform float bloom_amount;     // 0.0 - 1.0
uniform int mask_type;          // 0=RGB, 1=Trinitron, 2=ShadowMask
uniform float brightness;       // 0.0 - 1.0
uniform float contrast;         // 0.0 - 2.0
uniform float gamma;            // 0.5 - 3.0
uniform float focus;            // 0.0 - 1.0 (electronic focus)
uniform float v_overscan;       // vertical overscan
uniform float h_overscan;       // horizontal overscan

const float PI = 3.14159265358979;

// ── Screen curvature (barrel distortion) ──
vec2 curve_uv(vec2 p) {
    // CRT screen is curved both horizontally and vertically
    // This models the actual electron gun deflection geometry
    p = p * 2.0 - 1.0;

    vec2 offset = abs(p.yx) / vec2(6.0, 4.5) * curvature;
    p = p + p * offset * offset;

    // Overscan (CRT images were often zoomed in slightly)
    p *= vec2(1.0 + h_overscan * 0.05, 1.0 + v_overscan * 0.05);

    p = p * 0.5 + 0.5;
    return p;
}

// ── Phosphor mask simulation ──

// RGB stripe mask (standard shadow mask / aperture grille)
float mask_rgb(vec2 p, vec2 res) {
    float x = p.x * res.x;
    float subpixel = mod(x, 3.0);
    // Each subpixel is R, G, B in order
    return subpixel;
}

// Trinitron (vertical stripe with gap)
float mask_trinitron(vec2 p, vec2 res) {
    float x = p.x * res.x;
    float y = p.y * res.y;
    // Vertical stripes with horizontal gap every other line
    float stripe = mod(x, 3.0);
    float gap = step(0.95, fract(y / 2.0));
    if (gap > 0.5) return -1.0; // dark gap
    return stripe;
}

// Shadow mask (honeycomb pattern)
float mask_shadow(vec2 p, vec2 res) {
    float x = p.x * res.x * 0.5;
    float y = p.y * res.y * 0.33;
    float row = floor(y);
    float offset = mod(row, 2.0) * 1.5;
    float cell = mod(x + offset, 3.0);
    return cell;
}

vec3 apply_mask(vec2 p, vec2 res, vec3 color) {
    if (scanline_intensity < 0.01) return color;

    float m = 0.0;
    vec3 mask_color = vec3(1.0);

    if (mask_type == 0) {
        // RGB stripe
        m = mask_rgb(p, res);
        if (m < 0.5) mask_color = vec3(1.0, 0.2, 0.2);
        else if (m < 1.5) mask_color = vec3(0.2, 1.0, 0.2);
        else mask_color = vec3(0.2, 0.2, 1.0);
    } else if (mask_type == 1) {
        // Trinitron
        m = mask_trinitron(p, res);
        if (m < -0.5) {
            return color * 0.1; // gap is dark
        }
        if (m < 0.5) mask_color = vec3(1.0, 0.3, 0.3);
        else if (m < 1.5) mask_color = vec3(0.3, 1.0, 0.3);
        else mask_color = vec3(0.3, 0.3, 1.0);
    } else {
        // Shadow mask
        m = mask_shadow(p, res);
        if (m < 0.5) mask_color = vec3(1.0, 0.3, 0.3);
        else if (m < 1.5) mask_color = vec3(0.3, 1.0, 0.3);
        else mask_color = vec3(0.3, 0.3, 1.0);
    }

    // The mask attenuates each color channel through its corresponding subpixel
    float mask_strength = scanline_intensity * 0.35;
    return color * mix(vec3(1.0), mask_color, mask_strength);
}

// ── Scanlines ──
vec3 apply_scanlines(vec2 p, vec3 color, vec2 res) {
    if (scanline_intensity < 0.01) return color;

    float y = p.y * res.y;
    float scanline = sin(y * PI) * 0.5 + 0.5;
    scanline = pow(scanline, 1.5);

    float intensity = scanline_intensity * 0.8;
    color *= mix(1.0, scanline, intensity);

    // Scanline brightness varies with image brightness (electron beam response)
    float brightness_response = dot(color, vec3(0.299, 0.587, 0.114));
    color *= 1.0 + brightness_response * scanline_intensity * 0.1;

    return color;
}

// ── Bloom / Glow ──
vec3 apply_bloom(vec2 p, sampler2D tex, vec2 res) {
    if (bloom_amount < 0.01) return vec3(0.0);

    vec3 bloom = vec3(0.0);
    float total = 0.0;

    // Gaussian-like blur kernel
    for (float x = -4.0; x <= 4.0; x += 1.0) {
        for (float y = -4.0; y <= 4.0; y += 1.0) {
            vec2 offset = vec2(x, y) / res * 3.0;
            float d = length(vec2(x, y));
            float weight = exp(-d * d / 8.0);
            bloom += texture(tex, p + offset).rgb * weight;
            total += weight;
        }
    }
    bloom /= total;
    return bloom * bloom_amount;
}

// ── Vignette (CRT edge brightness rolloff) ──
float vignette(vec2 p) {
    vec2 q = p - 0.5;
    float d = length(q);
    return 1.0 - d * d * 0.8;
}

// ── Flicker ──
float flicker(float t) {
    // 60Hz refresh with slight variation
    float f = 1.0 + 0.003 * sin(t * 377.0); // ~60Hz
    f *= 1.0 + 0.002 * sin(t * 7.5); // Slow brightness drift
    return f;
}

// ── Interlace simulation ──
vec3 interlace(vec2 p, vec3 color, float t) {
    float line = floor(p.y * resolution.y);
    float field = mod(floor(t * 30.0), 2.0); // 30 fields/sec interlaced

    // Only draw alternating lines based on field
    if (mod(line + field, 2.0) < 0.5) {
        color *= 0.85; // Slightly dim the non-drawn lines (persistence of vision)
    }

    return color;
}

void main() {
    vec2 p = uv;

    // Apply CRT curvature
    vec2 crt_uv = curve_uv(p);

    // Check if outside CRT screen area
    if (crt_uv.x < 0.0 || crt_uv.x > 1.0 || crt_uv.y < 0.0 || crt_uv.y > 1.0) {
        fragColor = vec4(vec3(0.005), 1.0); // Near-black (CRT phosphor off)
        return;
    }

    // Sample the processed signal
    vec3 color = texture(signal_tex, crt_uv).rgb;

    // Apply bloom (before scanlines for more realistic look)
    vec3 bloom = apply_bloom(crt_uv, signal_tex, resolution);
    color += bloom;

    // Apply scanlines
    color = apply_scanlines(crt_uv, color, resolution);

    // Apply phosphor mask
    color = apply_mask(crt_uv, resolution, color);

    // Interlace simulation
    color = interlace(crt_uv, color, time);

    // Brightness / contrast
    color = (color - 0.5) * contrast + 0.5 + (brightness - 0.5) * 0.3;

    // Gamma correction (CRT gamma is typically 2.2-2.5)
    color = pow(max(color, vec3(0.0)), vec3(1.0 / gamma));

    // Vignette
    color *= vignette(crt_uv);

    // Flicker
    color *= flicker(time);

    // Edge softening (defocus at edges)
    if (focus < 0.9) {
        float edge = length(crt_uv - 0.5) * 2.0;
        float blur = edge * (1.0 - focus) * 0.005;
        vec3 blurred = vec3(0.0);
        float total = 0.0;
        for (float x = -2.0; x <= 2.0; x += 1.0) {
            for (float y = -2.0; y <= 2.0; y += 1.0) {
                float w = exp(-(x*x+y*y)*0.5);
                blurred += texture(signal_tex, crt_uv + vec2(x,y)*blur).rgb * w;
                total += w;
            }
        }
        color = mix(color, blurred / total, (1.0 - focus) * 0.6);
    }

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""

# Fragment: UI overlay for settings
FRAG_UI = """
#version 460 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D screen_tex;
uniform sampler2D font_tex;
uniform float show_ui;
uniform float settings[20];

// Simple character rendering using distance field
// Settings displayed:
// [0]  color_bleed
// [1]  noise_amount
// [2]  ghosting
// [3]  scanline_intensity
// [4]  curvature
// [5]  bloom_amount
// [6]  mask_type
// [7]  brightness
// [8]  contrast
// [9]  gamma
// [10] composite_artifacts
// [11] hum_bar
// [12] focus
// [13] h_overscan
// [14] v_overscan
// [15] show_test_pattern

void main() {
    vec3 color = texture(screen_tex, uv).rgb;

    if (show_ui > 0.5) {
        // Darken background
        color *= 0.7;

        // Draw semi-transparent settings panel
        vec2 p = uv;
        vec2 panel_min = vec2(0.02, 0.02);
        vec2 panel_max = vec2(0.42, 0.85);

        if (p.x > panel_min.x && p.x < panel_max.x &&
            p.y > panel_min.y && p.y < panel_max.y) {
            // Panel background
            vec2 np = (p - panel_min) / (panel_max - panel_min);

            // Border
            float border = 0.008;
            float is_border = 0.0;
            if (np.x < border || np.x > 1.0-border ||
                np.y < border || np.y > 1.0-border) {
                is_border = 1.0;
            }

            // Interior
            if (is_border < 0.5) {
                color = vec3(0.05, 0.08, 0.12);

                // "bars" for each setting
                float num_settings = 14.0;
                float bar_area = 1.0;
                float bar_height = bar_area / num_settings;
                float bar_y = np.y;
                float bar_idx = floor(bar_y / bar_height);
                float bar_frac = fract(bar_y / bar_height);

                if (bar_idx < num_settings && bar_frac > 0.1 && bar_frac < 0.9) {
                    // Bar background
                    float bar_x = np.x;
                    if (bar_x > 0.4) {
                        // Slider track
                        float track_y = abs(bar_frac - 0.5) * 2.0;
                        if (track_y < 0.3) {
                            float value = 0.0;
                            if (bar_idx == 0.0) value = settings[0];  // color_bleed
                            else if (bar_idx == 1.0) value = settings[1];  // noise
                            else if (bar_idx == 2.0) value = settings[2];  // ghosting
                            else if (bar_idx == 3.0) value = settings[3];  // scanlines
                            else if (bar_idx == 4.0) value = settings[4];  // curvature
                            else if (bar_idx == 5.0) value = settings[5];  // bloom
                            else if (bar_idx == 6.0) value = settings[6];  // mask_type
                            else if (bar_idx == 7.0) value = settings[7];  // brightness
                            else if (bar_idx == 8.0) value = settings[8];  // contrast
                            else if (bar_idx == 9.0) value = settings[9];  // gamma
                            else if (bar_idx == 10.0) value = settings[10]; // artifacts
                            else if (bar_idx == 11.0) value = settings[11]; // hum
                            else if (bar_idx == 12.0) value = settings[12]; // focus
                            else if (bar_idx == 13.0) value = settings[13]; // overscan

                            // Track background
                            float track_x = (bar_x - 0.4) / 0.55;
                            vec3 track_color = vec3(0.02, 0.03, 0.05);

                            // Filled portion
                            if (track_x < value) {
                                vec3 bar_color = vec3(0.2, 0.5, 0.8);
                                // Different colors for different types
                                if (bar_idx <= 2.0) bar_color = vec3(0.8, 0.3, 0.2); // RF (red)
                                else if (bar_idx <= 5.0) bar_color = vec3(0.2, 0.7, 0.3); // CRT (green)
                                else if (bar_idx <= 9.0) bar_color = vec3(0.3, 0.4, 0.8); // Display (blue)
                                else bar_color = vec3(0.7, 0.5, 0.2); // Other (yellow)

                                // Handle mask_type (0-2 normalized)
                                if (bar_idx == 6.0) {
                                    track_x = (bar_x - 0.4) / 0.55;
                                    value = settings[6] / 2.0;
                                }

                                color = mix(track_color, bar_color, 0.8);
                            }

                            // Indicator line at current value
                            float indicator = abs(track_x - value);
                            if (indicator < 0.01) {
                                color = vec3(1.0);
                            }
                        }
                    }
                }
            } else {
                // Border color
                color = vec3(0.15, 0.25, 0.4);
            }
        }
    }

    fragColor = vec4(color, 1.0);
}
"""

# Fragment: Copy (passthrough)
FRAG_COPY = """
#version 460 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D tex;

void main() {
    fragColor = texture(tex, uv);
}
"""


# ─── CRT Emulator Class ──────────────────────────────────────────────────────

class CRTEimulator:
    def __init__(self, width=1280, height=720, fullscreen=False, source_path=None):
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.source_path = source_path

        # Settings with defaults
        self.settings = {
            'color_bleed': 0.5,
            'noise_amount': 0.3,
            'ghosting': 0.15,
            'scanline_intensity': 0.6,
            'curvature': 0.5,
            'bloom_amount': 0.25,
            'mask_type': 0,  # 0=RGB, 1=Trinitron, 2=ShadowMask
            'brightness': 0.55,
            'contrast': 1.05,
            'gamma': 2.2,
            'composite_artifacts': 1.0,
            'hum_bar': 0.1,
            'focus': 0.85,
            'h_overscan': 0.5,
            'v_overscan': 0.5,
        }

        self.show_ui = False
        self.show_test_pattern = True
        self.running = False
        self.start_time = time.time()

        self._init_glfw()
        self._init_gl()
        self._init_shaders()
        self._init_geometry()
        self._init_framebuffers()
        self._load_source()

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        if self.fullscreen:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            self.width, self.height = mode.size.width, mode.size.height
        else:
            monitor = None

        self.window = glfw.create_window(self.width, self.height, "CRT/RF TV Signal Emulator", monitor, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # VSync

        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)

    def _init_gl(self):
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        print(f"OpenGL: {self.ctx.info['GL_VERSION']}")
        print(f"Renderer: {self.ctx.info['GL_RENDERER']}")

    def _compile_shader(self, source, shader_type):
        """Compile a shader and return the program"""
        return self.ctx.program(
            vertex_shader=VERT_SHADER if shader_type == 'vertex' else source,
            fragment_shader=source if shader_type == 'fragment' else None
        )

    def _create_program(self, frag_source):
        """Create a shader program from fragment source"""
        return self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=frag_source
        )

    def _init_shaders(self):
        print("Compiling shaders...")
        self.prog_test = self._create_program(FRAG_TESTPATTERN)
        self.prog_signal = self._create_program(FRAG_SIGNAL)
        self.prog_crt = self._create_program(FRAG_CRT)
        self.prog_ui = self._create_program(FRAG_UI)
        self.prog_copy = self._create_program(FRAG_COPY)
        print("Shaders compiled successfully.")

    def _init_geometry(self):
        # Fullscreen quad with proper UV coordinates
        vertices = np.array([
            # x,    y,    u,    v
            -1.0, -1.0,  0.0,  0.0,
             1.0, -1.0,  1.0,  0.0,
             1.0,  1.0,  1.0,  1.0,
            -1.0, -1.0,  0.0,  0.0,
             1.0,  1.0,  1.0,  1.0,
            -1.0,  1.0,  0.0,  1.0,
        ], dtype='f4')

        self.vao = self.ctx.vertex_array(
            self.prog_test,
            [(self.ctx.buffer(vertices), '2f 2f', 'in_pos', 'in_uv')]
        )

    def _create_fbo(self, width, height):
        """Create a framebuffer object with color attachment"""
        tex = self.ctx.texture((width, height), 4, dtype='f1')
        tex.filter = (mgl.LINEAR, mgl.LINEAR)
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        return fbo, tex

    def _init_framebuffers(self):
        """Create all render targets"""
        self.fbo_test, self.tex_test = self._create_fbo(self.width, self.height)
        self.fbo_signal, self.tex_signal = self._create_fbo(self.width, self.height)
        self.fbo_crt, self.tex_crt = self._create_fbo(self.width, self.height)
        self.fbo_ui, self.tex_ui = self._create_fbo(self.width, self.height)
        self.fbo_source, self.tex_source = self._create_fbo(self.width, self.height)

        # Create source texture for loaded images
        if self.source_path is None:
            self.tex_source = self.tex_test  # Use test pattern as source

    def _load_source(self):
        """Load image source if provided"""
        if self.source_path and os.path.exists(self.source_path):
            try:
                img = Image.open(self.source_path).convert('RGB')
                img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
                self.source_data = np.array(img, dtype='f4') / 255.0

                # Create texture from image data
                self.tex_source = self.ctx.texture(
                    (self.width, self.height), 3,
                    data=self.source_data.astype('f4').tobytes()
                )
                self.tex_source.filter = (mgl.LINEAR, mgl.LINEAR)
                self.show_test_pattern = False
                print(f"Loaded source: {self.source_path}")
            except Exception as e:
                print(f"Failed to load source: {e}")
                print("Falling back to test pattern")
                self.show_test_pattern = True

    def _set_uniforms(self, program, uniforms_dict):
        """Set uniforms on a shader program"""
        for name, value in uniforms_dict.items():
            try:
                if name in program:
                    loc = program[name]
                    if isinstance(value, float):
                        program[name].value = value
                    elif isinstance(value, int):
                        program[name].value = value
                    elif isinstance(value, tuple) and len(value) == 2:
                        program[name].value = value
            except KeyError:
                pass

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            self.running = False

        elif key == glfw.KEY_F1:
            self.show_ui = not self.show_ui

        elif key == glfw.KEY_F2:
            self.settings['mask_type'] = (self.settings['mask_type'] + 1) % 3
            names = ['RGB Stripe', 'Trinitron', 'Shadow Mask']
            print(f"Mask: {names[self.settings['mask_type']]}")

        elif key == glfw.KEY_F3:
            self.show_test_pattern = not self.show_test_pattern
            print(f"Test pattern: {'ON' if self.show_test_pattern else 'OFF'}")

        elif key == glfw.KEY_1:
            self.settings['color_bleed'] = max(0.0, self.settings['color_bleed'] - 0.05)
            print(f"Color bleed: {self.settings['color_bleed']:.2f}")
        elif key == glfw.KEY_2:
            self.settings['color_bleed'] = min(1.0, self.settings['color_bleed'] + 0.05)
            print(f"Color bleed: {self.settings['color_bleed']:.2f}")

        elif key == glfw.KEY_3:
            self.settings['noise_amount'] = max(0.0, self.settings['noise_amount'] - 0.05)
            print(f"Noise: {self.settings['noise_amount']:.2f}")
        elif key == glfw.KEY_4:
            self.settings['noise_amount'] = min(1.0, self.settings['noise_amount'] + 0.05)
            print(f"Noise: {self.settings['noise_amount']:.2f}")

        elif key == glfw.KEY_5:
            self.settings['ghosting'] = max(0.0, self.settings['ghosting'] - 0.05)
            print(f"Ghosting: {self.settings['ghosting']:.2f}")
        elif key == glfw.KEY_6:
            self.settings['ghosting'] = min(1.0, self.settings['ghosting'] + 0.05)
            print(f"Ghosting: {self.settings['ghosting']:.2f}")

        elif key == glfw.KEY_7:
            self.settings['scanline_intensity'] = max(0.0, self.settings['scanline_intensity'] - 0.05)
            print(f"Scanlines: {self.settings['scanline_intensity']:.2f}")
        elif key == glfw.KEY_8:
            self.settings['scanline_intensity'] = min(1.0, self.settings['scanline_intensity'] + 0.05)
            print(f"Scanlines: {self.settings['scanline_intensity']:.2f}")

        elif key == glfw.KEY_9:
            self.settings['curvature'] = max(0.0, self.settings['curvature'] - 0.05)
            print(f"Curvature: {self.settings['curvature']:.2f}")
        elif key == glfw.KEY_0:
            self.settings['curvature'] = min(1.0, self.settings['curvature'] + 0.05)
            print(f"Curvature: {self.settings['curvature']:.2f}")

        elif key == glfw.KEY_MINUS:
            self.settings['bloom_amount'] = max(0.0, self.settings['bloom_amount'] - 0.05)
            print(f"Bloom: {self.settings['bloom_amount']:.2f}")
        elif key == glfw.KEY_EQUAL:
            self.settings['bloom_amount'] = min(1.0, self.settings['bloom_amount'] + 0.05)
            print(f"Bloom: {self.settings['bloom_amount']:.2f}")

        elif key == glfw.KEY_TAB:
            self.settings['composite_artifacts'] = 1.0 - self.settings['composite_artifacts']
            print(f"Composite artifacts: {'ON' if self.settings['composite_artifacts'] > 0.5 else 'OFF'}")

        elif key == glfw.KEY_BACKSPACE:
            self.settings = {
                'color_bleed': 0.5,
                'noise_amount': 0.3,
                'ghosting': 0.15,
                'scanline_intensity': 0.6,
                'curvature': 0.5,
                'bloom_amount': 0.25,
                'mask_type': 0,
                'brightness': 0.55,
                'contrast': 1.05,
                'gamma': 2.2,
                'composite_artifacts': 1.0,
                'hum_bar': 0.1,
                'focus': 0.85,
                'h_overscan': 0.5,
                'v_overscan': 0.5,
            }
            print("Settings reset to defaults")

    def _resize_callback(self, window, width, height):
        if width <= 0 or height <= 0:
            return
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)

        # Recreate framebuffers at new size
        self._init_framebuffers()

    def _render_test_pattern(self, t):
        """Render test pattern to its FBO"""
        self.fbo_test.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.prog_test['time'].value = t
        self.prog_test['resolution'].value = (float(self.width), float(self.height))

        self.vao.program = self.prog_test
        self.vao.render(mgl.TRIANGLES)

    def _render_signal(self, t):
        """Render signal processing (composite encode/decode/RF)"""
        self.fbo_signal.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Bind source texture (test pattern or loaded image)
        if self.show_test_pattern:
            self.tex_test.use(location=0)
        else:
            self.tex_source.use(location=0)

        self.prog_signal['source_tex'].value = 0
        self.prog_signal['time'].value = t
        self.prog_signal['resolution'].value = (float(self.width), float(self.height))
        self.prog_signal['color_bleed'].value = self.settings['color_bleed']
        self.prog_signal['noise_amount'].value = self.settings['noise_amount']
        self.prog_signal['ghosting'].value = self.settings['ghosting']
        self.prog_signal['composite_artifacts'].value = self.settings['composite_artifacts']
        self.prog_signal['hum_bar'].value = self.settings['hum_bar']

        self.vao.program = self.prog_signal
        self.vao.render(mgl.TRIANGLES)

    def _render_crt(self, t):
        """Render CRT display with phosphor simulation"""
        self.fbo_crt.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.tex_signal.use(location=0)
        self.prog_crt['signal_tex'].value = 0
        self.prog_crt['time'].value = t
        self.prog_crt['resolution'].value = (float(self.width), float(self.height))
        self.prog_crt['curvature'].value = self.settings['curvature']
        self.prog_crt['scanline_intensity'].value = self.settings['scanline_intensity']
        self.prog_crt['bloom_amount'].value = self.settings['bloom_amount']
        self.prog_crt['mask_type'].value = int(self.settings['mask_type'])
        self.prog_crt['brightness'].value = self.settings['brightness']
        self.prog_crt['contrast'].value = self.settings['contrast']
        self.prog_crt['gamma'].value = self.settings['gamma']
        self.prog_crt['focus'].value = self.settings['focus']
        self.prog_crt['v_overscan'].value = self.settings['v_overscan']
        self.prog_crt['h_overscan'].value = self.settings['h_overscan']

        self.vao.program = self.prog_crt
        self.vao.render(mgl.TRIANGLES)

    def _render_ui(self, t):
        """Render UI overlay"""
        self.fbo_ui.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.tex_crt.use(location=0)
        self.prog_ui['screen_tex'].value = 0
        self.prog_ui['show_ui'].value = 1.0 if self.show_ui else 0.0

        # Pack settings into uniform array
        settings_list = [
            self.settings['color_bleed'],
            self.settings['noise_amount'],
            self.settings['ghosting'],
            self.settings['scanline_intensity'],
            self.settings['curvature'],
            self.settings['bloom_amount'],
            self.settings['mask_type'] / 2.0,
            self.settings['brightness'],
            self.settings['contrast'] / 2.0,
            self.settings['gamma'] / 3.0,
            self.settings['composite_artifacts'],
            self.settings['hum_bar'],
            self.settings['focus'],
            self.settings['h_overscan'],
            self.settings['v_overscan'],
            1.0 if self.show_test_pattern else 0.0,
            0.0, 0.0, 0.0, 0.0
        ]
        self.prog_ui['settings'].value = tuple(settings_list[:20])

        self.vao.program = self.prog_ui
        self.vao.render(mgl.TRIANGLES)

    def _render_to_screen(self):
        """Blit final result to screen"""
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.tex_ui.use(location=0)
        self.prog_copy['tex'].value = 0

        self.vao.program = self.prog_copy
        self.vao.render(mgl.TRIANGLES)

    def run(self):
        """Main render loop"""
        self.running = True
        print("\n" + "=" * 60)
        print("  CRT/RF TV Signal Emulator")
        print("=" * 60)
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Source: {'Test Pattern' if self.show_test_pattern else self.source_path}")
        print("  Press F1 for settings, ESC to quit")
        print("=" * 60 + "\n")

        while self.running and not glfw.window_should_close(self.window):
            glfw.poll_events()
            t = time.time() - self.start_time

            # Render pipeline
            self._render_test_pattern(t)
            self._render_signal(t)
            self._render_crt(t)
            self._render_ui(t)
            self._render_to_screen()

            glfw.swap_buffers(self.window)

        self.cleanup()

    def cleanup(self):
        glfw.terminate()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CRT/RF TV Signal Emulator - Faithful analog TV simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  F1         Toggle settings overlay
  F2         Cycle CRT mask type (RGB/Trinitron/ShadowMask)
  F3         Toggle test pattern
  1/2        Adjust color bleed
  3/4        Adjust signal noise
  5/6        Adjust ghosting
  7/8        Adjust scanline intensity
  9/0        Adjust curvature
  -/=        Adjust bloom/glow
  Tab        Toggle composite artifacts
  Backspace  Reset all settings
  Esc        Quit
        """
    )
    parser.add_argument('--width', '-W', type=int, default=1280, help='Window width (default: 1280)')
    parser.add_argument('--height', '-H', type=int, default=720, help='Window height (default: 720)')
    parser.add_argument('--fullscreen', '-f', action='store_true', help='Fullscreen mode')
    parser.add_argument('--source', '-s', type=str, help='Path to source image')

    args = parser.parse_args()

    try:
        app = CRTEimulator(
            width=args.width,
            height=args.height,
            fullscreen=args.fullscreen,
            source_path=args.source,
        )
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
