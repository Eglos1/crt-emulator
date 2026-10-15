# CRT / RF TV Signal Emulator

Faithfully emulates a CRT television **and the whole analog signal chain** that
feeds it — not just a scanline post-filter. A source image/video is encoded to
composite (NTSC/PAL YIQ/YPbPr), pushed through an RF channel (noise, multipath,
ghosting), demodulated with proper bandwidth limiting, decoded back with the
real color artifacts, and finally drawn by a simulated CRT electron beam
(phosphor mask, curvature, scanlines, bloom).

## Files
| file | purpose |
|------|---------|
| `crt_emulator.py` | interactive emulator / viewer (ModernGL + NumPy) |
| `crt_physics.py` | the analog signal-chain physics model |

## Run
```sh
pip install moderngl numpy
# optional, for video sources:
pip install opencv-python

python3 crt_emulator.py                       # built-in test pattern
python3 crt_emulator.py --source image.png
python3 crt_emulator.py --source video.mp4      # needs opencv
python3 crt_emulator.py --fullscreen
```

## Controls
`F1` overlay · `F2` mask type · `F3` test pattern · `F4` source ·
`1/2` color bleed · `3/4` noise · `5/6` ghosting · `7/8` scanlines ·
`9/0` curvature · `-`/`=` bloom · `Tab` composite artifacts · `Backspace` reset · `Esc` quit.

## License
MIT
