
## Usage Instructions
Run `3D Particle explorer.py` after installing dependencies: `pip install opencv-python mediapipe PyQt5 pyqtgraph numpy`. Ensure a webcam is connected. The app opens a 1200x800 window with a 3D view (left) and controls (right).

- Position one hand in frame: thumb-index pinch scales/rotates particles based on hand center and finger distance.
- Use two hands: distance between centers controls persistent zoom (closer hands = zoom in).
- Press 'q' in OpenCV window to quit; templates auto-decay to default when hands removed.

## Key Features
Teal glowing spheres (10k particles) with additive blending. Five templates switch via dropdown: particles (random cloud), hearts, flowers, saturn rings, fireworks bursts. Real-time pulsing animation via sine-based alpha. Hand landmarks drawn on webcam feed (640x480). 

## Customization Guide
**ParticleSystem class** (lines ~10-120): Edit `base_positions` generation in `init_*` methods for new shapes. Adjust `colors` array for hues (RGBA, 0-1 floats). Tweak `global_scale/size/zoom_offset` bounds in `update_loop` for gesture sensitivity.

**Gesture tuning** (update_loop ~250+): Modify `total_tension * 10 - 0.15` for pinch scaling, `dx * 60.0` for rotation speed, `dist_hands * 40.0` for zoom factor. Change `max_num_hands=2` or confidences for detection.

| Parameter | Location | Effect |
|-----------|----------|--------|
| particle_count | __init__ | Total spheres (default 10000)  |
| global_scale | update_loop | Pinch-based size (0.1-3.0) |
| rot_azimuth/elevation | update_loop | Hand position rotation (±60°/±40°) |
| zoom_offset | 2-hand block | Persistent camera distance |

**Adding templates**: Define `init_newshape`, add to `self.templates` dict and combo box.

