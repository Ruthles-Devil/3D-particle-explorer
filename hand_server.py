import sys
import numpy as np
import cv2
import mediapipe as mp

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class ParticleSystem:
    def __init__(self, particle_count=10000):
        self.particle_count = particle_count
        self.base_positions = np.random.uniform(-5, 5, (self.particle_count, 3)).astype(np.float32)
        self.positions = self.base_positions.copy()
        self.global_scale = 1.0
        self.global_size = 0.1
        self.rot_azimuth = 0.0
        self.rot_elevation = 0.0
        self.zoom_offset = 0.0

        # Colors (teal glowing spheres)
        self.colors = np.ones((self.particle_count, 4), dtype=np.float32)
        self.colors[:, 0] = 0.2  # R
        self.colors[:, 1] = 0.8  # G
        self.colors[:, 2] = 1.0  # B
        self.colors[:, 3] = 0.8  # A

        self.templates = {
            "particles": self.init_particles,
            "hearts": self.init_hearts,
            "flowers": self.init_flowers,
            "saturn": self.init_saturn,
            "fireworks": self.init_fireworks,
        }
        self.current_template = "particles"
        self.init_particles()

    def init_particles(self):
        self.base_positions = np.random.uniform(-5, 5, (self.particle_count, 3)).astype(np.float32)

    def init_hearts(self):
        t = np.linspace(0, 2 * np.pi, self.particle_count)
        r = 3 + np.sin(t * 3) * 1.5
        x = np.cos(t) * r * (np.sin(t) ** 3)
        y = np.sin(t) * r * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16
        z = np.sin(np.arange(self.particle_count) * 0.01) * 2
        self.base_positions = np.vstack([x, y, z]).T.astype(np.float32)

    def init_flowers(self):
        t = np.linspace(0, 8 * np.pi, self.particle_count)
        r = 2 + np.sin(t * 5) * 0.8
        x = np.cos(t) * r
        y = np.sin(t) * r * 0.6
        z = np.sin(np.arange(self.particle_count) * 0.02) * 1.5
        self.base_positions = np.vstack([x, y, z]).T.astype(np.float32)

    def init_saturn(self):
        i = np.arange(self.particle_count)
        phi = np.arccos(2 * i / self.particle_count - 1)
        theta = np.sqrt(self.particle_count * np.pi) * phi
        x = 3 * np.cos(theta) * np.sin(phi)
        y = 0.5 + np.sin(i * 0.1) * 0.3
        z = 3 * np.sin(theta) * np.sin(phi)
        self.base_positions = np.vstack([x, y, z]).T.astype(np.float32)

    def init_fireworks(self):
        i = np.arange(self.particle_count)
        angle = (i / self.particle_count) * 2 * np.pi
        dist = np.sqrt(i / self.particle_count) * 8
        x = np.cos(angle) * dist * 0.5
        y = dist * 0.3 + np.sin(i * 0.05) * 2
        z = np.sin(angle) * dist * 0.5
        self.base_positions = np.vstack([x, y, z]).T.astype(np.float32)

    def set_template(self, name: str):
        if name in self.templates:
            self.current_template = name
            self.templates[name]()

    def update_scaled_positions(self):
        self.positions = (self.base_positions * self.global_scale).astype(np.float32)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D model explorer")
        self.resize(1200, 800)

        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('k')
        self.view.opts['distance'] = 20
        self.view.setCameraPosition(azimuth=30, elevation=20)
        layout.addWidget(self.view, 3)

        # Controls
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        layout.addWidget(control_panel, 1)

        title = QtWidgets.QLabel("🎮 3D MODEL EXPLORER Controls")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        control_layout.addWidget(title)

        control_panel.setStyleSheet("background-color: #111; color: #ccc;")

        control_layout.addWidget(QtWidgets.QLabel("Template:"))
        self.template_combo = QtWidgets.QComboBox()
        self.template_combo.addItems(["particles", "hearts", "flowers", "saturn", "fireworks"])
        self.template_combo.setCurrentText("particles")
        control_layout.addWidget(self.template_combo)

        self.status_label = QtWidgets.QLabel("👋 1 hand: rotate/pinch\n2 hands: zoom (stays!)")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch(1)

        # Particle system
        self.ps = ParticleSystem(particle_count=10000)

        # FIXED: Proper glowing spheres (no invalid attributes)
        pos = self.ps.positions
        self.scatter = gl.GLScatterPlotItem(
            pos=pos,
            color=self.ps.colors,
            size=0.1,
            pxMode=False
        )
        self.scatter.setGLOptions('translucent')  # smooth alpha blending
        self.scatter.setGLOptions('additive')     # glowing effect
        self.view.addItem(self.scatter)

        # Grid reference
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.view.addItem(grid)

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(16)

        self.template_combo.currentTextChanged.connect(self.change_template)

    def change_template(self, name):
        self.ps.set_template(name)
        self.ps.update_scaled_positions()
        self.scatter.setData(pos=self.ps.positions, color=self.ps.colors, size=0.1 * self.ps.global_size)
        print(f"✅ Switched to {name}")

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            total_tension = 0.0
            hand_count = 0
            cx_sum, cy_sum = 0.0, 0.0
            hand_centers = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb = hand_landmarks.landmark[4]
                    index = hand_landmarks.landmark[8]
                    distance = np.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)
                    total_tension += float(distance)
                    hand_count += 1
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    cx = hand_landmarks.landmark[0].x
                    cy = hand_landmarks.landmark[0].y
                    cx_sum += cx
                    cy_sum += cy
                    hand_centers.append((cx, cy))

            if hand_count > 0:
                self.ps.global_scale = max(0.1, min(3.0, 1.0 + (total_tension * 10 - 0.15)))
                self.ps.global_size = max(0.1, min(5.0, 1.0 + hand_count * 0.8))

                cx_avg = cx_sum / hand_count
                cy_avg = cy_sum / hand_count
                dx = cx_avg - 0.5
                dy = cy_avg - 0.5
                self.ps.rot_azimuth = dx * 60.0
                self.ps.rot_elevation = -dy * 40.0

                # PERSISTENT 2-hand zoom
                if len(hand_centers) == 2:
                    (x1, y1), (x2, y2) = hand_centers
                    dist_hands = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    target_zoom = (0.5 - dist_hands) * 40.0
                    self.ps.zoom_offset += (target_zoom - self.ps.zoom_offset) * 0.1

                self.status_label.setText(f"✅ Hands: {hand_count}  Zoom: {self.ps.zoom_offset:.1f}")
                self.status_label.setStyleSheet("color: lime;")
            else:
                self.ps.global_scale += (1.0 - self.ps.global_scale) * 0.05
                self.ps.global_size += (1.0 - self.ps.global_size) * 0.05
                self.ps.rot_azimuth *= 0.9
                self.ps.rot_elevation *= 0.9
                # ZOOM PERSISTS - no decay!
                self.status_label.setText("👋 Zoom stays!")
                self.status_label.setStyleSheet("color: #ccc;")

            cv2.imshow("Hand Tracker (q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close()

        # Update 3D
        self.ps.update_scaled_positions()
        self.view.setCameraPosition(
            azimuth=30 + self.ps.rot_azimuth,
            elevation=20 + self.ps.rot_elevation,
            distance=20 + self.ps.zoom_offset  # PERSISTS!
        )

        t = pg.ptime.time()
        colors = self.ps.colors.copy()
        colors[:, 3] = 0.5 + 0.3 * np.sin(t + np.arange(self.ps.particle_count) * 0.01)
        colors = np.clip(colors, 0.0, 1.0)

        self.scatter.setData(pos=self.ps.positions, color=colors, size=0.5 * self.ps.global_size)

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
