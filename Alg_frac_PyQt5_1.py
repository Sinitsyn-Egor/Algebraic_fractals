import copy
import re
import sys
import time
from typing import NamedTuple

import numba as nb
import numpy as np
import PyQt5.QtWidgets as Pwgt
from vispy import app, scene, visuals

app.use_app("pyqt5")


class GridParms(NamedTuple):
    x_min: np.float64
    x_max: np.float64
    y_min: np.float64
    y_max: np.float64
    num_points_x: np.int64 = 1024
    num_points_y: np.int64 = 1024

    @classmethod
    def create(
        self: "GridParms",
        x_min: np.float64,
        x_max: np.float64,
        y_min: np.float64,
        y_max: np.float64,
        num_points_x: np.int64 = 1024,
        num_points_y: np.int64 = 1024,
    ) -> "GridParms":
        return self(
            x_min=np.float64(x_min),
            x_max=np.float64(x_max),
            y_min=np.float64(y_min),
            y_max=np.float64(y_max),
            num_points_x=np.int64(num_points_x),
            num_points_y=np.int64(num_points_y),
        )

    @nb.njit
    def check_grid(self: "GridParms", param: np.float64 = 0.1) -> bool:
        x_len: int = len(
            np.unique(np.linspace(self.x_min, self.x_max, self.num_points_x))
        )
        y_len: int = len(
            np.unique(np.linspace(self.y_min, self.y_max, self.num_points_y))
        )
        x_check: bool = abs(self.num_points_x - x_len) / self.num_points_x > param
        y_check: bool = abs(self.num_points_y - y_len) / self.num_points_y > param
        return x_check or y_check

    @property
    def size_x(self):
        return abs(self.x_min - self.x_max)

    @property
    def size_y(self):
        return abs(self.y_min - self.y_max)


class Draw_imshow:
    def __init__(self: "Draw_imshow", func_str):
        self.func_str: str = self.beaut_str_func(func_str)
        self.func_njit = nb.njit(eval(f"lambda Z, Z0:{self.func_str}"))
        self.usual_sizes: GridParms = GridParms.create(-2.5, 2.5, -2.5, 2.5)

    @staticmethod
    def beaut_str_func(instr: str):
        replacement_dict = {
            "\\": "",
            " ": "",
            "IM": "np.imag",
            "RE": "np.real",
            "EXP": "np.exp",
        }
        instr = re.sub("abs Re Im exp Z Z0 /+/*", "", re.escape(instr)).upper()
        for old, new in replacement_dict.items():
            instr = re.sub(re.escape(old), new, instr)
        return instr

    @property
    def ret_usual_sizes(self: "Draw_imshow") -> GridParms:
        return self.usual_sizes

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def fast_many_func_draw_wrapped(
        func, sizes: GridParms, max_iter: np.int64 = 100
    ) -> np.ndarray:
        x = np.linspace(sizes.x_min, sizes.x_max, sizes.num_points_x)
        y = np.linspace(sizes.y_min, sizes.y_max, sizes.num_points_y)
        image: np.ndarray = np.zeros((len(y), len(x)), dtype=np.int64)
        for i in nb.prange(len(x)):
            for j in nb.prange(len(y)):
                Z0 = np.complex128(x[i] + 1j * y[j])
                Z = np.complex128(Z0)
                for k in range(max_iter):
                    Z = np.complex128(func(Z, Z0))
                    if np.abs(Z) > 2:
                        image[j, i] = k
                        break
                else:
                    image[j, i] = max_iter
        return image / max_iter

    def fast_many_frac_draw(self: "Draw_imshow", sizes: GridParms):
        return self.fast_many_func_draw_wrapped(self.func_njit, sizes)


class MainWindow(Pwgt.QMainWindow):
    def __init__(self):
        super().__init__()
        central_widget = Pwgt.QWidget()
        self.setCentralWidget(central_widget)
        layout = Pwgt.QVBoxLayout(central_widget)

        up_panel = Pwgt.QHBoxLayout()
        button_save = Pwgt.QPushButton("Save")
        button_save.clicked.connect(self.button_save)
        up_panel.addWidget(button_save)
        text = Pwgt.QLabel("F(Z, Z0) = ")
        up_panel.addWidget(text)
        self.input_text = Pwgt.QLineEdit()
        self.str_inp: str = "Z**2 + Z0"
        self.input_text.setText(self.str_inp)
        up_panel.addWidget(self.input_text)
        button_draw = Pwgt.QPushButton("Draw")
        button_draw.clicked.connect(self.update_frack_all)
        up_panel.addWidget(button_draw)
        layout.addLayout(up_panel, stretch=0)

        self.canvas: scene.canvas.SceneCanvas = scene.SceneCanvas(
            keys="interactive", show=False, parent=central_widget
        )
        self.view = self.canvas.central_widget.add_view()
        layout.addWidget(self.canvas.native)
        self.view.camera = scene.PanZoomCamera()
        self.view.camera.aspect = None

        yaxis = scene.AxisWidget(orientation="right")
        self.view.add_widget(yaxis)
        yaxis.link_view(self.view)

        xaxis = scene.AxisWidget(orientation="top")
        self.view.add_widget(xaxis)
        xaxis.link_view(self.view)

        self.view.camera.set_range()

        self.drawer: Draw_imshow = Draw_imshow(self.input_text.text())
        self.sizes_regular: GridParms = self.drawer.ret_usual_sizes
        self.sizes: GridParms = copy.deepcopy(self.sizes_regular)
        self.image_data = self.drawer.fast_many_frac_draw(self.sizes).astype(np.float32)
        self.image = scene.visuals.Image(
            self.image_data, parent=self.view.scene, method="auto"
        )
        self.last_update = time.time()

        @self.view.scene.transform.changed.connect
        def on_transform_change(ev):
            if time.time() - self.last_update < 0.03:  # 30 fps
                return
            self.last_update = time.time()
            self.sizes = GridParms.create(
                self.view.camera.rect.left,
                self.view.camera.rect.right,
                self.view.camera.rect.bottom,
                self.view.camera.rect.top,
            )
            self.update_frack()

        self.view.camera.events.transform_change.connect(on_transform_change)
        self.update_frack()

    def button_save(self):
        print("o")
        print(self.sizes)

    def update_frack(self):
        self.image.parent = None
        self.image_data = self.drawer.fast_many_frac_draw(self.sizes).astype(np.float32)
        self.image = scene.visuals.Image(self.image_data, parent=self.view.scene)
        self.image.transform = visuals.transforms.STTransform(
            scale=(
                self.sizes.size_x / self.image_data.shape[0],
                self.sizes.size_y / self.image_data.shape[1],
            ),
            translate=(self.sizes.x_min, self.sizes.y_min),
        )
        self.view.update()

    def update_frack_all(self):
        if self.str_inp == self.input_text.text():
            self.sizes = GridParms.create(
                self.view.camera.rect.left,
                self.view.camera.rect.right,
                self.view.camera.rect.bottom,
                self.view.camera.rect.top,
                1000,
                1000,
            )
            self.update_frack()
        else:
            try:
                self.drawer = Draw_imshow(self.input_text.text())
                self.str_inp = self.input_text.text()
                self.sizes_regular = self.drawer.ret_usual_sizes
                self.sizes = GridParms.create(
                    self.view.camera.rect.left,
                    self.view.camera.rect.right,
                    self.view.camera.rect.bottom,
                    self.view.camera.rect.top,
                )
                self.update_frack()
            except Exception as e:
                print(f"Ошибка: {e}")


if __name__ == "__main__":
    app = Pwgt.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()
