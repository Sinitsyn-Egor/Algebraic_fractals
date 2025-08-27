import copy
from typing import NamedTuple

import matplotlib.backend_bases as mbabs
import matplotlib.pyplot as plt
import matplotlib.widgets as mwgt
import numba as nb
import numpy as np


class GridParms(NamedTuple):
    x_min: np.float64
    x_max: np.float64
    y_min: np.float64
    y_max: np.float64
    num_points_x: np.int64 = 400
    num_points_y: np.int64 = 400

    @classmethod
    def create(
        self: "GridParms",
        x_min: np.float64,
        x_max: np.float64,
        y_min: np.float64,
        y_max: np.float64,
        num_points_x: np.int64 = 400,
        num_points_y: np.int64 = 400,
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
    def generate_x_lin(self: "GridParms"):
        return np.linspace(self.x_min, self.x_max, self.num_points_x)

    @nb.njit
    def generate_y_lin(self: "GridParms"):
        return np.linspace(self.x_min, self.x_max, self.num_points_x)

    @nb.njit
    def check_grid(self: "GridParms", param: np.float64 = 0.3) -> bool:
        x_len = len(np.unique(np.linspace(self.x_min, self.x_max, self.num_points_x)))
        y_len = len(np.unique(np.linspace(self.y_min, self.y_max, self.num_points_y)))
        x_check: bool = abs(self.num_points_x - x_len) / self.num_points_x > param
        y_check: bool = abs(self.num_points_y - y_len) / self.num_points_y > param
        return x_check or y_check

    def ret_extent(
        self: "GridParms",
    ) -> tuple[np.float64, np.float64, np.float64, np.float64]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)


class Draw_imshow:
    def __init__(self: "Draw_imshow", func_str):
        self.func_str: str = self.beaut_str_func(func_str)
        self.func_njit = nb.njit(eval(f"lambda Z, Z0:{self.func_str}"))
        self.usual_sizes: GridParms = GridParms.create(-2.5, 2.5, -2.5, 2.5)

    @staticmethod
    def beaut_str_func(instr: str):
        return instr.replace(" ", "").replace("^", "**")

    @property
    def ret_usual_sizes(self: "Draw_imshow") -> GridParms:
        return self.usual_sizes

    def many_poly_frack_func(
        self: "Draw_imshow", sizes: GridParms, max_iter: np.int64 = 100
    ) -> np.ndarray[np.ndarray[np.float64]]:
        x = np.linspace(sizes.x_min, sizes.x_max, sizes.num_points_x)
        y = np.linspace(sizes.y_min, sizes.y_max, sizes.num_points_y)
        X, Y = np.meshgrid(x, y)
        Z0: np.ndarray = X + 1j * Y
        Z: np.ndarray = Z0
        image: np.ndarray = np.zeros(Z0.shape, dtype=int)
        for k in range(max_iter):
            Z = self.func(Z, Z0)
            mask = (np.abs(Z) > 4) & (image == 0)
            image[mask] = k
            Z[mask] = np.nan
        mask = (np.abs(Z) < 4) & (image == 0)
        image[mask] = k
        image[image == 0] = 1
        return np.log(image)

    @staticmethod
    @nb.jit(nopython=True, parallel=True, cache=True)
    def fast_many_func_draw_wrapped(
        func, sizes: GridParms, max_iter: np.int64 = 200
    ) -> np.ndarray:
        x = np.linspace(sizes.x_min, sizes.x_max, sizes.num_points_x)
        y = np.linspace(sizes.y_min, sizes.y_max, sizes.num_points_y)
        image: np.ndarray = np.zeros((len(y), len(x)), dtype=np.int64)
        for i in nb.prange(len(x)):
            for j in nb.prange(len(y)):
                Z0 = x[i] + 1j * y[j]
                Z = np.complex128(Z0)
                for k in range(max_iter):
                    Z = np.complex128(func(Z, Z0))
                    if np.abs(Z) > 2:
                        image[j, i] = k
                        break
                else:
                    image[j, i] = max_iter
        return image / max_iter

    def fast_many_func_draw(self: "Draw_imshow", sizes: GridParms):
        return self.fast_many_func_draw_wrapped(self.func_njit, sizes)


class Usual_ster:
    def __init__(self: "Usual_ster"):
        self.str_func = "Z**2 + Z0"
        self.drawer: Draw_imshow = Draw_imshow(self.str_func)
        self.sizes_regular: GridParms = self.drawer.ret_usual_sizes
        self.sizes: GridParms = copy.deepcopy(self.sizes_regular)

        self.fig = plt.figure()
        self.fig.subplots_adjust()
        self.ax = self.fig.add_subplot()

        axbox = plt.axes([0.224, 0.9, 0.4, 0.05])
        self.text_box = mwgt.TextBox(axbox, "F(Z, Z0) = ", initial=self.str_func)

        ax_add = plt.axes([0.624, 0.9, 0.176, 0.05])
        add_btn = mwgt.Button(ax_add, "Draw")

        self.text_box.on_submit(self.add_function)
        add_btn.on_clicked(self.add_function)

        self.rs = mwgt.RectangleSelector(
            self.ax,
            self.onselect,
            useblit=True,
            button=[1, 3],
            minspanx=10,
            minspany=10,
            spancoords="pixels",
        )

        self.redraw_all()
        plt.show()

    def onselect(
        self: "Usual_ster", eclick: mbabs.MouseEvent, erelease: mbabs.MouseEvent
    ):
        if eclick.button == 1:
            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])
            self.sizes = GridParms.create(x_min, x_max, y_min, y_max)
            self.redraw_all()
        elif eclick.button == 3:
            dx_min, dx_max = sorted([eclick.xdata, erelease.xdata])
            dy_min, dy_max = sorted([eclick.ydata, erelease.ydata])
            x_min = self.sizes.x_min - abs(self.sizes.x_min - dx_min)
            x_max = self.sizes.x_max + abs(self.sizes.x_max - dx_max)
            y_min = self.sizes.y_min - abs(self.sizes.y_min - dy_min)
            y_max = self.sizes.y_max + abs(self.sizes.y_min - dy_max)
            self.sizes = GridParms.create(x_min, x_max, y_min, y_max)
            self.redraw_all()

    def add_function(self: "Usual_ster", event: mbabs.MouseEvent = None):
        try:
            self.drawer = Draw_imshow(self.text_box.text)
            self.sizes_regular = self.drawer.ret_usual_sizes
            self.sizes = copy.deepcopy(self.sizes_regular)
            self.redraw_all()
        except Exception as e:
            print(f"Ошибка: {e}")

    def redraw_all(self: "Usual_ster") -> None:
        self.ax.cla()
        if self.sizes.check_grid():
            print(self.sizes)
        self.ax.set_xlim(self.sizes.x_min, self.sizes.x_max)
        self.ax.set_ylim(self.sizes.y_min, self.sizes.y_max)
        image_f = self.drawer.fast_many_func_draw(self.sizes)
        self.ax.imshow(
            image_f,
            extent=self.sizes.ret_extent(),
            vmin=0,
            vmax=1,
            cmap="plasma",
            origin="lower",
        )
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    Usual_ster()
