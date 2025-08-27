import matplotlib.backend_bases as mbabs
import matplotlib.pyplot as plt
import matplotlib.widgets as mwgt
import numpy as np


class Draw_imshow:
    def __init__(self, func_str: str = "Z**2+Z0"):
        self.func_str: str = func_str
        self.func = self.func_str_wrapper(self.func_str)
        self.usual_sizes: dict[str, float] = {
            "x_min": -2.5,
            "x_max": 2.5,
            "y_min": -2.5,
            "y_max": 2.5,
        }

    @staticmethod
    def func_str_wrapper(instr: str):
        return eval(f"lambda Z, Z0:{instr}")

    def many_poly_frack_func(
        self,
        sizes: dict[str, np.float64],
        width: int = 400,
        height: int = 400,
        max_iter: int = 100,
    ) -> np.ndarray[np.ndarray[np.float64]]:
        x: np.ndarray = np.linspace(sizes["x_min"], sizes["x_max"], width)
        y: np.ndarray = np.linspace(sizes["y_min"], sizes["y_max"], height)
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

    @property
    def ret_usual_sizes(self) -> dict[str, np.float64]:
        return self.usual_sizes


class Usual_ster:
    def __init__(self):
        self.drawer: Draw_imshow = Draw_imshow()
        self.sizes_regular: dict[str, np.float64] = self.drawer.ret_usual_sizes
        self.sizes: dict[str, np.float64] = self.sizes_regular.copy()

        self.fig = plt.figure()
        self.fig.subplots_adjust()
        self.ax = self.fig.add_subplot()

        axbox = plt.axes([0.224, 0.9, 0.4, 0.05])
        self.text_box = mwgt.TextBox(axbox, "F(Z) = ", initial="Z**2 + Z0")

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

    def onselect(self, eclick: mbabs.MouseEvent, erelease: mbabs.MouseEvent) -> None:
        if eclick.button == 1:
            self.sizes["x_min"], self.sizes["x_max"] = sorted(
                [eclick.xdata, erelease.xdata]
            )
            self.sizes["y_min"], self.sizes["y_max"] = sorted(
                [eclick.ydata, erelease.ydata]
            )
        elif eclick.button == 3:
            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])
            self.sizes["x_min"] = self.sizes["x_min"] - abs(self.sizes["x_min"] - x_min)
            self.sizes["x_max"] = self.sizes["x_max"] + abs(self.sizes["x_max"] - x_max)
            self.sizes["y_min"] = self.sizes["y_min"] - abs(self.sizes["y_min"] - y_min)
            self.sizes["y_max"] = self.sizes["y_max"] + abs(self.sizes["y_min"] - y_max)
        self.redraw_all()

    def add_function(self, event: mbabs.MouseEvent = None) -> None:
        try:
            self.drawer = Draw_imshow(self.text_box.text)
            self.sizes_regular = self.drawer.ret_usual_sizes
            self.sizes = self.sizes_regular.copy()
            self.redraw_all()
        except Exception as e:
            print(f"Ошибка: {e}")

    def redraw_all(self) -> None:
        self.ax.cla()
        self.ax.set_xlim(self.sizes["x_min"], self.sizes["x_max"])
        self.ax.set_ylim(self.sizes["y_min"], self.sizes["y_max"])
        image_f = self.drawer.many_poly_frack_func(self.sizes)
        self.ax.imshow(
            image_f,
            extent=(
                self.sizes["x_min"],
                self.sizes["x_max"],
                self.sizes["y_min"],
                self.sizes["y_max"],
            ),
            cmap="plasma",
            origin="lower",
        )
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    Usual_ster()
