import taichi as ti
import numpy as np
from Camera import Camera
from MassSpring import MassSpring
from SPHFluid import SPHFluid

WIDTH = 1024
HEIGHT = 720

# 3d pos -> ndc
def T(pos):
    M = camera.projection_matrix @ camera.view_matrix
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ndc_x = M[0, 0] * x + M[0, 1] * y + M[0, 2] * z + M[0, 3]
    ndc_y = M[1, 0] * x + M[1, 1] * y + M[1, 2] * z + M[1, 3]
    ndc_w = M[3, 0] * x + M[3, 1] * y + M[3, 2] * z + M[3, 3]
    ndc_x /= ndc_w
    ndc_y /= ndc_w
    ndc_x += 1
    ndc_x /= 2
    ndc_y += 1
    ndc_y /= 2
    return np.array([ndc_x, ndc_y]).swapaxes(0, 1)

if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    camera = Camera(45 * np.pi / 180, WIDTH / HEIGHT, 0.1, 100.0)

    gui = ti.GUI('TiEngine', res=(WIDTH, HEIGHT), background_color=0xdddddd)

    configs = {
        "title": "MassSpring",
        "model": MassSpring,
        "type": 2,
        "dt": 1e-3,
        "t": 8
    }
    substeps = int(1 / 40 // configs["dt"])
    model = configs["model"](configs["type"], configs["dt"])

    t = 0
    frame = 0

    while gui.running:
        if t > configs["t"]:
            frame = 0
            t = 0
            model.init_field()

        for i in range(substeps):
            model.substep()
            t += configs["dt"]

        gui.circles(T(model.x.to_numpy()), radius=5, color=0xffaa77)
        gui.text(content=configs["title"], pos=(0, 1), font_size=25, color=0x000000)
        gui.show(f"frame/{frame:04d}.png")
        frame += 1