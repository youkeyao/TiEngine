import taichi as ti
import numpy as np
from Camera import Camera
from MassSpring import MassSpring
from FEMSolid import FEMSolid
from SPHFluid import SPHFluid
from EulerianFluid import EulerianFluid
from MPMFluid import MPMFluid

WIDTH = 1024
HEIGHT = 720

if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    camera = Camera(45 * np.pi / 180, WIDTH / HEIGHT, 0.1, 100.0)

    gui = ti.GUI('TiEngine', res=(WIDTH, HEIGHT), background_color=0xdddddd)

    configs = {
        "title": "FEMSolid",
        "model": FEMSolid,
        "type": 2,
        "dt": 1e-3,
        "t": 10
    }
    substeps = int(1 / 20 // configs["dt"])
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

        gui.circles(camera.T(model.x.to_numpy()), radius=5, color=0xffaa77)
        gui.text(content=configs["title"], pos=(0, 1), font_size=25, color=0x000000)
        gui.show()
        # gui.show(f"frame/{frame:04d}.png")
        frame += 1