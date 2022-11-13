import taichi as ti
import numpy as np
from Camera import Camera
from MassSpring import MassSpring
from SPHFluid import SPHFluid
from EulerianFluid import EulerianFluid

WIDTH = 1024
HEIGHT = 720

if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    camera = Camera(45 * np.pi / 180, WIDTH / HEIGHT, 0.1, 100.0)

    # gui = ti.GUI('TiEngine', res=(WIDTH, HEIGHT), background_color=0xdddddd)
    

    configs = {
        "title": "WCSPH",
        "model": EulerianFluid,
        "type": 1,
        "dt": 1e-2,
        "t": 8
    }
    substeps = int(1 / 20 // configs["dt"])
    model = configs["model"](configs["type"], configs["dt"])
    gui = ti.GUI('TiEngine', res=(model.grid_shape[0], model.grid_shape[1]), background_color=0xdddddd)

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

        # gui.circles(camera.T(model.x.to_numpy()), radius=5, color=0xffaa77)
        gui.set_image(model._img)
        gui.text(content=configs["title"], pos=(0, 1), font_size=25, color=0x000000)
        gui.show()
        # gui.show(f"frame/{frame:04d}.png")
        frame += 1