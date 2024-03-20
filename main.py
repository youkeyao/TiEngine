import taichi as ti
import numpy as np
from Camera import Camera
from cases.MassSpring import MassSpringSolver
from cases.FEMSolid import FEMSolidSolver
from cases.SPHFluid import SPHFluidSolver
from cases.EulerianFluid import EulerianFluidSolver
from cases.MPMFluid import MPMFluidSolver
from cases.PDSolid import PDSolidSolver

WIDTH = 1024
HEIGHT = 720

if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    camera = Camera(45 * np.pi / 180, WIDTH / HEIGHT, 0.1, 100.0)

    gui = ti.GUI('TiEngine', res=(WIDTH, HEIGHT), background_color=0xdddddd)

    configs = {
        "solver": PDSolidSolver,
        "type": 2,
        "dt": 1e-2,
        "t": 10
    }
    substeps = int(1 / 20 // configs["dt"])
    solver = configs["solver"](configs["type"], configs["dt"])

    t = 0
    frame = 0

    while gui.running:
        if t > configs["t"]:
            frame = 0
            t = 0
            solver.reset()

        for i in range(substeps):
            solver.substep()
            t += configs["dt"]

        gui.circles(camera.T(solver.x.to_numpy()), radius=5, color=0xffaa77)
        gui.text(content=configs["solver"].name, pos=(0, 1), font_size=25, color=0x000000)
        gui.show()
        # gui.show(f"frame/{frame:04d}.png")
        frame += 1