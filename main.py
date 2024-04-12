import taichi as ti
import numpy as np
from Camera import Camera
from cases.MassSpring import MassSpringSolver
from cases.FEMSolid import FEMSolidSolver
from cases.SPHFluid import SPHFluidSolver
from cases.EulerianFluid import EulerianFluidSolver
from cases.MPMFluid import MPMFluidSolver
from cases.PDSolid import PDSolidSolver
from cases.CPDIPC import CPDIPCSolver
from cases.LBM import LBMSolver

WIDTH = 1024
HEIGHT = 720

if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    camera = Camera(45 * np.pi / 180, WIDTH / HEIGHT, 0.1, 100.0)

    configs = {
        "solver": LBMSolver,
        "type": 2,
        "dt": 1e-2,
        "t": 10,
        "showGui": True,
    }
    substeps = int(1 / 20 // configs["dt"])
    solver = configs["solver"](configs["type"], configs["dt"])

    t = 0
    frame = 0
    if configs["showGui"]:
        gui = ti.GUI('TiEngine', res=(WIDTH, HEIGHT), background_color=0xdddddd)

    while (configs["showGui"] and gui.running) or (not configs["showGui"]):
        if t > configs["t"]:
            if configs["showGui"]:
                frame = 0
                t = 0
                solver.reset()
            else:
                break

        for i in range(substeps):
            solver.substep()
            t += configs["dt"]

        if configs["showGui"]:
            gui.circles(camera.T(solver.x.to_numpy()), radius=5, color=0xffaa77)
            gui.text(content=configs["solver"].name + "-" + str(frame), pos=(0, 1), font_size=25, color=0x000000)
            # gui.show(f"frame/{frame:04d}.png")
            gui.show()
        else:
            verts = solver.x.to_numpy()
            writer = ti.tools.PLYWriter(num_vertices=verts.shape[0])
            writer.add_vertex_pos(verts[:, 0], verts[:, 1], verts[:, 2])
            writer.export_frame_ascii(frame, "frame/frame.ply")
            print("frame " + str(frame))

        frame += 1