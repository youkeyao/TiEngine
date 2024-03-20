import taichi as ti
import numpy as np
from utils import LinearSolver, Scene

@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.data_oriented
class FEMSolidSolver:
    name = "FEMSolid"
    def __init__(self, type, dt):
        self.type = type # 1: explicit, 2: Jacobi, 3: Conjugate Gradient, 4: MGPCG
        self.dt = dt

        self.scene = Scene()
        self.scene.add_cube([0.5, 0.5, 0.5], [-1, 1, -1])

        self.n = self.scene.n_verts
        self.n_elements = self.scene.n_elements
        E, nu = 1e5, 0.0
        self.mu, self.la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=self.n_elements)
        self.m = ti.field(dtype=ti.f32, shape=self.n)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.fe = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.f = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_elements)
        self.F_W = ti.field(dtype=ti.f32, shape=self.n_elements)

        if type == 2:
            self.solver = LinearSolver(LinearSolver.jacobi, self.v, self.n)
        elif type == 3:
            self.solver = LinearSolver(LinearSolver.cg, self.v, self.n)
        elif type == 4:
            self.solver = LinearSolver(LinearSolver.mgpcg, self.v, self.n)
        else:
            print("Invalid type!")

        self.reset()

    def reset(self):
        self.x.from_numpy(self.scene.x.astype(np.float32))
        self.elements.from_numpy(self.scene.elements.astype(np.int32))
        self.fe.from_numpy(self.scene.fe.astype(np.float32))
        self.m.from_numpy(self.scene.rho.astype(np.float32))
        self.init_field()

    @ti.kernel
    def init_field(self):
        self.v.fill(0)
        for e in self.elements:
            D = self.D(e)
            self.F_B[e] = D.inverse()
            self.F_W[e] = ti.abs(D.determinant()) / 6
            for i in ti.static(range(4)):
                self.v[self.elements[e][i]][0] += self.F_W[e] / 4
        for i in self.m:
            if self.v[i][0] == 0:
                self.v[i][0] = 1
            self.m[i] = self.v[i][0] * self.m[i]
        for f in self.fe:
            self.fe[f] = self.fe[f] * self.m[f]
        self.v.fill(0)

    @ti.func
    def D(self, e):
        return ti.Matrix.cols([self.x[self.elements[e][i]] - self.x[self.elements[e][3]] for i in range(3)])

    @ti.kernel
    def compute_f(self):
        for i in self.f:
            self.f[i] = self.fe[i]
        for e in self.elements:
            F = self.D(e) @ self.F_B[e]
            P = ti.Matrix.zero(ti.f32, 3, 3)
            U, sig, V = ssvd(F)
            P = 2 * self.mu * (F - U @ V.transpose())
            H = -self.F_W[e] * P @ self.F_B[e].transpose()
            for i in ti.static(range(3)):
                force = ti.Vector([H[j, i] for j in range(3)])
                self.f[self.elements[e][i]] += force
                self.f[self.elements[e][3]] -= force

    @ti.kernel
    def compute_b(self):
        for i in range(self.n):
            self.solver.b[i] = self.m[i] * self.v[i] + self.dt * self.f[i]

    @ti.kernel
    def compute_A(self):
        I = ti.Matrix.identity(ti.f32, 3)
        for i, j in ti.ndrange(self.n, self.n):
            if i == j:
                self.solver.A[i, j] = self.m[i] * I
            else:
                self.solver.A[i, j].fill(0)
        for e in self.elements:
            verts = self.elements[e]
            W_c = self.F_W[e]
            B_c = self.F_B[e]
            for u in ti.static(range(4)):
                for d in ti.static(range(3)):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if ti.static(u == 3):
                        for j in ti.static(range(3)):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ self.F_B[e]
                    dP = 2.0 * self.mu * dF
                    dH = -W_c * dP @ B_c.transpose()
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            self.solver.A[verts[u], verts[i]][d, j] += -self.dt**2 * dH[j, i]
                            self.solver.A[verts[u], verts[3]][d, j] += self.dt**2 * dH[j, i]

    @ti.kernel
    def update_xv(self):
        for i in range(self.n):
            if self.type == 1:
                self.v[i] += self.dt * (self.f[i] / self.m[i])
            self.x[i] += self.dt * self.v[i]
            if self.x[i].y < 0:
                self.x[i].y = 0
                if self.v[i].y < 0:
                    self.v[i].y = 0

    def substep(self):
        if self.type == 1:
            self.compute_f()
            self.update_xv()
        else:
            self.compute_f()
            self.compute_A()
            self.compute_b()
            self.solver.solve()
            self.update_xv()