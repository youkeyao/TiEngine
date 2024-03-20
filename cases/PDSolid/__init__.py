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
class PDSolidSolver:
    name = "PDSolid"
    def __init__(self, type, dt):
        self.type = type # 1: Jacobi, 3: Conjugate Gradient, 4: MGPCG
        self.dt = dt

        self.scene = Scene()
        self.scene.add_sphere([4, 4, 4], [0, 4, 0])

        self.n = self.scene.n_verts
        self.n_elements = self.scene.n_elements
        self.jacobi_alpha = 0.1
        self.stiffness = 5000
        self.max_iter = 5
        self.epsilon = 1e-4

        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=self.n_elements)
        self.m = ti.field(dtype=ti.f32, shape=self.n)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.x_prev = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.s = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.q = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.x_iter = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.fe = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.D_m = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_elements)
        self.F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_elements)
        self.F_W = ti.field(dtype=ti.f32, shape=self.n_elements)
        self.count = ti.field(dtype=ti.i32, shape=self.n)
        self.sum_SiTSi = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n, self.n))

        if type == 1:
            self.solver = LinearSolver(LinearSolver.jacobi, self.q, self.n)
        elif type == 2:
            self.solver = LinearSolver(LinearSolver.cg, self.q, self.n)
        elif type == 3:
            self.solver = LinearSolver(LinearSolver.mgpcg, self.q, self.n)
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
            self.D_m[e] = self.D(e)
            self.F_B[e] = self.D_m[e].inverse()
            self.F_W[e] = ti.abs(self.D_m[e].determinant()) / 6
            for i in ti.static(range(4)):
                self.v[self.elements[e][i]][0] += self.F_W[e] / 4
        for i in self.m:
            if self.v[i][0] == 0:
                self.v[i][0] = 1
            self.m[i] = self.v[i][0] * self.m[i]
        for f in self.fe:
            self.fe[f] = self.fe[f] * self.m[f]
        self.v.fill(0)
        self.compute_Si()
        self.compute_A()

    @ti.func
    def D(self, e):
        return ti.Matrix.cols([self.x[self.elements[e][i]] - self.x[self.elements[e][3]] for i in range(3)])

    @ti.func
    def compute_Si(self):
        I = ti.Matrix.identity(ti.f32, 3)
        self.sum_SiTSi.fill(0)
        for e in range(self.n_elements):
            for i in ti.static(range(4)):
                x_i = self.elements[e][i]
                self.sum_SiTSi[x_i, x_i] += I
    @ti.kernel
    def compute_b(self):
        for i in range(self.n):
            self.solver.b[i] = self.m[i] * self.s[i] / (self.dt ** 2) + self.stiffness * self.sum_SiTSi[i, i] @ self.x[i]
    @ti.func
    def compute_A(self):
        I = ti.Matrix.identity(ti.f32, 3)
        for i, j in ti.ndrange(self.n, self.n):
            if i == j:
                self.solver.A[i, j] = self.m[i] * I / (self.dt ** 2) + self.stiffness * self.sum_SiTSi[i, j]
            else:
                self.solver.A[i, j].fill(0)

    @ti.kernel
    def init_x(self):
        for i in range(self.n):
            self.x_prev[i] = self.x[i]
            self.x[i] = self.x[i] + self.dt * self.v[i] + (self.dt ** 2) * self.fe[i] / self.m[i]
            self.s[i] = self.x[i]

    @ti.kernel
    def local_step(self) -> ti.f32:
        # Jacobi Solver
        r = 0.0
        for i in range(self.n):
            self.x_iter[i] = ti.Vector([0.0, 0.0, 0.0])
            self.count[i] = 0
        for e in range(self.n_elements):
            F = self.D(e) @ self.F_B[e]
            U, S, V = ti.svd(F)
            S[0, 0] = min(max(0.95, S[0, 0]), 1.05)
            S[1, 1] = min(max(0.95, S[1, 1]), 1.05)
            S[2, 2] = min(max(0.95, S[2, 2]), 1.05)
            F_star = U @ S @ V.transpose()
            D_star = F_star @ self.D_m[e]

            e0 = self.elements[e][0]
            e1 = self.elements[e][1]
            e2 = self.elements[e][2]
            e3 = self.elements[e][3]

            # find the center of gravity
            center = (self.x[e0] + self.x[e1] + self.x[e2] + self.x[e3]) / 4

            # find the projected vector
            for n in ti.static(range(3)):
                x3_new = center[n] - (D_star[n, 0] + D_star[n, 1] + D_star[n, 2]) / 4
                self.x_iter[e3][n] += x3_new
                self.x_iter[e0][n] += x3_new + D_star[n, 0]
                self.x_iter[e1][n] += x3_new + D_star[n, 1]
                self.x_iter[e2][n] += x3_new + D_star[n, 2]

            self.count[e0] += 1
            self.count[e1] += 1
            self.count[e2] += 1
            self.count[e3] += 1

        for i in range(self.n):
            r += (self.x_iter[i] - self.x[i]).norm_sqr()
            self.x[i] = (self.x_iter[i] + self.jacobi_alpha * self.x[i]) / (self.count[i] + self.jacobi_alpha)

        return r

    @ti.kernel
    def update_x(self) -> ti.f32:
        r = 0.0
        for i in range(self.n):
            r += (self.q[i] - self.x[i]).norm_sqr()
            self.x[i] = self.q[i]
            if self.x[i].y < 0:
                self.x[i].y = 0
        return r
    @ti.kernel
    def update_v(self):
        for i in range(self.n):
            self.v[i] = (self.x[i] - self.x_prev[i]) / self.dt

    def substep(self):
        self.init_x()
        for i in range(self.max_iter):
            for k in range(self.max_iter):
                if self.local_step() < self.epsilon:
                    break
            self.compute_b()
            self.solver.solve()
            if self.update_x() < self.epsilon:
                break
        self.update_v()