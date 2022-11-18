import taichi as ti
from utils import LinearSolver

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
class FEMSolid:
    def __init__(self, type, dt):
        self.type = type # 1: explicit, 2: Jacobi, 3: Conjugate Gradient, 4: MGPCG
        self.dt = dt
        self.n = 5 * 5 * 5
        E, nu = 1e5, 0.0
        self.mu, self.la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        self.density = 1000.0
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=64 * 5)
        self.m = ti.field(dtype=ti.f32, shape=self.n)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.f = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=64 * 5)
        self.F_W = ti.field(dtype=ti.f32, shape=64 * 5)

        if type == 2:
            self.solver = LinearSolver(LinearSolver.jacobi, self.v, self.n)
        elif type == 3:
            self.solver = LinearSolver(LinearSolver.cg, self.v, self.n)
        elif type == 4:
            self.solver = LinearSolver(LinearSolver.mgpcg, self.v, self.n)

        self.init_field()

    @ti.kernel
    def init_field(self):
        for I in ti.grouped(ti.ndrange(4, 4, 4)):
            e = ((I.x * 4 + I.y) * 4 + I.z) * 5
            for i, j in ti.static(enumerate([0, 3, 5, 6])):
                self.set_element(e + i, I, (j, j ^ 1, j ^ 2, j ^ 4))
            self.set_element(e + 4, I, (1, 2, 4, 7))
        for I in ti.grouped(ti.ndrange(5, 5, 5)):
            id = (I.x * 5 + I.y) * 5 + I.z
            self.x[id] = I * 0.5 + ti.Vector([-1, 1, -1])
        self.v.fill(0)
        self.f.fill(0)
        self.m.fill(0)
        for e in self.elements:
            D = self.D(e)
            self.F_B[e] = D.inverse()
            self.F_W[e] = ti.abs(D.determinant()) / 6
            for i in ti.static(range(4)):
                self.m[self.elements[e][i]] += self.F_W[e] / 4 * self.density

    @ti.func
    def set_element(self, e, I, verts):
        for i in ti.static(range(4)):
            t = I + (([verts[i] >> k for k in range(3)] ^ I) & 1)
            self.elements[e][i] = (t.x * 5 + t.y) * 5 + t.z

    @ti.func
    def D(self, e):
        return ti.Matrix.cols([self.x[self.elements[e][i]] - self.x[self.elements[e][3]] for i in range(3)])

    @ti.kernel
    def compute_f(self):
        for i in self.f:
            self.f[i] = self.gravity * self.m[i]
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