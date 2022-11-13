import taichi as ti
from utils import LinearSolver

@ti.data_oriented
class MassSpring:
    def __init__(self, type, dt):
        self.type = type # 1: explicit, 2: Jacobi, 3: Conjugate Gradient
        self.dt = dt
        self.n = 8 * 8
        self.m = 1.0
        self.spring_stiffness = 1000
        self.damping = 0.5
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.l = ti.field(dtype=ti.f32, shape=(self.n, self.n))

        if type == 2:
            self.solver = LinearSolver(LinearSolver.jacobi, self.v, self.n)
        elif type == 3:
            self.solver = LinearSolver(LinearSolver.cg, self.v, self.n)

        self.init_field()

    @ti.kernel
    def init_field(self):
        for i, j in ti.ndrange(8, 8):
            self.x[i + j * 8] = [i - 4, 6, -j]
        for i, j in ti.ndrange(self.n, self.n):
            d = (self.x[i] - self.x[j]).norm()
            if i != j and d < ti.sqrt(2) + 0.01:
                self.l[i, j] = d
            else:
                self.l[i, j] = 0
        self.v.fill([0, 0, 0])

    @ti.func
    def compute_F(self, i):
        f = self.m * self.gravity - self.v[i] * self.damping
        for j in range(self.n):
            if self.l[i, j] != 0:
                x_ij = self.x[i] - self.x[j]
                f -= self.spring_stiffness * (x_ij.norm() - self.l[i, j]) * x_ij.normalized()
        return f

    @ti.kernel
    def compute_b(self):
        for i in range(self.n):
            self.solver.b[i] = self.v[i] + self.dt * (self.compute_F(i) / self.m)

    @ti.kernel
    def compute_A(self):
        I = ti.Matrix([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        for i, j in ti.ndrange(self.n, self.n):
            # compute Jacobi
            J = ti.Matrix.zero(ti.f32, 3, 3)
            for k in range(self.n):
                if self.l[i, k] != 0 and (j == i or j == k):
                    x_ik = self.x[i] - self.x[k]
                    x_ik_norm = x_ik.norm()
                    x_ik_normalized = x_ik.normalized()
                    x_ik_mat = x_ik_normalized.outer_product(x_ik_normalized)

                    if j == i:
                        J += -self.spring_stiffness * ((1 - self.l[i, k] / x_ik_norm) * (I - x_ik_mat) + x_ik_mat)
                    else:
                        J -= -self.spring_stiffness * ((1 - self.l[i, k] / x_ik_norm) * (I - x_ik_mat) + x_ik_mat)
            # compute A
            if i == j:
                self.solver.A[i, j] = I
            else:
                self.solver.A[i, j] = ti.Matrix.zero(ti.f32, 3, 3)
            self.solver.A[i, j] -= self.dt ** 2 * J / self.m

    @ti.kernel
    def update_xv(self):
        for i in range(self.n):
            if i == 0 or i == 7:
                self.v[i] = [0, 0, 0]
            self.x[i] += self.v[i] * self.dt

    @ti.kernel
    def explicit(self):
        for i in range(self.n):
            self.v[i] += self.dt * (self.compute_F(i) / self.m)
            if i == 0 or i == 7:
                self.v[i] = [0, 0, 0]
            self.x[i] += self.v[i] * self.dt

    def substep(self):
        # explicit
        if self.type == 1:
            self.explicit()
        # implicit
        else:
            self.compute_A()
            self.compute_b()
            
            self.solver.solve()
            self.update_xv()