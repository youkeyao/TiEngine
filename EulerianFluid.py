import taichi as ti
from utils import semi_lagrange, divergence, sample, LinearSolver

# advection: semi-lagrange
@ti.data_oriented
class EulerianFluid:
    def __init__(self, type, dt):
        self.type = type # 1: Jacobi, 2: Conjugate Gradient
        self.dt = dt
        self.grid_size = 0.08
        self.bound = ti.Vector([1, 4, 1])
        self.grid_shape = [int(2 * self.bound.x / self.grid_size), int(self.bound.y / self.grid_size), int(2 * self.bound.z / self.grid_size)]
        self.n = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        self.episilon = 1e-2

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_shape)
        self.v_new = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_shape)
        self.v_div = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.density = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.density_new = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.pressure = ti.field(dtype=ti.f32, shape=self.n)

        self.solver = LinearSolver(LinearSolver.sparse, self.pressure, self.n)
        self.compute_A(self.solver.A)
        self.solver.build_sparse()

        self.init_field()
        self._img = ti.Vector.field(3, dtype=ti.f32, shape=(self.grid_shape[0], self.grid_shape[1]))

    @ti.kernel
    def init_field(self):
        self.v.fill([0, 0, 0])
        self.v_new.fill([0, 0, 0])
        self.v_div.fill(0)
        self.density.fill(0)
        self.density_new.fill(0)
        self.pressure.fill(0)

    @ti.kernel
    def advection(self):
        for i, j, k in self.v:
            pos = ti.Vector([i, j, k]) + 0.5
            self.v_new[i, j, k] = semi_lagrange(self.v, self.v, pos, self.dt)
            self.density_new[i, j, k] = semi_lagrange(self.v, self.density, pos, self.dt)
        for i, j, k in self.v:
            self.v[i, j, k] = self.v_new[i, j, k]
            self.density[i, j, k] = self.density_new[i, j, k]

    @ti.kernel
    def v_divergence(self):
        divergence(self.v, self.v_div)

    @ti.kernel
    def compute_b(self):
        for i, j, k in ti.ndrange(self.grid_shape[0], self.grid_shape[1], self.grid_shape[2]):
            self.solver.b[i + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]] = -self.v_div[i, j, k]

    @ti.kernel
    def compute_A(self, A: ti.types.sparse_matrix_builder()):
        for i, j, k in ti.ndrange(self.grid_shape[0], self.grid_shape[1], self.grid_shape[2]):
            iA = i + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]
            if i != 0:
                A[iA, iA - 1] += -1.0
            if i != self.grid_shape[0] - 1:
                A[iA, iA + 1] += -1.0
            if j != 0:
                A[iA, iA - self.grid_shape[0]] += -1.0
            if j != self.grid_shape[1] - 1:
                A[iA, iA + self.grid_shape[0]] += -1.0
            if k != 0:
                A[iA, iA - self.grid_shape[0] * self.grid_shape[1]] += -1.0
            if k != self.grid_shape[2] - 1:
                A[iA, iA + self.grid_shape[0] * self.grid_shape[1]] += -1.0
            A[iA, iA] += 6.0

    @ti.kernel
    def pressure_jacobi(self, pf: ti.template(), new_pf: ti.template()):
        for i, j, k in pf:
            pl = sample(pf, i - 1, j, k)
            pr = sample(pf, i + 1, j, k)
            pb = sample(pf, i, j - 1, k)
            pt = sample(pf, i, j + 1, k)
            ph = sample(pf, i, j, k - 1)
            pq = sample(pf, i, j, k + 1)
            div = self.v_div[i, j, k]
            new_pf[i, j, k] = (pl + pr + pb + pt + ph + pq - div) * 1.0 / 6

    @ti.kernel
    def update_v(self):
        for i, j, k in self.v:
            imin, jmin, kmin = max(0, i - 1), max(0, j - 1), max(0, k - 1)
            imax, jmax, kmax = min(self.grid_shape[0] - 1, i + 1), min(self.grid_shape[1] - 1, j + 1), min(self.grid_shape[2] - 1, k + 1)
            pl = self.pressure[imin + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]]
            pr = self.pressure[imax + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]]
            pb = self.pressure[i + jmin * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]]
            pt = self.pressure[i + jmax * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]]
            ph = self.pressure[i + j * self.grid_shape[0] + kmin * self.grid_shape[0] * self.grid_shape[1]]
            pq = self.pressure[imin + j * self.grid_shape[0] + kmax * self.grid_shape[0] * self.grid_shape[1]]
            v = sample(self.v, i, j, k)
            self.v[i, j, k] = v - 0.5 * ti.Vector([pr - pl, pt - pb, pq - ph])

    @ti.kernel
    def update_x(self):
        for i, j, k in self.density:
            if self.density[i, j, k] > self.episilon:
                self.x[i + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]] = [i * self.grid_size - self.bound.x, j * self.grid_size, k * self.grid_size - self.bound.z]
            else:
                self.x[i + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]] = [0, 0, 0]

    @ti.kernel
    def source(self):
        a1 = self.grid_shape[0] // 2 - self.grid_shape[0] // 20
        b1 = self.grid_shape[0] // 2 + self.grid_shape[0] // 20
        c1 = self.grid_shape[2] // 2 - self.grid_shape[2] // 20
        d1 = self.grid_shape[2] // 2 + self.grid_shape[2] // 20

        for i, j, k in ti.ndrange((a1, b1), (0, 10), (c1, d1)):
            self.density[i, j, k] = 0.5
        for i, j, k in ti.ndrange((a1, b1), (0, 10), (c1, d1)):
            self.v[i, j, k] = [0, 50, 0]

    @ti.kernel
    def to_image(self):
        for i, j in ti.ndrange((0, self.grid_shape[0]), (0, self.grid_shape[1])):
            self._img[i, j] = self.density[i, j, self.grid_shape[0] // 2] * ti.Vector([1, 0.8, 0.8])

    def substep(self):
        self.source()
        self.advection()
        self.v_divergence()
        # projection
        self.compute_b()
        self.solver.solve()
        self.update_v()
        self.update_x()
        self.to_image()