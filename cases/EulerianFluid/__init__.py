import taichi as ti
from utils import semi_lagrange, bfecc, sample, LinearSolver

# projection: taichi sparse matrix solver
@ti.data_oriented
class EulerianFluidSolver:
    name = "EulerianFluid"
    def __init__(self, type, dt):
        self.type = type # 1: semi-lagrange, 2: BFECC
        self.dt = dt
        self.dx = 1.0
        self.grid_size = 0.04
        self.bound = ti.Vector([0.6, 4, 0.6])
        self.grid_shape = [int(2 * self.bound.x / self.grid_size), int(self.bound.y / self.grid_size), int(2 * self.bound.z / self.grid_size)]
        self.n = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        self.density_threshold = 0.3

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

        self.reset()

    def reset(self):
        self.init_field()

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
            if self.type == 1:
                self.v_new[i, j, k] = semi_lagrange(self.v, self.v, pos, self.dt)
                self.density_new[i, j, k] = semi_lagrange(self.v, self.density, pos, self.dt)
            elif self.type == 2:
                self.v_new[i, j, k] = bfecc(self.v, self.v, pos, self.dt)
                self.density_new[i, j, k] = bfecc(self.v, self.density, pos, self.dt)
            else:
                print("Invalid type!")
        for i, j, k in self.v:
            self.v[i, j, k] = self.v_new[i, j, k]
            self.density[i, j, k] = self.density_new[i, j, k]

    @ti.kernel
    def v_divergence(self):
        for i, j, k in self.v:
            vl = sample(self.v, i - 1, j, k)[0]
            vr = sample(self.v, i + 1, j, k)[0]
            vb = sample(self.v, i, j - 1, k)[1]
            vt = sample(self.v, i, j + 1, k)[1]
            vh = sample(self.v, i, j, k - 1)[2]
            vq = sample(self.v, i, j, k + 1)[2]
            vc = sample(self.v, i, j, k)
            if i == 0:
                vl = -vc[0]
            if i == self.v.shape[0] - 1:
                vr = -vc[0]
            if j == 0:
                vb = -vc[1]
            if j == self.v.shape[1] - 1:
                vt = -vc[1]
            if k == 0:
                vh = -vc[2]
            if k == self.v.shape[2] - 1:
                vq = -vc[2]
            self.v_div[i, j, k] = (vr - vl + vt - vb + vq - vh) / self.dx / 2

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
            self.v[i, j, k] = v - 0.5 / self.dx * ti.Vector([pr - pl, pt - pb, pq - ph])

    @ti.kernel
    def update_x(self):
        for i, j, k in self.density:
            if self.density[i, j, k] > self.density_threshold:
                self.x[i + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]] = [
                    i * self.grid_size - self.bound.x,
                    j * self.grid_size,
                    k * self.grid_size - self.bound.z
                ]
            else:
                self.x[i + j * self.grid_shape[0] + k * self.grid_shape[0] * self.grid_shape[1]] = [0, 0, 0]

    @ti.kernel
    def source(self):
        a1 = self.grid_shape[0] // 2 - 3
        b1 = self.grid_shape[0] // 2 + 3
        c1 = self.grid_shape[2] // 2 - 3
        d1 = self.grid_shape[2] // 2 + 3

        for i, j, k in ti.ndrange((a1, b1), (0, 3), (c1, d1)):
            self.density[i, j, k] = 0.5
        for i, j, k in ti.ndrange((a1, b1), (0, 3), (c1, d1)):
            self.v[i, j, k] = [0, 50, 0]

    def substep(self):
        self.advection()
        self.v_divergence()
        self.source()
        # projection
        self.compute_b()
        self.solver.solve()
        self.update_v()
        self.update_x()
