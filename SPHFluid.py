import taichi as ti
import numpy as np

@ti.data_oriented
class SPHFluid:
    def __init__(self, type, dt):
        self.type = type # 1: WCSPH
        self.dt = dt
        self.n = 16 * 16 * 16
        self.m = 1
        self.surface_tension = 0.01
        self.mu = 2
        self.bulk_modulus = 2000
        self.density0 = 700
        self.kernel_radius = 0.2
        self.grid_size = 0.1
        self.bound = ti.Vector([2, 6, 2])
        self.grid_shape = ti.Vector([int(2 * self.bound.x / self.grid_size) + 1, int(self.bound.y / self.grid_size) + 1, int(2 * self.bound.z / self.grid_size) + 1])
        self.gridn = self.grid_shape.x * self.grid_shape.y * self.grid_shape.z
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.density = ti.field(dtype=ti.f32, shape=self.n)
        self.pressure = ti.field(dtype=ti.f32, shape=self.n)
        self.F = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.grid = ti.field(dtype=ti.i32, shape=(self.gridn, self.n))
        self.gridCount = ti.field(dtype=ti.i32, shape=self.gridn)
        self.neighbour = ti.field(dtype=ti.i32, shape=(self.n, self.n))
        self.neighbourCount = ti.field(dtype=ti.i32, shape=self.n)

        self.init_field()

    @ti.kernel
    def init_field(self):
        for i, j, k in ti.ndrange(16, 16, 16):
            self.x[i + j * 16 + k * 16 * 16] = [(i - 7.5) * 0.1, j * 0.1 + 2, (k - 7.5) * 0.1]
        for i in range(self.n):
            self.v[i] = [0, 0, 0]
            self.density[i] = 0.0
            self.pressure[i] = 0.0

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.kernel_radius
        k = 8 / np.pi
        k /= h ** 3
        q = r_norm / h
        if q < 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.kernel_radius
        k = 8 / np.pi
        k = 6. * k / h ** 3
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0, 0.0, 0.0])
        if r_norm > 1e-5 and q < 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def hash(self, id):
        return id.x + id.y * self.grid_shape.x + id.z * self.grid_shape.x * self.grid_shape.y

    @ti.kernel
    def sph_init(self):
        for i in range(self.gridn):
            self.gridCount[i] = 0
        for i in range(self.n):
            self.density[i] = 0
            self.F[i] = self.m * self.gravity
            self.neighbourCount[i] = 0
            id = ti.cast((self.x[i] + ti.Vector([self.bound.x, 0, self.bound.z])) / self.grid_size, ti.i32)
            index = self.hash(id)
            self.grid[index, self.gridCount[index]] = i
            self.gridCount[index] += 1
        for i in range(self.n):
            id = ti.cast((self.x[i] + ti.Vector([self.bound.x, 0, self.bound.z])) / self.grid_size, ti.i32)
            for m in range(-2,3):
                for n in range(-2,3):
                    for q in range(-2,3):
                        id_n = id + ti.Vector([m, n, q])
                        if id_n.x >= 0 and id_n.x < self.grid_shape.x and \
                        id_n.y >= 0 and id_n.y < self.grid_shape.y  and \
                        id_n.z >= 0 and id_n.z < self.grid_shape.z :
                            index_n = self.hash(id_n)
                            for k in range(self.gridCount[index_n]):
                                j = self.grid[index_n, k]
                                if j != i:
                                    r = (self.x[i] - self.x[j]).norm()
                                    if r < self.kernel_radius:
                                        self.neighbour[i, self.neighbourCount[i]] = j
                                        self.neighbourCount[i] += 1

    @ti.kernel
    def compute_density(self):
        for i in range(self.n):
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                x_ij = self.x[i] - self.x[j]
                self.density[i] += self.m * self.cubic_kernel(x_ij.norm())
            self.density[i] = ti.max(self.density[i], self.density0)
            self.pressure[i] = self.bulk_modulus * (ti.pow(self.density[i] / self.density0, 7) - 1.0)

    @ti.kernel
    def compute_Force(self):
        for i in range(self.n):
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                x_ij = self.x[i] - self.x[j]
                v_ij = self.v[i] - self.v[j]
                # Surface Tension
                self.F[i] += -self.surface_tension / self.m * self.m * x_ij * self.cubic_kernel(x_ij.norm())
                # Viscosoty Force
                self.F[i] += self.mu * (self.m / (self.density[i] + self.density[j])) * v_ij.dot(x_ij) / (
                    x_ij.norm()**2 + 0.01 * self.kernel_radius**2) * self.cubic_kernel_derivative(x_ij)
                # Pressure Force
                dpi = self.pressure[i] / self.density[i] ** 2
                dpj = self.pressure[j] / self.density[j] ** 2
                self.F[i] += -self.m * (dpi + dpj) * self.cubic_kernel_derivative(x_ij)

    @ti.kernel
    def update_xv(self):
        for i in range(self.n):
            self.v[i] += self.dt * self.F[i] / self.m
            if self.x[i].x + self.v[i].x * self.dt < -self.bound.x:
                self.x[i].x = -self.bound.x
                self.v[i].x = 0
            if self.x[i].x + self.v[i].x * self.dt > self.bound.x:
                self.x[i].x = self.bound.x
                self.v[i].x = 0
            if self.x[i].y + self.v[i].y * self.dt < 0:
                self.x[i].y = 0
                self.v[i].y = 0
            if self.x[i].y + self.v[i].y * self.dt > self.bound.y:
                self.x[i].y = self.bound.y
                self.v[i].y = 0
            if self.x[i].z + self.v[i].z * self.dt < -self.bound.z:
                self.x[i].z = -self.bound.z
                self.v[i].z = 0
            if self.x[i].z + self.v[i].z * self.dt > self.bound.z:
                self.x[i].z = self.bound.z
                self.v[i].z = 0
            self.x[i] += self.v[i] * self.dt

    def substep(self):
        if type == 1:
            self.sph_init()
            self.compute_density()
            self.compute_Force()
            self.update_xv()