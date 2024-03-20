import taichi as ti
import numpy as np

@ti.data_oriented
class SPHFluidSolver:
    name = "SPHFluid"
    def __init__(self, type, dt):
        self.type = type # 1: WCSPH, 2: PCISPH
        self.dt = dt
        self.n = 20 * 20 * 20
        self.m = 1
        self.surface_tension = 0.02
        self.mu = 0.5
        self.bulk_modulus = 5000
        self.density0 = 1000
        self.kernel_radius = 0.2
        self.grid_size = 0.1
        self.bound = ti.Vector([2, 6, 2]) # -bound.x < x < bound.x, 0 < y < bound.y, -bound.z < z < bound.z
        self.grid_shape = ti.Vector([int(2 * self.bound.x / self.grid_size) + 1, int(self.bound.y / self.grid_size) + 1, int(2 * self.bound.z / self.grid_size) + 1])
        self.n_grid = self.grid_shape.x * self.grid_shape.y * self.grid_shape.z
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        # basic
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.density = ti.field(dtype=ti.f32, shape=self.n)
        self.pressure = ti.field(dtype=ti.f32, shape=self.n)
        self.f = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        # neighbour search
        self.grid = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n))
        self.gridCount = ti.field(dtype=ti.i32, shape=self.n_grid)
        self.neighbour = ti.field(dtype=ti.i32, shape=(self.n, self.n))
        self.neighbourCount = ti.field(dtype=ti.i32, shape=self.n)
        # pci
        self.epsilon = 1e-2
        self.x_star = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v_star = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.pci_factor = ti.field(dtype=ti.f32, shape=())

        self.reset()

    def reset(self):
        self.init_field()

    @ti.kernel
    def init_field(self):
        for i, j, k in ti.ndrange(20, 20, 20):
            self.x[i + j * 20 + k * 20 * 20] = [(i - 9.5) * 0.1, j * 0.1 + 2, (k - 9.5) * 0.1]
        self.v.fill([0, 0, 0])
        self.density.fill(0.0)
        self.pressure.fill(0.0)
        self.pci_factor[None] = self.compute_pci_factor()

    @ti.func
    def cubic_kernel(self, r_norm):
        res = 0.0
        h = self.kernel_radius / 2
        k = 1 / np.pi
        k /= h ** 3
        q = r_norm / h
        if q < 2.0:
            if q <= 1.0:
                q2 = q * q
                q3 = q2 * q
                res = k * (1 - 1.5 * q2 + 0.75 * q3)
            else:
                res = k * 0.25 * ti.pow(2 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.kernel_radius / 2
        k = 1 / np.pi
        k /= h ** 3
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0, 0.0, 0.0])
        if r_norm > 1e-5 and q < 2.0:
            grad_q = r / (r_norm * h)
            if q <= 1.0:
                res = k * q * (2.25 * q - 3.0) * grad_q
            else:
                factor = 2.0 - q
                res = k * 0.75 * (-factor * factor) * grad_q
        return res

    @ti.func
    def hash(self, id):
        return id.x + id.y * self.grid_shape.x + id.z * self.grid_shape.x * self.grid_shape.y

    @ti.kernel
    def neighbour_init(self):
        for i in range(self.n_grid):
            self.gridCount[i] = 0
        for i in range(self.n):
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

    @ti.func
    def compute_density(self, i, j):
        x_ij = self.x[i] - self.x[j]
        return self.m * self.cubic_kernel(x_ij.norm())

    @ti.func
    def compute_surface_tension(self, i, j):
        x_ij = self.x[i] - self.x[j]
        return -self.surface_tension * self.m * x_ij * self.cubic_kernel(x_ij.norm())

    @ti.func
    def compute_viscosoty_force(self, i, j):
        x_ij = self.x[i] - self.x[j]
        v_ij = self.v[i] - self.v[j]
        return self.m * self.m * (self.mu / (self.density[i] + self.density[j])) * v_ij.dot(x_ij) / (
                x_ij.norm()**2 + 0.01 * self.kernel_radius**2) * self.cubic_kernel_derivative(x_ij)

    @ti.func
    def compute_pressure_force(self, i, j):
        x_ij = self.x[i] - self.x[j]
        dpi = self.pressure[i] / self.density[i] ** 2
        dpj = self.pressure[j] / self.density[j] ** 2
        return -self.m * self.m * (dpi + dpj) * self.cubic_kernel_derivative(x_ij)

    @ti.func
    def compute_xv(self, i):
        self.v[i] += self.dt * self.f[i] / self.m
        if self.x[i].x + self.v[i].x * self.dt < -self.bound.x:
            self.x[i].x = -self.bound.x + 0.01 * ti.random()
            self.v[i].x = 0
        if self.x[i].x + self.v[i].x * self.dt > self.bound.x:
            self.x[i].x = self.bound.x - 0.01 * ti.random()
            self.v[i].x = 0
        if self.x[i].y + self.v[i].y * self.dt < 0:
            self.x[i].y = 0 + 0.01 * ti.random()
            self.v[i].y = 0
        if self.x[i].y + self.v[i].y * self.dt > self.bound.y:
            self.x[i].y = self.bound.y - 0.01 * ti.random()
            self.v[i].y = 0
        if self.x[i].z + self.v[i].z * self.dt < -self.bound.z:
            self.x[i].z = -self.bound.z + 0.01 * ti.random()
            self.v[i].z = 0
        if self.x[i].z + self.v[i].z * self.dt > self.bound.z:
            self.x[i].z = self.bound.z - 0.01 * ti.random()
            self.v[i].z = 0
        self.x[i] += self.dt * self.v[i]

    @ti.func
    def compute_pci_factor(self):
        beta = self.dt * self.dt * self.m * self.m * 2 / (self.density0 * self.density0)
        sum1 = ti.Vector([0.0, 0.0, 0.0])
        sum2 = 0.0
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    r = ti.Vector([i * 0.1, j * 0.1, k * 0.1])
                    gradW = self.cubic_kernel_derivative(r)
                    sum1 += gradW
                    sum2 += gradW.dot(gradW)
            
        return -1 / (beta * (- sum1.dot(sum1) - sum2))

    @ti.kernel
    def wcsph(self):
        for i in range(self.n):
            self.density[i] = 0
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                self.density[i] += self.compute_density(i, j)
            self.density[i] = ti.max(self.density[i], self.density0)
            self.pressure[i] = self.bulk_modulus * (ti.pow(self.density[i] / self.density0, 7) - 1.0)
        for i in range(self.n):
            self.f[i] = self.m * self.gravity
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                # Surface Tension
                self.f[i] += self.compute_surface_tension(i, j)
                # Viscosoty Force
                self.f[i] += self.compute_viscosoty_force(i, j)
                # Pressure Force
                self.f[i] += self.compute_pressure_force(i, j)
            self.compute_xv(i)
        
    @ti.kernel
    def pci_init(self):
        for i in range(self.n):
            self.pressure[i] = 0
            self.density[i] = 0
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                self.density[i] += self.compute_density(i, j)
            self.density[i] = ti.max(self.density[i], self.density0)
        for i in range(self.n):
            self.f[i] = self.m * self.gravity
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                # Surface Tension
                self.f[i] += self.compute_surface_tension(i, j)
                # Viscosoty Force
                self.f[i] += self.compute_viscosoty_force(i, j)

    @ti.kernel
    def pci_iteration(self) -> ti.f32:
        err = 0.0
        # predict xv
        for i in range(self.n):
            self.v_star[i] = self.v[i]
            self.x_star[i] = self.x[i]
            F_origin = self.f[i]
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                self.f[i] += self.compute_pressure_force(i, j)
            self.compute_xv(i)
            self.f[i] = F_origin
        for i in range(self.n):
            # density_star
            self.density[i] = 0
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                self.density[i] += self.compute_density(i, j)
            self.density[i] = ti.max(self.density[i], self.density0)
            density_err = self.density[i] / self.density0 - 1.0
            err += density_err
            # update pressure
            self.pressure[i] += self.pci_factor[None] * density_err
            self.v[i] = self.v_star[i]
            self.x[i] = self.x_star[i]
        return err / self.n

    @ti.kernel
    def pci_update(self):
        for i in range(self.n):
            for k in range(self.neighbourCount[i]):
                j = self.neighbour[i, k]
                self.f[i] += self.compute_pressure_force(i, j)
            self.compute_xv(i)

    def substep(self):
        if self.type == 1:
            self.neighbour_init()
            self.wcsph()
        elif self.type == 2:
            self.neighbour_init()
            self.pci_init()
            for iter in range(50):
                if self.pci_iteration() < self.epsilon:
                    break
            self.pci_update()
        else:
            print("Invalid type!")