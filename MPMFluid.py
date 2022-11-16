import taichi as ti

@ti.data_oriented
class MPMFluid:
    def __init__(self, type, dt):
        self.type = type # 1: sticky, 2: slip, 3: seperate
        self.dt = dt
        self.liquid_n = 15 * 15 * 15
        self.solid_n = 15 * 15 * 15
        self.n = self.liquid_n + self.solid_n
        self.rho = 1
        E, nu = 400, 0.2
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        self.grid_size = 0.1
        self.vol = (self.grid_size * 0.5) ** 3
        self.m = self.vol * self.rho
        self.bound = ti.Vector([2, 4, 2])
        self.bound_buf = 2
        self.grid_shape = [int(2 * self.bound.x / self.grid_size), int(self.bound.y / self.grid_size), int(2 * self.bound.z / self.grid_size)]

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.C = ti.Matrix.field(3, 3, dtype=float, shape=self.n)
        self.F = ti.Matrix.field(3, 3, dtype=float, shape=self.n)
        self.grid_v = ti.Vector.field(3, dtype=float, shape=self.grid_shape)
        self.grid_m = ti.field(dtype=float, shape=self.grid_shape)

        self.init_field()

    @ti.kernel
    def init_field(self):
        for i, j, k in ti.ndrange(15, 15, 15):
            self.x[i + j * 15 + k * 15 * 15] = ti.Vector([i * 0.1 + 0.4, j * 0.1 + 2, k * 0.1 - 0.4])
        for i, j, k in ti.ndrange(15, 15, 15):
            self.x[i + j * 15 + k * 15 * 15 + self.liquid_n] = ti.Vector([i * 0.1 - 0.4, j * 0.1 + 0.5, k * 0.1 + 0.4])
        self.v.fill(0)
        self.C.fill(0)
        self.F.fill(ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.grid_v.fill(0)
        self.grid_m.fill(0)

    @ti.func
    def BC(self, i, j, k):
        if self.type == 1:
            if i < self.bound_buf:
                self.grid_v[i, j, k] = [0, 0, 0]
            if i > self.grid_shape[0] - self.bound_buf:
                self.grid_v[i, j, k] = [0, 0, 0]
            if j < self.bound_buf:
                self.grid_v[i, j, k] = [0, 0, 0]
            if j > self.grid_shape[1] - self.bound_buf:
                self.grid_v[i, j, k] = [0, 0, 0]
            if k < self.bound_buf:
                self.grid_v[i, j, k] = [0, 0, 0]
            if k > self.grid_shape[2] - self.bound_buf:
                self.grid_v[i, j, k] = [0, 0, 0]
        elif self.type == 2:
            if i < self.bound_buf:
                self.grid_v[i, j, k][0] = 0
            if i > self.grid_shape[0] - self.bound_buf:
                self.grid_v[i, j, k][0] = 0
            if j < self.bound_buf:
                self.grid_v[i, j, k][1] = 0
            if j > self.grid_shape[1] - self.bound_buf:
                self.grid_v[i, j, k][1] = 0
            if k < self.bound_buf:
                self.grid_v[i, j, k][2] = 0
            if k > self.grid_shape[2] - self.bound_buf:
                self.grid_v[i, j, k][2] = 0
        elif self.type == 3:
            if i < self.bound_buf and self.grid_v[i, j, k][0] < 0:
                self.grid_v[i, j, k][0] = 0
            if i > self.grid_shape[0] - self.bound_buf and self.grid_v[i, j, k][0] > 0:
                self.grid_v[i, j, k][0] = 0
            if j < self.bound_buf and self.grid_v[i, j, k][1] < 0:
                self.grid_v[i, j, k][1] = 0
            if j > self.grid_shape[1] - self.bound_buf and self.grid_v[i, j, k][1] > 0:
                self.grid_v[i, j, k][1] = 0
            if k < self.bound_buf and self.grid_v[i, j, k][2] < 0:
                self.grid_v[i, j, k][2] = 0
            if k > self.grid_shape[2] - self.bound_buf and self.grid_v[i, j, k][2] > 0:
                self.grid_v[i, j, k][2] = 0

    @ti.kernel
    def P2G(self):
        for i, j, k in self.grid_m:
            self.grid_v[i, j, k] = [0, 0, 0]
            self.grid_m[i, j, k] = 0
        for p in self.x:
            pos = (self.x[p] + ti.Vector([self.bound.x, 0, self.bound.z])) / self.grid_size
            id = ti.cast(pos - 0.5, ti.i32)
            fx = pos - id
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            self.F[p] = (ti.Matrix.identity(float, 3) + self.dt * self.C[p]) @ self.F[p]            
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(3)):
                J *= sig[d, d]
            mu, la = self.mu_0, self.lambda_0
            if p < self.liquid_n:
                self.F[p] = ti.Matrix.identity(float, 3) * ti.pow(J, 1/3)
                mu = 0.0
            else:
                mu, la = self.mu_0 * 0.3, self.lambda_0 * 0.3
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
            stress = (-self.dt * self.vol * 4) * stress / self.grid_size ** 2
            affine = stress + self.m * self.C[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.grid_size
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_v[id + offset] += weight * (self.m * self.v[p] + affine @ dpos)
                self.grid_m[id + offset] += weight * self.m

    @ti.kernel
    def compute_grid(self):
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                self.grid_v[i, j, k] = \
                    (1 / self.grid_m[i, j, k]) * self.grid_v[i, j, k]
                self.grid_v[i, j, k][1] -= self.dt * 9.8
                self.BC(i, j, k)

    @ti.kernel
    def G2P(self):
        for p in self.x:
            pos = (self.x[p] + ti.Vector([self.bound.x, 0, self.bound.z])) / self.grid_size
            id = ti.cast(pos - 0.5, ti.i32)
            fx = pos - id
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset - fx) * self.grid_size
                g_v = self.grid_v[id + offset]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.grid_size ** 2
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]

    def substep(self):
        self.P2G()
        self.compute_grid()
        self.G2P()