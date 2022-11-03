import taichi as ti

@ti.data_oriented
class MassSpring:
    def __init__(self, type, dt):
        self.type = type # 1: explicit, 2: Jacobi, 3: Conjugate Gradient
        self.dt = dt
        self.n = 64
        self.m = 1.0
        self.spring_stiffness = 1000
        self.damping = 0.5
        self.epsilon = 1e-5
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.l = ti.field(dtype=ti.f32, shape=(self.n, self.n))

        self.A = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n, self.n))
        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=self.n)

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
        for i in range(self.n):
            self.v[i] = [0, 0, 0]

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
            self.b[i] = self.v[i] + self.dt * (self.compute_F(i) / self.m)

    @ti.kernel
    def compute_A(self):
        I = ti.Matrix([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        for i, j in self.A:
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
                self.A[i, j] = I
            else:
                self.A[i, j] = ti.Matrix.zero(ti.f32, 3, 3)
            self.A[i, j] -= self.dt ** 2 * J / self.m

    @ti.kernel
    def compute_rp(self):
        for i in range(self.n):
            self.r[i] = self.b[i]
            for j in range(self.n):
                self.r[i] -= self.A[i, j] @ self.v[j]
            self.p[i] = self.r[i]

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

    @ti.kernel
    def jacobi_iteration(self) -> ti.f32:
        e = 0.0
        for i in range(self.n):
            r = self.b[i]
            for j in range(self.n):
                r -= self.A[i, j] @ self.v[j]
            self.v[i] += self.A[i, i].inverse() @ r
            e += r.norm()
        return e

    @ti.kernel
    def cg_iteration(self) -> ti.f32:
        rr = 0.0
        pAp = 0.0
        rr1 = 0.0
        for i in range(self.n):
            rr += self.r[i].dot(self.r[i])
            self.Ap[i] = [0, 0, 0]
            for j in range(self.n):
                self.Ap[i] += self.A[i, j] @ self.p[j]
            pAp += self.p[i].dot(self.Ap[i])
        alpha = rr / pAp
        for i in range(self.n):
            self.v[i] += alpha * self.p[i]
            self.r[i] -= alpha * self.Ap[i]
        for i in range(self.n):
            rr1 += self.r[i].dot(self.r[i])
        beta = rr1 / rr
        for i in range(self.n):
            self.p[i] = self.r[i] + beta * self.p[i]
        return rr1

    def substep(self):
        # explicit
        if self.type == 1:
            self.explicit()
        # implicit
        else:
            self.compute_A()
            self.compute_b()
            
            # Jacobi
            if self.type == 2:
                for iter in range(10):
                    if self.jacobi_iteration() < self.epsilon:
                        break
            # Conjugate Gradient
            else:
                self.compute_rp()
                for iter in range(10):
                    if self.cg_iteration() < self.epsilon:
                        break
            self.update_xv()