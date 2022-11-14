import taichi as ti

@ti.func
def sample(qf, u, v, w):
    i, j, k = int(u), int(v), int(w)
    i = max(0, min(qf.shape[0] - 1, i))
    j = max(0, min(qf.shape[1] - 1, j))
    k = max(0, min(qf.shape[2] - 1, k))
    return qf[i, j, k]

@ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)

@ti.func
def trilerp(vf, u, v, w):
    s, t, n = u - 0.5, v - 0.5, w - 0.5
    iu, iv, iw = max(0, int(s)), max(0, int(t)), max(0, int(n))
    fu, fv, fw = s - iu, t - iv, n - iw
    a = sample(vf, iu, iv, iw)
    b = sample(vf, iu + 1, iv, iw)
    c = sample(vf, iu, iv + 1, iw)
    d = sample(vf, iu + 1, iv + 1, iw)
    e = sample(vf, iu, iv, iw + 1)
    f = sample(vf, iu + 1, iv, iw + 1)
    g = sample(vf, iu, iv + 1, iw + 1)
    h = sample(vf, iu + 1, iv + 1, iw + 1)

    bilerp1 = lerp(lerp(a, b, fu), lerp(c, d, fu), fv)
    bilerp2 = lerp(lerp(e, f, fu), lerp(g, h, fu), fv)
    return lerp(bilerp1, bilerp2, fw)

@ti.func
def back_trace_rk2(vf, pos, dt):
    mid = pos - 0.5 * dt * trilerp(vf, pos[0], pos[1], pos[2])
    return pos - dt * trilerp(vf, mid[0], mid[1], mid[2])

@ti.func
def semi_lagrange(vf, qf, pos, dt):
    coord = back_trace_rk2(vf, pos, dt)
    return trilerp(qf, coord[0], coord[1], coord[2])

@ti.func
def divergence(vf, v_div):
    for i, j, k in vf:
        vl = sample(vf, i - 1, j, k)[0]
        vr = sample(vf, i + 1, j, k)[0]
        vb = sample(vf, i, j - 1, k)[1]
        vt = sample(vf, i, j + 1, k)[1]
        vh = sample(vf, i, j, k - 1)[2]
        vq = sample(vf, i, j, k + 1)[2]
        vc = sample(vf, i, j, k)
        if i == 0:
            vl = -vc[0]
        if i == vf.shape[0] - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == vf.shape[1] - 1:
            vt = -vc[1]
        if k == 0:
            vh = -vc[2]
        if k == vf.shape[2] - 1:
            vq = -vc[2]
        v_div[i, j, k] = (vr - vl + vt - vb + vq - vh) * 0.5

@ti.func
def restrict_matrix(i, j, dim):
    v = 0.0
    dim1 = dim // 2
    if i == 0 and j == 0:
        v = 3/4
    elif i == 0 and j == 1:
        v = 1/4
    elif i == dim1 - 1 and j == dim - 1:
        v = 3/4
    elif i == dim1 - 1 and j == dim - 2:
        v = 1/4
    elif j == 2*i - 1 or j == 2*i + 1:
        v = 1/4
    elif j == 2*i:
        v = 1/2
    return v

@ti.func
def prolong_matrix(i, j, dim):
    v = 0.0
    dim2 = dim // 2
    if i == 0 and j == 0:
        v = 1
    elif i == 1 and j == 0:
        v = 1/2
    elif i == dim - 1 and j == dim2 - 1:
        v = 1
    elif i == dim - 2 and j == dim2 - 1:
        v = 1/2
    elif i == 2*j - 1 or i == 2*j + 1:
        v = 1/2
    elif i == 2*j:
        v = 1
    return v

@ti.data_oriented
class LinearSolver:
    jacobi = 1
    cg = 2
    mgpcg = 3
    sparse = 4
    def __init__(self, type, x, n, dim=3):
        self.type = type
        self.n = n
        self.dim = dim
        self.max_iter = 10
        self.smooth_iter = 2
        self.mg_levels = 4
        self.epsilon = 1e-5

        self.x = x
        if type == self.sparse:
            self.A = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=n * 7)
            self.b = ti.field(dtype=ti.f32, shape=self.n)
            self.solver = ti.linalg.SparseSolver(solver_type="LLT")
        elif type == self.cg:
            self.A = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(self.n, self.n))
            self.b = ti.Vector.field(dim, dtype=ti.f32, shape=self.n)
            self.r = ti.Vector.field(dim, dtype=ti.f32, shape=self.n)
            self.p = ti.Vector.field(dim, dtype=ti.f32, shape=self.n)
            self.Ap = ti.Vector.field(dim, dtype=ti.f32, shape=self.n)
        else:
            self.A = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(self.n, self.n))
            self.b = ti.Vector.field(dim, dtype=ti.f32, shape=self.n)
            self.r = [ti.Vector.field(dim, dtype=ti.f32, shape=self.n // (2 ** l)) for l in range(self.mg_levels)]
            self.z = [ti.Vector.field(dim, dtype=ti.f32, shape=self.n // (2 ** l)) for l in range(self.mg_levels)]
            self.p = ti.Vector.field(dim, dtype=ti.f32, shape=self.n)
            self.Ap = ti.Vector.field(dim, dtype=ti.f32, shape=self.n)

    def build_sparse(self):
        if self.type == self.sparse:
            A = self.A.build()
            self.solver.analyze_pattern(A)
            self.solver.factorize(A)

    @ti.kernel
    def init_rp(self):
        for i in range(self.n):
            self.r[i] = self.b[i]
            for j in range(self.n):
                self.r[i] -= self.A[i, j] @ self.x[j]
            self.p[i] = self.r[i]

    @ti.kernel
    def jacobi_iteration(self) -> ti.f32:
        e = 0.0
        for i in range(self.n):
            r = self.b[i]
            for j in range(self.n):
                r -= self.A[i, j] @ self.x[j]
            self.x[i] += self.A[i, i].inverse() @ r
            e += r.norm()
        return e

    @ti.kernel
    def cg_iteration(self) -> ti.f32:
        rr = 0.0
        pAp = 0.0
        rr1 = 0.0
        for i in range(self.n):
            rr += self.r[i].dot(self.r[i])
            self.Ap[i] = ti.Vector.zero(dt=ti.f32, n=self.dim)
            for j in range(self.n):
                self.Ap[i] += self.A[i, j] @ self.p[j]
            pAp += self.p[i].dot(self.Ap[i])
        alpha = rr / pAp
        for i in range(self.n):
            self.x[i] += alpha * self.p[i]
            self.r[i] -= alpha * self.Ap[i]
            rr1 += self.r[i].dot(self.r[i])
        beta = rr1 / rr
        for i in range(self.n):
            self.p[i] = self.r[i] + beta * self.p[i]
        return rr1

    # --------------MGPCG----------------
    @ti.func
    def A_coarsen(self, i, j, l):
        sum = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        dim = 2 ** l
        for m, n in ti.ndrange(dim, dim):
            sum += self.A[i*dim+m, j*dim+n]
        return sum / dim / dim

    @ti.kernel
    def init_mgpcg(self):
        for i in range(self.n):
            self.r[0][i] = self.b[i]
            for j in range(self.n):
                self.r[0][i] -= self.A[i, j] @ self.x[j]
        self.precondition()

    @ti.func
    def smooth(self, l: ti.template()):
        dim = self.n // (2 ** l)
        for i in range(dim):
            r = self.r[l][i]
            for j in range(dim):
                if i != j:
                    r -= self.A_coarsen(i, j, l) @ self.z[l][j]
            self.z[l][i] = self.A_coarsen(i, i, l).inverse() @ r

    @ti.func
    def coarsen(self, l: ti.template()):
        dim = self.n // (2 ** l)
        for i in range(dim):
            r = self.r[l][i]
            for j in range(dim):
                r -= self.A_coarsen(i, j, l) @ self.z[l][j]
            self.r[l+1][i // 2] += 0.5 * r

    @ti.func
    def refine(self, l: ti.template()):
        dim = self.n // (2 ** l)
        for i in range(dim):
            self.z[l][i] += self.z[l+1][i // 2]

    @ti.func
    def precondition(self):
        self.z[0].fill(0)
        for l in ti.static(range(self.mg_levels - 1)):
            for _ in range(self.smooth_iter):
                self.smooth(l)
            self.r[l+1].fill(0)
            self.z[l+1].fill(0)
            self.coarsen(l)
        for _ in range(self.smooth_iter):
            self.smooth(self.mg_levels - 1)
        for l in ti.static(range(self.mg_levels - 2, -1, -1)):
            self.refine(l)
            for _ in range(self.smooth_iter):
                self.smooth(l)

    @ti.kernel
    def mgpcg_iteration(self) -> ti.f32:
        zr = 0.0
        pAp = 0.0
        zr1 = 0.0
        rr = 0.0
        for i in range(self.n):
            zr += self.z[0][i].dot(self.r[0][i])
            self.Ap[i] = ti.Vector.zero(dt=ti.f32, n=self.dim)
            for j in range(self.n):
                self.Ap[i] += self.A[i, j] @ self.p[j]
            pAp += self.p[i].dot(self.Ap[i])
        alpha = zr / pAp
        for i in range(self.n):
            self.x[i] += alpha * self.p[i]
            self.r[0][i] -= alpha * self.Ap[i]
        self.precondition()
        for i in range(self.n):
            zr1 += self.z[0][i].dot(self.r[0][i])
        beta = zr1 / zr
        for i in range(self.n):
            self.p[i] = self.r[0][i] + beta * self.p[i]
            rr += self.r[0][i].dot(self.r[0][i])
        return rr

    def solve(self):
        # Jacobi
        if self.type == self.jacobi:
            for iter in range(self.max_iter):
                if self.jacobi_iteration() < self.epsilon:
                    break
        # Conjugate Gradient
        elif self.type == self.cg:
            self.init_rp()
            for iter in range(self.max_iter):
                if self.cg_iteration() < self.epsilon:
                    break
        # MGPCG
        elif self.type == self.mgpcg:
            self.init_mgpcg()
            self.p.copy_from(self.z[0])
            for iter in range(self.max_iter):
                if self.mgpcg_iteration() < self.epsilon:
                    break
        # sparse solver
        elif self.type == self.sparse:
            x = self.solver.solve(self.b)
            if self.solver.info():
                self.x.from_numpy(x)