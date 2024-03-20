import taichi as ti
import numpy as np
from utils import Scene
from cases.CPDIPC.ccd import point_triangle_ccd, Point_Triangle_Distance_Unclassified

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
class CPDIPCSolver:
    name = "CPDIPC"
    def __init__(self, type, dt):
        self.dt = dt

        self.scene = Scene()
        self.scene.add_sphere([4, 4, 4], [0, 4, 0])
        self.scene.add_mat([10, 20, 10])

        self.stiffness = 5000
        self.outer_iter = 250
        self.inner_iter = 20
        self.epsilon = 1e-8
        self.dhat = 1e-2
        self.dhat2 = self.dhat ** 2

        self.n_verts = self.scene.n_verts
        self.n_elements = self.scene.n_elements
        self.n_faces = self.scene.n_faces
        self.n_edges = self.scene.n_edges

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.x_prev= ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=self.n_elements)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=self.n_faces)
        self.edges = ti.Vector.field(2, dtype=ti.i32, shape=self.n_edges)

        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.m = ti.field(dtype=ti.f32, shape=self.n_verts)
        self.fe = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.y = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.q = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.x_iter = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.D_m = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_elements)
        self.F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_elements)
        self.count = ti.field(dtype=ti.i32, shape=self.n_verts)
        self.sum_SiTSi = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n_verts, self.n_verts))
        # self.toi = ti.field(dtype=ti.f32, shape=self.n_verts)
        self.mind = ti.field(dtype=ti.f32, shape=self.n_verts)
        self.rebound = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.contact_fe = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

        self.P = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.L = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.H = ti.field(dtype=ti.f32, shape=())
        self.K = ti.field(dtype=ti.f32, shape=())
        self.sum_m = ti.field(dtype=ti.f32, shape=())
        self.dC = ti.Matrix.field(3, 7, dtype=ti.f32, shape=self.n_verts)
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.S = ti.Matrix.field(7, 7, dtype=ti.f32, shape=())
        self.Sb = ti.Vector.field(7, dtype=ti.f32, shape=())
        self.lak = ti.Vector.field(7, dtype=ti.f32, shape=())
        self.eps = 1e-3

        self.A = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n_verts, self.n_verts))
        self.A_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n_verts, self.n_verts))
        self.dgx = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

        self.reset()

    def reset(self):
        self.x.from_numpy(self.scene.x.astype(np.float32))
        self.elements.from_numpy(self.scene.elements.astype(np.int32))
        self.faces.from_numpy(self.scene.faces.astype(np.int32))
        self.edges.from_numpy(self.scene.edges.astype(np.int32))
        self.fe.from_numpy(self.scene.fe.astype(np.float32))
        self.m.from_numpy(self.scene.rho.astype(np.float32))

        self.init_field()

    @ti.kernel
    def init_field(self):
        # ------------Elements & m---------------------------------
        self.mind.fill(0)
        for e in self.elements:
            self.D_m[e] = self.D(e)
            self.F_B[e] = self.D_m[e].inverse()
            F_W = ti.abs(self.D_m[e].determinant()) / 6
            for i in ti.static(range(4)):
                self.mind[self.elements[e][i]] += F_W / 4
        self.sum_m.fill(0)
        for i in self.m:
            if self.mind[i] == 0:
                self.mind[i] = 1
            self.m[i] = self.mind[i] * self.m[i]
            self.sum_m[None] += self.m[i]
        # --------------forces---------------------------------
        for f in self.fe:
            self.fe[f] = self.fe[f] * self.m[f]
        self.contact_fe.fill(0)
        # ---------------dCp----------------------------------
        self.dC.fill(0)
        for i in self.m:
            self.dC[i][0, 0] = self.dC[i][1, 1] = self.dC[i][2, 2] = self.m[i] / self.dt
        # ---------------PLH----------------------------------------
        self.P[None].fill(0)
        self.L[None].fill(0)
        self.H[None] = 0
        self.alpha[None] = 0
        # ------------------------------------------------------------
        self.v.fill(0)
        self.rebound.fill(0)
        self.alpha.fill(0)
        self.compute_Si()
        self.compute_A()

    @ti.func
    def D(self, e) -> ti.Matrix:
        return ti.Matrix.cols([self.x[self.elements[e][i]] - self.x[self.elements[e][3]] for i in range(3)])
    @ti.func
    def compute_Si(self):
        I = ti.Matrix.identity(ti.f32, 3)
        self.sum_SiTSi.fill(0)
        for e in range(self.n_elements):
            for i in ti.static(range(4)):
                x_i = self.elements[e][i]
                self.sum_SiTSi[x_i, x_i] += I
    @ti.func
    def compute_dgx(self):
        for i in range(self.n_verts):
            self.dgx[i] = self.m[i] * (self.q[i] - self.y[i]) / (self.dt ** 2) + self.stiffness * self.sum_SiTSi[i, i] @ (self.q[i] - self.x[i]) - self.contact_fe[i]
    @ti.func
    def compute_A(self):
        I = ti.Matrix.identity(ti.f32, 3)
        for i, j in ti.ndrange(self.n_verts, self.n_verts):
            if i == j:
                self.A[i, j] = self.m[i] * I / (self.dt ** 2) + self.stiffness * self.sum_SiTSi[i, j]
                self.A_inv[i, j] = self.A[i, j].inverse()
            else:
                self.A[i, j].fill(0)
    @ti.func
    def compute_W(self) -> ti.f32:
        W = 0.0
        I = ti.Matrix.identity(ti.f32, 3)
        for e in range(self.n_elements):
            for i in ti.static(range(4)):
                W += self.stiffness / 2 * (self.x[self.elements[e][i]] - self.q[self.elements[e][i]]).norm_sqr()
        return W
    @ti.func
    def compute_Bt(self) -> ti.f32:
        H = 0.0
        for i in range(self.n_verts):
            H += -(self.mind[i] - self.dhat2) * (self.mind[i] - self.dhat2) * ti.log(self.mind[i] / self.dhat2)
        return H
    # -----------------------CCD-----------------------------------------------------------------

    @ti.func
    def CCD(self):
        # PT
        alpha = 1.0
        for i, j in ti.ndrange(self.n_verts, self.n_faces):
            f0 = self.faces[j][0]
            f1 = self.faces[j][1]
            f2 = self.faces[j][2]
            if i != f0 and i != f1 and i != f2:
                # pos
                x0 = self.q[i]
                x1 = self.q[f0]
                x2 = self.q[f1]
                x3 = self.q[f2]
                # vel
                v0 = self.x[i] - x0
                v1 = self.x[f0] - x1
                v2 = self.x[f1] - x2
                v3 = self.x[f2] - x3
                toi = point_triangle_ccd(x0, x1, x2, x3, v0, v1, v2, v3, 0.8, 1.0)
                ti.atomic_min(alpha, toi)
        for i in range(self.n_verts):
            v0 = self.x[i] - self.q[i]
            self.x[i] = self.q[i] + alpha * v0

    # ---------------------kernels------------------------------------------------

    @ti.kernel
    def compute_PCH(self):
        I = ti.Matrix.identity(ti.f32, 3)
        Interia = ti.Matrix.zero(ti.f32, 3, 3)
        for i in range(self.n_verts):
            self.P[None] += self.dt * (self.fe[i]+self.contact_fe[i])
            self.L[None] += self.dt * (self.fe[i]+self.contact_fe[i]).cross(self.x_prev[i])
            self.H[None] += self.dt * (self.fe[i]+self.contact_fe[i]).dot(self.v[i])
            Interia += self.m[i] * (self.x[i].dot(self.x[i]) * I - self.x[i].outer_product(self.x[i]))

            self.dC[i][0, 4] = self.m[i] * self.x_prev[i].z / self.dt
            self.dC[i][0, 5] = -self.m[i] * self.x_prev[i].y / self.dt
            self.dC[i][1, 3] = -self.m[i] * self.x_prev[i].z / self.dt
            self.dC[i][1, 5] = self.m[i] * self.x_prev[i].x / self.dt
            self.dC[i][2, 3] = self.m[i] * self.x_prev[i].y / self.dt
            self.dC[i][2, 4] = -self.m[i] * self.x_prev[i].x / self.dt
        self.K[None] = self.P[None].dot(self.P[None]) / 2 / self.sum_m[None] + (Interia.inverse()@self.L[None]).dot(self.L[None]) / 2

    @ti.kernel
    def init_x(self):
        for i in range(self.n_verts):
            self.x_prev[i] = self.x[i]
            self.q[i] = self.x[i]
            self.y[i] = self.x[i] + self.dt * self.v[i] + (self.dt ** 2) * (self.fe[i]+self.contact_fe[i]) / self.m[i]
            self.x[i] = self.y[i]

    @ti.kernel
    def barrier_projection(self):
        I = ti.Matrix.identity(ti.f32, 3)
        self.contact_fe.fill(0)
        self.mind.fill(self.dhat2)
        for i, j in ti.ndrange(self.n_verts, self.n_faces):
            f0 = self.faces[j][0]
            f1 = self.faces[j][1]
            f2 = self.faces[j][2]
            if i != f0 and i != f1 and i != f2:
                x0 = self.x[i]
                x1 = self.x[f0]
                x2 = self.x[f1]
                x3 = self.x[f2]
                d = Point_Triangle_Distance_Unclassified(x0, x1, x2, x3)
                n = ((x2 - x1).cross(x3 - x1)).normalized()
                if d < self.dhat2:
                    self.mind[i] = ti.min(self.mind[i], d)
                    self.mind[f0] = ti.min(self.mind[f0], d)
                    self.mind[f1] = ti.min(self.mind[f1], d)
                    self.mind[f2] = ti.min(self.mind[f2], d)
                    self.contact_fe[i] += n.dot(3*x0-(x1+x2+x3)) * n / (d + 1e-8)
                    self.contact_fe[f0] += -n.dot(x0-x1) * n / (d + 1e-8)
                    self.contact_fe[f1] += -n.dot(x0-x2) * n / (d + 1e-8)
                    self.contact_fe[f2] += -n.dot(x0-x3) * n / (d + 1e-8)

    @ti.kernel
    def local_step(self) -> ti.f32:
        # Jacobi Solver
        r = 0.0
        for i in range(self.n_verts):
            self.x_iter[i] = ti.Vector([0.0, 0.0, 0.0])
            self.count[i] = 1
        for e in range(self.n_elements):
            F = self.D(e) @ self.F_B[e]
            U, S, V = ssvd(F)
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
                self.x_iter[e3][n] += x3_new - self.x[e3][n]
                self.x_iter[e0][n] += x3_new + D_star[n, 0] - self.x[e0][n]
                self.x_iter[e1][n] += x3_new + D_star[n, 1] - self.x[e1][n]
                self.x_iter[e2][n] += x3_new + D_star[n, 2] - self.x[e2][n]

            self.count[e0] += 1
            self.count[e1] += 1
            self.count[e2] += 1
            self.count[e3] += 1

        for i in range(self.n_verts):
            r += self.x_iter[i].norm_sqr()
            self.x[i] = (self.x[i] + self.x_iter[i] / self.count[i])

        return r

    @ti.kernel
    def compute_S(self) -> ti.f32:
        # dgx
        self.compute_dgx()

        # S
        self.S.fill(0)
        P = ti.Vector.zero(ti.f32, 3)
        L = ti.Vector.zero(ti.f32, 3)
        H = 0.0
        for i in range(self.n_verts):
            dCh = self.dgx[i]+self.dt*self.m[i]*self.v[i]+self.dt*self.dt*(self.fe[i])
            self.dC[i][0, 6] = dCh[0]
            self.dC[i][1, 6] = dCh[1]
            self.dC[i][2, 6] = dCh[2]
            self.S[None] += self.dC[i].transpose() @ self.A_inv[i, i] @ self.dC[i]
            P += self.m[i] * (self.q[i] - self.x_prev[i]) / self.dt
            L += self.m[i] * self.q[i].cross(self.q[i] - self.x_prev[i]) / self.dt
            H += self.m[i] / 2 * (self.q[i] - self.x_prev[i]).norm_sqr() / self.dt / self.dt
        H += self.compute_W()
        # H += self.compute_Bt()
        self.S[None][6, 6] += (self.H[None] - self.K[None]) * (self.H[None] - self.K[None]) / self.eps
        # Sb
        Cp = P - self.P[None]
        Cl = L - self.L[None]
        Ch = H - (1-self.alpha[None]) * self.H[None] - self.alpha[None] * self.K[None]
        self.Sb[None] = ti.Vector([Cp[0], Cp[1], Cp[2], Cl[0], Cl[1], Cl[2], Ch])
        for i in range(self.n_verts):
            self.Sb[None] -= self.dC[i].transpose() @ self.A_inv[i, i] @ self.dgx[i]
        self.Sb[None][6] -= (self.H[None] - self.K[None]) / self.eps * self.eps * self.alpha[None]

        return H

    @ti.kernel
    def compute_q(self):
        self.lak[None] = self.S[None] @ self.Sb[None]
        for i in range(self.n_verts):
            self.x[i] = self.q[i] - self.A_inv[i, i] @ (self.dgx[i] + self.dC[i] @ self.lak[None])
        self.alpha[None] -= (self.eps * self.alpha[None] + (self.H[None] - self.K[None]) * self.lak[None][6]) / self.eps

    @ti.kernel
    def update_x(self) -> ti.f32:
        r = 0.0
        self.CCD()
        for i in range(self.n_verts):
            r += (self.x[i] - self.q[i]).norm_sqr()
            self.q[i] = self.x[i]
        return r
    @ti.kernel
    def update_v(self):
        for i in range(self.n_verts):
            self.v[i] = (self.x[i] - self.x_prev[i]) / self.dt

    def substep(self):
        self.barrier_projection()
        self.init_x()
        self.compute_PCH()
        for i in range(self.outer_iter):
            self.barrier_projection()
            for j in range(self.inner_iter):
                if self.local_step() < self.epsilon:
                    break
            self.compute_S()
            self.S.from_numpy(np.linalg.inv(self.S.to_numpy()))
            self.compute_q()
            if self.update_x() < self.epsilon:
                break
        self.update_v()