import taichi as ti
import numpy as np
from utils.Scene import read_obj
from cases.LBM.sample_utils import uniform_sample
from cases.LBM.solid_utils import get_solid_pos, get_solid_vel

D3Q19_e = np.array([
    [0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
    [1,1,0], [-1,-1,0], [1,-1,0], [-1,1,0], [1,0,1], [-1,0,-1], [1,0,-1],
    [-1,0,1], [0,1,1], [0,-1,-1], [0,1,-1], [0,-1,1]
], dtype=np.int32)
D3Q19_w = np.array([
    1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
], dtype=np.float32)
D3Q19_M = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
    [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
    [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
    [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
    [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
    [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
    [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
    [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
    [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
    [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
    [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
    [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
    [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]
], dtype=np.float32)
D3Q19_LR = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], dtype=np.int32)

@ti.data_oriented
class LBMSolver:
    name = "LBM"
    def __init__(self, type, dt):
        self.dt = dt
        self.grid_size = 0.02
        self.bound = ti.Vector([4.0, 4.0, 2.0])
        self.rho_threshold = 0
        self.grid_shape = (int(2 * self.bound.x / self.grid_size) + 1, int(self.bound.y / self.grid_size) + 1, int(2 * self.bound.z / self.grid_size) + 1)
        self.fluid_shape = (self.grid_shape[0] // 4, 10, 10)
        self.n = self.fluid_shape[0] * self.fluid_shape[1] * self.fluid_shape[2]
        self.Q = 19

        vertices, normals, faces = read_obj("assets/cube.obj")
        self.sample_vertices, self.sample_normals = uniform_sample(self.grid_size/2, vertices, normals, faces, [0.1, 1, 0.1])

        self.x_fluid = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.x_solid = ti.Vector.field(3, dtype=ti.f32, shape=self.sample_vertices.shape[0])
        self.n_solid = ti.Vector.field(3, dtype=ti.f32, shape=self.sample_normals.shape[0])
        self.Y_solid = ti.Vector.field(7, dtype=ti.f32, shape=()) # position, origentation
        self.dY_solid = ti.Vector.field(6, dtype=ti.f32, shape=()) # Linear velocity, Angular velocity
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n+self.sample_vertices.shape[0])
        self.f = ti.Vector.field(self.Q, ti.f32, shape=self.grid_shape)
        self.f_star = ti.Vector.field(self.Q, ti.f32, shape=self.grid_shape)
        self.rho = ti.field(ti.f32, shape=self.grid_shape)
        self.cutcells = ti.field(ti.u1, shape=self.grid_shape)
        self.projected_points = ti.field(ti.i32, shape=self.grid_shape)
        self.d = ti.field(ti.f32, shape=self.grid_shape)
        self.fe = ti.Vector.field(3, ti.f32,shape=self.grid_shape)
        self.v = ti.Vector.field(3, ti.f32, shape=self.grid_shape)
        self.e = ti.Vector.field(3, ti.i32, shape=(self.Q))
        self.w = ti.field(ti.f32, shape=(self.Q))
        self.LR = ti.field(ti.i32, shape=(self.Q))
        self.M = ti.Matrix.field(self.Q, self.Q, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(self.Q, self.Q, ti.f32, shape=())
        self.S = ti.Vector.field(self.Q, ti.f32, shape=())

        self.reset()

    def reset(self):
        self.x_solid.from_numpy(self.sample_vertices)
        self.n_solid.from_numpy(self.sample_normals)
        self.e.from_numpy(D3Q19_e)
        self.w.from_numpy(D3Q19_w)
        self.LR.from_numpy(D3Q19_LR)
        self.M.from_numpy(D3Q19_M)
        self.inv_M.from_numpy(np.linalg.inv(D3Q19_M))
        self.init_field()

    @ti.kernel
    def init_field(self):
        self.rho.fill(1)
        self.v.fill(0)
        for i, j, k in ti.ndrange((0, self.fluid_shape[0]), (0, self.grid_shape[1]), (0, self.grid_shape[2])):
            self.v[i, j, k] = ti.Vector([0.1, 0, 0])
        # -------------init Y-------------------------------
        self.Y_solid.fill(0)
        self.dY_solid.fill(0)
        self.Y_solid[None][1] = 2
        self.Y_solid[None][3] = 1
        # ------------init pos-----------------------
        self.x.fill(0)
        self.x_fluid.fill(0)
        for i, j, k in ti.ndrange((0, self.fluid_shape[0]), (0, self.fluid_shape[1]), (0, self.fluid_shape[2])):
            self.x_fluid[i*self.fluid_shape[1]*self.fluid_shape[2] + j*self.fluid_shape[2] + k] = [
                self.grid_size * i - self.bound.x,
                self.grid_size * (j + self.grid_shape[1] // 2 - self.fluid_shape[1] // 2 + 20),
                self.grid_size * (k + self.grid_shape[2] // 2 - self.fluid_shape[2] // 2) - self.bound.z
            ]
        # -------------init f-----------------------------
        for Index in ti.grouped(self.f):
            self.f[Index] = self.f_star[Index] = self.feq(self.rho[Index], self.v[Index])
        self.fe.fill(0)

        tau_f=0.16667/3.0+0.5
        s_v=1.0/tau_f
        s_other=8.0*(2.0-s_v)/(8.0-s_v)
        self.S[None] = ti.Vector([0,s_v,s_v,0,s_other,0,s_other,0,s_other, s_v, s_v,s_v,s_v,s_v,s_v,s_v,s_other,s_other,s_other])

    @ti.func
    def feq(self, rho, v) -> ti.types.vector():
        feq = ti.Vector.zero(ti.f32, self.Q)
        for s in ti.static(range(self.Q)):
            ev = self.e[s].dot(v)
            v2 = v.dot(v)
            feq[s] = self.w[s]*rho*(1.0+3.0*ev+4.5*ev*ev-1.5*v2)
        return feq

    @ti.func
    def projected_fe(self, fe, v) -> ti.types.vector():
        projected_fe = ti.Vector.zero(ti.f32, self.Q)
        for s in ti.static(range(self.Q)):
            projected_fe[s] = self.w[s] * (3.0*(self.e[s]-v)+9.0*self.e[s].dot(v)*self.e[s]).dot(fe)
        projected_fe = self.M[None] @ projected_fe
        projected_fe = self.inv_M[None] @ (projected_fe - 0.5 * self.S[None] * projected_fe)
        return projected_fe

    @ti.func
    def compute_id(self, Index, e) -> ti.types.vector():
        ip = Index + e
        for i in ti.static(range(3)):
            if ip[i] < 0:
                ip[i] = self.grid_shape[i] - 1
            if ip[i] > self.grid_shape[i] - 1:
                ip[i] = 0
        return ip

    # ---------------------------------------------------------------------------------------------------------------------

    @ti.kernel
    def calculate_cutcells(self):
        self.cutcells.fill(False)
        self.projected_points.fill(-1)
        self.d.fill(-1)
        for p in self.x_solid:
            pos = get_solid_pos(self.x_solid[p], self.Y_solid[None])
            id = (pos + ti.Vector([self.bound.x, 0, self.bound.z])) / self.grid_size
            id = ti.math.clamp(id, 0, ti.Vector([self.grid_shape[0]-2, self.grid_shape[1]-2, self.grid_shape[2]-2]))
            for Index in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                cell_id = ti.cast(id - 0.5 + Index, ti.i32)
                self.cutcells[cell_id] = True
                if (cell_id - id).dot(self.n_solid[p]) >= 0:
                    d = (cell_id - id).norm()
                    # find nearest node
                    if self.d[cell_id] < 0 or ti.atomic_min(self.d[cell_id], d) > d:
                        self.d[cell_id] = d
                        self.projected_points[cell_id] = p

    @ti.kernel
    def streaming(self):
        for Index in ti.grouped(ti.ndrange((1, self.grid_shape[0]-1), (1, self.grid_shape[1]-1), (1, self.grid_shape[2]-1))):
            for s in ti.static(range(self.Q)):
                ip = Index - self.e[s]

                # bounce back
                both_cutcells = self.cutcells[Index] and self.cutcells[ip]
                has_solid_cell = self.projected_points[Index] < 0 or self.projected_points[ip] < 0
                has_fluid_cell = self.projected_points[Index] >= 0 or self.projected_points[ip] >= 0
                thin_solid = self.projected_points[Index] >= 0 and self.projected_points[ip] >= 0 \
                    and self.n_solid[self.projected_points[Index]].dot(self.n_solid[self.projected_points[ip]]) < 0
                if (both_cutcells and has_fluid_cell and has_solid_cell) or thin_solid:
                    us = get_solid_vel(self.x_solid[self.projected_points[Index]], self.dY_solid[None])
                    self.f_star[Index][self.LR[s]] = self.f[Index][s] - 6 * self.w[s] * self.rho[Index] * us.dot(self.e[s])
                else:
                    self.f_star[Index][s] = self.f[ip][s]

    @ti.kernel
    def boundary_condition(self):
        for i, j, k in ti.ndrange((0, self.fluid_shape[0]), (0, self.grid_shape[1]), (0, self.grid_shape[2])):
            self.v[i, j, k] = ti.Vector([0.1, 0, 0])
            self.f[i, j, k] = self.f_star[i, j, k] = self.feq(self.rho[i, j, k], self.v[i, j, k])
        for i, j, k in self.v:
            if i == self.grid_shape[0] - 1:
                self.v[i, j, k].x = 0
            if j == 0 or j == self.grid_shape[1] - 1:
                self.v[i, j, k].y = 0
            if k == 0 or k == self.grid_shape[2] - 1:
                self.v[i, j, k].z = 0

    @ti.kernel
    def update_rho_v(self):
        for Index in ti.grouped(self.rho):
            self.rho[Index] = self.f_star[Index].sum()
            self.v[Index] = ti.Vector([0, 0, 0])

            for s in ti.static(range(self.Q)):
                self.v[Index] += self.e[s] * self.f_star[Index][s]

            # + F/2
            self.v[Index] /= self.rho[Index]

    @ti.kernel
    def calculate_velocity_correction(self):
        self.fe.fill(0)
        for i, j, k in self.fe:
            if self.cutcells[i, j, k]:
                uhat = ti.Vector.zero(float, 3)
                if self.projected_points[i, j, k] < 0:
                    pos = ti.Vector([
                        self.grid_size * i - self.bound.x - self.Y_solid[None][0],
                        self.grid_size * j - self.Y_solid[None][1],
                        self.grid_size * k - self.bound.z - self.Y_solid[None][2]
                    ])
                    uhat = get_solid_vel(pos, self.dY_solid[None])
                else:
                    xs = get_solid_pos(self.x_solid[self.projected_points[i, j, k]], self.Y_solid[None])
                    us = get_solid_vel(self.x_solid[self.projected_points[i, j, k]], self.dY_solid[None])
                    xf = ti.Vector([
                        self.grid_size * i - self.bound.x,
                        self.grid_size * j,
                        self.grid_size * k - self.bound.z
                    ])
                    direction = (xf - xs).normalized()

                    # hit the nearest axis-aligned cell face at point xg
                    t = ti.min(1/direction[0], 1/direction[1], 1/direction[2])
                    xg = xf + direction * t
                    id = (xg + ti.Vector([self.bound.x, 0, self.bound.z])) / self.grid_size

                    # if all the nodes on that nearest intersected cell face do not belong to any cut-cell
                    has_cutcells = False
                    for Index in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                        cell_id = ti.cast(id + Index, ti.i32)
                        if cell_id[0] < 0 or cell_id[0] > self.grid_shape[0]-1 \
                            or cell_id[1] < 0 or cell_id[1] > self.grid_shape[1]-1 \
                                or cell_id[2] < 0 or cell_id[2] > self.grid_shape[2]-1:
                            has_cutcells = True
                        else:
                            has_cutcells = has_cutcells or self.cutcells[cell_id]

                    # if so, linear interpolation
                    ug = ti.Vector.zero(float, 3)
                    if not has_cutcells:
                        w = [id-ti.cast(id, ti.i32), 1-id+ti.cast(id, ti.i32)]
                        for Index in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                            cell_id = ti.cast(id + Index, ti.i32)
                            ug += w[Index.x][0] * w[Index.y][1] * w[Index.z][2] * self.v[cell_id]
                    # else, discard
                    else:
                        xg = xf
                        ug = self.v[i, j, k]
                    
                    # calculate expected velocity
                    a = (xf-xs).norm() / (xg-xs).norm()
                    uhat = (1-a) * us + a * ug
                self.fe[i, j, k] = uhat - self.v[i, j, k]

    @ti.kernel
    def collision(self):
        for i, j, k in self.rho:
            m_temp = self.M[None] @ (self.f_star[i, j, k] - self.feq(self.rho[i, j, k], self.v[i, j, k]))
            projected_fe = self.projected_fe(self.fe[i, j, k], self.v[i, j, k])
            self.f[i, j, k] = self.f_star[i, j, k] - self.inv_M[None] @ (self.S[None]*m_temp) + projected_fe

    @ti.kernel
    def calculate_rigid(self):
        pass

    @ti.kernel
    def update_x(self):
        # fluid
        for p in self.x_fluid:
            pos = (self.x_fluid[p] + ti.Vector([self.bound.x, 0, self.bound.z])) / self.grid_size
            pos = ti.math.clamp(pos, 0, ti.Vector([self.grid_shape[0]-2, self.grid_shape[1]-2, self.grid_shape[2]-2]))
            id = ti.cast(pos-0.5, ti.i32)
            fx = pos - id
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k], dt=ti.i32)
                g_v = self.v[id + offset]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
            self.x_fluid[p] += self.dt * new_v * 10
        # all
        for p in self.x_fluid:
            self.x[p] = self.x_fluid[p]
        for p in self.x_solid:
            pos = get_solid_pos(self.x_solid[p], self.Y_solid[None])
            self.x[p+self.n] = pos

    def substep(self):
        self.calculate_cutcells()
        self.streaming()
        self.boundary_condition()
        self.update_rho_v()
        self.calculate_velocity_correction()
        self.collision()
        # TODO: calculate rigid
        self.calculate_rigid()
        self.update_x()