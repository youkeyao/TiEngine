import taichi as ti
import numpy as np

@ti.data_oriented
class Camera:
    def __init__(self, fov, aspect, n, f):
        self.position = ti.Vector([6.0, 7.0, 8.0])
        self.forward = (self.position - ti.Vector([0, 2, 0])).normalized()

        self.fov = fov
        self.aspect = aspect
        self.n = n
        self.f = f

        self.view_matrix = ti.Matrix([[0] * 5 for _ in range(5)], ti.f32)
        self.projection_matrix = ti.Matrix([[0] * 5 for _ in range(5)], ti.f32)

        self.initialize()

    def initialize(self):
        left = ti.Vector([0, 1, 0]).cross(self.forward)
        up = self.forward.cross(left)

        self.view_matrix[0, 0] = left.x
        self.view_matrix[0, 1] = left.y
        self.view_matrix[0, 2] = left.z
        self.view_matrix[0, 3] = -left.x * self.position.x - left.y * self.position.y - left.z * self.position.z
        self.view_matrix[1, 0] = up.x
        self.view_matrix[1, 1] = up.y
        self.view_matrix[1, 2] = up.z
        self.view_matrix[1, 3] = -up.x * self.position.x - up.y * self.position.y - up.z * self.position.z
        self.view_matrix[2, 0] = self.forward.x
        self.view_matrix[2, 1] = self.forward.y
        self.view_matrix[2, 2] = self.forward.z
        self.view_matrix[2, 3] = -self.forward.x * self.position.x - self.forward.y * self.position.y - self.forward.z * self.position.z
        self.view_matrix[3, 3] = 1

        self.projection_matrix[0, 0] = 1 / self.aspect / ti.tan(self.fov / 2)
        self.projection_matrix[1, 1] = 1 / ti.tan(self.fov / 2)
        self.projection_matrix[2, 2] = (self.n + self.f) / (self.n - self.f)
        self.projection_matrix[2, 3] = 2 * self.n * self.f / (self.n - self.f)
        self.projection_matrix[3, 2] = -1

    # 3d pos -> ndc
    def T(self, pos):
        M = self.projection_matrix @ self.view_matrix
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        ndc_x = M[0, 0] * x + M[0, 1] * y + M[0, 2] * z + M[0, 3]
        ndc_y = M[1, 0] * x + M[1, 1] * y + M[1, 2] * z + M[1, 3]
        ndc_w = M[3, 0] * x + M[3, 1] * y + M[3, 2] * z + M[3, 3]
        ndc_x /= ndc_w
        ndc_y /= ndc_w
        ndc_x += 1
        ndc_x /= 2
        ndc_y += 1
        ndc_y /= 2
        return np.array([ndc_x, ndc_y]).swapaxes(0, 1)