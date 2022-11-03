import taichi as ti

@ti.data_oriented
class Camera:
    def __init__(self, fov, aspect, n, f):
        self.position = ti.Vector([6.0, 7.0, 8.0])
        self.forward = (self.position - ti.Vector([0, 2, 0])).normalized()

        self.fov = fov
        self.aspect = aspect
        self.n = n
        self.f = f

        self.view_matrix = ti.Matrix([[0] * 5 for _ in range(5)], ti.i32)
        self.projection_matrix = ti.Matrix([[0] * 5 for _ in range(5)], ti.i32)

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