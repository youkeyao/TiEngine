import taichi as ti

@ti.func
def get_solid_pos(x_solid : ti.types.vector(3), Y_solid : ti.types.vector(7)) -> ti.types.vector(3):
    pos = ti.Vector([Y_solid[0], Y_solid[1], Y_solid[2]])
    w = Y_solid[3]
    x = Y_solid[4]
    y = Y_solid[5]
    z = Y_solid[6]

    R = ti.Matrix.zero(ti.f32, 3, 3)
    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y + w*z)
    R[0, 2] = 2 * (x*z - w*y)
    R[1, 0] = 2 * (x*y - w*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z + w*x)
    R[2, 0] = 2 * (x*z + w*y)
    R[2, 1] = 2 * (y*z - w*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)

    pos += R @ x_solid

    return pos

@ti.func
def get_solid_vel(x_solid : ti.types.vector(3), dY_solid : ti.types.vector(6)) -> ti.types.vector(3):
    v = ti.Vector([dY_solid[0], dY_solid[1], dY_solid[2]])
    w = ti.Vector([dY_solid[3], dY_solid[4], dY_solid[5]])

    return w.cross(x_solid) + v