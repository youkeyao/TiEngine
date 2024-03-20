import taichi as ti
from utils.LinearSolver import LinearSolver
from utils.Scene import Scene

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
def bfecc(vf, qf, pos, dt):
    coord = back_trace_rk2(vf, pos, dt)
    x1 = trilerp(qf, coord[0], coord[1], coord[2])
    coord2 = back_trace_rk2(vf, coord, -dt)
    x2 = trilerp(qf, coord2[0], coord2[1], coord[2])
    return x1 + 0.5 * (x2 - sample(qf, pos[0], pos[1], pos[2]))

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