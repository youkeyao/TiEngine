import taichi as ti
from cases.CPDIPC.ipc_utils import d_type_PT, d_type_EE, d_PP, d_PE, d_PT, d_EE

MAX_ITER = 10

@ti.func
def Point_Triangle_Distance_Unclassified(p: ti.math.vec3, t0: ti.math.vec3, t1: ti.math.vec3, t2: ti.math.vec3) -> ti.f32:
    dtype = d_type_PT(p, t0, t1, t2)
    d = 0.0
    if dtype == 0:
        d = d_PP(p, t0)

    elif dtype == 1:
        d = d_PP(p, t1)

    elif dtype == 2:
        d = d_PP(p, t2)

    elif dtype == 3:
        d = d_PE(p, t0, t1)

    elif dtype == 4:
        d = d_PE(p, t1, t2)

    elif dtype == 5:
        d = d_PE(p, t2, t0)

    elif dtype == 6:
        d = d_PT(p, t0, t1, t2)

    return d


@ti.func
def Edge_Edge_Distance_Unclassified(ea0: ti.math.vec3, ea1: ti.math.vec3, eb0: ti.math.vec3, eb1: ti.math.vec3)->ti.f32:
    d_type = d_type_EE(ea0, ea1, eb0, eb1)
    d = 0.0
    if d_type == 0:
        d = d_PP(ea0, eb0)

    elif d_type == 1:
        d = d_PP(ea0, ea1)

    elif d_type == 2:
        d = d_PE(ea0, eb0, eb1)

    elif d_type == 3:
        d = d_PP(ea1, eb0)

    elif d_type == 4:
        d = d_PP(ea1, eb1)

    elif d_type == 5:
        d = d_PE(ea1, eb0, eb1)

    elif d_type == 6:
        d = d_PE(eb0, ea0, ea1)

    elif d_type == 7:
        d = d_PE(eb1, ea0, ea1)

    elif d_type == 8:
        d = d_EE(ea0, ea1, eb0, eb1)

    return d

@ti.func
def point_triangle_ccd(p  : ti.math.vec3,
                       t0 : ti.math.vec3,
                       t1 : ti.math.vec3,
                       t2 : ti.math.vec3,
                       dp : ti.math.vec3,
                       dt0: ti.math.vec3,
                       dt1: ti.math.vec3,
                       dt2: ti.math.vec3,
                       eta: ti.f32,
                       toc: ti.f32):
    mov = 0.25 * (dt0 + dt1 + dt2 + dp)
    dt0 -= mov
    dt1 -= mov
    dt2 -= mov
    dp -= mov

    maxDispMag = dp.norm() + ti.math.sqrt(ti.max(dt0.dot(dt0), dt1.dot(dt1), dt2.dot(dt2)))

    t = 1.0
    if (maxDispMag > 0.0):
        dist2_cur = Point_Triangle_Distance_Unclassified(p, t0, t1, t2)

        dist_cur = ti.sqrt(dist2_cur)
        # print(f'{dist_cur}')

        gap = eta * (dist2_cur) / (dist_cur)

        t = 0

        i = 0
        for iter in range(MAX_ITER):
            i += 1
            tocLowerBound = (1 - eta) * (dist2_cur) / ((dist_cur) * maxDispMag)
            # print(f'{tocLowerBound}')
            p  += tocLowerBound * dp
            t0 += tocLowerBound * dt0
            t1 += tocLowerBound * dt1
            t2 += tocLowerBound * dt2

            dist2_cur = Point_Triangle_Distance_Unclassified(p, t0, t1, t2)
            dist_cur = ti.math.sqrt(dist2_cur)
            if t > 0.0 and (dist2_cur) / (dist_cur) < gap:
                break

            t += tocLowerBound
            # print(t)
            if (t > toc):
                t = toc
                # print(i)
                break

    return t

@ti.kernel
def edge_edge_ccd( ea0  : ti.math.vec3,
                   ea1: ti.math.vec3,
                   eb0: ti.math.vec3,
                   eb1: ti.math.vec3,
                   dea0: ti.math.vec3,
                   dea1: ti.math.vec3,
                   deb0: ti.math.vec3,
                   deb1: ti.math.vec3,
                   eta: ti.f32, #0.1
                   thickness: ti.f32,
                   toc: ti.f32) -> ti.f32:

    mov = 0.25 * (dea0 + dea1 + deb0 + dea1)
    dea0 -= mov
    dea1 -= mov
    deb0 -= mov
    deb1 -= mov

    maxDispMag = ti.math.sqrt(ti.max(dea0.dot(dea0), dea1.dot(dea1))) + ti.math.sqrt(ti.max(deb0.dot(deb0), deb1.dot(deb1)))
    if maxDispMag > 0.0:

        dist2_cur = Edge_Edge_Distance_Unclassified(ea0, ea1, eb0, eb1)
        dFunc = dist2_cur - thickness * thickness
        if dFunc <= 0.0:
            dist2_cur = ti.min((ea0 - eb0).dot(ea0 - eb0), (ea0 - eb1).dot(ea0 - eb1), (ea1 - eb0).dot(ea1 - eb0), (ea1 - eb1).dot(ea1 - eb1))
            dFunc = dist2_cur - thickness * thickness

        dist_cur = ti.math.sqrt(dist2_cur)
        gap = eta * dFunc / (dist_cur + thickness)

        t = 0
        while True:

            tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) / ((dist_cur + thickness) * maxDispMag)
            # print(f'{tocLowerBound}')
            ea0 += tocLowerBound * dea0
            ea1 += tocLowerBound * dea1
            eb0 += tocLowerBound * deb0
            eb1 += tocLowerBound * deb1

            dist2_cur = Edge_Edge_Distance_Unclassified(ea0, ea1, eb0, eb1)
            dFunc = dist2_cur - thickness * thickness
            if (dFunc <= 0):
                dist2_cur = ti.math.min((ea0 - eb0).dot(ea0 - eb0), (ea0 - eb1).dot(ea0 - eb1), (ea1 - eb0).dot(ea1 - eb0), (ea1 - eb1).dot(ea1 - eb1))
                dFunc = dist2_cur - thickness * thickness

            dist_cur = ti.sqrt(dist2_cur)

            if t > 0.0 and (dFunc - thickness ** 2) / (dist_cur + thickness) < gap:
                break

            t += tocLowerBound
            # print(t)
            if (t > toc):
                t = toc
                break

    return t