import taichi as ti
import numpy as np

def get_S(v1, v2, v3):
    a = ti.Vector(v1)
    b = ti.Vector(v2)
    c = ti.Vector(v3)
    ab = (a - b).norm()
    bc = (b - c).norm()
    ca = (c - a).norm()
    p = (ab + bc + ca) / 2
    return ti.sqrt(p * (p - ab) * (p - bc) * (p - ca))

def get_tri_index(tri_areas, random_area):
    sum_area = 0
    for i, area in enumerate(tri_areas):
        sum_area += area
        if random_area < sum_area:
            return i

def uniform_sample(radius, vertices, normals, faces, S):
    sample_vertices = []
    sample_normals = []
    tri_areas = []
    for i in range(faces.shape[0]):
        v1 = faces[i][0][0]
        v2 = faces[i][1][0]
        v3 = faces[i][2][0]
        n1 = faces[i][0][2]
        n2 = faces[i][1][2]
        n3 = faces[i][2][2]
        assert v1 >= 0 and v2 >= 0 and v3 >= 0 and n1 >= 0 and n2 >= 0 and n3 >= 0
        tri_areas.append(get_S(vertices[v1] * S, vertices[v2] * S, vertices[v3] * S))
    sum_area = np.array(tri_areas).sum()
    for _ in range(int(sum_area/radius/radius)):
        r = np.random.random_sample() * sum_area
        index = get_tri_index(tri_areas, r)
        v1 = vertices[faces[index][0][0]] * S
        v2 = vertices[faces[index][1][0]] * S
        v3 = vertices[faces[index][2][0]] * S
        n1 = normals[faces[index][0][2]]
        n2 = normals[faces[index][1][2]]
        n3 = normals[faces[index][2][2]]
        u = np.random.random_sample()
        v = np.random.random_sample()
        sample_v = v1 * (1 - ti.sqrt(u)) + v2 * (ti.sqrt(u) * (1 - v)) + v3 * (v * ti.sqrt(u))
        sample_n = n1 * (1 - ti.sqrt(u)) + n2 * (ti.sqrt(u) * (1 - v)) + n3 * (v * ti.sqrt(u))
        sample_vertices.append(sample_v)
        sample_normals.append(sample_n)
    return np.array(sample_vertices, dtype=np.float32), np.array(sample_normals, dtype=np.float32)