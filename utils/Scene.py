import taichi as ti
import numpy as np

def read_msh(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    nodes = lines[lines.index('$Nodes\n') + 3:lines.index('$EndNodes\n')]
    mesh_elements = lines[lines.index('$Elements\n') + 3:lines.index('$EndElements\n')]
    surfaces = lines[lines.index('$Surface\n') + 2:lines.index('$EndSurface\n')]
    x = np.array(list(map(lambda x: list(map(float, x[:-1].split(' ')[1:])), nodes)))
    elements = np.array(list(map(lambda x: list(map(int, x[:-1].split(' ')[1:])), mesh_elements))) - 1
    faces = np.array(list(map(lambda x: list(map(int, x[:-1].split(' '))), surfaces))) - 1
    return x, elements, faces

def count_edges(faces):
    edges = []
    faces_num = len(faces)
    for i in range(faces_num):
        f1 = faces[i][0]
        f2 = faces[i][1]
        f3 = faces[i][2]
        f1, f2, f3 = sorted([f1, f2, f3])
        edges.append([f1, f2])
        edges.append([f1, f3])
        edges.append([f2, f3])
    edges = np.array(edges)
    edges = np.unique(edges, axis=0)
    return edges

@ti.data_oriented
class Scene:
    def __init__(self):
        self.n_verts = 0
        self.n_elements = 0
        self.n_faces = 0
        self.n_edges = 0
        
        self.x = np.empty(shape=(0, 3))
        self.rho = np.empty(shape=(0,))
        self.elements = np.empty(shape=(0, 4))
        self.faces = np.empty(shape=(0, 3))
        self.edges = np.empty(shape=(0, 2))
        self.fe = np.empty(shape=(0, 3))
    
    def add_sphere(self, S=[1,1,1], T=[0,0,0]):
        sphere_x, sphere_elements, sphere_faces = read_msh("assets/sphere1K.msh")
        sphere_edges = count_edges(sphere_faces)
        self.x = np.append(self.x, sphere_x * S + T, axis=0)
        self.elements = np.append(self.elements, sphere_elements + self.n_verts, axis=0)
        self.faces = np.append(self.faces, sphere_faces + self.n_verts, axis=0)
        self.edges = np.append(self.edges, sphere_edges + self.n_verts, axis=0)
        self.fe = np.append(self.fe, np.array([[0, -9.8, 0]] * len(sphere_x)), axis=0)
        self.rho = np.append(self.rho, np.array([1e3] * len(sphere_x)), axis=0)
        self.n_verts += len(sphere_x)
        self.n_elements += len(sphere_elements)
        self.n_faces += len(sphere_faces)
        self.n_edges += len(sphere_edges)

    def add_floor(self, S=[1,1,1], T=[0,0,0]):
        self.x = np.append(self.x, np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]])*S+T, axis=0)
        self.faces = np.append(self.faces, np.array([[0, 1, 2], [2, 3, 0]]) + self.n_verts, axis=0)
        self.edges = np.append(self.edges, np.array([[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]]) + self.n_verts, axis=0)
        self.fe = np.append(self.fe, np.array([[0, 0, 0]] * 4), axis=0)
        self.rho = np.append(self.rho, np.array([1e8] * 4), axis=0)
        self.n_verts += 4
        self.n_faces += 2
        self.n_edges += 5

    def add_nail(self, S=[1,1,1], T=[0, 0, 0]):
        for i in range(-S[0], S[0] + 1):
            for j in range(-S[2], S[2] + 1):
                self.x = np.append(self.x, np.array([[i, 0, j], [i, 1, j]])+T, axis=0)
                self.edges = np.append(self.edges, np.array([[0, 1]]) + self.n_verts, axis=0)
                self.fe = np.append(self.fe, np.array([[0, 0, 0]] * 2), axis=0)
                self.rho = np.append(self.rho, np.array([1e8] * 2), axis=0)
                self.n_verts += 2
                self.n_edges += 1

    def add_mat(self, S=[1,1,1], T=[0, 0, 0]):
        mat_x, mat_elements, mat_faces = read_msh("assets/mat40x40.msh")
        mat_edges = count_edges(mat_faces)
        self.x = np.append(self.x, mat_x * S + T, axis=0)
        self.elements = np.append(self.elements, mat_elements + self.n_verts, axis=0)
        self.faces = np.append(self.faces, mat_faces + self.n_verts, axis=0)
        self.edges = np.append(self.edges, mat_edges + self.n_verts, axis=0)
        mat_fe = np.array([[0, -9.8, 0]] * len(mat_x))
        mat_rho = np.array([1e3] * len(mat_x))
        for i in range(len(mat_x)):
            if mat_x[i, 0] < -0.45 or mat_x[i, 0] > 0.45:
                mat_fe[i, 1] = 0
                mat_rho[i] = 1e8
        self.fe = np.append(self.fe, mat_fe, axis=0)
        self.rho = np.append(self.rho, mat_rho, axis=0)
        self.n_verts += len(mat_x)
        self.n_elements += len(mat_elements)
        self.n_faces += len(mat_faces)
        self.n_edges += len(mat_edges)

    def add_cube(self, S=[1,1,1], T=[0, 0, 0]):
        n = 5
        cube_x = np.zeros((n**3, 3), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    cube_x[i * n * n + j * n + k, 0] = i
                    cube_x[i * n * n + j * n + k, 1] = j
                    cube_x[i * n * n + j * n + k, 2] = k
        cube_elements = np.zeros(((n - 1)**3 * 5, 4), dtype=np.int32)
        for i in range(n - 1):
            for j in range(n - 1):
                for k in range(n - 1):
                    I = np.array([i, j, k])
                    e = ((i * (n-1) + j) * (n-1) + k) * 5
                    for u, v in enumerate([0, 3, 5, 6]):
                        verts = [v, v ^ 1, v ^ 2, v ^ 4]
                        for p in range(4):
                            t = I + (([verts[p] >> q for q in range(3)] ^ I) & 1)
                            cube_elements[e+u, p] = (t[0] * n + t[1]) * n + t[2]
                    verts = [1, 2, 4, 7]
                    for p in range(4):
                        t = I + (([verts[p] >> q for q in range(3)] ^ I) & 1)
                        cube_elements[e+4, p] = (t[0] * n + t[1]) * n + t[2]
        self.x = np.append(self.x, cube_x * S + T, axis=0)
        self.elements = np.append(self.elements, cube_elements + self.n_verts, axis=0)
        self.fe = np.append(self.fe, np.array([[0, -9.8, 0]] * len(cube_x)), axis=0)
        self.rho = np.append(self.rho, np.array([1e3] * len(cube_x)), axis=0)
        self.n_verts += len(cube_x)
        self.n_elements += len(cube_elements)