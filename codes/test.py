nx = 20
ny = 15
nz = 5

origin = (-(nx - 1)*0.1/2, -(ny - 1)*0.1/2, -(nz - 1)*0.1/2)
mesh = pv.UniformGrid((nx, ny, nz), (.1, .1, .1), origin)
x = mesh.points[:, 0]
y = mesh.points[:, 1]
z = mesh.points[:, 2]
vectors = np.empty((mesh.n_points, 3))
vectors[:, 0] = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
vectors[:, 1] = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
vectors[:, 2] = (np.sqrt(3.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
                 np.sin(np.pi * z))

mesh['vectors'] = vectors

stream, src = mesh.streamlines('vectors', return_source=True,
                               terminal_speed=0.1, n_points=200,
                               source_radius=0.1)

cpos = [(1.2, 1.2, 1.2), (1, 1, 1), (0, 1, 1.0)]
stream.tube(radius=0.0015).plot(cpos=cpos)