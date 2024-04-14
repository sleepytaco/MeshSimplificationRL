import trimesh

mesh_name = "cow"
face_count = 500

# Load the input mesh
mesh = trimesh.load(f"meshes/{mesh_name}.obj")
print("before simplification:")
print("num vertices", len(mesh.vertices))
print("num faces", len(mesh.faces))
print("num edges", len(mesh.edges) // 2)

# Perform mesh simplification with the Quadric Edge Collapse Decimation (QECD) algorithm
simplified_mesh = mesh.simplify_quadric_decimation(face_count=face_count)
print("\nafter simplification:")
# Save the simplified mesh
simplified_mesh.export(f"meshes/simplified_{mesh_name}_mesh.obj")
mesh = trimesh.load(f"meshes/simplified_{mesh_name}_mesh.obj")
print("num vertices", len(mesh.vertices))
print("num faces", len(mesh.faces))
print("num edges", len(mesh.edges) // 2)

# 1000 faces bunny
# num vertices 502
# num faces 1000
# num edges 1500
# 1000 faces cow
# num vertices 501
# num faces 1000
# num edges 1500

# 800 faces bunny
# num vertices 402
# num faces 800
# num edges 1200

# 500 faces bunny
# num vertices 252
# num faces 500
# num edges 750
# 500 faces cow
# num vertices 251
# num faces 500
# num edges 750

# Render the mesh
import pyvista as pv

# Load the mesh from OBJ file
mesh = pv.read(f"meshes/simplified_{mesh_name}_mesh.obj")

# Plot the mesh
p = pv.Plotter(off_screen=True)
p.add_mesh(mesh)
p.camera_position = [(-30, 30, 30), (0, 0, 0), (0, 1, 0)]  # Example camera position (eye, focal point, view up)
p.show(screenshot="output_mesh.png")

