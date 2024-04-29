import trimesh

# Helper file to generate "ideal" simplified meshes using an off-the-shelf mesh library
mesh_name = "sphere"
face_count = 200

# Load the input mesh
mesh = trimesh.load(f"meshes/{mesh_name}.obj")
print("before simplification:")
print("num vertices", len(mesh.vertices))
print("num faces", len(mesh.faces))
print("num edges", len(mesh.edges) // 2)

if len(mesh.faces) > face_count:
    # Perform mesh simplification with the Quadric Edge Collapse Decimation (QECD) algorithm
    simplified_mesh = mesh.simplify_quadric_decimation(face_count=face_count)
    print("\nafter simplification:")
    # Save the simplified mesh
    simplified_mesh.export(f"meshes/{mesh_name}_{face_count}f_trimesh.obj")
    # mesh = trimesh.load(f"meshes/{mesh_name}_{face_count}f.obj")
    print("num vertices", len(simplified_mesh.vertices))
    print("num faces", len(simplified_mesh.faces))
    print("num edges", len(simplified_mesh.edges) // 2)
else:
    print("input mesh already has faces <", face_count)

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
# import pyvista as pv
#
# # Load the mesh from OBJ file
# mesh = pv.read(f"meshes/simplified_{mesh_name}_mesh.obj")
#
# # Plot the mesh
# p = pv.Plotter(off_screen=True)
# p.add_mesh(mesh)
# p.camera_position = [(-30, 30, 30), (0, 0, 0), (0, 1, 0)]  # Example camera position (eye, focal point, view up)
# p.show(screenshot="output_mesh.png")

