import numpy as np
import open3d as o3d

def read_point_cloud(path):
    # Initialize an empty PointCloud for safety
    point_cloud = o3d.geometry.PointCloud()

    if path.endswith(".ply") or path.endswith(".pcd"):
        point_cloud = o3d.io.read_point_cloud(path)
    elif path.endswith(".csv"):
        point_cloud_data = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=(1, 2, 3))
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    elif path.endswith(".bin"):
        with open(path, "rb") as file:
            data = file.read()
        array = np.frombuffer(data, dtype=np.float32)
        num_points = array.shape[0] // 4
        points = np.reshape(array, (num_points, 4))[:, :3]  # Extract XYZ coordinates
        point_cloud.points = o3d.utility.Vector3dVector(points)
    elif path.endswith(".off"):
        with open(path, 'r') as f:
            lines = f.readlines()

        # Check if the format is incorrect
        if "OFF" in lines[0] and not lines[0].startswith("OFF\n"):
            # Split the line after "OFF"
            header_line = lines[0].replace("OFF", "").strip()
            corrected_lines = ["OFF\n", header_line + "\n"] + lines[1:]

            # Write the corrected content back to the file
            with open(path, 'w') as f:
                f.writelines(corrected_lines)

        mesh = o3d.io.read_triangle_mesh(path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=40000)
                
    else:
        print("Unsupported file format!")
    return point_cloud
