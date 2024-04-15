import trimesh
import plotly.graph_objects as go
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt


def plot_3d_mesh_with_diaphragm_line_plotly(mesh, diaphragm_line_points):
    # Plot the mesh vertices
    trace_mesh = go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',  # Mesh vertices color
        ),
        name='Mesh Vertices')

    # Plot the diaphragm line if points are provided
    if diaphragm_line_points is not None and len(diaphragm_line_points) > 0:
        trace_line = go.Scatter3d(
            x=diaphragm_line_points[:, 0],
            y=diaphragm_line_points[:, 1],
            z=diaphragm_line_points[:, 2] + 2,
            mode='lines',
            line=dict(
                color='blue',  # Diaphragm line color
                width=5),
            name='Diaphragm Line')
        data = [trace_mesh, trace_line]
    else:
        data = [trace_mesh]

    # Define layout
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0),
                       scene=dict(
                           xaxis=dict(title='X'),
                           yaxis=dict(title='Y'),
                           zaxis=dict(title='Z'),
                       ))

    fig = go.Figure(data=data, layout=layout)
    fig.show()

def convert_to_axial(coronal_point, slice):
    axial_point = np.copy(coronal_point)
    axial_point[0] = (coronal_point[0] + slice.ImagePositionPatient[2]) / slice.SliceThickness
    return axial_point

def line_plane_intersection(plane_point, plane_normal, line_points):
    p0, n = np.array(plane_point), np.array(plane_normal)
    p1, p2 = np.array(line_points[0]), np.array(line_points[1])
    line_vec = p2 - p1
    t = np.dot((p0 - p1), n) / np.dot(line_vec, n)
    if 0 <= t <= 1:
        return p1 + t * line_vec
    return None


def mesh_plane_intersection(mesh, plane_point, plane_normal, slice):
    intersection_points = []
    for triangle in mesh.vertices[mesh.faces]:
        for i in range(3):
            point1, point2 = triangle[i], triangle[(i +1) % 3]
            intersect_point = line_plane_intersection(plane_point, plane_normal,
                                                      [point1, point2])
            if intersect_point is not None:
                intersect_point_dicom = np.array([
                    intersect_point[2],  # Z coordinate (assuming it's DICOM Z coordinate)
                    intersect_point[1],  # X coordinate
                    intersect_point[0]   # Y coordinate
                ])
                intersection_points.append(intersect_point_dicom)
    return np.array(intersection_points)


def plot_axial(intersection_points, image_3d, slice_0,x_index):
    axial_intersection_points = np.copy(intersection_points)
    # for i in range(axial_intersection_points.shape[0]):
    #     axial_intersection_points[i, 2] = axial_intersection_points[i, 2] / slice_0.PixelSpacing[1]  # Y coordinate
    #     axial_intersection_points[i, 1] = axial_intersection_points[i, 1] / slice_0.PixelSpacing[0]  # X coordinate
    #     axial_intersection_points[i, 0] = (axial_intersection_points[
    #                                            i, 0] + slice_0.SliceLocation) / slice_0.SliceThickness  # Z coordinate
    print("X: ", axial_intersection_points[:, 0])
    print("Y: ", axial_intersection_points[:, 1])
    print("Z: ", axial_intersection_points[:, 2])


    axial_slice = image_3d[x_index,:, :]
    print("Slice index", x_index)
    pixel_size = axial_slice.shape
    plt.imshow(axial_slice, cmap='gray', origin='upper', extent=[0, pixel_size[0], 0, pixel_size[1]])
    # Overlay intersection points
    # for point in intersection_points:
    #     plt.scatter(point[0], point[1], color='red', marker='o')
    if intersection_points is not None and len(intersection_points) > 0:
        plt.scatter(axial_intersection_points[:, 2], axial_intersection_points[:, 0], s=10, color='red')
    plt.show()


def plot_coronal_slice_with_normalized_intersection(patient_id, image_3d,slice_index,intersection_points,
                                                    slice_0, output_dir):
    for i in range(intersection_points.shape[0]):
        intersection_points[
            i, 0] = intersection_points[i, 0] / slice_0.PixelSpacing[0]
        intersection_points[
            i, 1] = intersection_points[i, 1] / slice_0.PixelSpacing[0]
        intersection_points[i, 2] = (intersection_points[i, 2] + 5) / slice_0.SliceThickness

    if slice_index < 0 or slice_index >= image_3d.shape[1]:
        print(f"Slice index {slice_index} is out of bounds for the image data.")
        return
    coronal_slice = image_3d[:, slice_index, :]
    pixel_size = coronal_slice.shape
    plt.figure()
    plt.imshow(coronal_slice,cmap='gray',origin='upper', extent=[0, pixel_size[0], 0, pixel_size[1]])
    plt.title(f'{patient_id} Coronal Slice at Index {slice_index} with Diaphragmatic Line')
    if intersection_points is not None and len(intersection_points) > 0:
        plt.scatter(intersection_points[:, 1],
                    intersection_points[:, 2],
                    s=10,
                    color='red')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    # plt.show()

    output_path = os.path.join(output_dir, f"{slice_index}.png")
    plt.savefig(output_path)
    plt.close()




def plot_axial_slice_with_normalized_intersection(patient_id, image_3d, slice_index, intersection_points,
                                                  slice_0, output_dir):
    # Normalize intersection points to match DICOM coordinates
    for i in range(intersection_points.shape[0]):
        # # Normalize Z coordinate (slice thickness)
        # intersection_points[i, 0] /= slice_0.SliceThickness
        # # Normalize X and Y coordinates (pixel spacing)
        # intersection_points[i, 1] /= slice_0.PixelSpacing[0]
        # intersection_points[i, 2] /= slice_0.PixelSpacing[1]

        # Normalize Z coordinate (slice position)
        intersection_points[i, 0] = (intersection_points[i, 0] + slice_0.ImagePositionPatient[2]) / slice_0.SliceThickness
        # Normalize X coordinate (pixel spacing)
        intersection_points[i, 1] /= slice_0.PixelSpacing[0]
        # Normalize Y coordinate (pixel spacing)
        intersection_points[i, 2] /= slice_0.PixelSpacing[1]

    if slice_index < 0 or slice_index >= image_3d.shape[0]:
        print(f"Slice index {slice_index} is out of bounds for the image data.")
        return

    # Plot axial slice
    axial_slice = image_3d[slice_index-70, :, :]
    pixel_size = axial_slice.shape
    plt.figure()
    plt.imshow(axial_slice, cmap='gray', origin='upper', extent=[0, pixel_size[1], 0, pixel_size[0]])
    plt.title(f'{patient_id} Axial Slice at Index {slice_index} with Intersection Points')

    if intersection_points is not None and len(intersection_points) > 0:
        plt.scatter(intersection_points[:, 1],  # X coordinate
                    intersection_points[:, 2],  # Y coordinate
                    s=10,
                    color='red')

    # Save and close the plot
    output_path = os.path.join(output_dir, f"{slice_index}.png")
    plt.savefig(output_path)
    plt.close()


def load_scan(directory):
    slices = [
        pydicom.read_file(os.path.join(directory, s))
        for s in os.listdir(directory)
    ]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


# Function to plot intersection points in 3D
def plot_intersection_points_3d(intersection_points, output_path=None):
    fig = go.Figure()

    # Add mesh vertices
    fig.add_trace(go.Scatter3d(
        x=diaphragm_mesh.vertices[:, 0],
        y=diaphragm_mesh.vertices[:, 1],
        z=diaphragm_mesh.vertices[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
        ),
        name='Mesh Vertices'
    ))

    # Add intersection points
    if intersection_points is not None and len(intersection_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=intersection_points[:, 0],
            y=intersection_points[:, 1],
            z=intersection_points[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
            ),
            name='Intersection Points'
        ))

    # Set layout
    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    ))

    # Show or save the plot
    if output_path:
        fig.write_html(output_path)
    else:
        fig.show()


# def convert_mesh_to_dicom_coordinate_system(vertices, image_orientation_patient, slice_thickness):
#     # Convert mesh vertices to DICOM coordinate system
#     vertices_dicom = np.dot(vertices, image_orientation_patient)
#     # Scale Z-coordinate by slice thickness to match DICOM coordinates
#     vertices_dicom[:, 2] *= slice_thickness
#     return vertices_dicom

# Directory where your DICOM files are stored
patient_id = '10003382'
respiration_phase = 'ex'
dicom_dir = f'/Users/Pascal/Downloads/ct-based-diaphragm-function-evaluation-main/images/{patient_id}/{respiration_phase}'
intermediate_dir = f'/Users/Pascal/Downloads/ct-based-diaphragm-function-evaluation-main/result/{patient_id}/{respiration_phase}'

# Load the scans in given folder, and get the pixel values in HU
slices = load_scan(dicom_dir)
image_3d = get_pixels_hu(slices)

# Load the 3D mesh
diaphragm_mesh = trimesh.load_mesh(
    os.path.join(intermediate_dir,
                 '11.lung_diaphragm_contact_surface_mesh(manually).ply'))

max_x = int(np.max(diaphragm_mesh.vertices[:, 0]) / slices[0].PixelSpacing[0])
min_x = int(np.min(diaphragm_mesh.vertices[:, 0]) / slices[0].PixelSpacing[0])

lung_base_mesh = trimesh.load_mesh(
    os.path.join(intermediate_dir, '2.lung_base_3d_point_cloud.ply'))

output_dir = '/Users/Pascal/Downloads/coronal/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
x_index = 90
dicom_slice = slices[x_index]
slice_image = dicom_slice.pixel_array
x0 = x_index
# x0 = x_index * slices[0].PixelSpacing[0]
pixel_spacing = slices[0].PixelSpacing[0]
slice_thickness = slices[0].SliceThickness
image_orientation_patient = slices[0].ImageOrientationPatient
dicom_orientation = np.array(image_orientation_patient).reshape((2, 3)).T


# # possible plane point for coronal
# plane_normal = [1, 0, 0]
# plane_point = [x0, 0, 0]

# Define parameters for axial plane
plane_normal = [1, 0, 0]  # Normal vector for axial plane
plane_point = [x0,0,0]  # Point on the axial plane (slice index determines z-coordinate)
# plane_point = [x0 * slice_thickness,0,0]  # Point on the axial plane (slice index determines z-coordinate)

print(pixel_spacing)
print(slice_thickness)
print(image_orientation_patient)
print(dicom_orientation)

print("plane normal", plane_normal)
print("plane point", plane_point)


# Convert mesh vertices to DICOM coordinate system
# mesh_vertices_dicom = convert_mesh_to_dicom_coordinate_system(diaphragm_mesh.vertices, image_orientation_patient, slice_thickness)

# intersection_points = mesh_plane_intersection(diaphragm_mesh, plane_point, plane_normal,slices[0])
intersection_points = mesh_plane_intersection(diaphragm_mesh, plane_point, plane_normal,slices[x_index])
# intersection_points = mesh_plane_intersection(mesh_vertices_dicom, plane_point, plane_normal,slices[0])
# Plot intersection points in 3D
# plot_intersection_points_3d(intersection_points)
print("Intersection points:", intersection_points)
# plot_axial(intersection_points,image_3d, slices[0],x_index)
# plot_axial_slice_with_normalized_intersection(patient_id, image_3d, x_index, intersection_points, slices[0], output_dir)
# plot_axial_slice_with_normalized_intersection(patient_id, image_3d, x_index, intersection_points, slices[x_index], output_dir)
plot_coronal_slice_with_normalized_intersection(patient_id, image_3d,slices[x_index],intersection_points,
                                                    slices[0], output_dir)
