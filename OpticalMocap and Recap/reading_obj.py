import trimesh
import numpy as np
from PIL import Image

mesh = trimesh.load("/home/giuliamartinelli/Documents/SportTech/OBJs/simplified_Ball.obj",force='mesh')

# Load texture (image)
texture_image = Image.open('/home/giuliamartinelli/Documents/SportTech/OBJs/lambert2SG_Base_Color.png')

# Apply texture by setting the visual property of the mesh
uv_coordinates = np.array([mesh.vertices[:, 0], mesh.vertices[:, 1]]).T  # Example UV mapping
# Ensure the UV coordinates are within the range [0, 1]
uv_coordinates = uv_coordinates - np.floor(uv_coordinates)

# Create a visual object with the texture
texture_visual = trimesh.visual.texture.TextureVisuals(uv=uv_coordinates, image=texture_image)

# Create a translation matrix to move the mesh by 10 cm along the x-axis
translation_matrix = trimesh.transformations.translation_matrix([0.1, 0, 0])

# Apply the transformation to the mesh
mesh.apply_transform(translation_matrix)

# Create a rotation matrix to rotate by 90 degrees (π/2 radians)
# For rotation along the z-axis:
rotation_matrix = trimesh.transformations.rotation_matrix(
    angle=np.pi / 2,  # 90 degrees in radians
    direction=[0, 0, 1],  # z-axis
    point=mesh.centroid  # Rotate around the mesh's centroid
)

# Apply the rotation to the mesh
mesh.apply_transform(rotation_matrix)

# Apply the texture to the mesh
# mesh.visual = texture_visual

# print(mesh.texture)
mesh.show()
