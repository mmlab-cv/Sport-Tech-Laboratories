import numpy as np

def euler_to_rotation_matrix(euler):
    """Convert Euler angles (in radians) to a rotation matrix."""
    roll, pitch, yaw = euler
    
    # Rotation matrices for each axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    return R

def rotation_matrix_to_euler(R):
    """Convert a rotation matrix to Euler angles (in radians)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    if sy > 1e-6:  # Non-gimbal lock case
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:  # Gimbal lock case
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])  # Roll, Pitch, Yaw

def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion."""
    w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return np.array([w, x, y, z])

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a rotation matrix."""
    w, x, y, z = q
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                   [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
                   [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]])
    return R

def quaternion_to_euler(q):
    """Convert a quaternion to Euler angles (in radians)."""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])  # Roll, Pitch, Yaw

def euler_to_quaternion(euler):
    """Convert Euler angles (in radians) to a quaternion."""
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

# Example angles (in radians)
pitch = np.radians(30)  # Rotation around the x-axis
yaw = np.radians(45)    # Rotation around the y-axis
roll = np.radians(60)   # Rotation around the z-axis

# Create an Euler array
euler_angles = np.array([roll, pitch, yaw])

# Convert Euler to rotation matrix
rotation_matrix_from_euler = euler_to_rotation_matrix(euler_angles)

# Convert rotation matrix to quaternions
quaternion_from_matrix = rotation_matrix_to_quaternion(rotation_matrix_from_euler)

# Convert rotation matrix to Euler angles
euler_from_matrix = rotation_matrix_to_euler(rotation_matrix_from_euler)

# Convert quaternions to rotation matrix
rotation_matrix_from_quaternion = quaternion_to_rotation_matrix(quaternion_from_matrix)

# Convert quaternions to Euler angles
euler_from_quaternion = quaternion_to_euler(quaternion_from_matrix)

# Convert Euler angles back to quaternions
quaternion_from_euler = euler_to_quaternion(euler_angles)

# Print the results
print("Euler Angles (in radians):", euler_angles)
print("Rotation Matrix from Euler:\n", rotation_matrix_from_euler)
print("Quaternion from Rotation Matrix:", quaternion_from_matrix)
print("Euler from Rotation Matrix (in radians):", euler_from_matrix)
print("Rotation Matrix from Quaternion:\n", rotation_matrix_from_quaternion)
print("Euler from Quaternion (in radians):", euler_from_quaternion)
print("Quaternion from Euler:", quaternion_from_euler)