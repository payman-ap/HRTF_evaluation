import numpy as np

def polar2cart(polar_matrix):
    import numpy as np
    cart_matrix = np.zeros((polar_matrix.shape[0], 3))
    azimuth = np.deg2rad(polar_matrix[:, 0])
    elevation = np.deg2rad(polar_matrix[:, 1])
    radius = polar_matrix[:, 2]
    
    cart_matrix[:, 0] = radius * np.cos(elevation) * np.cos(azimuth)
    cart_matrix[:, 1] = radius * np.cos(elevation) * np.sin(azimuth)
    cart_matrix[:, 2] = radius * np.sin(elevation)
    
    return cart_matrix




def cart2polar(cart_matrix):
    import numpy as np
    polar_matrix = np.zeros((cart_matrix.shape[0], 3))
    x = cart_matrix[:, 0]
    y = cart_matrix[:, 1]
    z = cart_matrix[:, 2]
    
    polar_matrix[:, 0] = np.rad2deg(np.arctan2(y, x))  # azimuth
    polar_matrix[:, 1] = np.rad2deg(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))  # elevation
    polar_matrix[:, 2] = np.sqrt(x**2 + y**2 + z**2)  # radius
    
    return polar_matrix

def rotate_azimuths(source_positions, rotation_angle):
    """
    Rotates the azimuth values in source_positions by rotation_angle (degrees).
    
    Parameters:
        source_positions (np.ndarray): Nx3 matrix with columns [Azimuth, Elevation, Radius].
        rotation_angle (float): The amount (in degrees) to rotate azimuth values.
        
    Returns:
        np.ndarray: New source_positions array with rotated azimuth values.
    """
    rotated = source_positions.copy()
    rotated[:, 0] = (rotated[:, 0] - rotation_angle) % 360

    rotated = rotated[np.argsort(rotated[:, 0])] # Sorting in order
    
    return rotated




if __name__ == "__main__":
    # Testing the code
    # Spherical to Cartesian
    # Azimuth, Elevation, Radius to Cartesian
    polar_coords = np.array([[45, 30, 1], [90, 45, 2], [180, 60, 3]])
    cart_coords = polar2cart(polar_coords)
    print("Cartesian Coordinates:\n", cart_coords)

    # Cartesian to Azimuth, Elevation, Radius
    cart_coords = np.array([[1, 1, 1], [0, 2, 2], [-3, 0, 3]])
    polar_coords = cart2polar(cart_coords)
    print("Polar Coordinates:\n", polar_coords)


