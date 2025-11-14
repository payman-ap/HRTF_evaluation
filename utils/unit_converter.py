import numpy as np

def delay_to_distance_mm(delay_times, speed_of_sound=343):
    """
    Convert delay times to distances in millimeters.

    Parameters:
    - delay_times: float, int, list, tuple, or numpy.ndarray
        Time delays in seconds. Can be scalar, vector, or matrix.
    - speed_of_sound: float (default: 343 m/s)
        Speed of sound in m/s. Default is for air at 20°C.

    Returns:
    - distances_mm: same type as input
        Distances in millimeters, maintaining input shape.
    """
    # Convert to numpy array for uniform handling
    input_array = np.asarray(delay_times)
    
    # Calculate distances (convert m to mm with *1000)
    result = input_array * speed_of_sound * 1000
    
    # Return the same type as input
    if isinstance(delay_times, (int, float)):
        return float(result)
    elif isinstance(delay_times, (list, tuple)):
        return result.tolist()
    else:  # numpy array or other array-like
        return result



if __name__=="__main__":
    # Scalar
    print(delay_to_distance_mm(0.001))  # 343.0

    # List
    print(delay_to_distance_mm([0.001, 0.002]))  # [343.0, 686.0]

    # NumPy matrix
    arr = np.array([[0.001, 0.002], [0.003, 0.004]])
    print(delay_to_distance_mm(arr))
    # Output:
    # [[ 343.  686.]
    #  [1029. 1372.]]

