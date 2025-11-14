import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat




def get_directional_hrtfs(ir_matrix, source_position, azimuth_directions=None, elevation_directions=None, accuracy=0.1):
    '''
    Filters the IR matrix and source positions based on specified azimuth and elevation directions.
    
    Parameters:
        ir_matrix (np.ndarray): The impulse response matrix.
        source_position (np.ndarray): A [N x 3] matrix with azimuth, elevation, and radius.
        azimuth_directions (float or list of floats, optional): Azimuth(s) to filter.
        elevation_directions (float or list of floats, optional): Elevation(s) to filter.
        accuracy (float): Tolerance for matching directions.
    
    Returns:
        ir_matrix_new (np.ndarray): Filtered IR matrix.
        source_position_new (np.ndarray): Filtered source positions.
    '''
    indices = np.ones(source_position.shape[0], dtype=bool)

    if azimuth_directions is not None:
        azimuth_directions = np.atleast_1d(azimuth_directions)
        azimuth_mask = np.any(np.abs(source_position[:, 0, None] - azimuth_directions) <= accuracy, axis=1)
        indices &= azimuth_mask

    if elevation_directions is not None:
        elevation_directions = np.atleast_1d(elevation_directions)
        elevation_mask = np.any(np.abs(source_position[:, 1, None] - elevation_directions) <= accuracy, axis=1)
        indices &= elevation_mask

    source_position_new = source_position[indices]
    ir_matrix_new = ir_matrix[indices]

    return ir_matrix_new, source_position_new







if __name__=="__main__":
    mat = loadmat(r'Y:\projects\Payman\python_workspace2\hrtf_dimension_reduction\data\HMS_full.mat', struct_as_record=False, squeeze_me=True)
    hrtfFull = mat['hrtfFull']

    source_position = hrtfFull.SourcePosition
    # Access the IR matrix inside the nested Data struct
    ir_matrix = hrtfFull.Data.IR

    # getting the horizontal directions at elevations 0, 5, 10 
    ir_matrix_new, source_position_new = get_directional_hrtfs(
    ir_matrix, 
    source_position, 
    azimuth_directions=0, 
    elevation_directions=[0, 5, 10], 
    accuracy=0.1
    )

    ir_matrix_5deg, source_position_5deg = get_directional_hrtfs(
    ir_matrix, 
    source_position, 
    azimuth_directions=np.arange(0,360,5), 
    elevation_directions=np.arange(-90,95,5), 
    accuracy=0.1
    )
    
    d = 1


