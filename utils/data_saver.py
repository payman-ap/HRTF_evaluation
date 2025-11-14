import scipy.io
import sofar as sf

# Note: when using *args as positional arguments the saved variables lose their true names, that's why I use kwargs
def quick_save_for_matlab(file_path, **kwargs):
    if not file_path:
        raise ValueError("File path must be provided.")
    
    if not kwargs:
        raise ValueError("At least one named variable must be provided.")
    
    # scipy.io.savemat(file_path, kwargs)
    # print(f"Variables saved to {file_path}")
    # Save 1-D numpy arrays as column vectors (Nx1) instead of the default row (1xN)
    scipy.io.savemat(file_path, kwargs, oned_as="column") # New fix for 1D arrays
    print(f"Variables saved to {file_path}")





def save_to_sofa(filepath, **kwargs):
    """
    Save HRIR data to a SOFA file using the SimpleFreeFieldHRIR convention.

    Parameters:
        filepath (str): Path where the SOFA file will be saved.
        **kwargs: Keyword arguments containing SOFA variables.

    Mandatory keyword arguments:
        Data_IR (array): (M, R, N)
        Data_SamplingRate (float): Scalar sampling rate
        SourcePosition (array): (M, 3)
        SourcePosition_Type (str): e.g. "spherical"
        ListenerPosition (array): (1, 3)
        ListenerPosition_Type (str): e.g. "cartesian"
        ReceiverPosition (array): (R, 3)
        ReceiverPosition_Type (str): e.g. "cartesian"

    Optional keyword arguments:
        ListenerUp (array): (1, 3)
        ListenerView (array): (1, 3)
        Data_Delay (array): (M, R) or (M, R, 1)
        Any other global attributes like GLOBAL_Title, GLOBAL_Comment, etc.
    """
    # Validate mandatory variables
    required_keys = [
        "Data_IR", "Data_SamplingRate", "SourcePosition", "SourcePosition_Type",
        "ListenerPosition", "ListenerPosition_Type",
        "ReceiverPosition", "ReceiverPosition_Type"
    ]
    missing = [key for key in required_keys if key not in kwargs]
    if missing:
        raise ValueError(f"Missing mandatory SOFA variables: {missing}")

    # Create SOFA object with the appropriate convention
    sofa = sf.Sofa("SimpleFreeFieldHRIR")

    # Assign mandatory variables
    sofa.Data_IR = kwargs["Data_IR"]
    sofa.Data_SamplingRate = kwargs["Data_SamplingRate"]
    sofa.SourcePosition = kwargs["SourcePosition"]
    sofa.SourcePosition_Type = kwargs["SourcePosition_Type"]
    sofa.ListenerPosition = kwargs["ListenerPosition"]
    sofa.ListenerPosition_Type = kwargs["ListenerPosition_Type"]
    sofa.ReceiverPosition = kwargs["ReceiverPosition"]
    sofa.ReceiverPosition_Type = kwargs["ReceiverPosition_Type"]

    # Optional fields (check if provided)
    optional_fields = [
        "ListenerUp", "ListenerView", "Data_Delay",
        "GLOBAL_Title", "GLOBAL_Comment", "GLOBAL_History", "GLOBAL_ApplicationName",
        "GLOBAL_Organization", "GLOBAL_AuthorContact", "GLOBAL_License"
    ]
    for key in optional_fields:
        if key in kwargs:
            setattr(sofa, key, kwargs[key])

    # Write to file
    sf.write_sofa(filepath, sofa)





if __name__ == "__main__":
    import numpy as np
    variable01 = [1, 2, 3]
    variable02 = {'a': 1, 'b': 2}
    t = np.linspace(0,np.pi,100)
    variable03 = np.concatenate((np.sin(t).reshape(-1,1), np.cos(t).reshape(-1,1)), axis=1)

    quick_save_for_matlab(r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\hrtf_eval_py\example.mat', variable01=variable01, variable02=variable02, variable03=variable03)


