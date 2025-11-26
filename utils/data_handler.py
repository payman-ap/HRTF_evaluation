import numpy as np

def get_hrtf_from_directions(
    hrtf_data, directions, *,
    azimuth=None, elevation=None, radius=None,
    azimuth_range=None, elevation_range=None, radius_range=None,
    condition=None,
    exact=True,
    decimals=2,   # <--- new parameter to control rounding
):
    """
    Filters HRTF data based on directional conditions.

    Parameters
    ----------
    hrtf_data : np.ndarray
        Shape [N, ...], first dimension must match directions.shape[0].
    directions : np.ndarray
        Shape [N, 3] with columns [azimuth, elevation, radius].
    azimuth, elevation, radius : float or list, optional
        Exact values to match.
    azimuth_range, elevation_range, radius_range : tuple(min, max), optional
        Numeric ranges for filtering.
    condition : callable, optional
        A function f(directions) -> boolean mask for advanced filtering.
    exact : bool
        If True, require exact match (after rounding) for azimuth/elevation/radius.
        If False, use np.isclose (after rounding).
    decimals : int
        Number of decimal places used for comparisons (default: 2).

    Returns
    -------
    hrtf_selected : np.ndarray
    directions_selected : np.ndarray
    indices : np.ndarray
    """
    if hrtf_data.shape[0] != directions.shape[0]:
        raise ValueError("First dimension mismatch between hrtf_data and directions")

    indices = np.arange(directions.shape[0])

    # round all directions once
    rounded_directions = np.round(directions, decimals=decimals)

    def filter_exact_or_close(idx, values, col):
        if values is None:
            return idx
        values = np.atleast_1d(values)
        values = np.round(values, decimals=decimals)  # round targets too
        if exact:
            mask = np.isin(rounded_directions[idx, col], values)
        else:
            mask = np.any(np.isclose(rounded_directions[idx, col][:, None], values[None, :]), axis=1)
        return idx[mask]

    def filter_range(idx, rng, col):
        if rng is None:
            return idx
        lo, hi = np.round(rng, decimals=decimals)
        mask = (rounded_directions[idx, col] >= lo) & (rounded_directions[idx, col] <= hi)
        return idx[mask]

    # Apply filters
    indices = filter_exact_or_close(indices, azimuth, 0)
    indices = filter_exact_or_close(indices, elevation, 1)
    indices = filter_exact_or_close(indices, radius, 2)

    indices = filter_range(indices, azimuth_range, 0)
    indices = filter_range(indices, elevation_range, 1)
    indices = filter_range(indices, radius_range, 2)

    if condition is not None:
        mask = condition(rounded_directions[indices])  # use rounded for condition
        if mask.dtype != bool:
            raise ValueError("Condition must return a boolean mask")
        indices = indices[mask]

    # Subset
    hrtf_selected = hrtf_data[indices]
    directions_selected = directions[indices]  # return original values, not rounded

    return hrtf_selected, directions_selected, indices






def match_directions(
    measured_directions,
    target_directions,
    tolerance=1e-3,
    metric="max"):
    """
    Returns indices in measured_directions that best match target_directions.

    Parameters
    ----------
    measured_directions : (N, 3) array
    target_directions   : (M, 3) array
    tolerance           : float, max acceptable error
    metric              : "euclidean" or "max"
    for max: max(|Δaz|, |Δel|, |Δr|) < tolerance

    Returns
    -------
    matched_indices : (K,) array
        Indices into measured_directions
    mask : (M,) boolean array
        True for target directions that were matched
    """

    measured = measured_directions[:, None, :]  # (N,1,3)
    target   = target_directions[None, :, :]    # (1,M,3)

    diff = np.abs(measured - target)            # (N,M,3)

    if metric == "euclidean":
        dist = np.linalg.norm(diff, axis=2)     # (N,M)
    elif metric == "max":
        dist = np.max(diff, axis=2)             # (N,M)
    else:
        raise ValueError("metric must be 'euclidean' or 'max'")

    best_match_idx = np.argmin(dist, axis=0)    # For each target
    best_match_err = dist[best_match_idx, np.arange(dist.shape[1])]

    mask = best_match_err <= tolerance

    return best_match_idx[mask], mask






# def get_hrtf_from_directions(
#     hrtf_data, directions, *,
#     azimuth=None, elevation=None, radius=None,
#     azimuth_range=None, elevation_range=None, radius_range=None,
#     condition=None,
#     exact=True
# ):
#     """
#     Filters HRTF data based on directional conditions.

#     Parameters
#     ----------
#     hrtf_data : np.ndarray
#         Shape [N, ...], first dimension must match directions.shape[0].
#     directions : np.ndarray
#         Shape [N, 3] with columns [azimuth, elevation, radius].
#     azimuth, elevation, radius : float or list, optional
#         Exact values to match.
#     azimuth_range, elevation_range, radius_range : tuple(min, max), optional
#         Numeric ranges for filtering.
#     condition : callable, optional
#         A function f(directions) -> boolean mask for advanced filtering.
#     exact : bool
#         If True, require exact match for azimuth/elevation/radius.
#         If False, use np.isclose.

#     Returns
#     -------
#     hrtf_selected : np.ndarray
#     directions_selected : np.ndarray
#     indices : np.ndarray
#     """
#     if hrtf_data.shape[0] != directions.shape[0]:
#         raise ValueError("First dimension mismatch between hrtf_data and directions")

#     indices = np.arange(directions.shape[0])

#     def filter_exact_or_close(idx, values, col):
#         if values is None:
#             return idx
#         values = np.atleast_1d(values)
#         if exact:
#             mask = np.isin(directions[idx, col], values)
#         else:
#             mask = np.any(np.isclose(directions[idx, col][:, None], values[None, :]), axis=1)
#         return idx[mask]

#     def filter_range(idx, rng, col):
#         if rng is None:
#             return idx
#         lo, hi = rng
#         mask = (directions[idx, col] >= lo) & (directions[idx, col] <= hi)
#         return idx[mask]

#     # Apply filters
#     indices = filter_exact_or_close(indices, azimuth, 0)
#     indices = filter_exact_or_close(indices, elevation, 1)
#     indices = filter_exact_or_close(indices, radius, 2)

#     indices = filter_range(indices, azimuth_range, 0)
#     indices = filter_range(indices, elevation_range, 1)
#     indices = filter_range(indices, radius_range, 2)

#     if condition is not None:
#         mask = condition(directions[indices])
#         if mask.dtype != bool:
#             raise ValueError("Condition must return a boolean mask")
#         indices = indices[mask]

#     # Subset
#     hrtf_selected = hrtf_data[indices]
#     directions_selected = directions[indices]

#     return hrtf_selected, directions_selected, indices




# Deprecated, to be removed after approval

# def get_hrtf_from_directions(hrtf_data, directions, *, azimuth=None, elevations=None, radius=None, exact=True):
#     """
#     Selects subsets of HRTF data based on directional filtering.
    
#     Parameters
#     ----------
#     hrtf_data : np.ndarray
#         Array of shape [N, ...], where N is the number of directions.
#     directions : np.ndarray
#         Array of shape [N, 3], with columns [azimuth, elevation, radius].
#     azimuth : float or list/array of floats, optional
#         Azimuth(s) to filter on.
#     elevations : float or list/array of floats, optional
#         Elevation(s) to filter on.
#     radius : float or list/array of floats, optional
#         Radius/radii to filter on.
#     exact : bool, default=True
#         If True, requires exact matches. If False, allows approximate matching 
#         using np.isclose().
        
#     Returns
#     -------
#     hrtf_selected : np.ndarray
#         Subset of hrtf_data corresponding to matching directions.
#     directions_selected : np.ndarray
#         Subset of directions corresponding to the same indices.
#     indices : np.ndarray
#         The indices of selected entries.
#     """
#     # Check dimension consistency
#     if hrtf_data.shape[0] != directions.shape[0]:
#         raise ValueError(
#             f"First dimension mismatch: hrtf_data.shape[0]={hrtf_data.shape[0]} "
#             f"but directions.shape[0]={directions.shape[0]}"
#         )

#     # Start with all indices
#     indices = np.arange(directions.shape[0])

#     def filter_indices(current_indices, values, col, name):
#         if values is None:
#             return current_indices
#         values = np.atleast_1d(values)
#         if exact:
#             mask = np.isin(directions[current_indices, col], values)
#         else:
#             mask = np.any(
#                 np.isclose(directions[current_indices, col][:, None], values[None, :]),
#                 axis=1
#             )
#         return current_indices[mask]

#     # Apply filters
#     indices = filter_indices(indices, azimuth,   col=0, name="azimuth")
#     indices = filter_indices(indices, elevations, col=1, name="elevation")
#     indices = filter_indices(indices, radius,    col=2, name="radius")

#     # Subset
#     hrtf_selected = hrtf_data[indices]
#     directions_selected = directions[indices]

#     return hrtf_selected, directions_selected, indices




class HRTFPolarPositions:
    '''Under Construction'''
    def __init__(self, positions):
        if positions.shape[1] != 3:
            raise ValueError("Positions must be a [N x 3] array with azimuth, elevation, and radius.")
        self.positions = positions
        self.azimuths = positions[:, 0]
        self.elevations = positions[:, 1]

    def get_indices(self, azimuth=None, elevations=None):
        try:
            indices = np.arange(self.positions.shape[0])

            if azimuth is not None:
                azimuth = np.atleast_1d(azimuth)
                indices = np.intersect1d(indices, np.where(np.isin(self.azimuths, azimuth))[0])

            if elevations is not None:
                elevations = np.atleast_1d(elevations)
                indices = np.intersect1d(indices, np.where(np.isin(self.elevations, elevations))[0])

            return indices.tolist()

        except Exception as e:
            print(f"Error in get_indices: {str(e)}")
            return []

if __name__=="__main__":
    # Example usage:
    positions = np.array([[10, 0, 1], [20, 0, 1], [10, 5, 1], [20, 5, 1]])  # Sample data
    positions_object = HRTFPolarPositions(positions)
    print(positions_object.get_indices(azimuth=[10, 20]))
    print(positions_object.get_indices(elevations=[0]))
    print(positions_object.get_indices(azimuth=[10], elevations=[0]))
