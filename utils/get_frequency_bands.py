import numpy as np


# ======================== | Bark Band Limits | ========================

def get_bark_bands(f_min: float, f_max: float, fs: float):
    """
    Calculate Bark band edges, centers, and edge matrix within a frequency range.

    Parameters
    ----------
    f_min : float
        Minimum frequency (Hz).
    f_max : float
        Maximum frequency (Hz).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    bark_centers : np.ndarray
        Array of Bark band center frequencies.
    bark_edges : np.ndarray
        Array of Bark band edge frequencies.
    bark_edge_matrix : np.ndarray
        Matrix of Bark band edges, shape (num_bands, 2).
    """

    # Fixed Bark edges (up to ~15.5 kHz, can be truncated depending on fs/2)
    bark_edges = np.array([
        10, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
        1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
        4400, 5300, 6400, 7700, 9500, 12000, 15500
    ], dtype=float)

    # Make sure f_max does not exceed Nyquist
    f_max = min(f_max, fs / 2.0)

    # Filter valid edges
    bark_edges = bark_edges[(bark_edges >= f_min) & (bark_edges <= f_max)]

    # Ensure first and last edges match requested frequency range
    if bark_edges[0] > f_min:
        bark_edges = np.insert(bark_edges, 0, f_min)
    if bark_edges[-1] < f_max:
        bark_edges = np.append(bark_edges, f_max)

    # Compute Bark band centers and edge matrix
    num_bands = len(bark_edges) - 1
    bark_centers = np.zeros(num_bands)
    bark_edge_matrix = np.zeros((num_bands, 2))

    for i in range(num_bands):
        bark_centers[i] = (bark_edges[i] + bark_edges[i+1]) / 2.0
        bark_edge_matrix[i, :] = [bark_edges[i], bark_edges[i+1]]

    return bark_centers, bark_edges, bark_edge_matrix


# ======================== | Octave Band Limits | ========================

def get_octave_bands(f_min: float,
                     f_max: float,
                     fs: float,
                     band_type: str = "1/3-Octave",
                     f_ref: float = 1000.0):
    """
    Calculate Octave or fractional-Octave band edges and centers.

    Parameters
    ----------
    f_min : float
        Minimum frequency (Hz).
    f_max : float
        Maximum frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    band_type : str, optional
        Band type: "1-Octave", "1/3-Octave", or "1/6-Octave".
        Default is "1/3-Octave".
    f_ref : float, optional
        Reference frequency (Hz). Default is 1000 Hz.

    Returns
    -------
    centers : np.ndarray
        Center frequencies of the octave bands.
    edges : np.ndarray
        Band edge frequencies (length = len(centers)+1).
    edge_matrix : np.ndarray
        Nx2 array of band edges for each band.
    """

    # Respect Nyquist
    f_max = min(f_max, fs / 2.0)

    # Define band ratio
    if band_type == "1-Octave":
        band_ratio = 2 ** (1 / 1)
    elif band_type == "1/3-Octave":
        band_ratio = 2 ** (1 / 3)
    elif band_type == "1/6-Octave":
        band_ratio = 2 ** (1 / 6)
    else:
        raise ValueError('Invalid band_type. Choose "1-Octave", "1/3-Octave", or "1/6-Octave".')

    # Compute index range dynamically
    n_min = int(np.floor(np.log(f_min / f_ref) / np.log(band_ratio)))
    n_max = int(np.ceil(np.log(f_max / f_ref) / np.log(band_ratio)))

    # Generate centers
    raw_centers = f_ref * band_ratio ** np.arange(n_min, n_max + 1)
    centers = raw_centers[(raw_centers >= f_min) & (raw_centers <= f_max)]

    # Compute edges
    edges = np.zeros(len(centers) + 1)
    edges[:-1] = centers * band_ratio ** (-0.5)
    edges[-1] = centers[-1] * band_ratio ** (0.5)

    # Edge matrix
    edge_matrix = np.column_stack((edges[:-1], edges[1:]))

    return centers, edges, edge_matrix




# Testing Barks
if __name__ == "__main__":
    import pprint
    
    # Example test parameters
    f_min = 100       # Hz
    f_max = 18500    # Hz
    fs = 44100       # Sampling frequency
    
    bark_centers, bark_edges, bark_edge_matrix = get_bark_bands(f_min, f_max, fs)
    
    print("Bark band edges (Hz):")
    print(bark_edges)
    
    print("\nBark band centers (Hz):")
    print(bark_centers)
    
    print("\nBark edge matrix (Hz):")
    pprint.pp(bark_edge_matrix)


# Testing Octaves
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    f_min = 150
    f_max = 18000
    fs = 44100

    centers, edges, edge_matrix = get_octave_bands(f_min, f_max, fs, band_type="1/3-Octave", f_ref=1000)

    print("Octave band centers (Hz):", centers)
    pprint.pp(edge_matrix)

    # Plot frequency spectrum view with octave bands
    plt.figure(figsize=(12, 6))
    for (low, high), c in zip(edge_matrix, centers):
        plt.axvspan(low, high, alpha=0.2, color="C0")
        plt.axvline(c, color="C1", linestyle="--")
        plt.text(c, 0.5, f"{int(round(c))}", rotation=90,
                 ha="center", va="center", fontsize=8)

    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Band indicator")
    plt.title("Octave Bands (1/3 Octave Example)")
    plt.ylim(0, 1)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()
