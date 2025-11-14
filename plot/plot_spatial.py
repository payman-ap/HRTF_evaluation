import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata

def flat_polar_plot(polarDirections, polarError, title_suffix="Right Ear", clim=(0, 20)):
    InputData = polarDirections

    # Convert polar angles to radians
    azimuths = (InputData[:, 0] - 180) * np.pi / 180
    elevations = InputData[:, 1] * np.pi / 180

    # Spherical to Cartesian
    x = np.cos(elevations) * np.cos(azimuths)
    y = np.cos(elevations) * np.sin(azimuths)
    z = np.sin(elevations)

    # Use only one column from polarError (e.g., first)
    intensity = polarError[:, 0]

    # Create sphere grid
    phi = np.linspace(0, 2 * np.pi, 101)
    theta = np.linspace(0, np.pi, 101)
    phi, theta = np.meshgrid(phi, theta)
    aa = np.sin(theta) * np.cos(phi)
    bb = np.sin(theta) * np.sin(phi)
    cc = np.cos(theta)

    # Interpolation
    v = griddata((x, y, z), intensity, (aa, bb, cc), method='nearest')
    v = np.nan_to_num(v, nan=0)

    # Plot
    plt.figure(figsize=(8, 3))
    im = plt.imshow(v, cmap='jet', origin='lower', aspect='auto')
    plt.title(f'Noise Analysis - {title_suffix}')
    cbar = plt.colorbar(im)
    cbar.set_label('Signal-to-Noise Ratio (dB)')
    plt.clim(*clim)
    plt.xticks(np.linspace(0, 100, 36), np.arange(0, 360, 10), rotation=90)
    plt.yticks(np.linspace(0, 100, 19), np.arange(-90, 100, 10))
    plt.xlabel('Azimuth Angles')
    plt.ylabel('Elevation Angles')
    plt.tight_layout()
    plt.show()





def plot_spherical_heatmap():
    pass



if __name__ == "__main__":
    polarDirections_data = loadmat(r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\hrtf_eval_py\plot\polarDirections.mat')
    polarDirections = polarDirections_data['directions']  # [440, 2]

    # Fake data for demo
    polarError = np.random.rand(*polarDirections.shape)  # [440, 2]

    flat_polar_plot(polarDirections, polarError)

    # flat_polar_plot(polarDirections, polarError)  # uses defaults
    # flat_polar_plot(polarDirections, polarError, title_suffix="Left Ear")
    # flat_polar_plot(polarDirections, polarError, clim=(5, 25))
    # flat_polar_plot(polarDirections, polarError, title_suffix="Custom", clim=(10, 30))

