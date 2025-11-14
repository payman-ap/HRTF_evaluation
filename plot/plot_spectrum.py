import numpy as np
import matplotlib.pyplot as plt


# def plot_spectrum(freq, mag, db=True, title="Spectrum"):

#     # mag = np.clip(mag, 1e-12, None)
#     if db:
#         # Protect from log(0)
#         mag = np.clip(mag, 1e-12, None)
#         mag = 20 * np.log10(mag)
#     plt.plot(freq, mag)
#     plt.title(title)
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude (dB)")
#     plt.grid(True)
#     plt.show()


def plot_spectrum(freq, mag, db=False, x_tick_db=False, title="Spectrum"):


    # mag = np.clip(mag, 1e-12, None)
    if db:
        # Protect from log(0)
        mag = np.clip(mag, 1e-12, None)
        mag = 20 * np.log10(mag)
        # Choose plotting style based on x_tick_db (log-x axis)
    if x_tick_db:
        plt.semilogx(freq, mag)
    else:
        plt.plot(freq, mag)
        
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()


