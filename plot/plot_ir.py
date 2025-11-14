import matplotlib.pyplot as plt
import numpy as np

def plot_ir(ir, fs, title="Impulse Response"):
    t = np.arange(len(ir)) / fs
    plt.plot(t, ir)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
