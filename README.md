
The General structure of the codes for evaluations

```txt
hrtf_eval_py/
├── __init__.py
│
├── core/                      # Data representations (main classes)
│   ├── __init__.py
│   └── hrtf_measurement.py   # HRTFMeasurement class
│
├── analysis/                  # Analysis utilities
│   ├── __init__.py
│   ├── spectrum.py           # FFT, magnitude, phase, etc.
│   ├── snr.py                # SNR, dB calculations
│   └── filters.py            # Windowing, filtering helpers
│
├── plot/                      # All plotting
│   ├── __init__.py
│   ├── plot_ir.py            # Time-domain plotting
│   ├── plot_spectrum.py      # Frequency-domain plots
│   └── plot_spatial.py       # Heatmaps, 3D spatial plots, etc.
│
├── utils/                     # Optional: for loaders, helpers
│   ├── __init__.py
│   └── mat_loader.py         # Load from .mat, convert to objects
│
├── config.py                  # Configs and constants (FS, thresholds, paths)
└── main.py                    # Entry point to run your workflow

```