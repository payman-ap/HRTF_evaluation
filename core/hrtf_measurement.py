


class HRTFMeasurement:
    """
    Class representing a single HRTF measurement and its analysis tools.

    Attributes:
        subject_id (str): Public ID of the subject.
        sample_rate (int): Sampling rate of the measurement (Hz).
        _hrtf_data (dict): Protected dictionary storing HRTFs per direction.
        __raw_signal (np.ndarray): Private raw signal data.
    """

    def __init__(self, ir_set, fs: int, SourcePositions, ReceiverPosition, ListenerPosition, subject_id:str, azimuth=None, elevation=None):
        """Initializes the HRTFMeasurement object.
        
        Args:
            ir_set, SourcePositions, ReceiverPosition, ListenerPosition
            subject_id (str): Identifier for the subject.
            sample_rate (int): Sampling rate in Hz.
        """
        self.ir_set = ir_set                                         # Public attribute
        self.fs = fs                                                 # Public attribute
        self.SourcePositions = SourcePositions                         # Protected: dict[direction] = HRTF array
        self._hrtf_data = {}
        self.__raw_signal = None                                        # Private: raw measurement data

        self.azimuth = azimuth
        self.elevation = elevation

    # ─────────────────────────────────────────────
    # Public Methods
    # ─────────────────────────────────────────────

    def get_hrtf_for_direction(self, azimuth: float, elevation: float):
        """
        Retrieve the HRTF for a specific direction.
        """
        # Use self._hrtf_data to retrieve relevant HRTF
        pass

    def compute_fft_for_direction(self, azimuth: float, elevation: float):
        """
        Compute and return FFT of the HRTF for a given direction.
        """
        pass


    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────

    @property
    def signal_length(self):
        """
        Compute and return the length of the raw signal.
        print(measurement.signal_length)  # No () needed, it's a property
        """
        if self.__raw_signal is None:
            return 0
        return len(self.__raw_signal)



    # ─────────────────────────────────────────────
    # Protected Methods (helper-like)
    # ─────────────────────────────────────────────

    def _interpolate_missing_directions(self):
        """
        Interpolate missing directions in the HRTF data grid.
        """
        pass

    def _normalize_hrtf(self, hrtf_array):
        """
        Normalize the amplitude of the HRTF data.
        """
        pass

    # ─────────────────────────────────────────────
    # Private Methods (internal logic only)
    # ─────────────────────────────────────────────


    def __calculate_signal_length(self):
        """
        Calculate the length of the raw signal.
        """
        pass


    def __parse_metadata(self, file_path: str):
        """
        Extract and store metadata from the HRTF file.
        """
        pass



    # ─────────────────────────────────────────────
    # Dunder Methods
    # ─────────────────────────────────────────────

    def __repr__(self):
        """
        Official string representation of the object.
        """
        return (f"HRTFMeasurement(subject_id='{self.subject_id}', "
                f"sample_rate={self.sample_rate}, "
                f"directions_loaded={len(self._hrtf_data)})")

    def __str__(self):
        """
        Human-readable summary of the object.
        """
        return (f"HRTF Measurement for subject '{self.subject_id}'\n"
                f"Sample rate: {self.sample_rate} Hz\n"
                f"Number of directions: {len(self._hrtf_data)}\n"
                f"Signal length: {self.signal_length} samples")

        """
        print(repr(measurement))
        # HRTFMeasurement(subject_id='subject_001', sample_rate=44100, directions_loaded=56)

        print(str(measurement))
        # HRTF Measurement for subject 'subject_001'
        # Sample rate: 44100 Hz
        # Number of directions: 56
        # Signal length: 32768 samples
        """





