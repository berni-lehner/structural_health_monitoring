"""Filterbank with log-scaled center frequencies.

 Acknowledgement: 
 Parts of this code were taken and modified from librosa.filters.mel
 https://librosa.org/doc/main/_modules/librosa/filters.html#mel
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")


import warnings
import numpy as np

import matplotlib.pyplot as plt


class LogFilterbank(object):
    """Filterbank with log-scaled center frequencies.

    Creates a set of 50% overlapping triangular filters according to the parameters.
    Can then be used to apply to spectra in order to scale them logarithmically in frequency
    as well as optionally in magnitude (dB scaling).
    Notice that you already need to know n_fft (window size for FFT that gives a linear spectrum)
    and the sampling rate in order to create the filterbank.
    
        
    Example:
        sr = 44100
        n_bins = 64
        n_fft = 1024    
        norm = 'area'
        fb = LogFilterbank(sr=sr, n_fft=n_fft, n_bins=n_bins, norm=norm)

        # plot filterbank
        fb.plot()
        
        spectrum = ... # some spectrum from FFT with window size n_fft
        
        # apply log-scaling in frequency and dB scaling in magnitude
        scaled_spec = fb.to_dB(fb.apply(spectrum))        
    """
    
 
    def __init__(self,
                 sr:int=None,
                 n_fft:int=None,
                 n_fft_bins:int=None,
                 n_log_bins:int=64,
                 f_min=0.0,
                 f_max=None,
                 norm=None,
                 dtype=np.float32):
        """Creates Filterbank that transforms a given linear spectrum into a log-scaled spectrum.
        
        Args:
          sr: sampling rate.
          n_fft: window size of FFT used for linear spectrum (to be scaled).
          n_fft_bins: number of spectrum bins (optional in case n_fft is not used).
          n_log_bins: number of bins of resulting log-scaled spectrum.
          f_min: minimum frequency of resulting log-scaled spectrum..
          f_max: maximum frequency of resulting log-scaled spectrum.
          norm: filter weight normalization [None, 'area', 'height']
          dtype: data type.
        """
        assert n_fft is not None or n_fft_bins is not None, "either parameter n_fft or n_fft_bins has to be given, both are missing"
        assert sr is not None, "parameter sr is missing"
        
        self._sr = sr
        
        if n_fft is None:
            self._n_fft = int(n_fft_bins-1)*2
        else:
            self._n_fft = n_fft
        
        if n_fft_bins is None:
            self._n_fft_bins = int(1+self._n_fft//2)
        else:
            self._n_fft_bins = int(n_fft_bins)
                
        self._n_log_bins = n_log_bins
        self._f_min = f_min
        self._f_max = f_max
        self._norm = norm
        
        self._center_bins = []
        
        if self._f_max is None:
            self._f_max = sr/2

        self._weights = np.zeros((self._n_log_bins, self._n_fft_bins), dtype=dtype)

        # center freqs of each FFT bin
        self._fft_freqs = self.__fft_frequencies(sr=self._sr, n_fft=self._n_fft)
    
        # center freqs of each log scale band - uniformly spaced between limits
        self._center_freqs = self.__log_frequencies(n_log_bins+2, f_min=self._f_min, f_max=self._f_max)
        
        # triangular filter creation
        self.__set_weights()      
        self._center_bins = np.argmax(self._weights, axis=1).astype(int)

        # normalization
        self.__norm_weights()
        
        # sanity checks
        self.__check_empty()
        self.__check_duplicates()


    def to_dB(self, spec):
        """Magnitude scaling to dB.

        Args:
          spec: the spectrum (linear scaled in magnitude).

        Returns:
          The dB scaled (in magnitude) spectrum.
        """
        spec_scaled = 10*np.log10(spec)  # dB

        return spec_scaled.T

    
    def apply(self, spec, to_dB=False):
        """Applies filterbank to a given spectrum.

        Args:
          spec: the spectrum (linear scaled in frequency).
          to_dB: whether the spectrum should be scaled to dB in magnitude as well.

        Returns:
          spec_scaled: the scaled spectrum.
        """
        spec_scaled = np.dot(spec, self._weights.T)
        spec_scaled = np.where(spec_scaled==0, np.finfo(float).eps, spec_scaled)  # Numerical Stability
        
        if to_dB:
            spec_scaled = self.to_dB(spec_scaled)
 
        return spec_scaled.T
        

    def bin2freq(self, x:int):
        return self._fft_freqs[x]

    def freq2bin(self, x:int):
        bins = np.where(self._fft_freqs <= x)
            
        if bins[-1].size > 1:
            bin = bins[0][-1]
        else:
            bin = bins[-1]
        return bin

    
    def plot(self, axs=None) -> None:
        """Plot filterbank.
        """
        

        if axs is None:
            fig, axs = plt.subplots(constrained_layout=True)
       
        axs.plot(self._fft_freqs, self._weights.T)
        axs.set_xlabel('Frequency in Hz')
        axs.set_ylabel('Weight')
        axs.set_xlim(0, self._fft_freqs[-1])
        
        ax2 = axs.twiny()
        ax2.plot(np.linspace(0,self._n_fft_bins-1,self._n_fft_bins).astype(int), self._weights.T)
        ax2.set_xlim(0, self._n_fft_bins-1)
        ax2.set_xlabel('bin')
        
        
        
    @property
    def f_min(self):
        return self._f_min

    @property
    def f_max(self):
        return self._f_max

    @property
    def norm(self):
        return self._norm

    @property
    def weights(self):
        return self._weights

    @property
    def center_bins(self):
        return self._center_bins

    
    @property
    def center_freqs(self):
        return self._center_freqs
    
    
    @property
    def fft_freqs(self):
        return self._fft_freqs

    @property
    def n_log_bins(self):
        return self._n_log_bins
    

    def __norm_weights(self) -> None:
        """Normalizes filter weights.
        """
        if self._norm == "area":
            # scaled to be approx constant energy per channel (Slaney-style)
            enorm = 2.0 / (self._center_freqs[2 : self._n_log_bins + 2] - self._center_freqs[:self._n_log_bins])
            self._weights *= enorm[:, np.newaxis]
        elif self._norm == "height":
            # scaled to have a height of 1 for each triangular filter
            self._weights /= np.max(self._weights, axis=-1)[:, None]
        else:
            if self._norm is not None:
                warnings.warn(
                    "Unknown norm (" + str(self._norm) + ")."
                    "Valid choices are 'area' or 'height' or None.",
                    stacklevel=2,
                )
        
        
    def __set_weights(self) -> None:
        """Creates triangular shaped, 50% overlapping filter weights.
        """
        fdiff = np.diff(self._center_freqs)
        ramps = np.subtract.outer(self._center_freqs, self._fft_freqs)

        for i in range(self._n_log_bins):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            self._weights[i] = np.maximum(0, np.minimum(lower, upper))
            
            
    def __check_empty(self) -> None:
        """Check if some filters (especially the lower) are empty due to misconfiguration.
        """
        # Only check weights if center_freqs[0] is positive
        if not np.all((self._center_freqs[:-2] == 0) | (self._weights.max(axis=1) > 0)):
            # we have empty channels somewhere
            warnings.warn(
                "Empty filters detected in log frequency basis. "
                "Some channels will produce empty responses. "
                "Try increasing your sampling rate (and fmax) or "
                "reducing n_log_bins.",
                stacklevel=2,
            )
        
        
    def __check_duplicates(self) -> None:
        """Check if some filters (especially the lower) are duplicates due to misconfiguration.
        """
        if(len(self._center_bins) != len(np.unique(self._center_bins))):
            # we have duplicate filters somewhere
            warnings.warn(
                "Duplicate filters detected in log frequency basis. "
                "Some channels will produce duplicate responses. "
                "Try increasing your sampling rate (and fmax), fmin, or "
                "reducing n_log_bins.",
                stacklevel=2,
            )
            
            
    def __log_frequencies(self, n_bins=None, f_min=0, f_max=None):
        """Compute a log-scaled array of frequencies.
        
        Args:
          n_bins: number of bins of resulting log-scale.
          f_min: minimum frequency of resulting log-scale.
          f_max: maximum frequency of resulting log-scale.

        Returns:
            log_freqs: log-scaled frequencies
        """
        assert n_bins is not None, "parameter n_bins is missing"
        assert f_max is not None, "parameter f_max is missing"

        # center freqs of bands - uniformly spaced between limits
        min_log_freqs = self.__hz_to_logscale(f_min)
        max_log_freqs = self.__hz_to_logscale(f_max)

        log_freqs = np.linspace(min_log_freqs, max_log_freqs, n_bins)
        log_freqs = self.__logscale_to_hz(log_freqs)

        return log_freqs


    def __fft_frequencies(self, sr=None, n_fft=None):
        """Compute corresponding center frequencies for bins of spectrum.
        
        Args:
          sr: sampling rate.
          n_fft: window size of FFT used for linear spectrum (to be scaled).

        Returns:
            frequencies: center frequencies
            
        """
        assert sr is not None, "parameter sr is missing"
        assert n_fft is not None, "parameter n_fft is missing"
        
        frequencies = np.fft.rfftfreq(n=n_fft, d=1.0/sr)
        
        return frequencies

    
    def __hz_to_logscale(self, frequencies):
        """Convert Hz to log scale.

        Args:
            frequencies : number or np.ndarray [shape=(n,)] , float
                scalar or array of frequencies

        Returns:
            logs : number or np.ndarray [shape=(n,)]
                input frequencies in log scale
        """
        logs = np.log(np.asanyarray(frequencies))

        return logs


    def __logscale_to_hz(self, logs):
        """Convert log scale bin numbers to frequencies.

        Args:
            logs : np.ndarray [shape=(n,)], float
                log bins to convert

        Returns:
            frequencies : np.ndarray [shape=(n,)]
                input logs in Hz
        """
        frequencies = np.exp(np.asanyarray(logs))

        return frequencies