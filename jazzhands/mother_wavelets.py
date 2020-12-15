import numpy as np
import scipy
import scipy.signal
from scipy.special import factorial

from .base_wavelet import BaseWavelet

class Morlet(BaseWavelet):

    def __init__(self, *args, **kwargs):
        # Initialise parent:
        super().__init__(*args, **kwargs)

    def weights(self, time, omega, tau, c):
        """
        Weighting function for each point at a given omega and tau; (5-3) in Foster (1996).

        Parameters
        ----------
        time : array-like
            times of observations

        omega : float
            angular frequency in radians per unit time.

        tau : float
            time shift in same units as t

        c : float
            Decay constant of the Gaussian envelope for the wavelet

        Returns
        -------
        array-like
            Statistical weights of data points

        """
        weights = np.exp(-c * np.power(omega * (time - tau), 2.0))
        return weights / np.sum(weights)

    def _construct_wavelet_time(self, time, tau, scale):
        """
        Construction function for a given mother wavelet in the time domain.

        Parameters
        ----------
        time : array-like
            times of observations

        tau : float
            time shift in same units as t

        scale : float
            scale ranges of the wavelet

        Returns
        -------
        array-like
            Mother wavelet evaluated either at given time stamps

        """
        # TODO: Depends on whether scale is really used, but ignore for the
        # moment
        # Compute eta following Torrence and Compo (1998)
        eta = (time - tau)/scale
        output = np.exp(1j*eta) * np.exp(-self.c * eta**2)
        return output

    def _construct_wavelet_frequency(self, omega, tau, scale):
        """
        Construction function for a given mother wavelet in the frequency 
        domain.

        Parameters
        ----------
        omega : float
            angular frequency in radians per unit time.

        tau : float
            time shift in same units as t

        scale : float
            scale ranges of the wavelet

        Returns
        -------
        array-like
            Mother wavelet evaluated either at given time stamps

        """
        # TODO: Again this is dependent on whether the actual scale is used or
        # not. Will come back to this later.
        x = omega * scale # Assuming scale is given
        # Mock heaviside function
        Heaviside = np.zeros_like(omega)
        Heaviside[omega > 0] = 1
        # Need to check that this is consistent with the formulation from 
        # Foster (1996)
        return Hw * np.exp((-x - self.w0) **2 / 2) 


    def __call__(self, *args, **kwargs):
        # Call method constructs the wavelet with given arguments and keyword
        # arguments
        return self.construct_wavelet(**args, **kwargs)
