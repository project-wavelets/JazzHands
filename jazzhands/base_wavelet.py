import numpy as np

class BaseWavelet:
    """
    Base class used to provide the functionality for the mother wavelets used
    in the wavelet transform

    Parameters
    ----------
    c : float
        Decay rate of Gaussian envelope of wavelet. If not given defaults
        to 0.0125.
    
    w0 : float
        Non-dimensional frequency, as given in eqn. 1 of Torrence and Compo 
        (1998). It is related to the parameter c.

    """
    def __init__(self, c=None, w0=None):

        if (c is None) and (w0 is None):
            print("No resolution factor given. Setting c=0.0125")
            self.c = 0.0125
            # Omega0 as defined in Torrence and Compo
            self.w0 = 1/np.sqrt(2*c)
        elif (c is not None) and (w0 is None):
            self.c = c
            self.w0 = 1/np.sqrt(2*c)
        elif (c is None) and (w0 is not None):
            self.w0 = w0
            self.c = 1/(2*w0**2)
        else:
            raise ValueError("Either c or w0 needs to be defined, "
                             "but not both!")

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
        raise NotImplementedError("Weights function has not been implemented"
                                  " for chosen mother wavelet.")

    def construct_wavelet(self, time, omega, tau, scale, domain='time'):
        """
        Construction function for a given mother wavelet. This can be done
        either in the time domain, or the frequency domain.

        Parameters
        ----------
        time : array-like
            times of observations

        omega : float
            angular frequency in radians per unit time.

        tau : float
            time shift in same units as t

        scale : float
            scale ranges of the wavelet

        Returns
        -------
        array-like
            Mother wavelet evaluated either at given time stamps or frequencies

        """
        if domain == 'time':
            return self._construct_wavelet_time(time, tau, scale)
        elif domain == 'frequency':
            return self._construct_wavelet_frequency(omega, tau, scale)
        else:
            # TODO this is just a placeholder - need to raise correct exception!
            raise ValueError("Incorrect domain keyword set. Must be either time"
                             " or frequency.")

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
        raise NotImplementedError("Current mother wavelet has no function "
                                  "implementing its construction in the time "
                                  "domain.")

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
        raise NotImplementedError("Current mother wavelet has no function "
                                  "implementing its construction in the "
                                  "frequency domain.")