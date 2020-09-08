import numpy as np

__all__ = ['phi_1', 'phi_2', 'phi_3']


def phi_1(time, omega, tau):
    """
    Returns 1 for all times; sets the coarse detail in the wavelet transform. Note that the parameters omega and tau have no use here; we only include them to loop over the basis functions.

    Parameters
    ----------
    time : array-like
        Times of observations

    omega : float
        angular frequency in radians per unit time.

    tau : float
        time shift in same units as time

    Returns
    -------
    out : array-like
        array of 1s of length time

    """
    return np.ones(len(time))


def phi_2(time, omega, tau):
    """
    Second basis function, cos(omega(t-tau)).

    Parameters
    ----------
    time : array-like
        times of observations

    omega : float
        angular frequency in radians per unit time.

    tau : float
        time shift in same units as t

    Returns
    -------
    out : array-like
        value of phi_2

    """
    return np.cos(omega * (time - tau))


def phi_3(time, omega, tau):
    """
    Third basis function, sin(omega(t-tau)).

    Parameters
    ----------
    time : array-like
        times of observations

    omega : float
        angular frequency in radians per unit time.

    tau : float
        time shift in same units as time

    Returns
    -------
    out : array-like
        value of phi_3

    """
    return np.sin(omega * (time - tau))
