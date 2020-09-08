import numpy as np
class ResolutionFrequency:
    def __init__(self):
        pass

    def _resolution_at_frequency(self, omega, c=0.0125):
        """
        Calculates the resolution of the Morlet wavelet in time and frequency as a function of the angular frequency, `omega`
        
        Parameter
        ---------
        omega : float
            Angular frequency
        c : float
            Third parameter of the Morlet wavelet, governs the time/frequency resolution tradeoff
            
        Result
        ------
        delta_tau : float
            Time resolution
        delta_omega : float
            Angular frequency resolution
        """
        
        dw = omega*np.sqrt(2.0*c)
        
        dt = 1.0/dw
        
        return dt, dw

    def _omegas_taus_from_min_max_nu(self, nu_min, nu_max, tau_min, tau_max, resolution_factor=3, c=0.0125):
        """
        Given a user-specified minimum and maximum frequency, finds the frequency 
        grid that gives approximately `resolution_factor` elements across a peak in
        the wavelet transform. Then calculates the time resolution at the highest 
        desired frequency, and returns `resolution_factor` elements per time 
        element.
        
        The way it does this is if the resolution of the Morlet wavelet is 
        `sqrt(2*c)*omega`, and we want `resolution_factor` points per resolution
        element, then the ratio between resolution elements is going to be
        `1+sqrt(2*c)/resolution_factor`, which amounts to a constant spacing in log
        space.
        
        Parameters
        ----------
        nu_min : float
            Lowest frequency of interest, in units of actual frequency
        nu_max : float
            Highest frequency of interest, in units of actual frequency
        tau_min : float
            Earliest time of interest.
        tau_max : float
            Latest time of interest
        resolution_factor : int
            Number of points per resolution element

        
        """
        log_omega_min = np.log2(2*np.pi*nu_min)
        log_omega_max = np.log2(2*np.pi*nu_max)
        
        delta_log_omega = np.log2(1.0 + (np.sqrt(2.0*c) / resolution_factor))
        
        n_omega = int((log_omega_max - log_omega_min) / delta_log_omega) + 1
        
        omegas = np.logspace(log_omega_min, log_omega_max, n_omega, base=2)
        
        dt_max = 1.0 / (2 * np.pi * nu_max * np.sqrt(2.0*c))
        
        n_tau = int(resolution_factor*(tau_max - tau_min)/dt_max) + 1
        
        taus= np.linspace(tau_min, tau_max, n_tau)
        
        return omegas, taus
