"""
Definition of :class::class:`Wavelet`.
:class::class:`Wavelet` is used to create a Weighted Wavelet Transform,
Based on Foster 1996
"""
import numpy as np

from jazzhands.utils import phi_1, phi_2, phi_3

__all__ = ["WaveletTransformer"]


class WaveletTransformer:
    """
    Calculate the Weighted Wavelet Transform of the data `y`, measured at
    times `t`, evaluated at a wavelet scale $\omega$ and shift $\tau$, for a
    decay factor of the Gaussian envelope `c`. Adapted from (5-11) in Foster (1996).

    Parameters
    ----------
    time : array-like
        Times of observations

    counts : array-like
        Observed counts

    func_list : array-like, optional
        Array or list containing the basis functions, not yet evaluated. If you
        are unfamiliar with how these basis functions are derived, be very
        careful with setting this parameter. Default None [phi1, phi2, phi3]

    f1 : function, optional
        First basis function. Should be equivalent to `lambda x: numpy.ones(len(x)). Default None

    omegas : array-like, optional
        Angular frequency. Default None

    nus : array-like, optional
        Actual frequency. Corresponds to omegas/2pi. Default None

    scales : array-like, optional
        Scales of the wavelet. Corresponds to 2pi/omegas. Default None

    taus : array-like
        Shift of wavelet; corresponds to a time. Default None

    c : float
        Decay rate of Gaussian envelope of wavelet. Default 0.0125

    """
    def __init__(self, time, counts, func_list=None, f1=None, omegas=None, nus=None, scales=None, taus=None, c=0.0125):

        # REVIEW: Force type checking. If needed then add to all public functions.
        for key, value in locals().items():
            if key in ['time', 'counts']:
                if not isinstance(value, (list, tuple, np.ndarray)):
                    raise TypeError((f"{key} should be of type list, tuple or np.ndarray but is of type {type(value)}"))
            elif key in ['omegas', 'nus', 'scales', 'taus']:
                if not isinstance(value, (list, tuple, np.ndarray)):
                    if value is not None:
                        raise TypeError((f"{key} should be of type list, tuple, np.ndarray or NoneType but is of type {type(value)}"))
            elif key == 'c':
                if not isinstance(value, (int, float, complex)):
                    raise TypeError((f"{key} should be of type int, float, complex but is of type {type(value)}"))

        self._time = np.asarray(time)
        self._counts = np.asarray(counts)

        if self._time.shape != self._counts.shape:
            raise ValueError("time and counts must have the same shape")

        if func_list is None:
            self.func_list = [phi_1, phi_2, phi_3]
            self.f1 = phi_1
        else:
            self.func_list = func_list
            self.f1 = f1

        if (omegas is not None and nus is not None) or \
           (omegas is not None and scales is not None) or \
           (scales is not None and nus is not None):
            raise ValueError("Please only supply either omegas, nus, or scales"
                             "and not a combination")

        elif omegas is not None:
            self._omegas = np.asarray(omegas)
            self._nus = self._omegas / 2.0 / np.pi

            if 0 in self._omegas:
                raise ZeroDivisionError("omegas cannot have zero")
            else:
                self._scales = 2.0 * np.pi / self._omegas

        elif nus is not None:
            self._nus = np.asarray(nus)
            self._omegas = 2.0 * np.pi * self._nus

            if 0 in self._nus:
                raise ZeroDivisionError("nus cannot have zero")
            else:
                # REVIEW: Changed from self._scales = 1.0 / np.asarray(scales)
                # as it seemed wrong
                self._scales = 1.0 / self._nus

        elif scales is not None:
            self._scales = np.asarray(scales)
            self._omegas = 2.0 * np.pi / self._scales

            if 0 in self._scales:
                raise ZeroDivisionError("scales cannot have zero")
            else:
                self._nus = 1.0 / self._scales

        elif omegas is None and scales is None and nus is None:
            self._omegas = None
            self._nus = None
            self._scales = None

        self._taus = None if taus is None else np.asarray(taus)
        self._c = float(c)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, new_time):
        new_time = np.asarray(new_time)
        if not new_time.shape == self._time.shape:
            raise ValueError("Can only assign new time of the same shape as "
                             "the original array")

        self._time = new_time

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, new_counts):
        new_counts = np.asarray(new_counts)
        if not new_counts.shape == self._counts.shape:
            raise ValueError("Can only assign new counts of the same shape as "
                             "the original array")

        self._counts = new_counts

    @property
    def omegas(self):
        if self._omegas is None:
            print(f"Omegas not set, type: {type(self._omegas)}")
        else:
            return self._omegas

    @omegas.setter
    def omegas(self, new_omegas):
        self._omegas = np.asarray(new_omegas)

        self._nus = self._omegas / 2.0 / np.pi
        self._scales = 2.0 * np.pi / self._omegas

    @property
    def nus(self):
        if self._nus is None:
            print(f"Nus not set, type: {type(self._nus)}")
        else:
            return self._nus

    @nus.setter
    def nus(self, new_nus):
        self._nus = np.asarray(new_nus)

        self._omegas = 2.0 * np.pi * self._nus
        self._scales = 1.0 / self._nus

    @property
    def scales(self):
        if self._scales is None:
            print(f"Scales not set, type: {type(self._scales)}")
        else:
            return self._scales

    @scales.setter
    def scales(self, new_scales):
        self._scales = np.asarray(new_scales)

        self._nus = 1.0 / self._scales
        self._omegas = 2.0 * np.pi / self._scales

    @property
    def taus(self):
        if self._taus is None:
            print(f"Taus not set, type: {type(self._taus)}")
        else:
            return self._taus

    @taus.setter
    def taus(self, new_taus):
        self._taus = np.asarray(new_taus)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, new_c):
        self._c = float(new_c)

    def _weight_alpha(self, time, omega, tau, c):
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
        return np.exp(-c * np.power(omega * (time - tau), 2.0))

    def _n_points(self, weights):
        """
        Effective number of points contributing to the transform; (5-4) in Foster (1996).

        Parameters
        ----------
        weights : array-like
            weights of observations, already calculated

        Returns
        -------
        float
            Effective number of data points

        """
        return np.power(np.sum(weights), 2.0) / np.sum(np.power(weights, 2.0))

    def _inner_product(self, func1, func2, weights):
        """
        Define the inner product of two functions; (4-2) in Foster (1996).

        Parameters
        ----------
        func1 : array-like
            Values of f at times corresponding to the weights

        func2 : array-like
            Values of g at times corresponding to the weights

        weights : array-like
            weights of observations, already calculated

        Returns
        -------
        float
            Inner product of func1 and func2

        """
        return np.sum(weights * func1 * func2) / np.sum(weights)

    def _inner_product_vector(self, func_vals, weights, counts):
        """
        Generates a column vector consisting of the inner products between the basis functions and the observed data

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the
            weights. Should have shape (number of basis functions,len(ws))

        weights : array-like
            weights of observations, already calculated

        counts : array-like
            Observed data

        Returns
        -------
        `numpy.array`
            Column vector where phi_y_i = phi_i * counts

        """
        return np.array([[
            self._inner_product(func, counts, weights) for func in func_vals
        ]]).T

    def _S_matrix(self, func_vals, weights):
        """
        Define the S-matrix; (4-2) in Foster (1996)
        Takes the values of the functions already evaluated at the times of observations.

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the
            weights. Should have shape (number of basis functions,len(ws))

        weights : array-like
            weights of observations, already calculated

        Returns
        -------
        `numpy.matrix`
            S-matrix; size len(func_vals)xlen(func_vals)

        """
        S = np.array([[
            self._inner_product(func1, func2, weights) for func1 in func_vals
        ] for func2 in func_vals])

        return np.matrix(S)

    def _calc_coeffs(self, func_vals, weights, counts):
        """
        Calculate the coefficients of each $\phi$. Adapted from (4-4) in Foster (1996).

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the
            weights. Should have shape (number of basis functions,len(ws))

        weights : array-like
            Weights of observations, already calculated

        counts : array-like
            Observed data

        Returns
        -------
        `numpy.array`
            Contains coefficients for each basis function

        """
        S_m = self._S_matrix(func_vals, weights)
        phi_y = self._inner_product_vector(func_vals, weights, counts)

        return np.linalg.solve(S_m, phi_y).T

    def _weight_var_x(self, f1_vals, weights, counts):
        """
        Calculate the weighted variation of the data. Adapted from (5-9) in Foster (1996).

        Parameters
        ----------
        f1_vals : array-like
            Array of values of the first basis function; should be equivalent
            to `numpy.ones(len(counts))`

        weights : array-like
            Weights of observations, already calculated

        counts : array-like
            Observed data

        Returns
        -------
        float
            Weighted variation of the counts

        """
        return self._inner_product(counts, counts, weights) - np.power(
            self._inner_product(f1_vals, counts, weights), 2.0)

    def _y_fit(self, func_vals, weights, counts):
        """
        Calculate the value of the model.

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the
            weights. Should have shape (number of basis functions,len(ws))

        weights : array-like
            Weights of observations, already calculated

        counts : array-like
            Observed data

        Returns
        -------
        array-like
            Values of the fit model

        y_coeffs : `numpy.array`
            The coefficients returned by `coeffs`
        """
        y_coeffs = self._calc_coeffs(func_vals, weights, counts)

        return y_coeffs.dot(func_vals), y_coeffs

    def _weight_var_y(self, func_vals, f1_vals, weights, counts):
        """
        Calculate the weighted variation of the model. Adapted from (5-10) in Foster (1996).

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the weights. Should have shape (number of basis functions, len(weights))

        f1_vals : array-like
            Array of values of the first basis function; should be equivalent to `numpy.ones(len(counts))`

        weights : array-like
            Weights of observations, already calculated

        counts : array-like
            Observed data

        Returns
        -------
        float
            Weighted variation of the model

        float
            Coefficients from `coeffs`

        """
        y_f, y_coeffs = self._y_fit(func_vals, weights, counts)

        return self._inner_product(y_f, y_f, weights) - np.power(
            self._inner_product(f1_vals, y_f, weights), 2.0), y_coeffs

    def _wavelet_transform(self, exclude, tau, omega):
        """
        Internal function to compute wavelet for one tau, omega pair.

        Parameters
        ----------
        exclude : bool
            If exclude is True, returns 0 if the nearest data point is more than one cycle away. Default True.

        omega : float
            angular frequency in radians per unit time.

        tau : float
            time shift in same units as time

        Returns
        -------
        float
            WWZ of the data at the given frequency/time.

        float
            Corresponding amplitude of the signal at the given frequency/time

        """
        if exclude and (np.min(np.abs(self._time - tau)) >
                        2.0 * np.pi / omega):
            return 0.0, 0.0

        weights = self._weight_alpha(self._time, omega, tau, self._c)
        num_pts = self._n_points(weights)

        func_vals = np.array(
            [func(self._time, omega, tau) for func in self.func_list])

        f1_vals = self.f1(self._time, omega, tau)

        x_var = self._weight_var_x(f1_vals, weights, self._counts)
        y_var, y_coeff = self._weight_var_y(func_vals, f1_vals, weights,
                                            self._counts)
        y_coeff_rows = y_coeff[0]

        return ((num_pts - 3.0) * y_var) / (2.0 * (x_var - y_var)), np.sqrt(
            np.power(y_coeff_rows[1], 2.0) + np.power(y_coeff_rows[2], 2.0))

    def compute_wavelet(self, exclude=True, parallel=False, n_processes=False):
        """
        Calculate the Weighted Wavelet Transform of the object. Note that this
        can be incredibly slow for a large enough light curve and a dense
        enough grid of omegas and taus, so we include multiprocessing to speed
        it up. You can update the omega/nu/scale and tau grids if you
        initialized the `WaveletTransformer` object with them, or set them now
        if you didn't.

        Parameters
        ----------
        exclude : bool, optional
            If exclude is True, returns 0 if the nearest data point is more than one cycle away. Default True.

        parallel : bool, optional
            If multiprocessing is to be used. Default False.

        n_processes : int, optional
            If `mp` is True, sets the `processes` parameter of `multiprocessing.Pool`. If not given, sets to `multiprocessing.cpu_count()-1`

        Returns
        -------
        WWZ : `numpy.ndarray`
            WWZ of the data.

        WWA : `numpy.ndarray`
            Corresponding wavelet amplitude

        """
        if self._taus is None:
            raise ValueError(f"taus not set. {type(self._taus)}")
        if self._omegas is None and self._nus is None and self._scales is None:
            raise ValueError(
                f"omegas and nus and scales not set. {type(self._omegas)}")

        from tqdm.autonotebook import tqdm

        if parallel:
            import multiprocessing as mp

            n_processes = mp.cpu_count() - 1 if n_processes is None else n_processes

            args = np.array([[exclude, tau, omega] for omega in self._omegas
                             for tau in tqdm(self._taus)])

            with mp.Pool(processes=n_processes) as pool:
                results = pool.starmap(
                    self._wavelet_transform,
                    args,
                    chunksize=int(len(self._omegas) * len(self._taus) / 10))

                transform = np.array(results).reshape(len(self._omegas),
                                                      len(self._taus), 2)
                wwz = transform[:, :, 0]
                wwa = transform[:, :, 1]

        else:
            transform = np.array([[
                self._wavelet_transform(exclude, tau, omega)
                for omega in self._omegas
            ] for tau in tqdm(self._taus)])

            wwz = transform[:, :, 0].T
            wwa = transform[:, :, 1].T

        return wwz, wwa

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

        After calculating omegas and taus, sets the corresponding attributes of
        the `WaveletTransformer`.

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

        Returns
        -------
        omegas : `numpy.ndarray`
            Frequencies

        taus : `numpy.ndarray`
            Time shifts

        """
        log_omega_min = np.log2(2 * np.pi * nu_min)
        log_omega_max = np.log2(2 * np.pi * nu_max)

        delta_log_omega = np.log2(1.0 + (np.sqrt(2.0 * c) / resolution_factor))
        n_omega = int((log_omega_max - log_omega_min) / delta_log_omega) + 1
        omegas = np.logspace(log_omega_min, log_omega_max, n_omega, base=2)

        dt_max = 1.0 / (2 * np.pi * nu_max * np.sqrt(2.0 * c))
        n_tau = int(resolution_factor * (tau_max - tau_min) / dt_max) + 1
        taus = np.linspace(tau_min, tau_max, n_tau)

        return omegas, taus

    def auto_compute(self, nu_min, nu_max, tau_min=None, tau_max=None, resolution_factor=3, exclude=True, parallel=False, n_processes=False):
        """
        Calculate the Weighted Wavelet Transform of the object in a user-
        specified frequency window. `auto_compute` then figures out the
        frequency and time spacing in order to ensure `resolution_factor`
        grid points per resolution element, and sets the `.omegas`, `.nus`,
        `.scales`, and `.taus` attributes of the `WaveletTransformer`, and runs
        the wavelet transformation.

        Parameters
        ----------
        nu_min : float
            Minimum frequency to calculate the wavelet on.

        nu_max : float
            Maximum frequency to calculate the wavelet on.

        tau_min : float, optional
            Minimum shift to calculate the wavelet on. Defaults to the first
            data point.

        tau_max : float, optional
            Maximum shift to calculate the wavelet on. Defaults to the last
            data point.

        resolution_factor : int, optional
            Number of points per resolution element

        exclude : bool, optional
            If exclude is True, returns 0 if the nearest data point is more than one cycle away. Default True.

        parallel : bool, optional
            If multiprocessing is to be used. Default False.

        n_processes : int, optional
            If `mp` is True, sets the `processes` parameter of `multiprocessing.Pool`. If not given, sets to `multiprocessing.cpu_count()-1`

        Returns
        -------
        omegas : `numpy.ndarray`
            Frequencies

        taus : `numpy.ndarray`
            Time shifts

        WWZ : `numpy.ndarray`
            WWZ of the data.

        WWA : `numpy.ndarray`
            Corresponding wavelet amplitude

        """
        if tau_min is None:
            tau_min = self._time.min()

        if tau_max is None:
            tau_max = self._time.max()

        self._omegas, self._taus = self._omegas_taus_from_min_max_nu(nu_min, nu_max, tau_min, tau_max, resolution_factor)

        wwz, wwa = self.compute_wavelet(exclude=exclude,
                                        parallel=parallel,
                                        n_processes=n_processes)

        return self._omegas, self._taus, wwz, wwa

    def resolution(self, nu):
        """
        Calculates the resolution of the Morlet wavelet in time and frequency
        as a function of frequency

        Parameter
        ---------
        nu : float
            Frequency

        Returns
        ------
        delta_tau : float
            Time resolution

        delta_nu : float
            Frequency resolution

        """
        omega = 2.0 * np.pi * nu

        dw = omega * np.sqrt(2.0 * self._c)
        dt = 1.0 / dw
        dnu = dw / 2.0 / np.pi

        return dt, dnu
