"""
Definition of :class::class:`Wavelet`.
:class::class:`Wavelet` is used to create a Weighted Wavelet Transform,
Based on Foster 1996
"""
import numpy as np

__all__ = ['WaveletTransformer']


class WaveletTransformer:
    """
    Calculate the Weighted Wavelet Transform of the data `y`, measured at
    times `t`, evaluated at a wavelet scale $\omega$ and shift $\tau$, for a
    decay factor of the Gaussian envelope `c`. Adapted from (5-11) in Foster (1996).

    Parameters
    ----------
    func_list : array-like
        Array or list containing the basis functions, not yet evaluated

    f1 : array-like
        First basis function. Should be equivalent to `lambda x: numpy.ones(len(x))`

    data : array-like
        Observed data

    time : array-like
        Times of observations

    omegas : array-like
        Scale of wavelet; corresponds to an angular frequency

    taus : array-like
        Shift of wavelet; corresponds to a time

    c : float
        Decay rate of Gaussian envelope of wavelet. Default 0.0125

    """
    def __init__(self, func_list, f1, data, time, omegas, taus, c=0.0125):

        self.func_list = func_list
        self.f1 = f1
        self._data = np.asarray(data)
        self._time = np.asarray(time)
        self._omegas = np.asarray(omegas)
        self._taus = np.asarray(taus)
        self.c = c

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        new_data = np.asarray(new_data)
        if not new_data.shape == self._data.shape:
            raise ValueError('Can only assign new data of the same shape as '
                             'the original array')

        self._data = new_data

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, new_time):
        new_time = np.asarray(new_time)
        if not new_time.shape == self._time.shape:
            raise ValueError('Can only assign new time of the same shape as '
                             'the original array')

        self._time = new_time

    @property
    def omegas(self):
        return self._omegas

    @omegas.setter
    def omegas(self, new_omegas):
        new_omegas = np.asarray(new_omegas)
        if not new_omegas.shape == self._omegas.shape:
            raise ValueError('Can only assign new data of the same shape as '
                             'the original array')

        self._omegas = new_omegas

    @property
    def taus(self):
        return self._taus

    @taus.setter
    def taus(self, new_taus):
        new_taus = np.asarray(new_taus)
        if not new_taus.shape == self._taus.shape:
            raise ValueError('Can only assign new data of the same shape as '
                             'the original array')

        self._taus = new_taus

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
        Define the inner product of two functions; (4-2) in Foster (1996)

        Parameters
        ----------
        func1 : array-like
            Values of f at times corresponding to the weights

        func2 : array-like
            Values of g at times corresponding to the weights

        ws : array-like
            weights of observations, already calculated

        Returns
        -------
        float
            Inner product of func1 and func2

        """
        return np.sum(weights * func1 * func2) / np.sum(weights)

    def _inner_product_vector(self, func_vals, weights, data):
        """
        Generates a column vector consisting of the inner products between the basis
        functions and the observed data

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the
            weights. Should have shape (number of basis functions,len(ws))

        weights : array-like
            weights of observations, already calculated

        data : array-like
            Observed data

        Returns
        -------
        `numpy.array`
            Column vector where phi_y_i = phi_i * data

        """
        return np.array([[
            self._inner_product(func, data, weights) for func in func_vals
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
        return np.matrix(
            np.array([[
                self._inner_product(func1, func2, weights)
                for func1 in func_vals
            ] for func2 in func_vals]))

    def _calc_coeffs(self, func_vals, weights, data):
        """
        Calculate the coefficients of each $\phi$. Adapted from (4-4) in Foster (1996).

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the
            weights. Should have shape (number of basis functions,len(ws))

        weights : array-like
            Weights of observations, already calculated

        data : array-like
            Observed data

        Returns
        -------
        `numpy.array`
            Contains coefficients for each basis function

        """
        S_m = self._S_matrix(func_vals, weights)
        phi_y = self._inner_product_vector(func_vals, weights, data)

        return np.linalg.solve(S_m, phi_y).T

    def _weight_var_x(self, f1_vals, weights, data):
        """
        Calculate the weighted variation of the data. Adapted from (5-9) in Foster (1996).

        Parameters
        ----------
        f1_vals : array-like
            Array of values of the first basis function; should be equivalent
            to `numpy.ones(len(data))`

        weights : array-like
            Weights of observations, already calculated

        data : array-like
            Observed data

        Returns
        -------
        float
            Weighted variation of the data

        """
        return self._inner_product(data, data, weights) - np.power(
            self._inner_product(f1_vals, data, weights), 2.0)

    def _y_fit(self, func_vals, weights, data):
        """
        Calculate the value of the model.

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the
            weights. Should have shape (number of basis functions,len(ws))

        weights : array-like
            Weights of observations, already calculated

        data : array-like
            Observed data

        Returns
        -------
        array-like
            Values of the fit model

        y_coeffs : `numpy.array`
            The coefficients returned by `coeffs`

        """
        y_coeffs = self._calc_coeffs(func_vals, weights, data)

        return y_coeffs.dot(func_vals), y_coeffs

    def _weight_var_y(self, func_vals, f1_vals, weights, data):
        """
        Calculate the weighted variation of the model. Adapted from (5-10) in Foster (1996).

        Parameters
        ----------
        func_vals : array-like
            Array of values of basis functions at times corresponding to the weights. Should have shape (number of basis functions, len(weights))

        f1_vals : array-like
            Array of values of the first basis function; should be equivalent to `numpy.ones(len(data))`

        weights : array-like
            Weights of observations, already calculated

        data : array-like
            Observed data

        Returns
        -------
        float
            Weighted variation of the model

        float
            Coefficients from `coeffs`

        """
        y_f, y_coeffs = self._y_fit(func_vals, weights, data)

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

        weights = self._weight_alpha(self._time, omega, tau, self.c)
        num_pts = self._n_points(weights)

        func_vals = np.array(
            [func(self._time, omega, tau) for func in self.func_list])

        f1_vals = self.f1(self._time, omega, tau)

        x_var = self._weight_var_x(f1_vals, weights, self._data)
        y_var, y_coeff = self._weight_var_y(func_vals, f1_vals, weights,
                                            self._data)
        y_coeff_rows = y_coeff[0]

        return ((num_pts - 3.0) * y_var) / (2.0 * (x_var - y_var)), np.sqrt(
            np.power(y_coeff_rows[1], 2.0) + np.power(y_coeff_rows[2], 2.0))

    def compute_wavelet(self, exclude=True, parallel=False, n_processes=False):
        """
        Calculate the Weighted Wavelet Transform of the object.

        Note that this can be incredibly slow for a large enough light curve and a dense enough grid of omegas and taus, so we include multiprocessing to speed it up.

        Parameters
        ---------_
        exclude : bool
            If exclude is True, returns 0 if the nearest data point is more than one cycle away. Default True.

        parallel : bool
            If multiprocessing is to be used. Default False.

        n_processes : int
            If `mp` is True, sets the `processes` parameter of `multiprocessing.Pool`. If not given, sets to `multiprocessing.cpu_count()-1`

        Returns
        -------
        WWZ : float
            WWZ of the data at the given frequency/time.

        WWA : float
            Corresponding amplitude of the signal at the given frequency/time

        """
        from tqdm.autonotebook import tqdm

        if parallel:
            import multiprocessing as mp

            n_processes = multiprocessing.cpu_count(
            ) - 1 if n_processes is None else n_processes

            args = np.array([[exclude, tau, omega] for omega in self._omegas
                             for tau in self._taus])

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
            transform = np.array(
                [[self._wavelet_transform(exclude, tau, omega)]
                 for omega in self._omegas for tau in tqdm(self._taus)])

            wwz = transform[:, :, 0].T
            wwa = transform[:, :, 1].T

        return wwz, wwa
