import os
import shutil
import unittest
import numpy as np
from tqdm import tqdm

from jazzhands import wavelets


class WaveletsTest(unittest.TestCase):

    def setUp(self):
        self.outdir = "test"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def make_test_time_signal(self):
        # Can switch this out for something else later
        num_steps = 100
        t = np.linspace(-1, 1, num_steps)
        freq = 1
        phase = np.pi / 4
        x = np.cos(2 * np.pi * freq * t + phase)
        return t, x

    def get_params_and_data(self):
        t, x = self.make_test_time_signal()
        omegas = [1.1, 1.4, 1.8, 2.0]
        taus = [0.0, 0.5]
        c = 1 / (8 * np.pi ** 2)
        return t, x, omegas, taus, c

    def make_transformer(self):
        # not its own test, but is executed by later tests
        t, x, omegas, taus, c = self.get_params_and_data()

        transformer = wavelets.WaveletTransformer(
            time = t, 
            data = x, 
            omegas = omegas, 
            taus = taus, 
            c = c
        )
        
        return transformer

    def test_make_transformer(self):
        self.make_transformer()

    def test_wavelet_transform_no_gap(self):
        self.transformer = self.make_transformer()
        t, x, omegas, taus, c = self.get_params_and_data()

        def slow_wwz_and_checks(t, x, omegas, taus, c):
            # naive/not well encapsulated/not parallelized implementation, just to check correctness
            # also has asserts in case any sub-function implementation changes later 
            transforms = np.zeros((len(omegas), len(taus)))
        
            for i, omega in enumerate(omegas):
                for j, tau in enumerate(taus):
                    zs = omega * (t - tau)
                    basis_functions = [np.ones_like, np.sin, np.cos]
                    basis_evals = np.array([f(zs) for f in basis_functions])
                    weights = np.exp(-c * zs ** 2)
                    weights /= np.sum(weights)
                    npoints = 1 / np.inner(weights, weights)
                    assert np.allclose(weights, self.transformer._weight_alpha(t, omega, tau, c))
                    assert np.isclose(self.transformer._n_points(weights), npoints)

                    S = np.array([[np.sum(basis_evals[k] * basis_evals[l] * weights) for k in range(3)] for l in range(3)])
                    assert np.allclose(self.transformer._S_matrix(basis_evals, weights), S)

                    sols = np.sum(basis_evals * weights * x, axis=1)
                    yc = np.linalg.inv(S).dot(sols)
                    y = yc.dot(basis_evals)
                    y_package, yc_package = self.transformer._y_fit(basis_evals, weights, x)
                    assert np.allclose(yc, yc_package)
                    assert np.allclose(y, y_package)
                    
                    vx = np.sum(weights * x * x) - np.dot(weights, x) ** 2
                    vy = np.sum(weights * y * y) - np.dot(weights, y) ** 2
                    
                    assert np.isclose(vx, self.transformer._weight_var_x(np.ones(len(x)), weights, x))
                    assert np.isclose(vy, self.transformer._weight_var_y(basis_evals, np.ones(len(x)), weights, x)[0])

                    transforms[i][j] = (npoints - 3) * vy / (2 * (vx - vy))
                    assert np.isclose(transforms[i][j], self.transformer._wavelet_transform(True, tau, omega)[0])
            return transforms

        for v in [False, True]:
            wwz, wwa = self.transformer.compute_wavelet(vectorized=v)
            wwz_check = slow_wwz_and_checks(t, x, omegas, taus, c)
            
            assert np.allclose(wwz, wwz_check)

    # def test_omegas_taus_from_min_max_nu(self):
    #     wav = wavelets.WaveletTransformer(func_list=[0, 0], f1=[0, 0], data=[0, 0], time=[0, 0], omegas=[0, 0], taus=[0, 0], c=0.0125)

    #     res = wav.auto_compute(1, 1, 1, 1, 1, 1)
    #     self.assertIsNotNone(res)
    #     self.assertIsInstance(res[0], np.ndarray)


if __name__ == '__main__':
    unittest.main()
