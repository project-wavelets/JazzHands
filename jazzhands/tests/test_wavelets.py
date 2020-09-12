import os
import shutil
import unittest

from jazzhands import wavelets


class WaveletsTest(unittest.TestCase):

    def setUp(self):
        self.outdir = "test"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_correct_constructor(self):
        wavelets.WaveletTransformer(func_list=[0, 0], f1=[0, 0], data=[0, 0],
                                    time=[0, 0], omegas=[0, 0], taus=[0, 0], c=0.0125)

    def test_wavelet_transform_no_gap(self):
        from tqdm import tqdm

        def make_test_time_signal():
            '''
            Can switch this out for something else later
            '''
            num_steps = 100
            t = np.arange(-1, 1, num_steps)
            freq = 1
            phase = np.pi / 4
            x = np.cos(2 * np.pi * freq * t + phase)
            return t, x

        omegas = [1]
        taus = [0]
        c = 1 / (8 * np.pi ** 2)

        def slow_wavelet_transform(t, x, omegas, taus):
            transforms = np.zeros((len(omegas), len(taus), len(x)))
            for i, omega in enumerate(omegas):
                for j, tau in enumerate(taus):
                    z = omega * (t - tau)
                    dwt = np.cumsum(np.exp(-c * z ** 2) * (np.exp(1j * z) - np.exp(-1 / (4 * c))) * np.diff(t)[0])



        transformer = wavelets.WaveletTransformer(
            time = t, 
            data = x, 
            omegas = omegas, 
            taus = omegas, 
            c = c
        )

        assert np.allclose(transformer.compute_wavelet())

    # def test_omegas_taus_from_min_max_nu(self):
    #     wav = wavelets.WaveletTransformer(func_list=[0, 0], f1=[0, 0], data=[0, 0], time=[0, 0], omegas=[0, 0], taus=[0, 0], c=0.0125)

    #     res = wav.auto_compute(1, 1, 1, 1, 1, 1)
    #     self.assertIsNotNone(res)
    #     self.assertIsInstance(res[0], np.ndarray)


if __name__ == '__main__':
    unittest.main()
