import numpy as np
import pytest

from jazzhands import wavelets


class TestWavelets(object):
    def test_correct_constructor(self):
        wavelets.WaveletTransformer(func_list=[0, 0], f1=[0, 0], counts=[0, 0],
                                    time=[0, 0], omegas=[10, 10], taus=[0, 0], c=0.0125)

    def test_incorrect_constructor(self):
        with pytest.raises(TypeError) as excinfo:
            wavelets.WaveletTransformer(func_list=[0, 0], f1=[0, 0], counts=[0, 0], time=[0, 0], omegas="hello", taus="blah", c=0.0125)
        assert str(excinfo.value)

    def test_omegas_taus_from_min_max_nu(self):
        wav = wavelets.WaveletTransformer(func_list=[0, 0], f1=[0, 0], counts=[0, 0], time=[0, 0], omegas=[10, 10], taus=[0, 0], c=0.0125)

        res = wav._omegas_taus_from_min_max_nu(1, 1, 1, 1)

        assert res is not None
        assert isinstance((res[0] and res[1]), np.ndarray)
