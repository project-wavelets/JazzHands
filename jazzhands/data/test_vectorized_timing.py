# finds WWZ with and without vectorization
import numpy as np
from tqdm import tqdm
import time
np.random.seed(sum(map(ord, "wavelets are cool!")))

from jazzhands import wavelets

config = {
    "num_steps" : 100,
    "num_omegas" : 300,
    "num_taus" : 100,
    "tmax" : 1,
    "xmax" : 100,
    "omega_max" : 25 * np.pi,
    "tau_max" : 2 * np.pi,
    "num_unvec_runs" : 1,
    "num_vec_runs" : 1
}

for k in config:
    exec(k + " = " + str(config.get(k)))

t = np.linspace(-tmax, tmax, num_steps)
x = np.random.uniform(-xmax, xmax, num_steps)
omegas = np.random.uniform(0, omega_max, (num_omegas,))
taus = np.random.uniform(0, tau_max, (num_taus,))

transformer = wavelets.WaveletTransformer(t, x, omegas=omegas, taus=taus)
print("Starting unvectorized computation of WWT of {0} datapoints over {1} omega values and {2} tau values".format(num_steps, num_omegas, num_taus))
t0 = time.time()
for _ in range(num_unvec_runs):
    transformer.compute_wavelet(vectorized=False)
t1 = time.time()

print("Starting vectorized computation of WWT of {0} datapoints over {1} omega values and {2} tau values".format(num_steps, num_omegas, num_taus))
t2 = time.time()
for _ in range(num_vec_runs):
    transformer.compute_wavelet(vectorized=True)
t3 = time.time()

print("Unvectorized computation took {0} seconds for {1} computation(s), for an average of {2} seconds per computation.".format(t1 - t0, num_unvec_runs, (t1 - t0) / num_unvec_runs))
print("Vectorized computation took {0} seconds for {1} computation(s), for an average of {2} seconds per computation.".format(t3 - t2, num_vec_runs, (t3 - t2) / num_vec_runs))
