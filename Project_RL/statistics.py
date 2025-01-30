import numpy as np
from scipy.stats import ttest_1samp

uniform_baseline = -4186919.5
sma = -3734779.52
ema = -3417281.6
tab_q = [-3486585, -3486585, -3486585, -3486585, -3487298]

print(f"Mean: {np.mean(tab_q)}, SD: {np.std(tab_q)}")

uniform_stats = ttest_1samp(tab_q, uniform_baseline)
sma_stats = ttest_1samp(tab_q, sma)
ema_stats = ttest_1samp(tab_q, ema)

print(f"Uniform stats: {uniform_stats}")
print(f"SMA stats: {sma_stats}")
print(f"EMA stats: {ema_stats}")