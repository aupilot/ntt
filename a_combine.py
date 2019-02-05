import pandas as pd
import numpy as np
import datetime as dt


dir = 'subs/01'


files = [
    'k666c_noaug_f1_fwd_1_fwd.csv',
    'k666c_noaug_f2_fwd_1_fwd.csv',
]
weights = [
    0.76,
    0.03,
    0.03,
    0.02,
    0.02,
    ]

clip = 0.0001

files = [dir + file for file in files]
print(weights)

subm  = pd.read_csv(files[0])
s0 = subm[subm.columns[1:]]                 # without filename
s0 = s0 * weights[0]

print('\nCombined:\n\t{}'.format(files[0]))
for w, f in zip(weights[1:], files[1:]):
    subm = pd.read_csv(f)
    s = subm[subm.columns[1:]]
    s = s * w
    s0 = s0 + s
    print('\t{}'.format(f))

s0 = s0.clip(clip, 1-clip)
# s0 = s0 / len(files)

# adjust 1.00s and 0.00s by 1e-4
# epsilon = 5e-4
# s0[s0 < epsilon] = epsilon
# s0[s0 > 1.0-epsilon] = 1.0-epsilon

subm[subm.columns[1:]] = s0

tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
file = 'comb_{}.csv'.format(tmp)
subm.to_csv('./{}{}'.format(dir,file), index=False, float_format='%.6f')

