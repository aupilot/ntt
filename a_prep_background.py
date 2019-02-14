import os
import librosa
from joblib import Parallel, delayed
import numpy as np
from dataloader import spectrum


sr = 16000
bg_wav_dir = './background/audio'
bg_wav_list = sorted(os.listdir(bg_wav_dir))

# bg_wav_list = bg_wav_list[0:100]

def convert_wav2spec(file_name):
    data, _ = librosa.load(os.path.join(bg_wav_dir,file_name), mono=True, sr=sr)
    spec = spectrum(data, sr)
    return spec


out = Parallel(n_jobs=6, verbose=1)(delayed(convert_wav2spec)(f_name) for f_name in bg_wav_list)

np.save('./background/backgrounds.npy', out)
