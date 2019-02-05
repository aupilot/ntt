import os
import librosa
import csv
import numpy as np

from dataloader import mu_law_encode

folds_total = 2
this_fold_no = 1
frame_len_sec = 0.25
cut_len = None   # None - for full len of fold

classes = ['MA_CH', 'MA_AD', 'MA_EL', 'FE_CH', 'FE_EL', 'FE_AD']


root_dir = "/Volumes/KProSSD/Datasets/ntt/"
if not os.path.isdir(root_dir):
    # windows
    root_dir = "D:/Datasets/ntt/"

os.makedirs("./data", exist_ok=True)

label_file = os.path.join(root_dir, "class_train.tsv")
sample_rate = 16000

np.random.seed(666)
sample_list = []

# load metadata
with open(label_file, newline='') as csvfile:
    my_reader = csv.DictReader(csvfile,
                               fieldnames=['hash', 'class'],
                               delimiter='\t')
    for row in my_reader:
        sample_list.append(row)

# split folds and take current fold only
num_samples = len(sample_list) // folds_total   # колич wav фрагментов
current_fold = sample_list[this_fold_no*num_samples:(this_fold_no+1)*num_samples]

frame_len = int(frame_len_sec * sample_rate)

# == we don't mix here! It's better of they are of different types kinda
# indices = np.random.permutation(num_samples)
#
# if cut_len is None:
#     cache_samples = [current_fold[i] for i in indices]
# else:
#     cache_samples = [current_fold[i] for i in indices[0:cut_len]]
if cut_len is None:
    cache_samples = current_fold
else:
    cache_samples = current_fold[0:cut_len]


cache_data = []
cache_labels = []

for i in range(len(cache_samples)):
    file_name = os.path.join(root_dir, "train", cache_samples[i]['hash'] + '.wav')
    data, _ = librosa.load(file_name, sr=None)  # keep sample rate the same
    data = mu_law_encode(data)

    cluss = cache_samples[i]['class']

    num_frames = len(data) // frame_len

    for c in range(num_frames):
        cache_data.append(data[c * frame_len: (c + 1) * frame_len])
        cache_labels.append(cluss)

cache_data_np = np.array(cache_data)
cache_labels_np = np.array([classes.index(value) for count, value in enumerate(cache_labels)])

# np.savez_compressed(f"./data/fold_{this_fold_no}_{folds_total}",cache_data=cache_data_np, cache_labels=cache_labels_np )
np.savez(f"./data/fold_{this_fold_no}_{folds_total}", cache_data=cache_data_np, cache_labels=cache_labels_np)



