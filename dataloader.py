import csv
import os
import librosa
import torch.utils.data as data
import numpy as np

def mu_law_encode(signal, quantization_channels=256):
    # Manual mu-law companding and mu-bits quantization
    mu = quantization_channels - 1

    magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    # signal = (signal + 1) / 2 * mu + 0.5
    # quantized_signal = signal.astype(np.int32)
    return signal

class NttDataset(data.Dataset):
    """
        The pieces have min len 2 sec. We split them y short frames L sec. So, the number of frames is not really known,
        but at least 2/L * num pieces.
    """
    def __init__(self, root_dir='data', folds_total=2, chunk_exclude=0, validation=False, frame_len_sec=0.25, add_noise=False, add_shift=False):
        self.add_noise = add_noise
        self.add_shift = add_shift
        self.samples = None

        # if validation, we only load one chunk chunk_exclude
        if validation:
            loaded = np.load(os.path.join(root_dir, f"fold_{chunk_exclude}_{folds_total}.npz"))
            self.samples = loaded['cache_data']
            self.labels = loaded['cache_labels']
        else:
            # start with the very first chunk
            if chunk_exclude == 0:
                i = 1
            else:
                i = 0
            loaded = np.load(os.path.join(root_dir, f"fold_{i}_{folds_total}.npz"))
            self.samples = loaded['cache_data']
            self.labels = loaded['cache_labels']
            # then load the rest
            for i in range(1, folds_total):
                if i != chunk_exclude:
                    loaded = np.load(os.path.join(root_dir, f"fold_{i}_{folds_total}.npz"))
                    chunk_samples= loaded['cache_data']
                    chunk_labels = loaded['cache_labels']
                    self.samples = np.concatenate((self.samples, chunk_samples), axis=0)
                    self.labels  = np.concatenate((self.labels, chunk_labels), axis=0)

        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        data = self.samples[index]
        if self.add_shift:
            # we modify only 50% of data
            if np.random.choice([True,False]):
                datalen = len(data)
                shift = np.random.randint(-datalen*0.1,datalen*0.1)
                if shift > 0:
                    data[0:datalen-shift] = data[shift:]
                else:
                    shift = -shift
                    data[shift:] = data[0:datalen-shift]

        if self.add_noise:
            data += np.random.randn(*data.shape) * np.random.rand(1) * 0.2
        return np.expand_dims(data, 0), self.labels[index].astype(np.int)

    def __len__(self):
        return self.num_samples


class NttTestDataset(data.Dataset):
    """
    """
    def __init__(self, frame_len_sec=0.25):
        self.root_dir = "/Volumes/KProSSD/Datasets/ntt/"
        if not os.path.isdir(self.root_dir):
            # windows
            self.root_dir = "D:/Datasets/ntt/"

        frame_len_sec = frame_len_sec
        sample_rate = 16000

        label_file = os.path.join(self.root_dir, "sample_submit.tsv")

        np.random.seed(666)
        self.sample_list = []

        # load metadata. We will use the hash, but ignore the class for now (replace it with proper one later)
        with open(label_file, newline='') as csvfile:
            my_reader = csv.DictReader(csvfile,
                                       fieldnames=['hash', 'class'],
                                       delimiter='\t')
            for row in my_reader:
                self.sample_list.append(row)

        self.num_samples = len(self.sample_list)
        self.frame_len = int(frame_len_sec * sample_rate)

    def __getitem__(self, index):
        # returns a set of frames cut from the wav-piece
        if isinstance(index, list):
            # we cannot make a batch, because the length of samples is different
            raise NotImplementedError
        else:
            file_name = os.path.join(self.root_dir, "test", self.sample_list[index]['hash'] + '.wav')
            data, _ = librosa.load(file_name, sr=None)
            data = mu_law_encode(data)

            cache_data = []
            num_frames = len(data) // self.frame_len

            for c in range(num_frames):
                cache_data.append(data[c * self.frame_len: (c + 1) * self.frame_len])
            cache_data_np = np.array(cache_data)

            return cache_data_np, self.sample_list[index]['hash']
            # return np.expand_dims(data,0), self.sample_list[index]['hash']

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    params = {'batch_size': 12,
              'shuffle': True,
              'num_workers': 0}
    training_set = NttDataset(folds_total=2, chunk_exclude=666, add_noise=True)
    training_generator = data.DataLoader(training_set, **params)
    for local_batch, local_labels in training_generator:
        print(local_batch)

    # ntt = NttTestDataset()
    # print(ntt[1])
