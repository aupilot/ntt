import csv
import os
import random
import secrets
import librosa
import torch.utils.data as data
import numpy as np
import threading

from librosa import filters
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


classes_list = ['MA_CH', 'MA_AD', 'MA_EL', 'FE_CH', 'FE_EL', 'FE_AD']


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
            # data = mu_law_encode(data)

            cache_data = []
            num_frames = len(data) // self.frame_len

            for c in range(num_frames):
                cache_data.append(data[c * self.frame_len: (c + 1) * self.frame_len])
            cache_data_np = np.array(cache_data)

            return cache_data_np, self.sample_list[index]['hash']
            # return np.expand_dims(data,0), self.sample_list[index]['hash']

    def __len__(self):
        return self.num_samples


class NttDataset2(data.Dataset):
    """
        Init:
        Load the piece list.
        Load 128 pieces in memory, augment them and set the reading head to random position.

        Get batch
        Prepare one frame from each wav (heads read wavs). Once a wav come to end, load a new wav and augment it

    """
    def __init__(self, root_dir=None, folds_total=2, chunk_exclude=0, validation=False, frame_len_sec=0.25):

        self.root_dir = root_dir

        # we wanted multiprocessing - each thread has its own seed -- does not help!!
        # np.random.seed(np.array(threading.get_ident(), dtype=np.int32))
        # np.random.seed(int.from_bytes(secrets.token_bytes(2),byteorder='big'))

        self.sr = sample_rate = 16000
        self.cache_size = 128
        self.cache_pieces = []
        self.cache_headpo = []
        self.cache_label  = []
        # self.frame_stride = 400         # we use random instead
        self.needs_init = True

        label_file = os.path.join(root_dir, "class_train.tsv")
        self.frame_len = int(frame_len_sec * sample_rate)

        sample_list = []

        # load metadata
        with open(label_file, newline='') as csvfile:
            my_reader = csv.DictReader(csvfile,
                                       fieldnames=['hash', 'class'],
                                       delimiter='\t')
            for row in my_reader:
                sample_list.append(row)

        chunk_size = len(sample_list) // folds_total
        chunk_list = [sample_list[i:i + chunk_size] for i in range(0, len(sample_list), chunk_size)]

        if not validation:
            # make a list with the chunk excluded
            new_list = []
            for i in range(folds_total):
                if i != chunk_exclude:
                    new_list.append(chunk_list[i])

            # convert chunk_list to a flat list of all avaialble pieces
            self.flat_list = [item for sublist in new_list for item in sublist]

        else:
            self.flat_list = chunk_list[chunk_exclude]

        self.num_pieces = len(self.flat_list)

        # fill the cache with an initial pieces and head positions
        for w in range(self.cache_size):
            # piece = random.randrange(self.num_pieces)   # this random uses OS urandom
            piece = np.random.randint(self.num_pieces)
            file_name = os.path.join(root_dir, "train", self.flat_list[piece]['hash'] + '.wav')
            # data, _ = librosa.load(file_name, sr=None)  # keep sample rate the same
            data = load_wav(file_name)  # load, notmalise, trim blank ends
            # self.cache_headpo.append(np.random.randint(len(data) - self.frame_len + 1))
            self.cache_headpo.append(0)     # we must use zero to be compatible with spectro
            self.cache_label.append(classes_list.index(self.flat_list[piece]['class']))
            # convert data to spectrum at the last step to preserve len() before spectogram
            spectrum = self.wav_preprocess(data)
            self.cache_pieces.append(spectrum)

        self.current_piece = 0
        self.fake_len = self.num_pieces * 8    # we don't know the real length. Just use something
        # we wanted to make it ridiculously high to prevent calling constructor every now and then,
        # but can't do that - it allocates too much RAM

    def __getitem__(self, index):
        # Here is a trick. If this is the very first call, we will init the random generator.
        # Otherwise all threads give same batch
        if self.needs_init:
            np.random.seed(int.from_bytes(secrets.token_bytes(2), byteorder='big') + index)
            self.needs_init = False

        frame, label = self.prep_item(index)
        return np.expand_dims(frame, 0), label

    def prep_item(self, index):
        # get sample from the current piece. We don't care about index
        st = self.cache_headpo[self.current_piece]
        en = st + self.frame_len
        frame = self.cache_pieces[self.current_piece][st:en]
        label = self.cache_label[self.current_piece]
        self.cache_headpo[self.current_piece] = st + np.random.randint(300,600)
        if en+self.frame_len >= len(self.cache_pieces[self.current_piece]):
            # if the current piece is finished
            piece = np.random.randint(self.num_pieces)
            file_name = os.path.join(self.root_dir, "train", self.flat_list[piece]['hash'] + '.wav')
            # data, _ = librosa.load(file_name, sr=None)  # keep sample rate the same
            data = load_wav(file_name)  # load, notmalise, trim blank ends
            data = self.wav_preprocess(data)
            self.cache_pieces[self.current_piece] = data
            self.cache_headpo[self.current_piece] = 0
            self.cache_label[self.current_piece] = classes_list.index(self.flat_list[piece]['class'])

        self.current_piece = np.random.randint(0, self.cache_size)

        return frame, label

    def __len__(self):
        return self.fake_len

    def wav_preprocess(self, data):

        # amplitude
        # if np.random.choice([True, False]):
        #     amplification_rate = np.random.randn() * 0.3 + 1
        #     data = data * amplification_rate

        # time stretch
        # if np.random.choice([True, False, False]):
        #     stretch_rate = np.random.rand() * 0.4 + 0.8
        #     data = librosa.effects.time_stretch(data, stretch_rate) # positive - faster

        # pitch shift
        # if np.random.choice([True, False, False]):
        #     shift_steps = np.random.choice([-6, -4, -2, 2, 4, 6])
        #     data = librosa.effects.pitch_shift(data, sr=self.sr, n_steps=shift_steps, bins_per_octave=200)

        # resample
        # https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf
        # - bad idea? confusing!
        # resample_rate = np.random.choice([0.9, 1, 1.1])
        # if resample_rate != 1:
        #     data = librosa.resample(data, self.sr, (self.sr * resample_rate))


        # compress - bad idea. quality gets shit
        # data = mu_law_encode(data)

        return data


# ================================================================================
def spectrum(data, sr):
    data = librosa.feature.melspectrogram(data, sr=sr, n_fft=1024, hop_length=256)
    data = np.log10(data + 1e-6)
    return data


def load_wav(file_name):
    data, sr = librosa.load(file_name, sr=None)
    # librosa.output.write_wav('_orig.wav', data, sr=sr)

    # Normalise volume
    data = librosa.util.normalize(data)
    # librosa.output.write_wav('_pcen.wav', data, sr=sr)

    # # Trim the beginning and ending silence
    # data, _ = librosa.effects.trim(data, top_db=25)
    # librosa.output.write_wav('_trim.wav', data, sr=sr)

    # remove all silences
    data = cut_silence(data, top_db=25)
    # librosa.output.write_wav('_trim.wav', data, sr=sr)

    return data


def cut_silence(data, top_db=25):
    intervals = librosa.effects.split(data, top_db=top_db,  frame_length=1024, hop_length=256)
    new_data = np.zeros(1)
    for interval in intervals:
        start = interval[0]
        end = interval[1]
        audio_chunk = data[start:end]

        new_data = np.concatenate((new_data, audio_chunk),0)

    return new_data




# ================================================================================


# def inverse_mel(mel, sr):
#     mel = np.power(10, mel)
#     mel_basis = filters.mel(sr, n_fft=1024)
#     inverted_spectrogram = np.dot(mel_basis.T, mel)


class NttDataset3(NttDataset2):
    """
    Spectrograms.
    We will pad the spectrogram with blanks to make a constant length (or trim it at the end)
    """
    def __init__(self, root_dir=None, folds_total=2, chunk_exclude=0, validation=False, frame_len_sec=0.25, add_bg=False):
        if add_bg:
            self.backgrounds = np.load('./background/backgrounds.npy')
        self.add_bg = add_bg
        print(f"Dataloader constructor called{np.random.rand(1)}")
        super(NttDataset3, self).__init__(root_dir, folds_total, chunk_exclude, validation, frame_len_sec)
        self.frame_len = 128
        self.frame_stride = 8

    def __getitem__(self, index):
        # Here is a trick. If this is the very first call, we will init the random generator.
        # Otherwise all threads give same batch
        if self.needs_init:
            np.random.seed(int.from_bytes(secrets.token_bytes(2), byteorder='big') + index)
            self.needs_init = False

        frame, label = self.prep_spec(index)
        # return np.expand_dims(frame, 0), label

        # make it compatible with resnet
        frame = np.expand_dims(frame, 0)
        # frame = np.repeat(frame, 3, 0)     # uncomment for a proper ResNet with 3 channel input
        return frame, label

    def wav_preprocess(self, data):
        '''
        augmentation:
        1. Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
        2. http://www.mirlab.org/conference_papers/International_Conference/ISMIR%202015/website/articles_splitted/264_Paper.pdf
        - dynamic range compression
        - add background sounds
        - added noise and dropout
        - random filters
        - loudness (to spectrogram)
        - pitch shift +-10% ???
        - tempo shift ???
        '''


        # == resample
        # if np.random.choice([True, False, False]):
        #     resample_rate = np.random.choice([0.9, 1.1])
        #     if resample_rate != 1:
        #         data = librosa.resample(data, self.sr, (self.sr * resample_rate))

        # == time stretch +-20%
        # if np.random.choice([True, False, False]):
        #     stretch_rate = np.random.rand() * 0.4 + 0.8
        #     data = librosa.effects.time_stretch(data, stretch_rate) # positive - faster

        # data = librosa.feature.mfcc(y=data, sr=self.sr)
        # data = librosa.feature.melspectrogram(data, sr=self.sr, n_fft=1024, hop_length=256)
        # data = np.log10(data + 1e-6)
        # data = data - data.mean()

        data = spectrum(data, self.sr)

        # amplify spectrogram
        if np.random.choice([True, True]):
            amplify = np.random.rand() * 0.6 + 0.7   # +-30%
            data = data * amplify

        # add gaussian noise
        if np.random.choice([True, True]):
            noise_level = np.random.rand() * data.std() * 1.0
            data = data + np.random.randn(data.shape[0],data.shape[1]) * noise_level

        # zoom the spectrogram
        if np.random.choice([True, False]):
            zoom_len = np.random.rand() * 0.6 + 0.7   # 30%
            zoom_frq = 1 # np.random.rand() * 0.1 + 0.95   # 5%
            zoomed = zoom(data, (zoom_frq, zoom_len))
            if zoom_frq >= 1:
                data = zoomed[0:128, 0:zoomed.shape[1]]
            else:
                data = np.random.randn(128,zoomed.shape[1])
                data[0:zoomed.shape[0],0:zoomed.shape[1]] = zoomed[:, :]

        pitch_aug = np.random.choice(['none', 'up', 'down'])
        
        # shift pitch UP
        if pitch_aug is 'up':
            shift = np.random.choice([3,2,1])
            data[0:128-shift, :] = data[shift:128, :]
            data[128-shift:128,:] = np.random.randn(shift, data.shape[1])

        # shift pitch DOWN
        if pitch_aug is 'down':
            shift = np.random.choice([1,2,3])
            data[shift:128, :] = data[0:128-shift, :]
            data[0:shift,:] = np.random.randn(shift, data.shape[1])

        # add no more than 25% of bg sounds
        if self.add_bg:
            bg = self.backgrounds[np.random.randint(0,self.backgrounds.shape[0]),:,:]
            bg = bg / bg.std() * data.std()
            if data.shape[1] < bg.shape[1]:
                data = data + bg[:, 0:data.shape[1]] * np.random.rand() * 0.25
            else:
                data[:, 0:bg.shape[1]] = data[:, 0:bg.shape[1]] + bg * np.random.rand() * 0.25

        # plt.imshow(data)
        # plt.show()

        return data

    def prep_spec(self, index):
        # get sample from the current piece. We don't care about index
        st = self.cache_headpo[self.current_piece]
        en = st + self.frame_len

        piece_len = self.cache_pieces[self.current_piece].shape[1]
        if piece_len >= self.frame_len:
            frame = self.cache_pieces[self.current_piece][:, st:en]
        else:
            dif_left = np.floor((self.frame_len - piece_len) / 2).astype('int')
            dif_right = np.ceil((self.frame_len - piece_len) / 2).astype('int')
            frame = np.pad(self.cache_pieces[self.current_piece], ((0, 0), (dif_left, dif_right)), 'wrap')

        label = self.cache_label[self.current_piece]
        self.cache_headpo[self.current_piece] = st + self.frame_stride
        if en+self.frame_len >= piece_len:
            # if the current piece is finished
            piece_idx = np.random.randint(self.num_pieces)
            file_name = os.path.join(self.root_dir, "train", self.flat_list[piece_idx]['hash'] + '.wav')
            # data, _ = librosa.load(file_name, sr=None)  # keep sample rate the same
            data = load_wav(file_name)  # load, notmalise, trim blank ends
            spectrum = self.wav_preprocess(data)
            self.cache_pieces[self.current_piece] = spectrum
            self.cache_headpo[self.current_piece] = 0
            self.cache_label[self.current_piece] = classes_list.index(self.flat_list[piece_idx]['class'])

        self.current_piece = np.random.randint(0, self.cache_size)

        return frame, label


class NttTestDataset3(data.Dataset):
    """
    """
    def __init__(self):
        self.root_dir = "/Volumes/KProSSD/Datasets/ntt/"
        # self.root_dir = "./data"
        if not os.path.isdir(self.root_dir):
            # windows
            self.root_dir = "D:/Datasets/ntt/"

        self.sr = 16000
        self.frame_len = 128
        self.stride = 16

        label_file = os.path.join(self.root_dir, "sample_submit.tsv")

        self.sample_list = []

        # load metadata. We will use the hash, but ignore the class for now (replace it with proper one later)
        with open(label_file, newline='') as csvfile:
            my_reader = csv.DictReader(csvfile,
                                       fieldnames=['hash', 'class'],
                                       delimiter='\t')
            for row in my_reader:
                self.sample_list.append(row)

        self.num_samples = len(self.sample_list)
        # self.frame_len = int(frame_len_sec * sample_rate)

    def __getitem__(self, index):
        # returns a set of frames cut from the wav-piece
        if isinstance(index, list):
            # we cannot make a batch, because the length of samples is different
            raise NotImplementedError
        else:
            file_name = os.path.join(self.root_dir, "test", self.sample_list[index]['hash'] + '.wav')

            data = load_wav(file_name)      # loads, normalises and trims
            spec = spectrum(data, self.sr)

            cache_data = []

            piece_len = spec.shape[1]
            if piece_len >= self.frame_len:
                # frame = self.cache_pieces[self.current_piece][:, st:en]
                # num_frames = (piece_len - self.frame_len) // self.stride + 1
                s = 0
                for c in range(32):      # no more than 32 frames
                    # s = c * self.frame_len
                    e = s + self.frame_len

                    # sanity check
                    if e > piece_len:
                        break

                    frame = spec[:, s:e]
                    frame = np.expand_dims(frame, 0)
                    # frame = np.repeat(frame, 3, axis=0) # uncomment for a proper ResNet with 3 channel input
                    cache_data.append(frame)

                    if e == piece_len:
                        break

                    if e+self.frame_len//3 > piece_len:
                        s = s + piece_len - e           # if less than wanted stride - we still use the tail!
                    else:
                        s = s + self.frame_len // 3     # we wanted stride = 1/3 of the frame len

                cache_data_np = np.array(cache_data)
                return cache_data_np, self.sample_list[index]['hash'], data
            else:
                dif_left = np.floor((self.frame_len - piece_len) / 2).astype('int')
                dif_right = np.ceil((self.frame_len - piece_len) / 2).astype('int')
                spec = np.pad(spec, ((0, 0), (dif_left, dif_right)), 'mean')
                # spectrum = np.pad(spectrum, ((0, 0), (dif_left, dif_right)), 'wrap')
                spec = np.expand_dims(spec, 0)
                # spectrum = np.repeat(spectrum, 3, axis=0)    # uncomment for a proper ResNet with 3 channel input
                return spec, self.sample_list[index]['hash'], data

    def __len__(self):
        return self.num_samples



if __name__ == "__main__":
    params = {'batch_size': 3,
              'shuffle': True,
              'num_workers': 0}

    # training_set = NttDataset(folds_total=2, chunk_exclude=666, add_noise=True)
    # training_generator = data.DataLoader(training_set, **params)
    # for local_batch, local_labels in training_generator:
    #     print(local_batch)

    # ntt = NttTestDataset()
    # print(ntt[1])

    # training_set = NttDataset2(root_dir="D:/Datasets/ntt/", folds_total=3, chunk_exclude=666)
    # training_generator = data.DataLoader(training_set, **params)

    # training_set = NttDataset3(root_dir="/Volumes/KProSSD/Datasets/ntt/", folds_total=3, chunk_exclude=666, add_bg=True)
    training_set = NttDataset3(root_dir="D:/Datasets/ntt/", folds_total=3, chunk_exclude=666, add_bg=True)
    training_generator = data.DataLoader(training_set, **params)

    for local_batch, local_labels in training_generator:
        print(local_batch)
