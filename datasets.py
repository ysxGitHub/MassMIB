"""
@time: 2021/12/

@ author: ysx
"""
import utils
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from data_process import load_dataset, compute_label_aggregations, select_data, preprocess_signals, data_slice, \
    load_raw_data_hf, load_data_chapman  # del_data_too_zero
from librosa.filters import mel as librosa_mel_fn
from data_augmentation import time_out, random_resized_crop


def hf_dataset(root='../data/hf/', resample_num=1000, num_classes=34):
    data, label = load_raw_data_hf(root, resample_num, num_classes)
    # print(label.sum(axis=0), label.sum(axis=0).sum(), len(label))
    data_num = len(label)
    shuffle_ix = np.random.permutation(np.arange(data_num))
    data = data[shuffle_ix]
    labels = label[shuffle_ix]

    X_train = data[int(data_num * 0.2):int(data_num * 0.8)]
    y_train = labels[int(data_num * 0.2):int(data_num * 0.8)]

    X_val = data[int(data_num * 0.8):]
    y_val = labels[int(data_num * 0.8):]

    X_test = data[:int(data_num * 0.2)]
    y_test = labels[:int(data_num * 0.2)]

    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        masks1 = []
        for _ in range(12):
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            temp = []
            for j in mask:
                for k in range(10):
                    temp.append(j)
            masks1.append(temp)

        masks2 = []
        for _ in range(12):
            img = np.ones((50, 20))
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            mask = mask.reshape(10, 10)
            for index_i, i in enumerate(mask):
                for index_j, j in enumerate(i):
                    if j == 0:
                        for k in range(5):
                            img[index_i * 5 + k][index_j * 2] = 0
                            img[index_i * 5 + k][index_j * 2 + 1] = 0
            masks2.append(img)

        # mask = np.hstack([
        #     np.zeros(self.num_patches - self.num_mask),
        #     np.ones(self.num_mask),
        # ])
        # np.random.shuffle(mask)
        return np.array(masks1), np.array(masks2)  # [196]


class ECGDatasetPretrain(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals, window_size, mask_ratio=0.):
        super(ECGDatasetPretrain, self).__init__()
        self.data = signals

        if mask_ratio != 0.:
            self.mask = RandomMaskingGenerator(window_size, mask_ratio)
        else:
            self.mask = None

    def __getitem__(self, index):
        x1 = self.data[index]

        x1 = x1.transpose()
        # x2 = x2.transpose()

        x1 = torch.tensor(x1.copy(), dtype=torch.float)
        # x2 = torch.tensor(x2.copy(), dtype=torch.float)
        x2 = mel_spectrogram(x1)

        return x1, x2, self.mask()

    def __len__(self):
        return len(self.data)


class ECGDatasetFinetuneZ(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals, labels):
        super(ECGDatasetFinetuneZ, self).__init__()
        self.data1 = signals
        self.label = labels
        self.num_classes = len(np.unique(labels))
        # self.num_classes = self.label.shape[1]
        # self.cls_num_list = np.sum(self.label, axis=0)
        # self.trans = transforms.RandomResizedCrop((50, 20))

    def __getitem__(self, index):
        x1 = self.data1[index]

        y = self.label[index]

        # data aug
        # x1 = time_out(random_resized_crop(x1))

        x1 = x1.transpose()
        # x2 = x2.transpose()

        x1 = torch.tensor(x1.copy(), dtype=torch.float)
        # x2 = torch.tensor(x2.copy(), dtype=torch.float)
        x2 = mel_spectrogram(x1)
        # x2 = self.trans(x2)

        #                   dtype = torch.long
        y = torch.tensor(y, dtype=torch.long)
        y = y.squeeze()
        return (x1, x2), y

    def __len__(self):
        return len(self.data1)


class ECGDatasetFinetune(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals, labels):
        super(ECGDatasetFinetune, self).__init__()
        self.data1 = signals
        self.label = labels
        # self.num_classes = len(np.unique(labels))
        self.num_classes = self.label.shape[1]
        # self.cls_num_list = np.sum(self.label, axis=0)
        # self.trans = transforms.RandomResizedCrop((50, 20))

    def __getitem__(self, index):
        x1 = self.data1[index]

        y = self.label[index]

        # data aug
        # x1 = time_out(random_resized_crop(x1))

        x1 = x1.transpose()
        # x2 = x2.transpose()

        x1 = torch.tensor(x1.copy(), dtype=torch.float)
        # x2 = torch.tensor(x2.copy(), dtype=torch.float)
        x2 = mel_spectrogram(x1)
        # x2 = self.trans(x2)

        #                   dtype = torch.long
        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return (x1, x2), y

    def __len__(self):
        return len(self.data1)


class DownLoadTwoECGDataSets:
    '''
        All experiments data
    '''

    def __init__(self, experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
                 train_fold=8, val_fold=9, test_fold=10):
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.experiment_name = experiment_name
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

    def preprocess_data(self):
        # Load PTB-XL data
        data, raw_labels = load_dataset(self.datafolder, self.sampling_frequency)
        # Preprocess label data
        labels = compute_label_aggregations(raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        data, labels, Y, _ = select_data(data, labels, self.task, self.min_samples)

        if self.datafolder == '../data/CPSC/':
            data = data_slice(data)

        # 10th fold for testing (9th for now)
        X_test = data[labels.strat_fold == self.test_fold]
        y_test = Y[labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        X_val = data[labels.strat_fold == self.val_fold]
        y_val = Y[labels.strat_fold == self.val_fold]
        # rest for training
        X_train = data[labels.strat_fold <= self.train_fold]
        y_train = Y[labels.strat_fold <= self.train_fold]

        # X_train, y_train = del_data_too_zero(X_train, y_train)
        # X_val, y_val = del_data_too_zero(X_val, y_val)
        # X_test, y_test = del_data_too_zero(X_test, y_test)

        # Preprocess signal data
        X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test


mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft=200, num_mels=50, sampling_rate=100, hop_size=50, win_size=200, fmin=1, fmax=45,
                    center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_datasets_finetune(datafolder=None, experiment=None):
    '''
    Load the final dataset
    '''
    experiment = experiment

    if datafolder == '../data/ptbxl/':
        experiments = {
            'exp0': ('exp0', 'all'),
            'exp1': ('exp1', 'diagnostic'),
            'exp1.1': ('exp1.1', 'subdiagnostic'),
            'exp1.1.1': ('exp1.1.1', 'superdiagnostic'),
            'exp2': ('exp2', 'form'),
            'exp3': ('exp3', 'rhythm')
        }
        name, task = experiments[experiment]
        ded = DownLoadTwoECGDataSets(name, task, datafolder)
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
        # X_train_f, X_val_f, X_test_f = multi_data_stft(X_train, X_val, X_test)
    elif datafolder == '../data/CPSC/':
        ded = DownLoadTwoECGDataSets('exp_CPSC', 'all', datafolder)
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
        # X_train_f, X_val_f, X_test_f = multi_data_stft(X_train, X_val, X_test)
    elif datafolder == '../data/ZheJiang/':
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_chapman(datafolder)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
        assert X_train != None

    if datafolder != '../data/ZheJiang/':
        ds_train = ECGDatasetFinetune(X_train, y_train)
        ds_val = ECGDatasetFinetune(X_val, y_val)
        ds_test = ECGDatasetFinetune(X_test, y_test)
    else:
        ds_train = ECGDatasetFinetuneZ(X_train, y_train)
        ds_val = ECGDatasetFinetuneZ(X_val, y_val)
        ds_test = ECGDatasetFinetuneZ(X_test, y_test)

    num_classes = ds_train.num_classes

    return ds_train, ds_val, ds_test, num_classes


def load_datasets_pretrain(datafolder=None, window_size=(1, 100), mask_ratio=0.75):
    '''
    Load the final dataset
    '''
    if 'ptbxl' in datafolder:
        ded = DownLoadTwoECGDataSets('exp0', 'all', '../data/ptbxl/')
        X_train, _, _, _, _, _ = ded.preprocess_data()
    elif 'CPSC' in datafolder:
        ded = DownLoadTwoECGDataSets('exp_CPSC', 'all', '../data/CPSC/')
        X_train, _, _, _, _, _ = ded.preprocess_data()
    elif 'ZheJiang' in datafolder:
        X_train, _, _, _, _, _ = load_data_chapman('../data/ZheJiang/')
    # elif 'hf' in datafolder:
    #     X_train, y_train, X_val, y_val, X_test, y_test = hf_dataset(datafolder)
    else:
        ded = DownLoadTwoECGDataSets('exp0', 'all', '../data/ptbxl/')
        X_train1, _, _, _, _, _ = ded.preprocess_data()
        ded = DownLoadTwoECGDataSets('exp_CPSC', 'all', '../data/CPSC/')
        X_train2, _, _, _, _, _ = ded.preprocess_data()
        X_train3, _, _, _, _, _ = load_data_chapman('../data/ZheJiang/')
        X_train = np.concatenate((X_train1, X_train2, X_train3), axis=0)
    ds_train = ECGDatasetPretrain(X_train, window_size, mask_ratio)
    return ds_train



