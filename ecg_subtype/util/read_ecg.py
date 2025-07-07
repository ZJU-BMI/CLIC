from typing import Optional, Union, Tuple, List, Any

import numpy as np
import torch
from bs4 import BeautifulSoup
from scipy.signal import sosfiltfilt, butter, resample
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def get_ecg(ecg_path):
    ecg_file = open(ecg_path).read()
    bs = None
    try:
        bs = BeautifulSoup(ecg_file, features="lxml")
    except Exception as e:
        print(f"Error when parsing {ecg_path}: {e}")
        return None
    ecg_waveform_length = 5000
    if ecg_waveform_length == 600:
        waveform = bs.body.cardiologyxml.mediansamples
    else:
        waveform = bs.body.cardiologyxml.stripdata
    # print(waveform)
    # print(type(waveform))
    data_numpy = None
    bs_measurement = bs.body.cardiologyxml.restingecgmeasurements
    heartbeat = int(bs_measurement.find_all("VentricularRate".lower())[0].string)

    for each_wave in waveform.find_all("waveformdata"):
        each_data = each_wave.string.strip().split(",")
        each_data = [s.replace('\n\t\t', '') for s in each_data]
        each_data = np.array(each_data, dtype=np.float32)
        # plt.plot(each_data)
        seasonal_decompose_result = seasonal_decompose(each_data, model="additive",
                                                       period=int(ecg_waveform_length * 6 / heartbeat))
        trend = seasonal_decompose_result.trend
        start, end = 0, ecg_waveform_length - 1
        sflag, eflag = False, False
        for i in range(ecg_waveform_length):
            if np.isnan(trend[i]):
                start += 1
            else:
                sflag = True
            if np.isnan(trend[ecg_waveform_length - 1 - i]):
                end -= 1
            else:
                eflag = True
            if sflag and eflag:
                break
        trend[:start] = trend[start]
        trend[end:] = trend[end]
        # trend[np.isnan(trend)] = 0.0
        result = np.array(seasonal_decompose_result.observed - trend)
        # plt.plot(result)
        # plt.show()
        # exit()
        if data_numpy is None:
            data_numpy = result
        else:
            data_numpy = np.vstack((data_numpy, result))

    return data_numpy


# 一种预处理代码，叫st-meme模型ICLR2024

class Resample:
    """Resample the input sequence.
    """

    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x


class SOSFilter:
    """Apply SOS filter to the input sequence.
    """

    def __init__(self,
                 fs: int,
                 cutoff: float,
                 order: int = 5,
                 btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)


class RandomCrop:
    """Crop randomly the input sequence.
    """

    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.random.randint(0, x.shape[1] - self.crop_length + 1)
        return x[:, start_idx:start_idx + self.crop_length]


class NCrop:
    """Crop the input sequence to N segments with equally spaced intervals.
    """

    def __init__(self, crop_length: int, num_segments: int) -> None:
        self.crop_length = crop_length
        self.num_segments = num_segments

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.arange(start=0,
                              stop=x.shape[1] - self.crop_length + 1,
                              step=(x.shape[1] - self.crop_length) // (self.num_segments - 1))
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)


class HighpassFilter(SOSFilter):
    """Apply highpass filter to the input sequence.
    """

    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')


class LowpassFilter(SOSFilter):
    """Apply lowpass filter to the input sequence.
    """

    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')


class Standardize:
    """Standardize the input sequence.
    """
    def __init__(self, axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2)) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        # Set rst = 0 if std = 0
        return np.divide(x - loc, scale, out=np.zeros_like(x), where=scale != 0)


class Compose:
    """Compose several transforms together.
    """

    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x


class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(self, dtype: Union[str, torch.dtype] = torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)


def get_ecg_transforms() -> Tuple[Compose, Compose]:
    """
    Get ECG transforms for training or testing.
    """
    train_transforms = Compose([
            Resample(target_fs=250),
            RandomCrop(2250),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])
    val_transforms = Compose([
            Resample(target_fs=250),
            NCrop(2250, 3),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])

    return train_transforms, val_transforms
