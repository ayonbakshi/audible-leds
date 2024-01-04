import collections
import math
import threading
import time
from contextlib import contextmanager
from multiprocessing import Lock, Manager, Process

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import librosa
import matplotlib
import numpy as np
import pyaudio
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from functools import cache
# from spleeter.separator import Separator

matplotlib.use("TkAgg")


class Timer:
    def __init__(self):
        self.start = time.time()

    def stop(self):
        self.t = time.time() - self.start
        return self.t


@contextmanager
def timer(name=None):
    t = Timer()

    yield t

    t.stop()


def get_channel_id(name):
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxInputChannels"] > 0 and name in device_info["name"]:
            return i

    raise KeyError(f"Can't find {name=} in list of devices")


class AudioInputStream:
    def __init__(
        self,
        channel_name,
        read_chunk_size=256,
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        exception_on_overflow=True,
    ):
        device_index = get_channel_id(channel_name)

        self.stream_args = dict(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
        )
        self.stream = None

        self.channels = channels
        self.chunk_size = read_chunk_size
        self.rate = rate
        self.exception_on_overflow = exception_on_overflow

    def initialize(self):
        if self.stream is not None:
            raise RuntimeError("Can't initialize twice")

        p = pyaudio.PyAudio()
        self.stream = p.open(**self.stream_args)

    def read(self):
        data = self.stream.read(
            self.chunk_size, exception_on_overflow=self.exception_on_overflow
        )
        return np.frombuffer(data, dtype=np.int16)

    def init_streaming_mode(self):
        if self.stream is not None:
            raise RuntimeError("Can't initialize twice")

        self.q = collections.deque([])
        self.total_frames = 0
        self.lock = threading.Lock()

        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.int16)
            data = data.reshape(frame_count, self.channels).transpose()
            with self.lock:
                self.q.append(data)

            return in_data, pyaudio.paContinue

        p = pyaudio.PyAudio()
        self.stream = p.open(
            **self.stream_args,
            frames_per_buffer=self.chunk_size,
            stream_callback=callback,
        )

    @contextmanager
    def get_data(self):
        with self.lock:
            yield self.q


class InteractivePlot:
    def __init__(self, xscale="linear", plot_type="line"):
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        if xscale == "log":
            self.ax.set_xscale("log")
        self.ax.set_ylim(bottom=0, top=1)
        self.line = None

        if plot_type not in ("line", "scatter", "scatter2"):
            raise RuntimeError("cant do that :(")
        self.plot_type = plot_type

    def show(self):
        self.fig.show()

    def update(self, Y, X=None):
        if X is None:
            X = np.arange(len(Y))

        if self.line is None:
            if self.plot_type == "line":
                # (self.line,) = self.ax.plot(X, Y)
                self.line = self.ax.bar(x=X, height=Y, width=1)
            elif self.plot_type == "scatter":
                self.line = self.ax.scatter(X, Y)
            elif self.plot_type == "scatter2":
                _Y = 0.5 * np.ones_like(Y)
                self.line = self.ax.scatter(X, _Y, s=3)

        if self.plot_type == "line":
            for rect, h in zip(self.line, Y):
                rect.set_height(h)
            # self.line.set_xdata(X)
            # self.line.set_ydata(Y)
        elif self.plot_type == "scatter":
            self.line.set_offsets(np.stack([X, Y], axis=-1))
        elif self.plot_type == "scatter2":
            self.line.set_alpha(Y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class InteractiveDoublePlot:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.ax1.set_ylim(bottom=-1, top=1)
        self.ax2.set_ylim(bottom=-1, top=1)
        self.line1, self.line2 = None, None

    def show(self):
        self.fig.show()

    def update(self, Y1, Y2):
        if self.line1 is None:
            (self.line1,) = self.ax1.plot(Y1)
            (self.line2,) = self.ax2.plot(Y2)

        self.line1.set_ydata(Y1)
        self.line2.set_ydata(Y2)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def populate_data(data_queue, lock, audio_stream: AudioInputStream, evict_interval=1):
    audio_stream.initialize()

    while True:
        try:
            data = audio_stream.read()
            with lock:
                data_queue.append(data)
        except KeyboardInterrupt:
            break

def normalize(data):
    data = data.astype(np.float32) / (np.iinfo(np.uint16).max / 2)
    return data

def preprocess_data(data, hann_tail=0.25):
    # apply hanning window
    # hann = np.hanning(round(len(data) * 1))
    # hann = np.pad(hann, pad_width=(len(data) - len(hann), 0))
    recency = np.arange(1, len(data)+1) / len(data)
    # multiplier = hann * recency

    data = data * recency

    # add padding
    padding_size = len(data) // 2
    data = np.pad(data, pad_width=padding_size)

    return data


def create_spectrum(data, sampling_freq):
    spectrum = np.fft.rfft(data)
    spectrum = np.abs(spectrum)

    freqs = np.fft.rfftfreq(len(data), 1 / sampling_freq)

    return spectrum, freqs


@cache
def mel_transform_and_freqs(n_fft, fmin, fmax, sampling_freq, bands):
    mel_filterbank = librosa.filters.mel(
        sr=sampling_freq, n_fft=n_fft, n_mels=bands, fmin=fmin, fmax=fmax
    )

    mel_frequencies = librosa.mel_frequencies(n_mels=bands, fmin=fmin, fmax=fmax)

    return mel_filterbank, mel_frequencies


def mel_filterbank(dft, sampling_freq, bands=128):
    n_fft = (len(dft) - 1) * 2  # since we only looked at positive values
    fmin = 0
    fmax = sampling_freq / 2

    mel_filterbank, mel_frequencies = mel_transform_and_freqs(
        n_fft=n_fft, fmin=fmin, fmax=fmax, sampling_freq=sampling_freq, bands=bands
    )

    s = time.time()
    mel_spectrum = np.dot(mel_filterbank, dft)

    return mel_spectrum, mel_frequencies


@cache
def compute_mel_bands(bins, desired_fmax, fmax):
    assert fmax > desired_fmax

    mel_desired_fmax, mel_fmax = librosa.hz_to_mel([desired_fmax, fmax])
    ratio = mel_desired_fmax / mel_fmax

    return math.ceil(bins / ratio)


def bin_with_mel(num_bins, desired_fmax, dft, sampling_freq):
    mel_bands = compute_mel_bands(num_bins, desired_fmax, sampling_freq / 2)
    mel_spectrum, freqs = mel_filterbank(dft, sampling_freq, bands=mel_bands)

    bins, freqs = mel_spectrum[:num_bins], freqs[:num_bins]

    return bins, freqs


class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.data = None

    def update(self, new_data):
        if self.data is None:
            self.data = new_data

        ret = self.data * (1 - self.decay) + new_data * self.decay
        self.data = ret
        return ret
    
class TimeBasedEMA:
    def __init__(self, decay):
        self.decay = decay
        self.data = None
        self.t = 0
    
    def update(self, new_data):
        cur_time = time.time()
        if self.data is None:
            self.data = new_data
            self.t = cur_time

        time_decay = (1 - self.decay)**(cur_time - self.t)
        # print("time", time_decay)
        self.t = cur_time

        self.data = self.data * time_decay + new_data * (1 - time_decay)
        return self.data

    @classmethod
    def decay_in_n_secs(cls, total_decay, secs):
        # (1 - decay)^secs = (1 - total_decay)
        decay = 1 - (1 - total_decay)**(1/secs)
        return cls(decay)


def convolve_with_padding(signal, kernel):
    # Calculate the amount of padding needed on each side
    left_pad = len(kernel) // 2
    right_pad = len(kernel) // 2 - (len(kernel) % 2 == 0)

    # Apply padding to the signal
    padded_signal = np.pad(signal, (left_pad, right_pad))

    # Perform convolution with padded signal and kernel
    convolved = np.convolve(padded_signal, kernel, mode="valid")

    # normalize
    padded_ones = np.pad(np.ones_like(signal), (left_pad, right_pad))
    convolved_ones = np.convolve(padded_ones, kernel, mode="valid")

    normalized_convolved = convolved / convolved_ones

    return normalized_convolved


# def smooth_signal(signal, kernel=None):
#     smoothed_signal = convolve_with_padding(signal, kernel) / sum(kernel)
#     return smoothed_signal


def mirror_bins(bins, reduce=True):
    mirrored = np.concatenate([bins, bins[::-1]])

    if reduce:
        mirrored = np.sum(mirrored.reshape(-1, 2), axis=1).reshape(-1)

    return mirrored


def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

def bass_jump(bins, jump_strength=2):
    hann = 0.5 + np.hanning(len(bins)) / 2
    bins = bins * hann

    inv_hann = 1 - hann
    bins += np.average(inv_hann * bins) * jump_strength

    return bins

def dynamic_range(bins, max_inv_scale=1/3):
    inv_scale = max(max_inv_scale, np.max(bins))
    return bins / inv_scale

class DynamicSensitivity:
    def __init__(self, min_sensitivity=0.15, baseline=0.99, baseline_adj_s=5):
        self.ema = TimeBasedEMA.decay_in_n_secs(total_decay=baseline, secs=baseline_adj_s)
        self.ema.data = 0

        self.min_sensitivity = min_sensitivity
        self.sens_range = 1 - min_sensitivity
    
    def update(self, bins):
        self.ema.update(0) # we decay down to 0
        self.ema.data = np.maximum(self.ema.data, np.max(bins))
        return self.get_sensitivities()
    
    def get_sensitivities(self):
        return self.min_sensitivity + self.sens_range * self.ema.data


def audio_spectrum():
    audio_stream = AudioInputStream(
        channel_name="BlackHole",
        read_chunk_size=128,
        channels=2,
        exception_on_overflow=False,
    )
    audio_stream.init_streaming_mode()

    def trailing_window(window_size):
        chunks_in_window = math.ceil(window_size / audio_stream.chunk_size)
        audio_stream.stream.start_stream()
        try:
            while audio_stream.stream.is_active():
                with audio_stream.get_data() as data:
                    if len(data) < chunks_in_window:
                        continue

                    for _ in range(len(data) - chunks_in_window):
                        data.popleft()

                data = np.concatenate(data, axis=-1)
                yield data
        finally:
            if audio_stream.stream.is_active():
                audio_stream.stream.stop_stream()

    # plot = InteractivePlot(plot_type="scatter2")
    # plot = InteractiveDoublePlot()
    # plot.show()

    print("Recording...")

    num_bins = 50
    fmax = 1250

    window_size = 1024*8
    decay_s = 0.5
    smooth_window = 20

    iterations = 0

    ema = TimeBasedEMA.decay_in_n_secs(total_decay=0.99, secs=decay_s)
    sens = DynamicSensitivity()
    # separator = Separator('spleeter:2stems')

    s = time.time()
    for data in trailing_window(window_size):
        with timer() as t:
            data = normalize(data)

            channel_bins = []
            for channel in data:
                data = preprocess_data(channel, hann_tail=1)
                spectrum, freqs = create_spectrum(data, sampling_freq=audio_stream.rate)
                bins, freqs = bin_with_mel(num_bins, fmax, spectrum, audio_stream.rate)
                channel_bins.append(bins)

            # POWER
            # data = np.average(data, axis=0)
            # rms = librosa.feature.rms(y=data)
            # avg_rms = rms.mean() * 4
            # filled_bins = round(avg_rms * num_bins) 
            # bins = np.array([0] * (num_bins - filled_bins) + [0.5] * filled_bins)
            # channel_bins.append(bins)

            if len(channel_bins) == 1:
                bins = mirror_bins(bins, reduce=False)
            else:
                bins = np.concatenate([channel_bins[0], channel_bins[1][::-1]])

            bins = convolve_with_padding(bins, np.hanning(smooth_window))

            bins = ema.update(bins)

            bins = bass_jump(bins)

            bins = (sigmoid(bins / window_size * 1500) - 0.5) * 2

            bins = bins / sens.update(bins)

            # s=time.time()
            # pred = separator.separate(data.transpose())
            # print(f"splitting took {time.time()-s}s")

            

        # with timer() as t_plot:
        #     plot.update(Y=bins)
        #     # plot.update(data[0], data[1])
        #     # plot.update(pred['accompaniment'].sum(axis=1)/2, pred['vocals'].sum(axis=1)/2)
        #     pass
        
        yield bins

        # iterations += 1
        # # print(f"Processed {window_size*iterations} samples ({window_size*iterations/(time.time()-s)}/s)")

        # t_total = t.t + t_plot.t
        # print(f"{t_total=}, fps {1/t_total}, analysis latency {t.t} fps {1/t.t}")

def main():
    for bins in audio_spectrum():
        print(max(bins))

if __name__ == "__main__":
    main()
