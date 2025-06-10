import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import random


class PreprocessedLibriSpeech(Dataset):
    def __init__(
        self,
        root="../",
        url="test-clean",
        sample_rate=16000,
        duration=5,
        augment=True,
        shift_limit=0.2,
        rate=1.2,
        n_steps=1.2,
        download=False,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=download
        )
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        self.num_channels = None
        self.augment = augment
        self.shift_limit = shift_limit
        self.rate = rate
        self.n_steps = n_steps

        self.vocab = list("-abcdefghijklmnopqrstuvwxyz '")
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=80,
        )

        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 80},
        )

        self.spec_augment = (
            torch.nn.Sequential(
                T.FrequencyMasking(freq_mask_param=80),
                T.TimeMasking(time_mask_param=80),
            )
            if augment
            else torch.nn.Identity()
        )

    @staticmethod
    def time_shift(waveform, shift_limit):
        """moves the entire audio forward or backward slightly in time"""
        shift_amt = int(random.uniform(-shift_limit, shift_limit) * waveform.size(1))
        return torch.roll(waveform, shifts=shift_amt, dims=1)

    @staticmethod
    def speed_change(waveform, rate):
        """makes the audio play faster or slower"""
        effects = [["speed", str(rate)], ["rate", "16000"]]
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, 16000, effects
        )
        return waveform

    @staticmethod
    def pitch_shift(waveform, n_steps):
        """changes how high or low the speaker sounds"""
        effects = [["pitch", str(n_steps * 100)], ["rate", "16000"]]
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, 16000, effects
        )
        return waveform

    def raw_augment(self, waveform):
        waveform = self.pitch_shift(
            self.speed_change(
                self.time_shift(waveform=waveform, shift_limit=self.shift_limit),
                rate=self.rate,
            ),
            n_steps=self.n_steps,
        )
        return waveform

    def preprocess_waveform(self, waveform, original_sr):
        """
        Resample output for original sample to 16000
        Convert to stereo
        pad or truncate th fixed duration
        Input shape: [channels, num_samples]
        Output shape: [2, TARGET_NUM_SAMPLES]
        """

        if original_sr != self.sample_rate:
            resampler = T.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
            waveform = torch.stack(
                [resampler(wf.unsqueeze(0)).squeeze(0) for wf in waveform]
            )

        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, : self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            pad_length = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        waveform, sample_rate, transcript, _, _, _ = self.dataset[index]
        waveform = self.preprocess_waveform(waveform=waveform, original_sr=sample_rate)
        waveform = self.raw_augment(waveform=waveform)
        mel_spec = self.mel_transform(waveform)
        target_time_steps = self.num_samples // 512 + 1  # Based on your hop_length=512
        if mel_spec.shape[1] > target_time_steps:
            mel_spec = mel_spec[:, :target_time_steps]
        else:
            mel_spec = torch.nn.functional.pad(
                mel_spec, (0, target_time_steps - mel_spec.shape[1])
            )
        mel_spec = self.spec_augment(mel_spec)
        # mel_spec  = self.mfcc_transform(mel_spec)

        transcript = transcript.lower()
        label = torch.tensor([self.char2idx.get(c, 0) for c in transcript])

        return mel_spec, label
