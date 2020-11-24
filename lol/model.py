"""
Models and features stuff
"""

from typing import List

import librosa
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from lol.types import Audio


def melspectrogram(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
    power=2.0,
    **kwargs,
):
    """
    Patched melspectrogram function.
    """

    S, n_fft = librosa.core.spectrum._spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr, n_fft, **kwargs)
    mel_basis = np.flip(mel_basis, axis=1)

    return np.dot(mel_basis, S)


class Featurizer(BaseEstimator, TransformerMixin):
    """
    Default featurizer for audios. Uses inverted mel filters for MFCCs.
    """

    def fit(self, X: List[Audio], y=None):
        return self

    def transform(self, X: List[Audio], y=None):
        n_mfcc = 20
        output = np.zeros((len(X), n_mfcc))
        for i, (y, sr) in enumerate(X):
            S = librosa.power_to_db(melspectrogram(y, sr))
            output[i, :] = librosa.feature.mfcc(sr=sr, S=S, n_mfcc=n_mfcc).mean(axis=1)

        return output
