"""
Models and features stuff
"""

from typing import List

import librosa
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from eh.types import Audio


class Featurizer(BaseEstimator, TransformerMixin):
    """
    Default featurizer for audios. Uses inverted mel filters for MFCCs.
    """

    def fit(self, X: List[Audio], y=None):
        return self

    def transform(self, X: List[Audio], y=None):
        # TODO: Invert filters

        n_mfcc = 20
        output = np.zeros((len(X), n_mfcc))
        for i, (y, sr) in enumerate(X):
            output[i, :] = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc).mean(axis=1)

        return output
