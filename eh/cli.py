"""
Eh!

Usage:
  eh train --audio-dir=<audio-dir> --config-yaml=<config-yaml> --output-model=<output-model>
  eh --audio-dir=<audio-dir> --model=<model> --output-csv=<output-csv>
"""

import os
import pickle
from glob import glob
from typing import List

import ffmpeg
import librosa
import pandas as pd
import yaml
from docopt import docopt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from eh import __version__
from eh.model import Featurizer


def list_audios(directory: str) -> List[str]:
    return glob(os.path.join(directory, "*.wav"))


def prepare_lossy_examples(files: List[str], output_dir: str, filters: List[str]):
    # HACK:
    tmp_file = "/tmp/eh.mp3"
    for f in tqdm(files):
        basename = os.path.basename(f)
        ffmpeg.input(f).output(tmp_file, format="mp3").overwrite_output().run()
        ffmpeg.input(tmp_file).output(os.path.join(output_dir, basename), format="wav", ar="8k").run()


def main():
    args = docopt(__doc__, version=__version__)

    if args["train"]:
        with open(args["--config-yaml"]) as fp:
            filters = yaml.safe_load(fp)
            files = list_audios(args["--audio-dir"])

            staging_dir = os.path.join(args["--audio-dir"], "eh-staging")
            if not os.path.exists(staging_dir):
                os.makedirs(staging_dir)

            prepare_lossy_examples(files, staging_dir, filters)

            # TODO: Chunk audios
            audios = [librosa.load(f, sr=8000, mono=True, duration=10) for f in tqdm(files)]
            lossy_audios = [librosa.load(f, sr=8000, mono=True, duration=10) for f in tqdm(list_audios(staging_dir))]

            X = audios + lossy_audios
            y = [1] * len(audios) + [0] * len(lossy_audios)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

            f = Featurizer()
            clf = SVC(kernel="linear")
            pipeline = make_pipeline(f, clf)
            pipeline.fit(X_train, y_train)

            y_test_pred = pipeline.predict(X_test)
            print(classification_report(y_test, y_test_pred))

            with open(args["--output-model"], "wb") as fp:
                pickle.dump(pipeline, fp)

    else:
        files = list_audios(args["--audio-dir"])

        with open(args["--model"], "rb") as fp:
            pipeline = pickle.load(fp)

        audios = [librosa.load(f, sr=8000, mono=True, duration=10) for f in tqdm(files)]
        preds = pipeline.predict(audios)
        pd.DataFrame({"filepath": files, "pred": preds}).to_csv(args["--output-csv"], index=None)
