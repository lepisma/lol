"""
lol

Usage:
  lol train --audio-dir=<audio-dir> --transforms-file=<transforms-file> --output-model=<output-model>
  lol --audio-dir=<audio-dir> --model=<model> --output-csv=<output-csv>

Options:
  --transforms-file=<transforms-file>      Plain text file with lines mapping to ffmpeg lossy transforms.
"""

import os
import pickle
import random
import shlex
import subprocess as sp
from glob import glob
from typing import List

import librosa
import pandas as pd
from docopt import docopt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from lol import __version__
from lol.model import Featurizer


def list_audios(directory: str) -> List[str]:
    return glob(os.path.join(directory, "*.wav"))


def prepare_lossy_examples(files: List[str], output_dir: str, transforms: List[str]):
    random.seed(1234)

    # HACK:
    tmp_file = "/tmp/lol.mp3"
    base_transform = "-f wav -ar 8k"

    for f in tqdm(files):
        transform = random.choice(transforms)
        command = f"ffmpeg -i {shlex.quote(f)} {transform} {tmp_file} -y"
        sp.run(command, shell=True)

        output_file = os.path.join(output_dir, os.path.basename(f))
        command = f"ffmpeg -i {tmp_file} {base_transform} {shlex.quote(output_file)}"
        sp.run(command, shell=True)


def main():
    args = docopt(__doc__, version=__version__)

    if args["train"]:
        with open(args["--transforms-file"]) as fp:
            transforms = [l.strip() for l in fp.readlines()]

        files = list_audios(args["--audio-dir"])

        staging_dir = os.path.join(args["--audio-dir"], "lol-staging")
        if not os.path.exists(staging_dir):
            os.makedirs(staging_dir)

        prepare_lossy_examples(files, staging_dir, transforms)

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
