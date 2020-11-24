"""
Eh!

Usage:
  eh train --audio-dir=<audio-dir> --config-yaml=<config-yaml> --output-model=<output-model>
  eh --audio-dir=<audio-dir> --model=<model> --output-csv=<output-csv>
"""

from docopt import docopt

from eh import __version__


def main():
    args = docopt(__doc__, version=__version__)
