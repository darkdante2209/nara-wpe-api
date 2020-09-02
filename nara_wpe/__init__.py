try:
    import pathlib
except ImportError:
    # Python 2.7
    import pathlib2 as pathlib

import os

project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

import sys
# print(sys.path)
# sys.path.append("E:\Program Files (x86)\Anaconda\envs")


import tensorflow as tf
print(tf.__version__)
import torch
print(torch.__version__)
from tensorflow import keras