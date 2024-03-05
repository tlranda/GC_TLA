from GC_TLA.Utils.findReplaceRegex import findReplaceRegex
import os
import uuid
import re
import time
import subprocess
import warnings
import stat
import signal
import math
import pathlib
import pickle
import atexit
import itertools
# MODULES
import numpy as np

class Plopper():
    def __init__(self, template, outputDir=None, outputExtension='.tmp',
