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
# Own library
from GC_TLA.Utils.findReplaceRegex import findReplaceRegex
from GC_TLA.Plopper.basic_executor import Executor

class Plopper():
    """
        Class that writes variations of a template file out based on a configuration
        Also utilizes an executor to evaluate the templated file
    """
    def __init__(self, template, outputDir=None, outputExtension='.tmp',
                 force_write=False, findReplace=None, executor=None, **kwargs):
        self.template = pathlib.Path(template)
        # While not directly used in the basic Plopper, this attribute is typically useful
        # for subclasses to have at their disposal
        self.template_dir = self.template.parents[0]
        self.output_extension = output_extension
        self.force_write = force_write

        # Default outputDir to CWD
        if outputdir is None:
            outputdir = pathlib.Path(".").resolve()
        # Shelter any generated files from cluttering outputdir's top level
        self.outputdir = pathlib.Path(outputdir).joinpath('tmp_files')
        # Ensure the path is valid
        self.outputdir.mkdir(exist_ok=True)

        # Ensure we can work with the provided substitution object
        if findReplace is not None and not isinstance(findReplace, findReplaceRegex):
            raise TypeError(f"Type findReplace ({type(findReplace)}) is not a GC_TLA findReplaceRegex")
        self.findReplace = findReplace

        # Executor must be saved but ensure we can work with it
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError(f"Type executor ({type(executor)}) is not a GC_TLA Executor")
        self.executor = executor
        # Initial buffer is empty
        self.buffer = None

        # Str-able attributes list
        self.str_attrs = ['template','force_write','outputdir','findReplace','executor']

    def __str__(self):
        return "{"+",\n".join([f"{attr}: {getattr(self,attr)}" for attr in self.str_attrs])+"}"

    def buildTemplateString(self, outfile, *args, **kwargs):
        """
            Return None to skip executing any commands after templating a file
            Otherwise, return a list of string commands to execute via the Python3 subprocess library
        """
        return None

    def buildExecutorString(self, outfile, *args, **kwargs):
        """
            Return None to skip executing any commands after templating a file
        """
