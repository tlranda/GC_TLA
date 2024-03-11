import os
import uuid
import subprocess
import warnings
import stat
import pathlib
import copy
# Own library
from GC_TLA.utils import (Configurable, FindReplaceRegex)
from GC_TLA.plopper import (Arch, Executor, EphemeralPlopper)

class Plopper(EphemeralPlopper):
    """
        Class that writes variations of a template file out based on a configuration
        Also utilizes an executor to evaluate the templated file
    """
    def __init__(self, template, *args, output_dir=None, output_extension='.tmp',
                 retain_buffer_in_memory=True, findReplace=None,
                 force_write=False, base_substitution=None,
                 executor=None, architecture=None, **kwargs):
        super().__init__(*args, force_write=force_write, base_substitution=base_substitution,
                         executor=executor, architecture=architecture, **kwargs)
        self.template = pathlib.Path(template)
        # While not directly used in the basic Plopper, this attribute is typically useful
        # for subclasses to have at their disposal
        self.template_dir = self.template.parents[0]
        self.output_extension = output_extension
        self.retain_buffer_in_memory = retain_buffer_in_memory

        # Default output_dir to CWD
        if output_dir is None:
            output_dir = pathlib.Path(".").resolve()
        # Shelter any generated files from cluttering outputdir's top level
        self.output_dir = pathlib.Path(output_dir).joinpath('tmp_files')
        # Ensure the path is valid
        self.output_dir.mkdir(exist_ok=True)

        # Ensure we can work with the provided substitution object
        if findReplace is not None and not isinstance(findReplace, FindReplaceRegex):
            raise TypeError(f"Type findReplace ({type(findReplace)}) is not a GC_TLA FindReplaceRegex")
        self.findReplace = findReplace

        # Load initial buffer contents
        if self.retain_buffer_in_memory:
            with open(self.template,'r') as f:
                self.buffer = f.readlines()

        # Str-able attributes list
        self.str_attrs = ['template','output_dir','findReplace']+self.str_attrs

    def buildTemplateCmds(self, outfile, *args, **kwargs):
        """
            Return None to skip executing any commands after templating a file
            Otherwise, return a list of string commands to execute via the Python3 subprocess library
        """
        return None

    def buildExecutorCmds(self, outfile, *args, **kwargs):
        """
            Return None to skip executing any commands after templating a file
            Otherwise, return a list of string commands to execute for evaluation via Executor instance
        """
        return None

    def fillTemplate(self, destination, substitution="", *args, lookup_match_substitution=None, **kwargs):
        """
            Create the file at destination and copy contents from the template file there, making edits as necessary through the FindReplaceRegex object

            substitution and lookup_match_substitution are directly passed to the findReplace object's .findReplace()
        """
        if self.findReplace is None and not self.force_write:
            return
        substitutions = copy.deepcopy(self.base_substitution)
        if lookup_match_substitution is not None:
            substitutions.update(lookup_match_substitution)

        # Ensure buffer exists if not retained
        if not self.retain_buffer_in_memory:
            with open(self.template,'r') as f:
                self.buffer = f.readlines()

        with open(destination, 'w') as f:
            for line in self.buffer:
                if self.findReplace is not None:
                    line = self.findReplace.findReplace(line, substitution, lookup_match_substitution=substitutions)
                f.write(line)
        # Ensure proper permissions on generated files (755)
        os.chmod(destination, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

        # Purge buffer it not retaining in memory
        if not self.retain_buffer_in_memory:
            del sel.buffer

    def setDestination(self, destination=None, use_raw_template=False, *args, **kwargs):
        if use_raw_template:
            destination = self.template
        elif destination is None:
            destination = self.output_dir.joinpath(str(uuid.uuid4())).with_suffix(self.output_extension)
        return destination

