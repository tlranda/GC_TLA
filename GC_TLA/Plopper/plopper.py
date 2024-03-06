import os
import uuid
import subprocess
import warnings
import stat
import pathlib
# Own library
from GC_TLA.Utils.findReplaceRegex import findReplaceRegex
from GC_TLA.Plopper.executor import Executor
from GC_TLA.Plopper.architecture import Architecture

class Plopper():
    """
        Class that writes variations of a template file out based on a configuration
        Also utilizes an executor to evaluate the templated file
    """
    def __init__(self, template, output_dir=None, output_extension='.tmp',
                 force_write=False, retain_buffer_in_memory=True,
                 findReplace=None, executor=None, architecture=None, **kwargs):
        self.template = pathlib.Path(template)
        # While not directly used in the basic Plopper, this attribute is typically useful
        # for subclasses to have at their disposal
        self.template_dir = self.template.parents[0]
        self.output_extension = output_extension
        self.force_write = force_write
        self.retain_buffer_in_memory = retain_buffer_in_memory

        # Default output_dir to CWD
        if output_dir is None:
            output_dir = pathlib.Path(".").resolve()
        # Shelter any generated files from cluttering outputdir's top level
        self.output_dir = pathlib.Path(output_dir).joinpath('tmp_files')
        # Ensure the path is valid
        self.output_dir.mkdir(exist_ok=True)

        # Ensure we can work with the provided substitution object
        if findReplace is not None and not isinstance(findReplace, findReplaceRegex):
            raise TypeError(f"Type findReplace ({type(findReplace)}) is not a GC_TLA findReplaceRegex")
        self.findReplace = findReplace

        # Executor must be saved but ensure we can work with it
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError(f"Type executor ({type(executor)}) is not a GC_TLA Executor")
        self.executor = executor

        # Architecture can be passed in, but might not be
        if architecture is not None and not isinstance(architecture, Architecture):
            raise TypeError(f"Type architecture ({type(architecture)}) is not a GC_TLA Architecture")
        elif architecture is None:
            architecture = Architecture() # Default system detection
        self.architecture = architecture

        # Load initial buffer contents
        if self.retain_buffer_in_memory:
            with open(self.template,'r') as f:
                self.buffer = f.readlines()

        # Str-able attributes list
        self.str_attrs = ['template','force_write','outputdir','findReplace','executor']

    def __str__(self):
        return "{"+",\n".join([f"{attr}: {getattr(self,attr)}" for attr in self.str_attrs])+"}"

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
            Create the file at destination and copy contents from the template file there, making edits as necessary through the findReplaceRegex object

            substitution and lookup_match_substitution are directly passed to the findReplace object's .findReplace()
        """
        if self.findReplace is None and not self.force_write:
            return

        # Ensure buffer exists if not retained
        if not self.retain_buffer_in_memory:
            with open(self.template,'r') as f:
                self.buffer = f.readlines()

        with open(destination, 'w') as f:
            for line in self.buffer:
                if self.findReplace is None:
                    f.write(line)
                else:
                    f.write(self.findReplace.findReplace(line, substitution, lookup_match_substitution=lookup_match_substitution))
        # Ensure proper permissions on generated files (755)
        os.chmod(destination, stat.S_IRWXU | stat.IRGRP | stat.IXGRP | stat.S_IROTH | stat.S_IXOTH)

        # Purge buffer it not retaining in memory
        if not self.retain_buffer_in_memory:
            del sel.buffer

    def templateExecute(self, destination=None, *args, use_raw_template=False, **kwargs):
        if use_raw_template:
            destination = self.template
        elif destination is None:
            destination = self.output_dir.joinpath(str(uuid.uuid4())).with_suffix(self.output_extension)
        # Track last-edited file in case it is useful to know (unit testing, etc)
        self.previous_destination = destination

        # Produce commands to finalize this template
        template_cmds = self.buildTemplateCmds(destination, *args, **kwargs)

        # Indicators that template should be filled in
        if not use_raw_template and (self.force_write or template_cmds is not None):
            self.fillTemplate(destination, *args, **kwargs)

        # Perform any commands necessary to finalize the template
        if template_cmds is not None:
            env = None if self.executor is None else self.executor.set_os_environ(None)
            for cmd_i, cmd in enumerate(template_cmds):
                status = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, env=env)
                if status.returncode != 0:
                    print(f"Template Command Failed: {cmd}")
                    print(status.stderr)
                    if self.executor is None:
                        return
                    else:
                        return self.executor.unable_to_execute()

        # Use executor to determine metrics of interest
        if self.executor is None:
            return
        else:
            # Pass buildExecutorCmds as Callable to generate the execution commands
            return self.executor.execute(destination, self.buildExecutorCmds, *args, **kwargs)

