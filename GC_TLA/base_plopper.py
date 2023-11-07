import os, uuid, re, time, subprocess, numpy as np, warnings, stat, signal, math

"""
    Expected usage:
    * Override the runString() call to produce the string command that evaluates the outputfile
        + If your benchmark reports its own measurement, override getTime() call to extract it from the process information
            * If you don't want to use the best-case time as objective value, override metric()
    * If your benchmark requires compilation, subclass this and override the compileString() call
        + If you do not use compilation but need to make use of plotValues(), set force_plot=True on initialization
        + Create a findReplaceRegex object to handle plotting values and pass them in for initialization
            * find should be a list of regexes to match
            * prefix|suffix should be a list of tuples of equal length, where each tuple has 2 strings
                + As of now, only static strings are supported (not regexes)
                + The first string is matched from the original input (removed)
                + The second string is replaced in the new output (substitution for removed)
    * Startup checks should be run after super().__init__(*args, **kwargs), for checking things like:
        + Host or GPU architecture
        + Presence of CUDA in source code (cache in self.buffer if you don't mind)
        + Compiler availability or dependent default flags for the compiler

    * Note that (args, kwargs) propagation pattern in findRuntime() allows for access to additional parameters as such:
      + compileString()
      + execute()
        * getTime()
        * runString()
      As such, these functions should have UNIQUE (args, kwargs) unless it is INTENDED for them to be shared across
      some or all of the above. This is intended for transparency if these functions are overridden and need additional
      information that is not tracked via the Plopper's self
      Note that these functions ALWAYS see the dictVal dictionary formed from dict(params: x)
"""


class findReplaceRegex:
    empty_from_to = [tuple(["",""])]
    """
        * find should be a list of regexes to match
        * prefix|suffix should be a list of tuples of equal length, where each tuple has 2 strings
            + As of now, only static strings are supported (not regexes)
            + The first string is matched from the original input (removed)
            + The second string is replaced in the new output (substitution for removed)
    """

    def __init__(self, find, prefix=None, suffix=None):
        if type(find) is str:
            find = tuple([find])
        self.find = find
        self.nitems = len(self.find)
        REQUIRED_ELEMS = 2*self.nitems

        # Repeated code for each of these attributes
        for attrName, passedValue in zip(['prefix', 'suffix'], [prefix, suffix]):
            # Be nice about wrapping/replacing default values
            if passedValue is None:
                passedValue = findReplaceRegex.empty_from_to * self.nitems
            elif type(passedValue) is tuple and type(passedValue[0]) is str:
                passedValue = [passedValue]
            # Validation of required length for each find-regex
            nAttrItems = sum(map(len, passedValue))
            if nAttrItems != REQUIRED_ELEMS:
                raise ValueError(f"{attrName} must have 2-element tuple per element in the find regex list (got {nAttrItems}, needed {REQUIRED_ELEMS})")
            else:
                self.__setattr__(attrName, passedValue)

        # Magic variables that can try to predict common use patterns and ease function paramaterization
        self.iter_idx = None
        self.invert_direction = 0

    def __str__(self):
        return str({'find': self.find,
                    'prefix': self.prefix,
                    'suffix': self.suffix,
                    'iter_idx': self.iter_idx,
                    'invert_direction': self.invert_direction})

    def __iter__(self):
        # Enumeration just to set up the magic variable
        for idx, regex in enumerate(self.find):
            self.iter_idx = idx
            yield regex

    def replace(self, match, to, string):
        # Automatically handle the expected replacement patterns
        if to is None or to == "":
            return re.sub(self.wrap(match, noInvert=True), "", string)
        else:
            return re.sub(self.wrap(match), self.wrap(to), string)

    def wrap(self, wrap, direction=None, idx=None, noInvert=False):
        # When direction|idx are None, attempt to use magic variables to predict correct output
        # Actual values passed to direction may be ['from'==0,'to'==1]
        # Actual values passed to idx may be in the range of values in self.find
        if direction is None:
            direction = self.invert_direction
        if type(direction) is str:
            if direction.lower() == 'from':
                direction = 0
            elif direction.lower() == 'to':
                direction = 1
        if direction not in [0, 1]:
            raise ValueError(f"Could not parse direction '{direction}', must be in ['from', 'to'] or [0, 1]")
        if idx is None:
            if self.iter_idx is None:
                if self.nitems == 1:
                    # Only case where these can both be None and we unambiguously match user expectations
                    idx = 0
                else:
                    raise ValueError(f"Index to wrap is poorly defined! Please define an index")
            else:
                idx = self.iter_idx
        # Magic updates for next call to match (usually expect opposite direction)
        if not noInvert:
            self.invert_direction = int(not direction)
        return self.prefix[idx][direction] + wrap + self.suffix[idx][direction]

class Plopper:
    def __init__(self, sourcefile, outputdir=None, output_extension='.tmp',
                 evaluation_tries=3, retries=0, findReplace=None,
                 infinity=1, force_plot=False, ignore_runtime_failure=False, **kwargs):
        self.sourcefile = sourcefile # Basis for runtime / plotting values
        self.kernel_dir = os.path.abspath(self.sourcefile[:self.sourcefile.rfind('/')])

        if outputdir is None:
            # Use CWD as basis
            outputdir = os.path.abspath(".")
        self.outputdir = outputdir+"/tmp_files" # Where temporary files will be generated
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        self.output_extension = output_extension # In case compilers are VERY picky about the extension on your intermediate files
        self.evaluation_tries = evaluation_tries # Number of executions to average amongst
        self.retries = retries # Number of failed evaluations to re-attempt before giving up
        if findReplace is not None and type(findReplace) is not findReplaceRegex:
            raise ValueError("Only support findReplaceRegex type for the findReplace attribute at this time")
        self.findReplace = findReplace # findReplaceRegex object
        self.infinity = infinity # Very large value to return on failure to compile or execute
        self.force_plot = force_plot # Always utilize plotValues() even if there is no compilation string
        self.ignore_runtime_failure = ignore_runtime_failure # Some processes may permit bad return codes

        self.buffer = None

    def seed(self, SEED):
        pass

    def __str__(self):
        return str("{"+",\n".join([str(_[0])+": "+str(_[1]) for _ in dict((k,str(v)) for (k,v) in self.__dict__.items() if not callable(v)).items()])+"}")
        #return str({'sourcefile': self.sourcefile,
        #            'kernel_dir': self.kernel_dir,
        #            'outputdir': self.outputdir,
        #            'output_extension': self.output_extension,
        #            'evaluation_tries': self.evaluation_tries,
        #            'retries': self.retries,
        #            'findReplace': self.findReplace,
        #            'infinity': self.infinity,
        #            'force_plot': self.force_plot,
        #            'buffer': self.buffer is not None})

    def compileString(self, outfile, dictVal, *args, **kwargs):
        # Return None to skip compilation
        # Override with compiling string rules to make a particular compilation occur (includes plotValues)
        # Final executable MUST be written to `outfile`
        return None

    def runString(self, outfile, attempt, dictVal, *args, **kwargs):
        # Return the string used to execute the attempt
        # outfile is the temporary filename that is generated for this particular instance, ignore it if no compilation/plotted values were used
        # Override as needed
        return outfile

    # Replace the Markers in the source file with the corresponding values
    def plotValues(self, outputfile, dictVal, *args, findReplace=None, **kwargs):
        if findReplace is None:
            if self.findReplace is None:
                # Compiling may be all that is necessary (-D switches, etc)
                return
            findReplace = self.findReplace

        # Use cache to save I/O time on repeated plots
        if self.buffer is None:
            with open(self.sourcefile, "r") as f1:
                self.buffer = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in self.buffer:
                # For each regex in the findReplaceRegex object
                for idx, find in enumerate(findReplace):
                    # While it matches in the line
                    while re.search(find, line):
                        # Cache substitutions as they may appear multiple times in a line, but will all be handled on first encounter
                        foundGroups = []
                        for m in re.finditer(find, line):
                            match = m.group(1)
                            if match in foundGroups:
                                continue
                            # String-ify-ing must be supported AND intended operation, else this is not going to work
                            line = findReplace.replace(match, str(dictVal[match]), line)
                            foundGroups.append(match)
                f2.write(line)
        # Ensure proper permissions on files
        os.chmod(outputfile,
                 stat.S_IRWXU |
                 stat.S_IRGRP | stat.S_IXGRP |
                 stat.S_IROTH | stat.S_IXOTH)

    def getTime(self, process, out, errs, outfile, attempt, dictVal, *args, **kwargs):
        # Define how to recover self-attributed objective values from the subprocess object
        # Return None to default to the python-based time library's timing of the event
        try:
            if out is None:
                return float(process.stdout.decode('utf-8'))
            else:
                return float(out.decode('utf-8'))
        except ValueError:
            try:
                if errs is None:
                    return float(process.stderr.decode('utf-8'))
                else:
                    return float(errs.decode('utf-8'))
            except ValueError:
                return None

    def metric(self, timing_list):
        # Allows for different interpretations of repeated events
        # Defaults to best-case scenario
        return min(timing_list)

    def execute(self, outfile, dictVal, *args, **kwargs):
        times = []
        failures = 0
        attempt = 0
        while failures <= self.retries and len(times) < self.evaluation_tries:
            run_str = self.runString(outfile, attempt, dictVal, *args, **kwargs)
            start = time.time()
            env = self.set_os_environ() if hasattr(self, 'set_os_environ') else None
            out, errs = None, None
            if hasattr(self, 'app_timeout'):
                execution_status = subprocess.Popen(run_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                try:
                    out, errs = execution_status.communicate(timeout=self.app_timeout)
                except subprocess.TimeoutExpired:
                    execution_status.kill()
                    # May need to clean up child processes here
                    # for proc in psutil.proccess_iter(attrs=['pid','name']):
                    #   if 'exe.pl' in proc.info['name']:
                    #       proc.kill()
                    out, errs = execution_status.communicate()
            else:
                execution_status = subprocess.run(run_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            duration = time.time() - start
            if not self.ignore_runtime_failure and execution_status.returncode != 0:
                # FAILURE
                failures += 1
                attempt += 1
                print(f"FAILED: {run_str}")
                continue
            # Find the execution time
            try:
                derived_time = self.getTime(execution_status, out, errs, outfile, attempt, dictVal, *args, **kwargs)
                if derived_time is not None:
                    duration = derived_time
            except:
                # Any exception in the getTime call should be treated as an evaluation failure
                failures += 1
                print(f"FAILED: {run_str}")
            else:
                if duration == 0.0:
                    failures += 1
                    print(f"FAILED: {run_str}")
                else:
                    times.append(duration)
            attempt += 1
        # Unable to evaluate this execution
        if failures > self.retries:
            print(f"OVERALL FAILED: {run_str}")
            return self.metric([self.infinity])
        return self.metric(times)

    def createDict(self, x, params, *args, **kwargs):
        return dict((k,v) for (k,v) in zip(params, x))

    # Function to find the execution time of the interim file, and return the execution time as cost to the search module
    # Additional args provided here will propagate to:
    # * compileString
    # * execute
    #   + getTime
    #   + runString
    def findRuntime(self, x, params, *args, **kwargs):
        # Generate non-colliding name to write outputs to:
        if len(x) > 0:
            interimfile = self.outputdir+"/"+str(uuid.uuid4())+self.output_extension
        else:
            interimfile = self.sourcefile

        # Generate intermediate file
        dictVal = self.createDict(x, params, *args, **kwargs)
        # If there is a compiling string, we need to run plotValues
        compile_str = self.compileString(interimfile, dictVal, *args, **kwargs)
        if len(x) > 0 and (self.force_plot or compile_str is not None):
            self.plotValues(interimfile, dictVal, *args, **kwargs)
            # Compilation
            if compile_str is not None:
                env = self.set_os_environ() if hasattr(self, 'set_os_environ') else None
                compilation_status = subprocess.run(compile_str, shell=True, stderr=subprocess.PIPE, env=env)
                # Find execution time ONLY when the compiler return code is zero, else return infinity
                if compilation_status.returncode != 0:
                # and len(compilation_status.stderr) == 0: # Second condition is to check for warnings
                    print(compilation_status.stderr)
                    print("Compile failed")
                    print(compile_str)
                    return self.metric([self.infinity])
        elif len(x) == 0 and (self.force_plot or compile_str is not None):
            # SKIP Plotting values
            # Compilation
            if compile_str is not None:
                env = self.set_os_environ() if hasattr(self, 'set_os_environ') else None
                compilation_status = subprocess.run(compile_str, shell=True, stderr=subprocess.PIPE, env=env)
                # Find execution time ONLY when the compiler return code is zero, else return infinity
                if compilation_status.returncode != 0:
                # and len(compilation_status.stderr) == 0: # Second condition is to check for warnings
                    print(compilation_status.stderr)
                    print("Compile failed")
                    print(compile_str)
                    return self.metric([self.infinity])
        # Evaluation
        return self.execute(interimfile, dictVal, *args, **kwargs)

class LibE_Plopper(Plopper):
    """ Call to findRuntime should be: (x, params) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.findReplace is None:
            self.findReplace = findReplaceRegex([r"#([CP][0-9]+[XYZ]*)"], prefix=[tuple(["#",""])])

    def set_architecture_info(self, **kwargs):
        # Architecture information
        if 'threads_per_node' in kwargs:
            self.threads_per_node = kwargs['threads_per_node']
        else:
            proc = subprocess.run(['nproc'], capture_output=True)
            if proc.returncode == 0:
                self.threads_per_node = int(proc.stdout.decode('utf-8').strip())
            else:
                proc = subprocess.run(['lscpu'], capture_output=True)
                for line in proc.stdout.decode('utf-8'):
                    if 'CPU(s):' in line:
                        self.threads_per_node = int(line.rstrip().rsplit(' ', 1)[1])
                        break
        if 'gpus' in kwargs:
            self.gpus = kwargs['gpus']
        else:
            self.gpus = 0
            proc = subprocess.run('nvidia-smi -L'.split(' '), capture_output=True)
            if proc.returncode == 0:
               self.gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
        if self.gpus > 0:
            self.ranks_per_node = self.gpus
        else:
            self.ranks_per_node = self.threads_per_node
        if 'nodes' in kwargs:
            self.nodes = kwargs['nodes']
        if 'mpi_ranks' in kwargs:
            self.mpi_ranks = kwargs['mpi_ranks']
        else:
            if hasattr(self, 'nodes'):
                self.mpi_ranks = self.nodes * self.ranks_per_node
        # Set machine name
        if 'machine_identifier' in kwargs:
            self.machine_identifier = kwargs['machine_identifier']
        else:
            # Automatic hostname identification
            import platform
            self.machine_identifier = platform.node()
            if 'polaris' in self.machine_identifier:
                self.machine_identifier = 'polaris'
                if self.gpus > 0:
                    self.machine_identifier += '-gpu'
                else:
                    self.machine_identifier += '-cpu'
            elif 'theta' in self.machine_identifier:
                self.machine_identifier = 'theta-knl'
        if 'formatSTR' in kwargs and kwargs['formatSTR'] is not None:
            self.cmd_template = kwargs['formatSTR']
        else:
            self.cmd_template = "mpiexec -n {mpi_ranks} --ppn {ranks_per_node} --depth {depth} --cpu-bind depth --env OMP_NUM_THREADS={depth} sh ./set_affinity_gpu_polaris.sh {interimfile}"

    def createDict(self, x, params, *args, **kwargs):
        dictVal = {}
        for p, v in zip(params, x):
            if type(v) is np.ndarray and v.shape == ():
                v = v.tolist()
            dictVal[p] = v
        return dictVal

    def runString(self, outfile, attempt, dictVal, *args, **kwargs):
        j = math.ceil(self.ranks_per_node * int(dictVal['P9']) / 64)
        required_sysinfo = ['mpi_ranks', 'ranks_per_node']
        assert all([hasattr(self,attr) for attr in required_sysinfo]), \
               "Insufficient architecture information to form runString -- " +\
               "specify or auto-determine properties via problem.set_architecture_info()"
        cmd = self.cmd_template.format(mpi_ranks=self.mpi_ranks, ranks_per_node=self.ranks_per_node,
                                       depth=int(dictVal['P9']), j=j, interimfile=outfile)
        return cmd

    def execute(self, outfile, dictVal, *args, **kwargs):
        times = []
        failures = 0
        attempt = 0
        while failures <= self.retries and len(times) < self.evaluation_tries:
            logfile = outfile.rsplit(".",1)[0] + f"_{attempt}.log"
            run_str = self.runString(outfile, attempt, dictVal, *args, **kwargs)
            attempt += 1
            start = time.time()
            env = self.set_os_environ() if hasattr(self, 'set_os_environ') else None
            out, errs = None, None
            logged = False
            with open(logfile, "w") as logs:
                if hasattr(self, 'app_timeout'):
                    execution_status = subprocess.Popen(run_str, shell=True, stdout=logs, stderr=logs, env=env)
                    child_pd = execution_status.pid
                    try:
                        # TODO: Set timeout from dictVal
                        execution_status.communicate(timeout=self.app_timeout)
                    except subprocess.TimeoutExpired:
                        #os.kill(child_pid, signal.SIGTERM)
                        try:
                            execution_status.kill()
                        except:
                            pass
                        time.sleep(1)
                    else:
                        logged = True
                else:
                    execution_status = subprocess.run(run_str, shell=True, stdout=logs, stderr=logs, env=env)
                    logged = True
                # GPU cleanup
                if hasattr(self, 'gpu_cleanup'):
                    try:
                        self.gpu_cleanup(outfile, attempt, dictVal, *args, **kwargs)
                    except Exception as e:
                        BadGPUCleanup = f"Bad GPU cleanup ({e.__class__})"
                        for attr in ['msg', 'message', 'args']:
                            if hasattr(e, attr):
                                BadGPUCleanup += f" -- {getattr(e,attr)}"
                        warnings.warn(BadGPUCleanup)
            duration = time.time() - start
            if logged and not self.ignore_runtime_failure and execution_status.returncode != 0:
                # FAILURE
                failures += 1
                times.append(2. + execution_status.returncode / 1000)
                print(f"FAILED: {run_str}")
                eval_error_warning = f"Error Code {execution_status.returncode}"
                warnings.warn(eval_error_warning)
                continue
            # Find the execution time
            elif logged:
                logged = self.getTime(execution_status, out, errs, outfile, attempt-1, dictVal, *args, **kwargs)
                if logged is not None:
                    times.append(logged)
                else:
                    bad_logs_warning = f"Failed to read logs of successful evaluation of {outfile}"
                    warnings.warn(bad_logs_warning)
                    failures += 1
                    times.append(1.1)
            else:
                # Timed out evaluations MAY be recoverable
                to_result = self.getTime(execution_status, out, errs, outfile, attempt-1, dictVal, *args, **kwargs)
                if to_result is None:
                    failures += 1
                    times.append(1.0)
                    to_warning = f"Evaluation of {outfile} TIMED OUT; non-recoverable"
                    warnings.warn(to_warning)
                else:
                    times.append(to_result)
        # Unable to evaluate this execution
        if failures > self.retries:
            print(f"OVERALL FAILED: {run_str}")
            return self.metric([self.infinity])
        return self.metric(times)

    def metric(self, figures_of_merit):
        usable = [_ for _ in figures_of_merit if _ < 0]
        if len(usable) > 0:
            # Best observed result
            return min(usable)
        else:
            # Gravest error is reported
            return max(figures_of_merit)

    def getTime(self, process, out, errs, outfile, attempt, dictVal, *args, **kwargs):
        try:
            logfile = outfile.rsplit(".",1)[0] + f"_{attempt}.log"
            with open(logfile, "r") as logged:
                lines = [_.rstrip() for _ in logged.readlines()]
                for line in lines:
                    if "Performance: " in line:
                        split = [_ for _ in line.split(' ') if len(_) > 0]
                        return -1 * float(split[1])
        except Exception as e:
            eval_warning = f"Evaluation raised {e.__class__.__name__}: {e.args}"
            warnings.warn(eval_warning)
            return 3.0 # Sentinel value - Python log processing error
        eval_warning = f"Evaluation failed to locate performance metric"
        warnings.warn(eval_warning)
        return None # Sentinel value for this failure

class ECP_Plopper(Plopper):
    """ Call to findRuntime should be: (x, params, d_size) """,
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.findReplace is None:
            self.findReplace = findReplaceRegex(r"#(P[0-9]+)", prefix=tuple(["#",""]))

    def compileString(self, outfile, dictVal, *args, **kwargs):
        # Drop extension in the output file name to prevent clobber
        clang_cmd = f"clang {outfile} {self.kernel_dir}/material.c {self.kernel_dir}/utils.c -I{self.kernel_dir} "+\
                    "-fopenmp -DOPENMP -fno-unroll-loops -O3 -mllvm -polly -mllvm "+\
                    "-polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math -lm "+\
                    f"-march=native -o {outfile[:-len(self.output_extension)]} "+\
                    "-I/lcrc/project/EE-ECP/jkoo/sw/clang13.2/llvm-project/release_pragma-clang-loop/projects/openmp/runtime/src"
        return clang_cmd

    def runString(self, outfile, attempt, dictVal, *args, **kwargs):
        return "srun -n1 "+outfile[:-len(self.output_extension)]+" ".join([str(_) for _ in args])

    def getTime(self, process, out, errs, outfile, attempt, dictVal, *arg, **kwargs):
        # Return last 3 floating point values from output by line
        if out is None:
            return [float(s) for s in process.stdout.decode('utf-8').split('\n')[-3:]]
        else:
            return [float(s) for s in out.decode('utf-8').split('\n')[-3:]]


class Polybench_Plopper(Plopper):
    """ Call to findRuntime should be: (x, params, d_size) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.findReplace is None:
            self.findReplace = findReplaceRegex(r"#(P[0-9]+)", prefix=tuple(["#",""]))

    def compileString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        # Drop extension in the output file name to prevent clobber
        clang_cmd = f"clang {outfile} {self.kernel_dir}/polybench.c -I{self.kernel_dir} {d_size} "+\
                    "-DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm "+\
                    "-polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math "+\
                    f"-march=native -o {outfile[:-len(self.output_extension)]}"
        return clang_cmd

    def runString(self, outfile, attempt, dictVal, *args, **kwargs):
        return "srun -n1 "+outfile[:-len(self.output_extension)]

    def getTime(self, process, out, errs, outfile, attempt, dictVal, *arg, **kwargs):
        # Return last 3 floating point values from output by line
        if out is None:
            return [float(s) for s in process.stdout.decode('utf-8').split('\n')[-4:-1]]
        else:
            return [float(s) for s in out.decode('utf-8').split('\n')[-4:-1]]

class Dummy_Plopper(Plopper):
    def __init__(self, *args, dummy_low=0, dummy_high=1, **kwargs):
        self.outputdir=""
        self.output_extension=""
        self.sourcefile=""
        self.force_plot=False
        self.retries=1
        self.evaluation_tries=1
        self.kernel_dir = os.path.abspath(".")
        # Castable range
        self.dummy_low = dummy_low
        self.dummy_range = np.abs(dummy_high - dummy_low)
    def __str__(self):
        return "DUMMY"
    def getTime(self, process, out, errs, outfile, attempt, dictVal, *args, **kwargs):
        return self.dummy_low + (self.dummy_range * np.random.rand())
    def runString(self, outfile, attempt, dictVal, *args, **kwargs):
        return "echo"


def __getattr__(name):
    if name == 'Plopper':
        return Plopper
    elif name == 'ECP_Plopper':
        return ECP_Plopper
    elif name == 'Polybench_Plopper':
        return Polybench_Plopper
    elif name == 'Dummy_Plopper':
        return Dummy_Plopper
    elif name == 'findReplaceRegex':
        return findReplaceRegex
    elif name == 'LazyPlopper':
        import torch # Currently used for serialization
        import atexit # Python < 3.10 bugfix for LazyPloppers
        import itertools # Chain fix for LazyPloppers
        class LazyPlopper(Plopper):
            def __init__(self, *args, cachefile=None, randomizeCacheName=False, lazySaveInterval=5, **kwargs):
                super().__init__(*args, **kwargs)
                # Make cache available
                if cachefile is None:
                    cachefile = "lazyplopper_cache"
                    if randomizeCacheName:
                        cachefile += "_"+str(uuid.uuid4())
                    cachefile += ".cache"
                self.cachefile = cachefile
                # Load cache
                if os.path.exists(self.cachefile):
                    self.load()
                    self.lastSaved = dict((k,v) for (k,v) in self.cache.items())
                else:
                    self.cache = dict()
                    self.lastSaved = dict()
                # Define a checkpoint interval to save new information at prior to object deletion
                self.lazySaveInterval = lazySaveInterval
                # Bug fix for Python < 3.10: Make sure there is a save before python deletes the open() method
                atexit.register(self.__del__)

            @property
            def nSaved(self):
                return len(self.lastSaved.keys())

            @property
            def nCached(self):
                return len(self.cache.keys())

            def __str__(self):
                return super().__str__()+"\n"+str({'cachefile': self.cachefile,
                                                   'saved': self.nSaved,
                                                   'cached': self.nCached,
                                                   'lazySaveInterval': self.lazySaveInterval})

            def __del__(self):
                # Prevent redundant saves
                if self.nSaved != self.nCached:
                    self.save()

            # Currently implemented using Pytorch serialization. Override these functions to use something else
            def load(self):
                self.cache = torch.load(self.cachefile)

            def save(self):
                try:
                    torch.save(self.cache, self.cachefile)
                except NameError:
                    missing_entries = self.nCached - self.nSaved
                    if missing_entries == 1:
                        print(f"!WARNING: FAILED to save final cache entry (garbage collection misordering likely)")
                    elif missing_entries > 1:
                        print(f"!WARNING: FAILED to save final {missing_entries} cache entries (garbage collection misordering likely)")
                else:
                    self.lastSaved = dict((k,v) for (k,v) in self.cache.items())

            def findRuntime(self, x, params, *args, **kwargs):
                searchtup = tuple(list(itertools.chain.from_iterable([xx,pp] for (xx,pp) in zip(x, params)))+
                                  list(args)+
                                  list(kwargs.values()))
                # Lazy evaluation doesn't call findRuntime() when it has seen the runtime before
                if searchtup in self.cache.keys():
                    return self.cache[searchtup]
                else:
                    rval = super().findRuntime(x, params, *args, **kwargs)
                    self.cache[searchtup] = rval
                    # Checkpoint new save values every interval to avoid catastrophic loss
                    if self.nCached - self.nSaved >= self.lazySaveInterval:
                        self.save()
                    return rval
        return LazyPlopper

