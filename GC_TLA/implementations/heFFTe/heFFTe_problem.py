import pathlib
from collections.abc import Mapping
from itertools import product as itertools_product
from math import ceil as math_ceil
# Dependent modules
from ConfigSpace import ConfigurationSpace as CS
from ConfigSpace.hyperparameters import (CategoricalHyperparameter as Categorical, Constant, OrdinalHyperparameter as Ordinal)
from sdv.constraints import ScalarRange
import numpy as np
# Own library
from GC_TLA.utils import (Factory, FindReplaceRegex)
from GC_TLA.plopper import (Arch, BOOLEAN_TYPES, Executor, Plopper)
from GC_TLA.problem import RuntimeProblem

"""
    Structure

    The ultimate top-level factory that will be importable from this file is:
        Factory(RuntimeProblem)

    The RuntimeProblem requires extensive configuration, so the Factory is subclassed to facilitate proper building:
        Positional arguments for Arch, Executor, and Plopper need to be factory-built at build-time based on name
        Append tunable args as a positional parameter

        Architecture:
            Needs a subclass to define the CPU thread sequence and MPI topology choices
            A basic factory can pass in limiting parameters as part of the configuration

        Executor:
            Requires a subclass of Executor (details closer to implementation)
            Requires a Factory subclass to handle options for initialization

        Plopper:
            Requires a subclass of Plopper (details closer to implementation)
            Requires a Factory subclass to statically set several values, including the tuple for problem identifier
            RECEIVES arch/executor from RuntimeProblem's factory
"""

# Hyperparameters for heFFTe Search are based upon the given FFT dims XYZ and detected architecture
def build_xyz_configuration_space_based_on_arch(x, y, z, arch, seed=None):
    tunable_params = CS(seed=seed)
    parameters = [
        Categorical(name='P0', choices=["double", "float"], default_value="float"),
        Constant(name='P1X', value=x),
        Constant(name='P1Y', value=y),
        Constant(name='P1Z', value=z),
        # Default ordering changes based on FFT backend impl, detected from architecture GPU Enabled = cuFFT, else FFTW
        Categorical(name='P2', choices=["-no-reorder", "-reorder"], default_value="-no-reorder" if arch.gpu_enabled else "-reorder"),
        Categorical(name='P3', choices=["-a2a", "-a2av", "-p2p", "-p2p_pl"], default_value="-a2av"),
        Categorical(name='P4', choices=["-pencils", "-slabs"], default_value="-pencils"),
        Categorical(name='P5', choices=["-r2c_dir 0", "-r2c_dir 1","-r2c_dir 2"], default_value="-r2c_dir 0"),
        Categorical(name='P6', choices=[f"-ingrid {top}" for top in arch.mpi_topologies], default_value=f"-ingrid {arch.default_mpi_topology}"),
        Categorical(name='P7', choices=[f"-outgrid {top}" for top in arch.mpi_topologies], default_value=f"-outgrid {arch.default_mpi_topology}"),
    ]
    if not arch.gpu_enabled:
        parameters.append(Ordinal(name='P8', sequence=arch.thread_sequence, default_value=arch.max_thread_depth))
    parameters.append(Constant(name='C0', value="cufft" if arch.gpu_enabled else "fftw"))
    tunable_params.add_hyperparameters(parameters)
    return tunable_params

# Weak Scaling problem identifiers are mapped in a special way due to their multi-dimensional problem space
class heFFTeProblemIDMapper(Mapping):
    """
        Why is this a class instead of a dictionary?
        There are way too many valid combinations of these scales, and can be even more combinatinos if more scales
        enter the scope of the search.

        However, there is absolutely no need to place all combinations in memory at any given time. So don't.
        We just need to validate the set of node and app scale parameterizations during the __getattr__ name lookup
        step. If the provided name matches a combination then accept the name, else reject it.

        Thus, this mapping will "include" a key any time that it should match, but only manifests keys when it is
        demanded to do so, and only specifically manifests required keys.
    """
    # Problems are designated by (NODE_SCALE, APP_X, APP_Y, APP_Z)
    APP_SCALES = [64,128,256,512,1024,1400,2048]
    APP_SCALE_NAMES = ['N','S','M','L','XL','H','XH']
    NODE_SCALES = [1,2,4,8,16,32,64,128]
    @property
    def app_scale_range(self):
        # Useful for creating SDV constraints
        return min(self.APP_SCALES), max(self.APP_SCALES)
    @property
    def node_scale_range(self):
        # Useful for creating SDV constraints
        return min(self.NODE_SCALES), max(self.NODE_SCALES)

    def __getitem__(self, key):
        """
            Maps a tuple of (NODE_SCALE, APP_X, APP_Y, APP_Z) (or a '_'-delimited string of the same fields)
            between APP_SCALES and APP_NAMES (opposite direction of whatever the input key is)
        """
        inverted = []
        if type(key) is str:
            converted = []
            for field in key.split('_'):
                try:
                    converted.append(int(field))
                except ValueError:
                    converted.append(field)
            fields = tuple(converted)
        elif type(key) is tuple:
            fields = key
        else:
            raise ValueError("Keys should be tuples or strings")
        if len(fields) != 4:
            raise KeyError("Did not identify 4 fields (node scale, app scale x3)")
        if fields[0] not in self.NODE_SCALES:
            raise KeyError(f"Node scale {fields[0]} not in known scales: {self.NODE_SCALES}")
        inverted.append(fields[0])
        for idx, field in enumerate(fields[1:]):
            if field in self.APP_SCALES:
                inverted.append(self.APP_SCALE_NAMES[self.APP_SCALES.index(field)])
            elif field in self.APP_SCALE_NAMES:
                inverted.append(self.APP_SCALES[self.APP_SCALE_NAMES.index(field)])
            else:
                raise KeyError(f"App scale {idx} ({field}) not in known app scales (as integer: {self.APP_SCALES}) or (as string: {self.APP_SCALE_NAMES})")
        return tuple(inverted)

    def __iter__(self):
        # If you REALLY want them all, I'll let you have it via itertools to be somewhat efficient
        return itertools_product(self.NODE_SCALES, *(([self.APP_SCALES],)*3))

    def __len__(self):
        # Number of accepted keys
        return len(self.NODE_SCALES) * (len(self.APP_SCALES) ** 3)

heFFTeProblemID_mapping = heFFTeProblemIDMapper()
min_app, max_app = heFFTeProblemID_mapping.app_scale_range
constraints = [ScalarRange(column_name=f'P1{LETTER}', low_value=min_app, high_value=max_app, strict_boundaries=False) for LETTER in "XYZ"]
# Node Scale != Ranks Scale, must be determined later
#min_ranks, max_ranks = heFFTeProblemID_mapping.node_scale_range
#constraints.append(ScalarRange(column_name='mpi_ranks', low_value=min_ranks, high_value=max_ranks, strict_boundaries=False))
IMPORT_AS='heFFTe'

class heFFTeInstanceFactory(Factory):
    def build(self, name, *args, **kwargs):
        # Identify X,Y,Z using mapping
        size = name.split("_",1)[1]
        identifier = self.mapping[size]
        # Name may have been integers, re-invert if so
        nodes,x,y,z = identifier
        if type(x) is str:
            nodes,x,y,z = self.mapping[identifier]
        new_args = list()
        if self.arch_factory is None:
            raise ValueError("Sub-factory for arch was not configured!")
        new_args.append(self.arch_factory.build(name, x=x, y=y, z=z))
        tunable_params = build_xyz_configuration_space_based_on_arch(x,y,z,new_args[-1])
        self._update_from_core(tunable_params=tunable_params)
        if self.exe_factory is None:
            raise ValueError("Sub-factory for exe was not configured!")
        new_args.append(self.exe_factory.build(name))
        if self.plopper_factory is None:
            raise ValueError("Sub-factory for plopper was not configured!")
        # Set GPU AWARE behavior by default
        if any([size >= 1024 for size in [x,y,z]]):
            base_substitution = {'GPU_AWARE': '-no-gpu-aware'}
        else:
            base_substitution = {'GPU_AWARE': ''}
        new_args.append(self.plopper_factory.build(name,
                                                   architecture=new_args[0],
                                                   executor=new_args[1],
                                                   base_substitution=base_substitution))
        new_args.append(tunable_params)
        # Append mapping identifier from earlier
        new_args.append(identifier)
        return super().build(name, *new_args, **kwargs)
heFFTeInstanceFactory._configure(arch_factory=None, exe_factory=None, plopper_factory=None, mapping=heFFTeProblemID_mapping)
heFFTe_instance_factory = heFFTeInstanceFactory(RuntimeProblem,
                                                factory_name=IMPORT_AS,
                                                initial_configure={'problem_mapping': heFFTeProblemID_mapping},)

class heFFTeArchitecture(Arch):
    def make_thread_sequence(self):
        max_depth = self.threads_per_node // self.ranks_per_node
        sequence = [2**_ for _ in range(1,10) if (2**_) <= max_depth]
        if len(sequence) >= 2:
            intermediates = []
            prevpow = sequence[1]
            for rawpow in sequence[2:]:
                if rawpow+prevpow >= max_depth:
                    break
                intermediates.append(rawpow+prevpow)
                prevpow = rawpow
            sequence = sorted(intermediates + sequence)
        if max_depth not in sequence:
            sequence += [max_depth]
        return max_depth, sequence

    @staticmethod
    def surface(fft_dims, grid):
        # Volume of FFT assigned to each process
        box_size = (np.asarray(fft_dims) / np.asarray(grid)).astype(int)
        # Sum of exchanged surface areas
        return (box_size * np.roll(box_size, -1)).sum()

    def minSurfaceSplit(self, X, Y, Z):
        fft_dims = (X, Y, Z)
        best_grid = (1, 1, self.mpi_ranks)
        best_surface = self.surface(fft_dims, best_grid)
        best_grid = " ".join([str(_) for _ in best_grid])
        topologies = []
        # Consider other topologies that utilize all ranks
        for i in range(1, self.mpi_ranks+1):
            if self.mpi_ranks % i == 0:
                remainder = int(self.mpi_ranks / float(i))
                for j in range(1, remainder+1):
                    candidate_grid = (i, j, int(remainder/j))
                    if np.prod(candidate_grid) != self.mpi_ranks:
                        continue
                    strtopology = " ".join([str(_) for _ in candidate_grid])
                    topologies.append(strtopology)
                    candidate_surface = self.surface(fft_dims, candidate_grid)
                    if candidate_surface < best_surface:
                        best_surface = candidate_surface
                        best_grid = strtopology
        # Topologies are reversed such that the topology order is X-1-1 to 1-1-X
        # This matches previous version ordering
        return best_grid, list(reversed(topologies))

    def init_derivable(self, **kwargs):
        # Do normal stuff first
        super().init_derivable(**kwargs)

        # Get the sequence list for # threads per node
        if 'thread_sequence' in kwargs:
            raise ValueError("Thread sequence is derived from threads_per_node and ranks_per_node")
        self.max_thread_depth, self.thread_sequence = self.make_thread_sequence()

        # Get MPI topology options for given number of ranks per node
        if 'mpi_topologies' in kwargs:
            raise ValueError("MPI topologies are derived from FFT dimensions and number of mpi ranks (derived from hostfile and ranks_per_node)")
        if 'default_mpi_topology' in kwargs:
            raise ValueError("Default MPI topology is determined by minimum surface splitting")
        # As long as the case is consistent I don't care to enforce upper or lower
        try:
            x, y, z = kwargs['x'],kwargs['y'],kwargs['z']
        except KeyError:
            x, y, z = kwargs['X'], kwargs['Y'], kwargs['Z']
        self.default_mpi_topology, self.mpi_topologies = self.minSurfaceSplit(x,y,z)

        # GPU-enabled flag could be set by user
        if 'gpu_enabled' not in kwargs or not self.default_assign('gpu_enabled', kwargs['gpu_enabled'], BOOLEAN_TYPES):
            self.gpu_enabled = self.gpus > 0

heFFTe_arch_factory = Factory(heFFTeArchitecture)
heFFTe_instance_factory._update_from_core(arch_factory=heFFTe_arch_factory)

# after arch factory builds, pass x,y,z,arch to build_xyz_configuration_space_based_on_arch()

class heFFTeExecutor(Executor):
    def getMetric(self, logfile, outfile, attempt, *args, aggregator_fn=None, **kwargs):
        if logfile is None:
            return None
        with open(logfile, 'r') as f:
            data = [_.rstrip() for _ in f.readlines()]
        for line in data:
            if "Performance: " in line:
                split = [_ for _ in line.split(' ') if len(_) > 0]
                try:
                    return -1 * float(split[1])
                except:
                    return None
        return None

    def produceMetric(self, metric_list):
        # Send greatest absolute value, but if any negative use the negative side
        sorted_metrics = sorted(metric_list)
        if sorted_metrics[0] < 0:
            return sorted_metrics[0]
        # Maximum detected error
        return sorted_metrics[-1]

heFFTe_exe_factory = Factory(heFFTeExecutor)
heFFTe_instance_factory._update_from_core(exe_factory=heFFTe_exe_factory)

class heFFTePlopper(Plopper):
    # There are no compilation steps, but ensure that the template is always filled by passing
    # force_write=True at construction
    def buildExecutorCmds(self, outfile, *args, lookup_match_substitution=None, **kwargs):
        format_args = {'self':self, 'outfile':outfile}
        if self.architecture.gpu_enabled:
            basic_format_string = "mpiexec -n {self.architecture.mpi_ranks} "+\
                                  "--ppn {self.architecture.ranks_per_node} "+\
                                  "sh ./set_affinity_gpu_polaris.sh {outfile}"
        else:
            # For Theta cluster, but I'm missing the format string with the j argument
            raise ValueError("Not Fully Implemented")
            # Math prefered over Numpy here as it converts to integer automatically
            #j = math_ceil(self.architecture.ranks_per_node * (int(lookup_match_substitution['p8']) / 64))
            #format_args['depth'] = int(lookup_match_substitution['p8'])
            #basic_format_string = "mpiexec -n {self.architecture.mpi_ranks} "
            #                      "--ppn {self.ranks_per_node} --depth {depth} "
            #                      "--cpu-bind depth --env OMP_NUM_THREADS={depth} "
            #                      "sh {outfile}"
        return ["echo "+basic_format_string.format(**format_args), "echo Performance: 3.14"]

heFFTe_FindReplaceRegex = FindReplaceRegex([r"([CP][0-9]+[XYZ]?)",r"(GPU_AWARE)"],prefix=(("#",""),("#","")))

heFFTe_plopper_factory = Factory(heFFTePlopper,
                                 initial_args=[pathlib.Path(__file__).parents[0].joinpath('speed3d.sh')],
                                 initial_kwargs={'output_extension': '.sh',
                                                 'findReplace': heFFTe_FindReplaceRegex,
                                                 'force_write': True,},)
heFFTe_instance_factory._update_from_core(plopper_factory=heFFTe_plopper_factory)

# Finally, set the import method for this module using the instance factory
__getattr__ = heFFTe_instance_factory.getattr

