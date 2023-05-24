from GC_TLA.base_problem import ecp_problem_builder
from GC_TLA.base_plopper import ECP_Plopper
# Used to locate kernel for ploppers
import os
HERE = os.path.dirname(os.path.abspath(__file__))

input_space = [('Ordinal',
                {'name': 'p0',
                 'sequence': ['2','4','8','16','32','48','64','96','128','192','256'],
                 'default_value': '64',}),
                ('Categorical',
                 {'name': 'p1',
                  'choices': ['cores','threads','sockets'],
                  'default_value': 'cores',}),
                ('Categorical',
                 {'name': 'p2',
                  'choices': ['close','spread','master'],
                  'default_value': 'close',}),
                ('Categorical',
                 {'name': 'p3',
                  'choices': ['dynamic','static'],
                  'default_value': 'static',}),
                ('Categorical',
                 {'name': 'p4',
                  'choices': ['#pragma omp parallel for', ' '],
                  'default_value': ' ',}),
                ('Categorical',
                 {'name': 'p5',
                  'choices': ['#pragma unroll (6)','#pragma unroll',' '],
                  'default_value': ' ',}),
                ('Categorical',
                 {'name': 'p6',
                  'choices': ['#pragma omp for', '#pragma omp for nowait'],
                  'default_value': '#pragma omp for',}),
                ('Categorical',
                 {'name': 'p7',
                  'choices': ['MPI_Barrier(MPI_COMM_WORLD);',' '],
                  'default_value': 'MPI_Barrier(MPI_COMM_WORLD);',})
              ]

def get_gpu_ids_from_section(section):
    ids = []
    for line in section:
        try:
            after_pipe_char = line.split('|', 1)[1].lstrip()
        except IndexError:
            continue
        try:
            gpu_id = int(after_pipe_char.split(' ',1)[0])
        except ValueError:
            pass
        else:
            ids.append(gpu_id)
    return ids

class SW4Lite_Plopper(ECP_Plopper):
    retries=1
    def set_os_environ(self):
        import os, subprocess
        nvsmi = subprocess.run('nvidia-smi', shell=True, stdout=subprocess.PIPE)
        important = nvsmi.stdout.decode('utf-8').split('+')
        exists = get_gpu_ids_from_section("".join(important[3:-4]).split('\n'))
        occupied = get_gpu_ids_from_section("".join(important[-3:-2]).split('\n'))
        ok = set(exists).difference(set(occupied))
        if len(ok) > 0:
            least_used = list(ok)[0]
        else:
            import numpy as np
            least_used = exists[np.argmin([occupied.count(k) for k in exists])]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(least_used)
        return dict(os.environ) #{'CUDA_VISIBLE_DEVICES': str(least_used)}
    def compileString(self, outfile, dictVal, *args, **kwargs):
        cmds = ["nvcc -O3 -x cu -Isrc -c -dc -arch=sm_60 -DSW4_CROUTINES -DSW4_CUDA -DSW4_NONBLOCKING "+\
                "-ccbin mpicxx -Xptxas -v -Xcompiler -fopenmp -DSW4_OPENMP -Isrc/double -c " +\
                f"{outfile} -o {outfile[:-len(self.output_extension)]}.o",

                "nvcc -arch=sm_60 -Xcompiler -fopenmp -Xlinker -arch=sm_60 -ccbin mpicxx -o " +\
                f"{outfile[:-len(self.output_extension)]} main.o {outfile[:-len(self.output_extension)]}.o " +\
                "Source.o SuperGrid.o GridPointSource.o time_functions_cu.o ew-cfromfort.o EW_cuda.o "+\
                "Sarray.o device-routines.o EWCuda.o CheckPoint.o Parallel_IO.o EW-dg.o "+\
                "MaterialData.o MaterialBlock.o Polynomial.o SecondOrderSection.o TimeSeries.o sacsubc.o curvilinear-c.o "+\
                "-lcuda -lnvcpumath -lnvc -lcudart -llapack -lm -lblas -lgfortran"]
        return ";".join(cmds) #compile_cmd
    def runString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        return f'mpirun -np 1 {outfile[:-len(self.output_extension)]} loh1/LOH.1-h100_s{d_size}.in'
    def getTime(self, process, dictVal, *args, **kwargs):
        try:
            return float(process.stdout.decode('utf-8').split(' ')[-1])
        except ValueError:
            try:
                return float(process.stderr.decode('utf-8').split(' ')[-1])
            except ValueError:
                return None

# Based on
lookup_ival = {1: ("NN", "MICRO"), 2: ("N", "TINY"), 3: ("S", "SMALL"), 4: ("SM", "SM"),
               5: ("M", "MEDIUM"), 6: ("ML", "ML"), 7: ("L", "LARGE"), 8: ("XL", "EXTRALARGE"),
               9: ("H", "HUGE"), 10: ("XH", "EXTRAHUGE")}

__getattr__ = ecp_problem_builder(lookup_ival, input_space, HERE, name="SW4Lite_Problem", plopper_class=SW4Lite_Plopper)

