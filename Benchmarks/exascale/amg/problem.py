from GC_TLA.base_problem import ecp_problem_builder
from GC_TLA.base_plopper import ECP_Plopper
# Used to locate kernel for ploppers
import os
HERE = os.path.dirname(os.path.abspath(__file__))

input_space = [('Ordinal',
    {'name': 'p0',
     'sequence': ['4','5','6','7','8'],
     'default_value': '8',
    }),
    ('Ordinal',
    {'name': 'p1',
     'sequence': ['10','20','40','64','80','100','128','160','200'],
     'default_value': '100',
    }),
    ('Categorical',
    {'name': 'p2',
     #'choices': ["#pragma clang loop unrolling full", " "],
     'choices': ["#pragma clang unrolling full", " "],
     'default_value': ' ',
    }),
    ('Categorical',
    {'name': 'p3',
     'choices': ["#pragma omp parallel for", " "],
     'default_value': ' ',
    }),
    ('Ordinal',
    {'name': 'p4',
     'sequence': ['2','4','8','16','32','64','96','128','256'],
     'default_value': '96',
    }),
    ('Ordinal',
    {'name': 'p5',
     'sequence': ['2','4','8','16','32','64','96','128','256'],
     'default_value': '256',
    }),
    ('Ordinal',
    {'name': 'p6',
     'sequence': ['10','20','40','64','80','100','128','160','200'],
     'default_value': '100',
    }),
    ('Categorical',
    {'name': 'p7',
     'choices': ['compact','scatter','balanced'],
     'default_value': 'compact',
    }),
    ('Categorical',
    {'name': 'p8',
     'choices': ['cores','threads','sockets'],
     'default_value': 'cores'
    }),
    ]

# Special compile string for plopper
class AMG_Plopper(ECP_Plopper):
    def compileString(self, outfile, dictVal, *args, **kwargs):
        compile_cmd = "mpicc -fopenmp -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm "+\
                      "-polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math "+\
                      f"-march=native -o {outfile[:-len(self.output_extension)]} {outfile} -I./ "+\
                      "-I./struct_mv -I./sstruct_mv -I./IJ_mv -I./seq_mv -I./parcsr_mv -I./utilities "+\
                      "-I./parcsr_ls -I./krylov -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG "+\
                      "-DHYPRE_NO_GLOBAL_PARTITION -DHYPRE_USING_PERSISTENT_COMM -DHYPRE_HOPSCOTCH "+\
                      "-DHYPRE_BIGINT -DHYPRE_TIMING -L./parcsr_ls -L./parcsr_mv -L./IJ_mv -L./seq_mv "+\
                      "-L./sstruct_mv -L./struct_mv -L./krylov -L./utilities -lparcsr_ls -lparcsr_mv "+\
                      "-lseq_mv -lsstruct_mv -lIJ_mv -lHYPRE_struct_mv -lkrylov -lHYPRE_utilities -lm "
        return compile_cmd
    def runString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        return f"mpirun -np 1 {outfile[:-len(self.output_extension)]} -laplace -n {d_size} {d_size} {d_size} -P 1 1 1"
    def getTime(self, process, dictVal, *args, **kwargs):
        try:
            return float(process.stdout.decode('utf-8').split('\n')[-1].split(' ')[-1])
        except ValueError:
            try:
                return float(process.stderr.decode('utf-8').split('\n')[-1].split(' ')[-1])
            except ValueError:
                return None

# Based on
lookup_ival = {50: ("S", "SMALL"), 75: ("SM", "SM"), 100: ("M", "MEDIUM"),
               125: ("ML", "ML"), 150: ("L", "LARGE"), 175: ("XL", "EXTRALARGE"), 200: ("H", "HUGE")}
__getattr__ = ecp_problem_builder(lookup_ival, input_space, HERE, name="AMG_Problem", plopper_class=AMG_Plopper)

