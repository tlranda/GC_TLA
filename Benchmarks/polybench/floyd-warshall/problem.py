from GC_TLA.base_problem import polybench_problem_builder
# Used to locate kernel for ploppers
import os
HERE = os.path.dirname(os.path.abspath(__file__))

input_space = [('Categorical',
    {'name': 'p1',
    'choices': ["#pragma clang loop(j2) pack array(path) allocate(malloc)", " "],
    'default_value': ' ',
    }),
    ('Categorical',
    {'name': 'p2',
    'choices': ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "],
    'default_value': ' ',
    }),
    ('Ordinal',
    {'name': 'p3',
    'sequence': ['4','8','16','20','32','50','64','80','96','100','128'],
    'default_value': '96',
    }),
    ('Ordinal',
    {'name': 'p4',
    'sequence': ['4','8','16','20','32','50','64','80','100','128','2048'],
    'default_value': '2048',
    }),
    ('Ordinal',
    {'name': 'p5',
    'sequence': ['4','8','16','20','32','50','64','80','100','128','256'],
    'default_value': '256',
    }),
    ]

# Based on N in floyd-warshall.h
lookup_ival = {60: ('N', "MINI"), 180: ('S', "SMALL"), 340: ('SM', "SM"), 500: ('M', "MEDIUM"),
               1650: ('ML', "ML"), 2800: ('L', "LARGE"), 5600: ('XL', "EXTRALARGE"),
               8600: ('H', "HUGE"),}
# Floyd-Warshall actually needs a different compile statement compared to most polybench ploppers
from GC_TLA.base_plopper import Polybench_Plopper
class Floyd_Warshall_Plopper(Polybench_Plopper):
    def compileString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        clang_cmd = f"clang -fno-caret-diagnostics {outfile} {self.kernel_dir}/polybench.c "+\
                    f"-I{self.kernel_dir} {d_size} -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops "+\
                    "-O3 -mllvm -polly -mllvm -polly-process-unprofitable "+\
                    "-mllvm -polly-use-llvm-names -mllvm -polly-reschedule=0 "+\
                    "-ffast-math -march=native "+\
                    f"-o {outfile[:-len(self.output_extension)]}"
                    #"-mllvm --polly-postops=0 "+\
        print(clang_cmd)
        return clang_cmd

__getattr__ = polybench_problem_builder(lookup_ival, input_space, HERE, name="Floyd-Warshall_Problem",
                                        plopper_class=Floyd_Warshall_Plopper)

