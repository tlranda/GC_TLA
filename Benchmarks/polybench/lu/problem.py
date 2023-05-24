from GC_TLA.base_problem import polybench_problem_builder
# Used to locate kernel for ploppers
import os
HERE = os.path.dirname(os.path.abspath(__file__))

input_space = [('Categorical',
    {'name': 'p1',
    'choices': ["#pragma clang loop(i1) pack array(A) allocate(malloc)", " "],
    'default_value': ' '
    }),
    ('Categorical',
    {'name': 'p2',
    'choices': ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "],
    'default_value': ' '
    }),
    ('Ordinal',
    {'name': 'p3',
    'sequence': ['4','8','16','20','32','50','64','80','96','100','128'],
    'default_value': '96'
    }),
    ('Ordinal',
    {'name': 'p4',
    'sequence': ['4','8','16','20','32','50','64','80','100','128','2048'],
    'default_value': '2048'
    }),
    ('Ordinal',
    {'name': 'p5',
    'sequence': ['4','8','16','20','32','50','64','80','100','128','256'],
    'default_value': '256'
    })
    ]

lookup_ival = {40: ("N", "MINI"), 120: ("S", "SMALL"), 260: ("SM", "SM"), 400: ("M", "MEDIUM"),
               1200: ("ML", "ML"), 2000: ("L", "LARGE"), 4000: ("XL", "EXTRALARGE"),
               6000: ("H", "HUGE")}
__getattr__ = polybench_problem_builder(lookup_ival, input_space, HERE, name="LU_Problem")

