from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
#from GPTune.gptune_tl import GPTune # Some modifications/QOL from Jaehoon to help run as expected
from GPTune.gptune import GPTune
from GPTune.computer import Computer
from GPTune.data import Categoricalnorm, Data
from GPTune.database import HistoryDB
from GPTune.options import Options
from GPTune.database import GetMachineConfiguration

import openturns as ot
import argparse, os, sys
from GC_TLA.base_problem import ecp_problem_builder, polybench_problem_builder

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('-benchmark', type=str, required=True, help="Benchmark name")
    parser.add_argument('-size', type=str, required=True, help="Problem size name to train")
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-seed', type=int, default=1234, help='Set seed')
    parser.add_argument('-builder', choices=['polybench', 'ecp'], default='polybench', help="Problem builder")
    parser.add_argument('-output', default='gptune_results.csv', help="Output file name")
    return parser

def parse(parser, args=None):
    if args is None:
        args = parser.parse_args()
    return args

def localized_load(benchmark):
    HERE = os.getcwd()
    sys.path.insert(0, HERE)
    from problem import input_space, lookup_ival
    kwargs = {}
    if benchmark == 'sw4lite':
        from problem import SW4Lite_Plopper
        kwargs.update({'plopper_class': SW4Lite_Plopper,})
    elif benchmark == 'amg':
        from problem import AMG_Plopper
        kwargs.update({'plopper_class': AMG_Plopper,})
    elif benchmark == 'xsbench':
        from problem import XSBench_Plopper
        kwargs.update({'plopper_class': XSBench_Plopper, 'ignore_runtime_failure': True, })
    elif benchmark == 'rsbench':
        from problem import RSBench_Plopper
        kwargs.update({'plopper_class': RSBench_Plopper, 'ignore_runtime_failure': True, })
    elif benchmark == 'floyd_warshall':
        from problem import Floyd_Warshall_Plopper
        kwargs.update({'plopper_class': Floyd_Warshall_Plopper,})
    os.chdir(HERE)
    return HERE, input_space, lookup_ival, kwargs

def main():
    args = parse(build())
    ot.RandomGenerator.SetSeed(args.seed)

    if args.builder == 'polybench':
        problem_lookup = polybench_problem_builder
    elif args.builder == 'ecp':
        problem_lookup = ecp_problem_builder
    else:
        raise ValueError(f"Unsupported problem builder {args.builder}")
    HERE, input_space, lookup_ival, kwargs = localized_load(args.benchmark)
    problem_lookup = problem_lookup(lookup_ival, input_space, HERE, name=args.benchmark+"_Problem", returnmode='GPTune', selflog=HERE+'/'+args.output, **kwargs)
    my_problem = problem_lookup(args.size.upper())
    objectives = my_problem.objective
    def seqchoice(obj):
        if hasattr(obj, 'sequence') and obj.sequence is not None:
            return obj.sequence
        elif hasattr(obj, 'choices') and obj.choices is not None:
            return obj.choices
        raise ValueError(f"Object {obj} lacks or has NONE for sequences and choices")
    PS = Space([Categoricalnorm(seqchoice(my_problem.input_space[x]), transform='onehot', name=x) for x in my_problem.input_space.get_hyperparameter_names()])
    OS = my_problem.output_space
    IS = Space([Integer(min(my_problem.dataset_lookup.keys()), max(my_problem.dataset_lookup.keys()), transform='normalize', name='isize')])
    constraints = {}
    problem = TuningProblem(IS,PS,OS, objectives, constraints, None)

    tuning_metadata = {
        "tuning_problem_name": my_problem.name.split('Problem')[0][:-1].replace('/','__'),
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "swing",
            "intel": { "nodes": 1, "cores": 128 }
        },
        "software_configuration": {},
        "loadable_machine_configurations": {
            "swing": {
                "intel": { "nodes": 1, "cores": 128 },
            }
        },
        "loadable_software_configurations": {}
    }
    machine, processor, nodes, cores = GetMachineConfiguration(meta_dict = tuning_metadata)
    print(f"Machine: {machine} | Processor: {processor} | Num_Nodes: {nodes} | Num_Cores: {cores}")
    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    options.update({'model_restarts': 1,
                    'distributed_memory_parallelism': False,
                    'shared_memory_parallelism': False,
                    'objective_evaluation_parallelism': False,
                    'objective_multisample_threads': 1,
                    'objective_multisample_Processes': 1,
                    'objective_nprocmax': 1,
                    'model_processes': 1,
                    'model_class': 'Model_GPy_LCM',
                    'verbose': False, # True
                    'sample_class': 'SampleOpenTURNS',
                   })
    options.validate(computer=computer)

    giventask = [[my_problem.problem_class]]
    print(f"Problem size is {args.size} --> {giventask}")
    data = Data(problem)
    historydb = HistoryDB(meta_dict = tuning_metadata)
    historydb.load_func_eval = False # We DO NOT WANT to have history loaded -- GPTune needs to evaluate things it proposes
    gt = GPTune(problem, computer=computer, data=data, options=options,
                historydb=historydb, driverabspath=os.path.abspath(__file__))
    data, modeler, stats = gt.MLA(NS=args.nrun, Igiven=giventask, NI=1, NS1=int(max(args.nrun//2,1)))
    print(f"stats: {stats}")

if __name__ == '__main__':
    main()

