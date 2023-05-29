from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
from GPTune.gptune import GPTune, BuildSurrogateModel_tl as BuildSurrogateModel
from GPTune.computer import Computer
from GPTune.data import Categoricalnorm, Data
from GPTune.database import HistoryDB, GetMachineConfiguration
from GPTune.options import Options
from GPTune.model import GPy

import openturns as ot
import argparse, sys, os
import numpy as np, pandas as pd, uuid, time, copy
from GC_TLA.base_problem import ecp_problem_builder, polybench_problem_builder

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('-benchmark', type=str, required=True, help='Benchmark name')
    parser.add_argument('-inputs', type=str, nargs='+', required=True, help='Problem sizes as predefined knowledge')
    parser.add_argument('-target', type=str, required=True, help='Target task to train on')
    parser.add_argument('-nrun', type=int, default=2, help='Number of runs per task')
    parser.add_argument('-ninit', type=int, default=-1, help='Set initial configs')
    parser.add_argument('-seed', type=int, default=1234, help='Set seed')
    parser.add_argument('-builder', choices=['polybench', 'ecp'], default='polybench', help='Problem builder')
    parser.add_argument('-output', type=str, default="results.csv", help="Output CSV filename in the benchmark directory (default: results.csv)")
    parser.add_argument('-experiment', action='store_false', help='Substitute TLA for MLA')
    parser.add_argument('-preserve-history', action='store_true', help='Rename rather than remove old history files (gptune.db/benchmark.json)')
    return parser

def parse(parser, args=None):
    if args is None:
        args = parser.parse_args()
    # Rebind to correct factory object
    if args.builder == 'polybench':
        args.builder = polybench_problem_builder
    elif args.builder == 'ecp':
        args.builder = ecp_problem_builder
    else:
        raise ValueError(f"Unsupported problem builder {args.builder}")

    return args

# Special rules for certain benchmarks that are otherwise hard to rip out of the problem import
def localized_load(benchmark):
    HERE = os.getcwd()
    sys.path.insert(0, HERE)
    from problem import input_space, lookup_ival
    kwargs = {}
    # TODO: Make these ploppers unnecessary to import for kwargs
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
    print(f"Benchmark {benchmark} loaded")
    return HERE, input_space, lookup_ival, kwargs

def infer_size(fname, lookup_ival):
    fname = os.path.basename(fname) # Drop directories
    fname = fname.rsplit('.',1)[0] # Drop file extension
    fname = fname.rsplit('_',1)[1].upper() # Isolate size name
    inv_lookup = dict((v[0], k) for (k,v) in lookup_ival.items())
    return inv_lookup[fname]

def csvs_to_gptune(fnames, tuning_metadata, lookup_ival):
    # Top-level JSON info, func_eval will be filled based on data
    json_dict = {'tuning_problem_name': tuning_metadata['tuning_problem_name'],
                 'tuning_problem_category': None,
                 'surrogate_model': [],
                 'func_eval': [],
                }
    # Template for a function evaluation
    func_template = {'constants': {},
                     'machine_configuration': tuning_metadata['machine_configuration'],
                     'software_configuration': tuning_metadata['software_configuration'],
                     'additional_output': {},
                     'source': 'measure',
                    }
    # Loop safety
    parameters = None
    if type(fnames) is str:
        fnames = [fnames]
    # Prepare return structures
    sizes = []
    dicts = []
    for fname in fnames:
        # Make basic copy
        gptune_dict = dict((k,v) for (k,v) in json_dict.items())
        csv = pd.read_csv(fname)
        # Only set parameters once -- they'll be consistent throughout different files
        if parameters is None:
            parameters = [_ for _ in csv.columns if _.startswith('p') and _ != 'predicted']
        for index, row in csv.iterrows():
            new_eval = dict((k,v) for (k,v) in func_template.items())
            try:
                new_eval['task_parameter'] = {'isize': row['isize']}
            except KeyError:
                new_eval['task_parameter'] = {'isize': infer_size(fname, lookup_ival)}
            # SINGLE update per task size
            if index == 0:
                sizes.append(new_eval['task_parameter']['isize'])
            new_eval['tuning_parameter'] = dict((col, str(row[col])) for col in parameters)
            new_eval['evaluation_result'] = {'time': row['objective']}
            new_eval['evaluation_detail'] = {'time': {'evaluations': row['objective'],
                                                      'objective_scheme': 'average'}}
            new_eval['uid'] = uuid.uuid4()
            gptune_dict['func_eval'].append(new_eval)
        dicts.append(gptune_dict)
        print(f"GPTune-ified {fname}")
    return dicts, sizes

# Return either the sequence or choice attribute based on what 'obj' actually defines
def seqchoice(obj):
    if hasattr(obj, 'sequence') and obj.sequence is not None:
        return obj.sequence
    elif hasattr(obj, 'choices') and obj.choices is not None:
        return obj.choices
    raise ValueError(f"Object {obj} lacks or has NONE for sequences and choices")

def GPTune_TLA1_patched(self, Tnew, NS, normalized=False, max_frustrate=100, reject_generate=1000):
        print('\n\n\n------Starting TLA1 for task: ',Tnew)
        stats = {
            "time_total": 0,
            "time_fun": 0
        }
        time_fun=0

        t3=time.time_ns()
        # Initialization
        kwargs = copy.deepcopy(self.options)
        ntso = len(self.data.I)
        ntsn = len(Tnew)

        if(self.data.O[0].shape[1]>1):
            raise Exception("TLA1 only works for single-objective tuning")

        PSopt =[]
        for i in range(ntso):
            PSopt.append(self.data.P[i][np.argmin(self.data.O[i])])
        # YSopt = np.array([[self.data.O[k].min()] for k in range(ntso)])
        MSopt = []

        # Data may already be normalized -- only normalize UNNORMALIZED data
        if normalized:
            INorms = self.data.I
        else:
            # convert the task spaces to the normalized spaces
            INorms=[]
            for t in self.data.I:
                INorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
                INorms.append(INorm.reshape((-1, self.problem.DI)))
            INorms = np.vstack([INorms[i] for i in range(ntso)]).reshape((ntso,self.problem.DI))

        tmp=[]
        for t in Tnew:
            INorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
            tmp.append(INorm.reshape((-1, self.problem.DI)))
        InewNorms=np.vstack([tmp[i] for i in range(ntsn)]).reshape((ntsn,self.problem.DI))

        if normalized:
            PSoptNorms = PSopt
        else:
            # convert the parameter spaces to the normalized spaces
            PSoptNorms = self.problem.PS.transform(PSopt)
        columns = []
        for j in range(self.problem.DP):
            columns.append([])
        for i in range(ntso):
            for j in range(self.problem.DP):
                columns[j].append(PSoptNorms[i][j])
        PSoptNorms = []
        for j in range(self.problem.DP):
            PSoptNorms.append(np.asarray(columns[j]).reshape((ntso, -1)))

        # Predict optimums of new tasks
        stacks = []
        meanvars = []
        for k in range(self.problem.DP):
            K = GPy.kern.RBF(input_dim=self.problem.DI)
            M = GPy.models.GPRegression(INorms, PSoptNorms[k], K)
            # M.optimize_restarts(num_restarts = 10, robust=True, verbose=False, parallel=False, num_processes=None, messages="False")
            M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust=True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = 'lbfgs', start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)
            MSopt.append(M)
            # Create NS-1 samples drawn around the mean
            mean, var = MSopt[-1].predict_noiseless(InewNorms)
            stacks.append(np.vstack((mean, np.random.normal(mean, var, (NS-1,1)))))
            meanvars.append((mean,var))

        #aprxoptsNorm=np.hstack([MSopt[k].predict_noiseless(InewNorms)[0] for k in range(self.problem.DP)])  # the index [0] is the mean value, [1] is the variance
        aprxoptsNorm = np.hstack(stacks)
        aprxoptsNorm=np.minimum(aprxoptsNorm,(1-1e-12)*np.ones((ntsn,self.problem.DP)))
        aprxoptsNorm=np.maximum(aprxoptsNorm,(1e-12)*np.ones((ntsn,self.problem.DP)))
        # print('aprxoptsNorm',aprxoptsNorm,type(aprxoptsNorm))
        aprxopts = self.problem.PS.inverse_transform(aprxoptsNorm)
        # print('aprxopts',aprxopts,type(aprxopts),type(aprxopts[0]))

        # Ensure we end up having enough unique samples
        tired = 0
        n_remain = lambda : NS - len(set([tuple(a) for a in aprxopts]))
        prev = n_remain()
        while prev > 0 and tired < max_frustrate:
            tired += 1
            new_sample = np.hstack([np.random.normal(m, v, (reject_generate,1)) for (m,v) in meanvars])
            new_sample = np.minimum(new_sample,(1-1e-12)*np.ones((ntsn,self.problem.DP)))
            new_sample = np.maximum(new_sample,(1e-12)*np.ones((ntsn,self.problem.DP)))
            new_sample_inv = self.problem.PS.inverse_transform(new_sample)
            aprxopts.extend(new_sample_inv)
            remain = n_remain()
            # Iterations that produce any number of new results are "free"
            if prev > remain:
                tired -= 1
                aprxoptsNorm = np.vstack((aprxoptsNorm, new_sample))
            else:
                aprxopts = aprxopts[:-reject_generate]
            prev = remain
        # Find the actual ones that matter
        lookup = [tuple(a) for a in aprxopts]
        lids = [lookup.index(a) for a in set(lookup)][:NS] # Limit to NS such configurations
        aprxopts = [lookup[i] for i in lids]
        aprxoptsNorm = np.asarray([aprxoptsNorm[i,:] for i in lids])

        aprxoptsNormList=[[_ for _ in aprxoptsNorm[:,]]]
        # TnewNormList=[]
        #for i in range(ntsn):
        #    aprxoptsNormList[i].append(aprxoptsNorm)  # this makes sure for each task, there is only one sample parameter set
        #    # InewNormList.append(InewNorms[i,:])

        t1 = time.time_ns()
        O = self.computer.evaluate_objective(problem = self.problem, I = InewNorms, P =aprxoptsNormList, history_db = self.historydb, options = kwargs)
        t2 = time.time_ns()
        time_fun = time_fun + (t2-t1)/1e9

        #        print(aprxopts)
        #        pickle.dump(aprxopts, open('TLA1.pkl', 'w'))

        t4 = time.time_ns()
        stats['time_total'] = (t4-t3)/1e9
        stats['time_fun'] = time_fun

        return (aprxopts, O, stats)

def wrap_objective(objective, surrogate_to_size_dict):
    def new_objective(point: dict):
        # Task identifier is 'isize'
        task = point['isize']
        if task in surrogate_to_size_dict.keys():
            result = surrogate_to_size_dict[task](point)
            # NO NEED TO LOG RESULTS
        else:
            # Should auto-log results
            result = objective(point)
            # BUG: GPTune's second configuration is unique despite same seed/input. Attempt static eval
            #print(point)
            #result = [1]
        return result
    return new_objective

def cleanup_history(args, problem_name):
    historyfile = f'gptune.db/{problem_name}.json'
    if os.path.exists(historyfile):
        if args.preserve_history:
            contents = os.listdir('gptune.db')
            next_avail = 0
            while os.path.exists(f'gptune.db/{problem_name}_{next_avail}.json'):
                next_avail += 1
            print(f"--PRESERVE HISTORY-- Move {historyfile} --> gptune.db/{problem_name}_{next_avail}.json")
            import shutil
            # Preserves metadata as best as possible in case that is relevant to user
            shutil.copy2(historyfile, f'gptune.db/{problem_name}_{next_avail}.json')
        os.remove(historyfile)

def main():
    args = parse(build())
    ot.RandomGenerator.SetSeed(args.seed)
    np.random.seed(args.seed)

    # Move into directory and fetch the relevant input space and problem description, perhaps other relevant
    # kwargs for the specified benchmark
    HERE, input_space, lookup_ival, kwargs = localized_load(args.benchmark)
    # FIRST, indirect lookup the factory builder in GPTune mode
    problem_lookup = args.builder(lookup_ival,
                                  input_space,
                                  HERE,
                                  name=args.benchmark+"_Problem",
                                  returnmode='GPTune',
                                  selflog=HERE+'/'+args.output,
                                  **kwargs)
    # Next build the actual instance for evaluating the target problem
    target_problem = problem_lookup(args.target.upper())
    print(f"Target problem {args.target} constructed")

    # *S are passed to GPTune objects directly
    # *_space are used to build surrogate models and MOSTLY share kwargs
    # As such the *S_options define common options when this saves re-specification

    # Steal the parameter names / values from Problem object's input space
    PS_options = [{'name': x,
                   'transform': 'onehot',
                   'categories': seqchoice(target_problem.input_space[x])
                  } for x in target_problem.input_space.get_hyperparameter_names()]
    PS = Space([Categoricalnorm(**options) for options in PS_options])
    parameter_space = []
    # Parameter space requires some alteration due to inconsistencies
    for options in PS_options:
        options['transformer'] = options.pop('transform') # REALLY?! Keyname alteration
        options['type'] = 'categorical' # Bonus key
        # Categories key MAY need to become list instead of tuple
        # options['categories'] = list(options['categories'])
        parameter_space.append(options)

    # Able to steal this entirely from Problem object API
    OS = target_problem.output_space
    output_space = [{'name': 'time',
                     'type': 'real',
                     'transformer': 'identity',
                     'lower_bound': float(0.0),
                     'upper_bound': float('Inf')}]

    # Steal input space limits from Problem object API
    input_space = [{'name': 'isize',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': min(target_problem.dataset_lookup.keys()),
                    'upper_bound': max(target_problem.dataset_lookup.keys())}]
    IS = Space([Integer(low=input_space[0]['lower_bound'],
                        high=input_space[0]['upper_bound'],
                        transform='normalize',
                        name='isize')])

    # Meta Dicts are part of building surrogate models for each input, but have a lot of common
    # specification templated here
    base_meta_dict = {'tuning_problem_name': target_problem.name.split('Problem')[0][:-1].replace('/','__'),
                      'modeler': 'Model_GPy_LCM',
                      'input_space': input_space,
                      'output_space': output_space,
                      'parameter_space': parameter_space,
                      'loadable_machine_configurations': {'swing': {'intel': {'nodes': 1, 'cores': 128}}},
                      'loadable_software_configurations': {}
                     }
    # Used to have consistent machine definition
    tuning_metadata = {
        "tuning_problem_name": base_meta_dict['tuning_problem_name'],
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "swing",
            "intel": { "nodes": 1, "cores": 128 }
        },
        "software_configuration": {},
        "loadable_machine_configurations": base_meta_dict['loadable_machine_configurations'],
        "loadable_software_configurations": base_meta_dict['loadable_software_configurations'],
    }
    # IF there is already a historyDB file, it can mess things up. Clean it up nicely
    cleanup_history(args, base_meta_dict['tuning_problem_name'])

    constraints = {}
    objectives = target_problem.objective
    # Load prior evaluations in GPTune-ready format
    prior_traces, prior_sizes = csvs_to_gptune(args.inputs, tuning_metadata, lookup_ival)
    # Teach GPTune about these prior evaluations
    surrogate_metadata = dict((k,v) for (k,v) in base_meta_dict.items())
    model_functions = {}
    for size, data in zip(prior_sizes, prior_traces):
        surrogate_metadata['task_parameter'] = [[size]]
        model_functions[size] = BuildSurrogateModel(metadata_path=None,
                                                    metadata=surrogate_metadata,
                                                    function_evaluations=data['func_eval'])
    wrapped_objectives = wrap_objective(objectives, model_functions)
    #func_evals = []
    #for prior_data in prior_traces:
    #    func_evals.extend(prior_data['func_eval'])
    #models, model_functions = gt.GenSurrogateModel([[s] for s in prior_sizes], func_evals)

    problem = TuningProblem(IS,PS,OS, wrapped_objectives, constraints, None) # None = models (dict of names : func(point_dict) -> list(outputs)

    machine, processor, nodes, cores = GetMachineConfiguration(meta_dict=tuning_metadata)
    print(f"Machine: {machine} | Processor: {processor} | Num_Nodes: {nodes} | Num_Cores: {cores}")

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    data = Data(problem)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    options  = Options()
    # These options inherited from Jaehoon's script
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
    # Create the GPTune object
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb)

    # Set up the actual transfer learning task
    if args.experiment:
        # THIS is what GPTune's HistoryDB says you should do for TLA; same # evals in all problems,
        # but leverage model functions on prior tasks to simulate their results
        transfer_task = [[target_problem.problem_class]]
        transfer_task.extend([[s] for s in prior_sizes])
        if args.ninit == -1:
            NS1 = max(args.nrun//2,1)
        else:
            NS1 = args.ninit
        data, modeler, stats = gt.MLA(Igiven=transfer_task, NS=args.nrun, NI=len(transfer_task),
                                      NS1=NS1)
    else:
        # THIS is a patched implementation of GPTune's actual TLA api call that allows TLA to produce
        # multiple results in the new target problem rather than just one
        transfer_task = [[target_problem.problem_class]]
        # Normalized = True is NOT a typical argument -- I modified GPTune.TLA1() to use this to
        # SKIP the normalization on self.data.I and self.data.P because the surrogate function already
        # normalizes this
        data, modeler, stats = GPTune_TLA1_patched(gt, transfer_task, args.nrun, normalized=True)
        #data, modeler, stats = gt.TLA1(transfer_task, args.nrun, normalized=True)
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()

