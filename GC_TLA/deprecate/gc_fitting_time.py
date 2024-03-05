"""
    Demonstrate how fitting time for GC scales with number of variables for a customizable problem set
"""

import pandas as pd
import numpy as np
from time import time
import argparse
import warnings

# Command line interface with --help explanations
def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--model', choices=['GaussianCopula','GPTune'], default='GaussianCopula', help="Model to measure")
    prs.add_argument('--n-data', type=int, default=100, help="Number of simulated rows of fitting data")
    prs.add_argument('--max-power', type=int, default=3, help="Largest power of variables to attempt fitting (base 10)")
    prs.add_argument('--powers', type=int, nargs="*", action='append', help="Explicit power list (supercedes --max-power when specified)")
    prs.add_argument('--field-type', choices=['float', 'categorical',],  default='float', help="Treat data as this kind of fittable variable")
    prs.add_argument('--seed', type=int, default=1234, help="Set RNG seeds")
    return prs

# Adjustments to parsed args
def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Used as a range, ergo +1 to ensure maximum power is represented
    args.max_power += 1
    # Replace experiment with function call
    args.experiment = globals()[f'experiment_{args.model}']
    if len(args.powers[0]) > 0:
        args.powers = args.powers[0]
    else:
        args.powers = None
    return args

def experiment_GaussianCopula(args):
    # Define all experiment powers as same # rows but increasing # of variables
    import sdv
    from sdv.tabular import GaussianCopula
    if args.powers is None:
        experiments = [(args.n_data, 10**power) for power in range(args.max_power)]
    else:
        experiments = [(args.n_data, power) for power in args.powers]
    fitting_times = []
    for (M,N) in experiments:
        names = [str(_) for _ in range(N)]
        transformers = dict((k,args.field_type) for k in names)
        data = pd.DataFrame(dict((k, np.random.randn(M)) for k in names))
        # FRESH model
        model = GaussianCopula(field_names=names,
                               field_transformers=transformers)
        # Time fitting and reduce output -- we aren't using these models so
        # simple warnings can generally be ignored
        warnings.simplefilter("ignore")
        time_start = time()
        model.fit(data)
        time_stop = time()
        warnings.simplefilter("default")
        fitting_times.append(time_stop-time_start)
        print(f"Fit {M} rows of {args.field_type} data with {N} variables in {fitting_times[-1]} seconds")
    # In case something ever interacts with outputs, may be nice to provide key returnables
    return fitting_times, experiments

def experiment_GPTune(args):
    from autotune.space import Space, Integer, Real
    from autotune.problem import TuningProblem
    from GPTune.gptune import GPTune, BuildSurrogateModel_tl as BuildSurrogateModel
    from GPTune.computer import Computer
    from GPTune.data import Categoricalnorm, Data
    from GPTune.database import HIstoryDB, GetMachineConfiguration
    from GPTune.options import Options
    from GPTune.model import GPy
    import openturns as ot
    ot.RandomGenerator.SetSeed(args.seed)
    def data_to_gptune(data, tuning_metadata, lookup_ival):
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
    base_meta_dict = {'tuning_problem_name': target_problem.name.split('Problem')[0][:-1],
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
    historyfile = f'gptune.db/fitting_time.json'
    if os.path.exists(historyfile):
        os.remove(historyfile)

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

def main(args):
    np.random.seed(args.seed)
    args.experiment(args)

if __name__ == '__main__':
    main(parse(build()))

