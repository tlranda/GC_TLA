import os, argparse
import numpy as np
import pandas as pd
import pdb

def stacker(size, li, dirname, label):
    sized_fs = [f"{dirname}/{_}" for _ in li if size in _.lower()]
    stacked = []
    try:
        stacked = pd.concat(tuple([pd.read_csv(_) for _ in sized_fs]))
        stacked = stacked.groupby(stacked.index)['objective'].mean()
        stacked = pd.DataFrame(stacked)
        stacked.insert(0,'label', [label for _ in range(len(stacked))])
    except:
        stacked = pd.DataFrame({'objective': None, 'label': None}, index=[0])
    return stacked

budget = {'sw4lite': 15,
          'xsbench': 7,
          'amg': 5,
          'rsbench': 3,
          '_3mm': None,
          'covariance': None,
          'lu': None,
          'floyd_warshall': 15,
          'heat3d': 8,
          'syr2k': 3,
          }

local_avg = []
global_avg = []
for d in os.listdir():
    # Filter directories
    if d.endswith('_exp') and not d.startswith('dummy'):
        # Group files by technique
        thomas_files = []
        thomas_dir = f"{d}/data/thomas_experiments"
        for f in os.listdir(thomas_dir):
            if '5555' not in f and '1337' not in f and 'GaussianCopula' in f and 'NO_REFIT' in f and 'trace' not in f:
                thomas_files.append(f)
        gptune_files = []
        gptune_dir = f"{d}/data/gptune_experiments"
        for f in os.listdir(gptune_dir):
            if 'eval' not in f:
                gptune_files.append(f)
        bo_files = []
        bo_dir = f"{d}/data/jaehoon_experiments"
        try:
            for f in os.listdir(bo_dir):
                if f.startswith('results') and 'rf' in f and 'eval' not in f and ('xl' in f.lower() or 'ml' in f.lower() or 'sm' in f.lower()):
                    bo_files.append(f)
        except:
            pass
        baselines = []
        baseline_names = []
        for f in os.listdir(f"{d}/data"):
            if f.startswith('DEFAULT') and f != "DEFAULT.csv":
                baseline_names.append(f)
                baselines.append(pd.read_csv(f"{d}/data/{f}").iloc[0]['objective'])
        baselines = dict((k.strip('DEFAULT_').rstrip('.csv').lower(),v) for k,v in zip(baseline_names, baselines))

        # Per size filtering
        for size in ['sm', 'ml', 'xl']:
            t_size_f = stacker(size, thomas_files, thomas_dir, 't')[['objective','label']]
            g_size_f = stacker(size, gptune_files, gptune_dir, 'g')[['objective','label']]
            b_size_f = stacker(size, bo_files, bo_dir, 'b')[['objective','label']]

            print(d, size)
            gc_first_idx = np.where(t_size_f.index == 0)[0]
            gc_first = baselines[size] / t_size_f.iloc[gc_first_idx]['objective']
            print(f"\tGC First: {sum(gc_first)/len(gc_first)}")
            gc_budget = budget[d[:-4]]
            if gc_budget is None:
                gc_budget = max(t_size_f.index)+1
            pre_budget_idx = np.where(t_size_f.index < gc_budget)[0]
            pre_budget_idx = np.argmin(t_size_f.iloc[pre_budget_idx]['objective'])
            pre_budget = baselines[size] / t_size_f.iloc[pre_budget_idx]['objective']
            print(f"\tGC Budget ({pre_budget_idx+1}/{gc_budget}): {pre_budget}")
            gc_best_idx = np.argmin(t_size_f['objective'])
            gc_best = baselines[size] / t_size_f.iloc[gc_best_idx]['objective']
            print(f"\tGC Best ({gc_best_idx+1}): {gc_best}")
            try:
                bo_best_idx = np.argmin(b_size_f['objective'])
                bo_best = baselines[size] / b_size_f.iloc[bo_best_idx]['objective']
                print(f"\tBO Best ({bo_best_idx+1}): {bo_best}")
            except:
                print(f"\tNo BO Best")
            gptune_best_idx = np.argmin(g_size_f['objective'])
            gptune_best = baselines[size] / g_size_f.iloc[gptune_best_idx]['objective']
            print(f"\tGPTune Best ({gptune_best_idx+1}): {gptune_best}")

