import os, pandas as pd, sys
from importlib import import_module

def load_from_problem(obj, problemName=None):
    if problemName is None:
        problemName = "UNK_PROBLEM"
    fname = obj.plopper.kernel_dir+"/Data"
    if obj.use_oracle:
        fname += "/oracle/all_"
    else:
        clsname = obj.__class__.__name__
        fname += "/ytopt_bo_source_tasks/"+clsname[:clsname.rindex('_')].lower()+"_"
    fname += obj.dataset_lookup[obj.problem_class][0].upper()+".csv"
    if not os.path.exists(fname):
        # First try backup
        backup_fname = fname.rsplit('/',1)
        backup_fname.insert(1, 'Data')
        backup_fname = "/".join(backup_fname)
        if not os.path.exists(backup_fname):
            # Next try replacing '-' with '_'
            dash_fname = "_".join(fname.split('-'))
            if not os.path.exists(dash_fname):
                dash_backup_fname = "_".join(backup_fname.split('-'))
                if not os.path.exists(dash_backup_fname):
                    # Execute the input problem and move its results files to the above directory
                    raise ValueError(f"Could not find {fname} for '{problemName}' "
                                     f"[{obj.name}] and no backup at {backup_fname}"
                                     "\nYou may need to run this problem or rename its output "
                                     "as above for the script to locate it")
                else:
                    print(f"WARNING! {problemName} [{obj.name}] is using backup data rather "
                            "than original data (Dash-to-Underscore Replacement ON)")
                    fname = dash_backup_fname
            else:
                print("Dash-to-Underscore Replacement ON")
                fname = dash_fname
        else:
            print(f"WARNING! {problemName} [{obj.name}] is using backup data rather "
                    "than original data")
            fname = backup_fname
    return pd.read_csv(fname)

def load_problem_module(name):
    mod, attr = name.split('.')
    mod += '.py'
    dirname, basename = os.path.split(mod)
    sys.path.insert(0, dirname)
    module_name = os.path.splitext(basename)[0]
    module = import_module(module_name)
    return module.__getattr__(attr)

def load_without_problem(name):
    return load_from_problem(load_problem_module(name), name)

