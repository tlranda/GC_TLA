import pathlib
import time
import warnings

import pandas as pd
#from sdv.constraints import ScalarRange
from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.single_table import GaussianCopulaSynthesizer

sizes = {'syr2k_L': 1000,
         'syr2k_M': 200,
         'syr2k_S': 60,
         }
constraints = [{'constraint_class': 'ScalarRange',
                'constraint_parameters': {'column_name': 'size',
                                          'low_value': 20,
                                          'high_value': 3000,
                                          'strict_boundaries': False,
                                          },
                },]

start = time.time()
frames = []
for frame in pathlib.Path('ytopt_bo_source_tasks').iterdir():
    if frame.suffix != '.csv':
        continue
    csv = pd.read_csv(frame).drop(columns=['elapsed_sec','objective'])
    csv.insert(0,'size',[sizes[frame.stem]] * len(csv))
    frames.append(csv)
data = pd.concat(frames).reset_index(drop=True)
end = time.time()
print("Data preparation time:", end-start)

warnings.simplefilter('ignore')
start = time.time()
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
# Weirdly size is detected as categorical despite being numeric
metadata.update_column(column_name='size',sdtype='numerical')
model = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=False)
model.add_constraints(constraints)
model.fit(data)
end = time.time()
warnings.simplefilter('default')
print("Fitting time:", end-start)

n_inference = 30
start = time.time()
model.sample(n_inference)
end = time.time()
print(f"Sample {n_inference} time:", end-start)

cond = Condition({'size': 80}, num_rows=n_inference)
start = time.time()
model.sample_from_conditions([cond])
end = time.time()
print(f"Conditionally sample {n_inference} rows time:", end-start)

