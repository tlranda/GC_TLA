"""
This module is a wrapper around an example GC_TLA objective function call
"""
__all__ = ['init_obj']

import numpy as np
import os
import time
import itertools

start_time = time.time()

def init_obj(H, persis_info, sim_specs, libE_info):
    point = {}
    for field in sim_specs['in']:
        point[field] = np.squeeze(H[field])
    # Pass along machine info to point for topology preparation
    user_specs = sim_specs['user']
    machine_info = user_specs['machine_info']
    point['machine_info'] = machine_info

    y = user_specs['problem'].objective(point, sim_specs['in'], libE_info['workerID'])

    H_o = np.zeros(len(sim_specs['out']), dtype=sim_specs['out'])
    H_o['FLOPS'] = y
    H_o['elapsed_sec'] = time.time() - start_time
    H_o['machine_identifier'] = [machine_info['identifier']]
    for wrapped_field in ['mpi_ranks', 'ranks_per_node', 'gpu_enabled', 'libE_workers']:
        H_o[wrapped_field] = [machine_info[wrapped_field]]
    H_o['libE_id'] = [libE_info['workerID']]

    return H_o, persis_info

