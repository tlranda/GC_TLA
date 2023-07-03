import numpy as np

from libensemble.message_numbers import (STOP_TAG,
                                         PERSIS_STOP,
                                         FINISHED_PERSISTENT_GEN_TAG,
                                         EVAL_GEN_TAG,)

from libensemble.tools.persistent_support import PersistentSupport
import logging
logger = logging.getLogger(__name__)

__all__ = ['persistent_model']

def persistent_model(H, persis_info, gen_specs, libE_info):
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    user_specs = gen_specs['user']
    model = user_specs['model']

    tag = None
    calc_in = None
    first_write = True
    fields = [i[0] for i in gen_specs['out']]

    # Send batches until the manager sends stop tag
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Generate search samples from the model
        samples = model.sample_conditions(user_specs['conditions'], num_rows=user_specs['num_sim_workers'])
        # Hand off information
        H_o = np.zeros(len(samples), dtype=gen_specs['out'])
        for i, entry in samples.iterrows():
            for key, value in entry.items():
                H_o[i][key] = value

        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            if len(calc_in):
                b = []
                for field_name, entry in zip(gen_specs['persis_in'], calc_in[0]):
                    try:
                        b += [str(entry[0])]
                    except Exception as e:
                        from inspect import currentframe
                        libE_asktell_warning = f"Field '{field_name}' with value '{entry}' produced exception {e.__class__} during persistent output in {__file__}:{currentframe().f_back.f_lineno}"
                        logger.warning(libE_asktell_warning)
                        b += [str(entry)]
                # Drop in the ensemble directory
                if first_write:
                    with open('../results.csv', 'w') as f:
                        f.write(','.join(calc_in.dtype.names) + "\n")
                    first_write = False
                with open('../results.csv', 'a') as f:
                    f.write(','.join(b) + "\n")

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
