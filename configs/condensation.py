import os


os.sys.path.insert(0, "/home/schirrmr/code/tabpfn/")
import time
import logging

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)

import torch as th
import torch.backends.cudnn as cudnn

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s")


log = logging.getLogger(__name__)
log.setLevel("INFO")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge1/schirrmr-tabpfn/exps/condensation/",
        }
    ]

    debug_params = [
        {
            "debug": False,
            "version": "precomputed_mean_std",
        }
    ]

    data_params = dictlistprod(
        {
            "dataset_id": range(10),
            "n_samples": [2,4,8,16,32,64,128,256,], #2,4,8,16,32 check missing
            "data_collection": ["cafee"],
        }
    )

    train_params = dictlistprod(
        {
            "proxy_labels": ["tabpfn"],
            "n_epochs": [1000],
            "weight_synthetic_points": [False],
            "backprop_preproc": [True],
            "N_ensemble_configurations": [1],
            "synthesize_targets": [False],
            "zero_nonexistent_features": [True],
            "init_syn_random": [True],
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            train_params,
            debug_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    dataset_id,
    proxy_labels,
    n_samples,
    n_epochs,
    weight_synthetic_points,
    debug,
    version,
    backprop_preproc,
    N_ensemble_configurations,
    data_collection,
    synthesize_targets,
    zero_nonexistent_features,
    init_syn_random,
):
    kwargs = locals()
    kwargs.pop("ex")
    kwargs.pop("version")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    kwargs["output_dir"] = output_dir
    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False
    from tabpfn.experiments.condensation import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
