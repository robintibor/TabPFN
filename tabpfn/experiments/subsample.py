import logging
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd

from ..datasets import (
    load_openml_list,
    open_cc_dids,
    open_cc_valid_dids,
)
from ..scripts.transformer_prediction_interface import (
    TabPFNClassifier,
)

log = logging.getLogger(__name__)


def run_exp(
    dataset_id,
    np_th_seed,
    proxy_labels,
    output_dir,
    debug,
):
    max_samples = 10000
    bptt = 10000

    cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(
        open_cc_dids,
        multiclass=True,
        shuffled=True,
        filter_for_nan=False,
        max_samples=max_samples,
        num_feats=100,
        return_capped=True,
    )
    cc_valid_datasets_multiclass, cc_valid_datasets_multiclass_df = load_openml_list(
        open_cc_valid_dids,
        multiclass=True,
        shuffled=True,
        filter_for_nan=False,
        max_samples=max_samples,
        num_feats=100,
        return_capped=True,
    )

    # Loading longer OpenML Datasets for generalization experiments (optional)
    # test_datasets_multiclass, test_datasets_multiclass_df = load_openml_list(test_dids_classification, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = 10000, num_feats=100, return_capped=True)

    random.seed(0)
    random.shuffle(cc_valid_datasets_multiclass)

    def get_datasets(selector, task_type, suite="cc"):
        if task_type == "binary":
            ds = valid_datasets_binary if selector == "valid" else test_datasets_binary
        else:
            if suite == "openml":
                ds = (
                    valid_datasets_multiclass
                    if selector == "valid"
                    else test_datasets_multiclass
                )
            elif suite == "cc":
                ds = (
                    cc_valid_datasets_multiclass
                    if selector == "valid"
                    else cc_test_datasets_multiclass
                )
            else:
                raise Exception("Unknown suite")
        return ds

    model_string, longer, task_type = "", 1, "multiclass"
    eval_positions = [1000]
    bptt = 2000

    test_datasets, valid_datasets = get_datasets(
        "test", task_type, suite="cc"
    ), get_datasets("valid", task_type, suite="cc")

    ds = test_datasets[dataset_id]
    print(f"Evaluation dataset name: {ds[0]} shape {ds[1].shape}")

    xs, ys = ds[1].clone(), ds[2].clone()
    eval_position = xs.shape[0] // 2
    train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
    test_xs, test_ys = xs[eval_position:], ys[eval_position:]
    if debug:
        train_xs = train_xs[:16]
        train_ys = train_ys[:16]
        test_xs = test_xs[:16]
        test_ys = test_ys[:16]
    classifier = TabPFNClassifier(device="cuda")

    classifier.fit(train_xs, train_ys)
    orig_prediction_ = classifier.predict_proba(test_xs)
    orig_predicted_labels = orig_prediction_.argmax(axis=1)

    wanted_train_inds = np.arange(len(train_xs))

    rng = np.random.RandomState(np_th_seed)
    intermediate_train_inds = []
    intermediate_results = []
    for i_removal in range(int(np.ceil(np.log2(len(wanted_train_inds)))) - 1):
        best_acc = -np.inf
        best_acc_to_test_labels = -np.inf
        best_acc_to_orig_preds = -np.inf
        i_tries = len(wanted_train_inds)
        next_train_inds = []
        for i_try in range(i_tries):
            n_samples = int(np.ceil(len(wanted_train_inds) / 2))
            this_train_inds = rng.choice(wanted_train_inds, n_samples, replace=False)
            included_classes = np.unique(train_ys[this_train_inds])
            if len(included_classes) == 1:
                missing_class = 1 - included_classes[0]
                ind_other_class = rng.choice(
                    wanted_train_inds[train_ys[wanted_train_inds] == missing_class],
                    1,
                    replace=False,
                )[0]
                this_train_inds[-1] = ind_other_class

            classifier.fit(train_xs[this_train_inds], train_ys[this_train_inds])
            prediction_ = classifier.predict_proba(test_xs)
            # impadd
            acc_to_test_labels = np.mean(test_ys.numpy() == prediction_.argmax(axis=1))
            acc_to_orig_preds = np.mean(
                orig_predicted_labels == prediction_.argmax(axis=1)
            )
            acc_to_use = {"tabpfn": acc_to_orig_preds, "test": acc_to_test_labels}[
                proxy_labels
            ]
            if acc_to_use > best_acc:
                best_acc = acc_to_use
                best_acc_to_orig_preds = acc_to_orig_preds
                best_acc_to_test_labels = acc_to_test_labels
                next_train_inds = this_train_inds
        wanted_train_inds = next_train_inds
        intermediate_train_inds.append(deepcopy(wanted_train_inds))
        intermediate_results.append(
            dict(
                test_acc=best_acc_to_test_labels,
                tabpfn_acc=best_acc_to_orig_preds,
                n_samples=len(wanted_train_inds),
            )
        )

        print(best_acc)
        print(len(wanted_train_inds))
    results = {}
    for i_dict, res_dict in enumerate(intermediate_results):
        for key, val in res_dict.items():
            results[key + f"_{i_dict + 1}"] = val

    classifier.fit(train_xs, train_ys)
    prediction_ = classifier.predict_proba(test_xs)
    acc = np.mean(test_ys.numpy() == prediction_.argmax(axis=1))
    results[f"test_acc_0"] = acc
    results[f"tabpfn_acc_0"] = 1  # by definition
    results["n_samples_0"] = len(train_xs)

    intermediate_df = pd.DataFrame(intermediate_results)
    intermediate_df.to_csv(
        os.path.join(output_dir, "intermediate_results.csv"),
    )
    np.save(
        os.path.join(output_dir, "intermediate_inds.npy"),
        np.array(intermediate_train_inds, dtype=object),
    )

    return results
