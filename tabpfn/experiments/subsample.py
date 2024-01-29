import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from ..datasets.load_data import load_from_all_datasets, load_from_well_described_datasets
from ..scripts.tabular_metrics import auc_metric
from ..scripts.transformer_prediction_interface import (
    TabPFNClassifier,
)

log = logging.getLogger(__name__)


def run_exp(
    dataset_id,
    np_th_seed,
    proxy_labels,
    output_dir,
    data_collection,
    debug,
    N_ensemble_configurations,
):
    tqdm = lambda x: x
    trange = range
    if data_collection == 'all':
        train_xs, train_ys, test_xs, test_ys = load_from_all_datasets(dataset_id)
    else:
        assert data_collection == 'cafee'
        train_xs, train_ys, test_xs, test_ys = load_from_well_described_datasets(dataset_id)
    if debug:
        train_xs = train_xs[:16]
        train_ys = train_ys[:16]
        test_xs = test_xs[:16]
        test_ys = test_ys[:16]
    classifier = TabPFNClassifier(device="cuda", N_ensemble_configurations=N_ensemble_configurations)

    classifier.fit(train_xs, train_ys, overwrite_warning=True)
    orig_prediction_ = classifier.predict_proba(test_xs)
    orig_predicted_labels = orig_prediction_.argmax(axis=1)

    # impadd
    orig_prediction_train = classifier.predict_proba(train_xs)
    orig_predicted_train_labels = orig_prediction_train.argmax(axis=1)

    wanted_train_inds = np.arange(len(train_xs))
    n_intermediate_samples_per_step = np.maximum(
        np.round(len(train_xs) / (2 ** np.arange(1, np.ceil(np.log2(len(train_xs)))))),
        2,
    )
    n_intermediate_samples_per_step = np.int64(n_intermediate_samples_per_step)

    rng = np.random.RandomState(np_th_seed)
    intermediate_train_inds = []
    intermediate_results = []
    for n_intermediate_samples in tqdm(n_intermediate_samples_per_step):
        best_acc = -np.inf
        i_tries = len(wanted_train_inds)
        next_train_inds = []
        for i_try in trange(i_tries):
            this_train_inds = rng.choice(
                wanted_train_inds, n_intermediate_samples, replace=False
            )
            included_classes = np.unique(train_ys[this_train_inds])
            if len(included_classes) == 1:
                ind_other_class = rng.choice(
                    wanted_train_inds[train_ys[wanted_train_inds] != included_classes[0]],
                    1,
                    replace=False,
                )[0]
                this_train_inds[-1] = ind_other_class

            classifier.fit(train_xs[this_train_inds], train_ys[this_train_inds], overwrite_warning=True)
            prediction_ = classifier.predict_proba(test_xs)
            prediction_train = classifier.predict_proba(train_xs)

            acc_to_test_labels = np.mean(test_ys.numpy() == prediction_.argmax(axis=1))
            acc_to_orig_preds = np.mean(
                orig_predicted_labels == prediction_.argmax(axis=1)
            )
            acc_to_train_labels = np.mean(
                train_ys.numpy() == prediction_train.argmax(axis=1)
            )
            train_auc = auc_metric(
                train_ys.numpy(), prediction_train, multi_class='ovo')
            test_auc = auc_metric(
                test_ys.numpy(), prediction_, multi_class='ovo')
            acc_to_use = {
                "tabpfn": acc_to_orig_preds,
                "test": acc_to_test_labels,
                "train": acc_to_train_labels,
            }[proxy_labels]
            if acc_to_use > best_acc:
                best_acc = acc_to_use
                best_acc_to_orig_preds = acc_to_orig_preds
                best_acc_to_test_labels = acc_to_test_labels
                best_auc_to_test_labels = test_auc
                best_acc_to_train_labels = acc_to_train_labels
                best_auc_to_train_labels = train_auc
                next_train_inds = this_train_inds

        wanted_train_inds = next_train_inds
        intermediate_train_inds.append(deepcopy(wanted_train_inds))
        intermediate_results.append(
            dict(
                test_auc=best_auc_to_test_labels,
                test_acc=best_acc_to_test_labels,
                tabpfn_acc=best_acc_to_orig_preds,
                train_acc=best_acc_to_train_labels,
                train_auc=best_auc_to_train_labels,
                n_samples=len(wanted_train_inds),
            )
        )

        print(best_acc)
        print(len(wanted_train_inds))
    results = {}
    for i_dict, res_dict in enumerate(intermediate_results):
        for key, val in res_dict.items():
            results[key + f"_{i_dict + 1}"] = val

    classifier.fit(train_xs, train_ys, overwrite_warning=True)
    prediction_ = classifier.predict_proba(test_xs)
    acc = np.mean(test_ys.numpy() == prediction_.argmax(axis=1))
    acc_train = np.mean(train_ys.numpy() == orig_predicted_train_labels)
    train_auc = auc_metric(
        train_ys.numpy(), orig_predicted_train_labels, multi_class='ovo')
    test_auc = auc_metric(
        test_ys.numpy(), prediction_, multi_class='ovo')
    results[f"test_acc_0"] = acc
    results[f"test_auc_0"] = test_auc
    results[f"tabpfn_acc_0"] = 1  # by definition
    results[f"train_auc_0"] = train_auc
    results[f"train_acc_0"] = acc_train
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
