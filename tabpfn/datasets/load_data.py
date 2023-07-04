from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids, test_dids_classification
import random
from tabpfn.datasets.caafedata import load_all_data
from tabpfn.datasets.caafedata import get_data_split
from tabpfn.datasets.caafedata import get_X_y

def load_from_all_datasets(dataset_id):
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
    return train_xs, train_ys, test_xs, test_ys


def load_from_well_described_datasets(dataset_id):
    cc_test_datasets_multiclass = load_all_data()
    ds = cc_test_datasets_multiclass[dataset_id]
    ds, df_train, df_test, _, _ = get_data_split(ds, seed=0)
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]
    train_xs, train_ys = get_X_y(df_train, target_column_name)
    test_xs, test_ys = get_X_y(df_test, target_column_name)
    return train_xs, train_ys, test_xs, test_ys