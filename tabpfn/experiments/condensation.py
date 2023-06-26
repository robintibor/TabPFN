import logging
import os
import random
import time

import numpy as np
import torch

from ..datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from ..scripts.transformer_prediction_interface import TabPFNClassifier
from ..utils import normalize_data, remove_outliers, normalize_by_used_features_f
from sklearn.preprocessing import PowerTransformer
from ..utils import NOP
from copy import deepcopy
from tabpfn.masked_multihead_attention import MaskedMultiHeadAttention

log = logging.getLogger(__name__)


def kl_divergence(clf_out_a, clf_out_b, reduction="mean"):
    assert clf_out_a.shape == clf_out_b.shape
    kl_divs_per_example = torch.sum(
        torch.nn.functional.softmax(clf_out_a, dim=1)
        * (
            torch.nn.functional.log_softmax(clf_out_a, dim=1)
            - torch.nn.functional.log_softmax(clf_out_b, dim=1)
        ),
        dim=1,
    )
    if reduction == "mean":
        kl_div = torch.mean(kl_divs_per_example)
    elif reduction == "sum":
        kl_div = torch.sum(kl_divs_per_example)
    else:
        assert reduction is None or reduction == "none"
        kl_div = kl_divs_per_example
    return kl_div


def predict_outputs(model, X, y, num_classes, ensemble_configurations):
    outputs_model = model((None, X, y), single_eval_pos=len(y))[:, :, 0:num_classes]

    reshuffled_outputs = []
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration
        output_ = outputs_model[:, i : i + 1, :]
        output_ = torch.cat(
            [
                output_[..., class_shift_configuration:],
                output_[..., :class_shift_configuration],
            ],
            dim=-1,
        )
        reshuffled_outputs.append(output_.squeeze(1))

    merged_output = torch.mean(torch.stack(reshuffled_outputs), dim=0)
    return merged_output


def get_inputs_for_ensemble(
    eval_xs,
    eval_ys,
    ensemble_configurations,
    style,
    num_classes,
    max_features,
    extend_features,
    normalize_with_test,
    eval_position,
    normalize_with_sqrt,
    normalize_to_ranking,
    yeo_params_per_transform,
):

    output = None

    eval_xs_transformed = {}
    inputs, labels = [], []
    if yeo_params_per_transform is None:
        yeo_params_per_transform = {}
    start = time.time()
    for ensemble_configuration in ensemble_configurations:
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration

        style_ = (
            style[styles_configuration : styles_configuration + 1, :]
            if style is not None
            else style
        )
        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed[preprocess_transform_configuration].clone()
        else:
            if preprocess_transform_configuration in yeo_params_per_transform:
                yeo_params = yeo_params_per_transform[
                    preprocess_transform_configuration
                ]
            else:
                yeo_params = None
            eval_xs_, yeo_params = preprocess_input(
                eval_xs_,
                eval_ys,
                preprocess_transform=preprocess_transform_configuration,
                max_features=max_features,
                yeo_johnson_params=yeo_params,
                normalize_with_test=normalize_with_test,
                eval_position=eval_position,
                normalize_with_sqrt=normalize_with_sqrt,
                normalize_to_ranking=normalize_to_ranking,
            )
            eval_xs_transformed[preprocess_transform_configuration] = eval_xs_
            if preprocess_transform_configuration not in yeo_params_per_transform:
                yeo_params_per_transform[
                    preprocess_transform_configuration
                ] = yeo_params

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat(
            [
                eval_xs_[..., feature_shift_configuration:],
                eval_xs_[..., :feature_shift_configuration],
            ],
            dim=-1,
        )

        # Extend X
        if extend_features:
            eval_xs_ = torch.cat(
                [
                    eval_xs_,
                    torch.zeros(
                        (
                            eval_xs_.shape[0],
                            eval_xs_.shape[1],
                            max_features - eval_xs_.shape[2],
                        )
                    ).to(eval_xs_.device),
                ],
                -1,
            )
        inputs += [eval_xs_]
        labels += [eval_ys_]
    batch_size_inference = 16
    inputs = torch.cat(inputs, 1)
    inputs = torch.split(inputs, batch_size_inference, dim=1)
    labels = torch.cat(labels, 1)
    labels = torch.split(labels, batch_size_inference, dim=1)
    return inputs, labels, yeo_params_per_transform


def preprocess_input(
    eval_xs,
    eval_ys,
    preprocess_transform,
    max_features,
    yeo_johnson_params,
    normalize_with_test,
    eval_position,
    normalize_with_sqrt,
    normalize_to_ranking,
):
    import warnings

    orig_device = eval_xs.device
    if eval_xs.shape[1] > 1:
        raise Exception("Transforms only allow one batch dim - TODO")
    if preprocess_transform != "none":
        if preprocess_transform == "power" or preprocess_transform == "power_all":
            pt = PowerTransformer(standardize=True)
        elif (
            preprocess_transform == "quantile" or preprocess_transform == "quantile_all"
        ):
            assert False
            pt = QuantileTransformer(output_distribution="normal")
        elif preprocess_transform == "robust" or preprocess_transform == "robust_all":
            assert False
            pt = RobustScaler(unit_variance=True)

    # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
    eval_xs = normalize_data(
        eval_xs, normalize_positions=-1 if normalize_with_test else eval_position
    )

    # Removing empty features
    eval_xs = eval_xs[:, 0, :]
    sel = [
        len(torch.unique(eval_xs[0 : eval_ys.shape[0], col])) > 1
        for col in range(eval_xs.shape[1])
    ]
    eval_xs = eval_xs[:, sel]

    warnings.simplefilter("error")
    if preprocess_transform != "none":
        if yeo_johnson_params is None:
            yeo_johnson_params = []
            eval_xs = eval_xs.cpu().numpy()
            feats = (
                set(range(eval_xs.shape[1]))
                if "all" in preprocess_transform
                else set(range(eval_xs.shape[1])) - set(categorical_feats)
            )
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_position, col : col + 1])
                    yeo_johnson_params.append(
                        deepcopy(
                            {
                                "lambda": pt.lambdas_.item(),
                                "mean": pt._scaler.mean_.item(),
                                "scale": pt._scaler.scale_.item(),
                            }
                        )
                    )
                    trans = pt.transform(eval_xs[:, col : col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs[:, col : col + 1] = trans
                except:
                    yeo_johnson_params.append(None)
            eval_xs = torch.tensor(eval_xs).float()
        else:
            feats = set(range(eval_xs.shape[1]))
            assert len(yeo_johnson_params) == len(feats)
            for col, params in zip(feats, yeo_johnson_params):
                if params is not None:
                    yeo_transformed = _yeo_johnson_transform_th(
                        eval_xs[:, col : col + 1], params["lambda"]
                    )
                    standardized = (yeo_transformed - params["mean"]) / (
                        params["scale"]
                    )
                    eval_xs[:, col : col + 1] = standardized

    warnings.simplefilter("default")

    eval_xs = eval_xs.unsqueeze(1)

    # TODO: Cautian there is information leakage when to_ranking is used, we should not use it
    eval_xs = (
        remove_outliers(
            eval_xs,
            normalize_positions=-1 if normalize_with_test else eval_position,
        )
        if not normalize_to_ranking
        else normalize_data(to_ranking_low_mem(eval_xs))
    )
    # Rescale X
    eval_xs = normalize_by_used_features_f(
        eval_xs,
        eval_xs.shape[-1],
        max_features,
        normalize_with_sqrt=normalize_with_sqrt,
    )

    return eval_xs.to(orig_device), yeo_johnson_params


def _yeo_johnson_transform_th(x, lmbda):
    """Return transformed input x following Yeo-Johnson transform with
    parameter lambda.
    """

    out = torch.zeros_like(x)
    pos = x >= 0  # binary mask

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        out[pos] = torch.log1p(x[pos])
    else:  # lmbda != 0
        out[pos] = (torch.pow(x[pos] + 1, lmbda) - 1) / lmbda

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        out[~pos] = -(torch.pow(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    else:  # lmbda == 2
        out[~pos] = -torch.log1p(-x[~pos])

    return out


def run_exp(
    dataset_id,
    n_samples,
    proxy_labels,
    output_dir,
    debug,
    n_epochs,
    weight_synthetic_points,
):
    tqdm = lambda x: x
    trange = range
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
    model = classifier.model[2]
    from sklearn.utils.validation import check_array

    device = "cuda"
    X = test_xs

    # Input validation
    X = check_array(X, force_all_finite=False)
    X_full = np.concatenate([train_xs, X], axis=0)
    X_full = torch.tensor(X_full, device=device).float().unsqueeze(1)
    y_full = np.concatenate([train_ys, np.zeros_like(X[:, 0])], axis=0)
    y_full = torch.tensor(y_full, device=device).float().unsqueeze(1)

    eval_pos = train_xs.shape[0]
    style = None
    inference_mode = False
    preprocess_transform = "mix"
    normalize_with_test = False
    N_ensemble_configurations = 3
    softmax_temperature = None
    combine_preprocessing = False
    multiclass_decoder = "permutation"
    feature_shift_decoder = True
    differentiable_hps_as_style = False
    seed = 0
    max_features = 100
    rescale_features = True
    normalize_to_ranking = False
    normalize_with_sqrt = False
    extend_features = True  # from function defaults
    batch_size_inference = 16

    eval_xs = X_full
    eval_ys = y_full
    eval_position = eval_pos
    num_classes = len(torch.unique(eval_ys))

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)

    import itertools

    if not differentiable_hps_as_style:
        style = None

    if style is not None:
        style = style.to(device)
        style = style.unsqueeze(0) if len(style.shape) == 1 else style
        num_styles = style.shape[0]
        softmax_temperature = (
            softmax_temperature
            if softmax_temperature.shape
            else softmax_temperature.unsqueeze(0).repeat(num_styles)
        )
    else:
        num_styles = 1
        style = None
        softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(0, num_styles)

    preprocess_transform_configurations = (
        ["none", "power_all"]
        if preprocess_transform == "mix"
        else [preprocess_transform]
    )

    if seed is not None:
        torch.manual_seed(seed)
    print(preprocess_transform_configurations)

    feature_shift_configurations = (
        torch.randperm(eval_xs.shape[2]) if feature_shift_decoder else [0]
    )
    class_shift_configurations = (
        torch.randperm(len(torch.unique(eval_ys)))
        if multiclass_decoder == "permutation"
        else [0]
    )
    ensemble_configurations = list(
        itertools.product(class_shift_configurations, feature_shift_configurations)
    )
    # default_ensemble_config = ensemble_configurations[0]
    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(
        itertools.product(
            ensemble_configurations,
            preprocess_transform_configurations,
            styles_configurations,
        )
    )
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]
    # if N_ensemble_configurations == 1:
    #    ensemble_configurations = [default_ensemble_config]

    inputs, labels, yeo_params = get_inputs_for_ensemble(
        eval_xs, eval_ys, ensemble_configurations,
        style,
        num_classes=num_classes,
        max_features=max_features, extend_features=extend_features,
        normalize_with_test=normalize_with_test,
        eval_position=eval_position,
        normalize_with_sqrt=normalize_with_sqrt,
        normalize_to_ranking=normalize_to_ranking,
        yeo_params_per_transform=None)

    assert len(inputs) == 1
    batch_input = inputs[0]
    batch_label = labels[0]
    train_input = batch_input.cuda()[: len(train_xs)]
    test_input = batch_input.cuda()[len(train_xs) :]

    transformed_model = deepcopy(model)

    if weight_synthetic_points:
        for module in transformed_model.transformer_encoder.modules():
            if hasattr(module, "self_attn"):
                th_attn = module.self_attn
                own_attn = MaskedMultiHeadAttention(
                    th_attn.embed_dim, th_attn.num_heads, activation=None
                ).cuda()
                w_q, w_k, w_v = th_attn.in_proj_weight.chunk(3)
                b_q, b_k, b_v = th_attn.in_proj_bias.chunk(3)
                assert w_q.shape == own_attn.linear_q.weight.shape
                assert w_k.shape == own_attn.linear_k.weight.shape
                assert w_v.shape == own_attn.linear_v.weight.shape
                assert b_q.shape == own_attn.linear_q.bias.shape
                assert b_k.shape == own_attn.linear_k.bias.shape
                assert b_v.shape == own_attn.linear_v.bias.shape

                own_attn.linear_q.weight.data[:] = w_q.data[:]
                own_attn.linear_q.bias.data[:] = b_q.data[:]
                own_attn.linear_k.weight.data[:] = w_k.data[:]
                own_attn.linear_k.bias.data[:] = b_k.data[:]
                own_attn.linear_v.weight.data[:] = w_v.data[:]
                own_attn.linear_v.bias.data[:] = b_v.data[:]
                own_attn.linear_o.weight.data[:] = th_attn.out_proj.weight.data[:]
                own_attn.linear_o.bias.data[:] = th_attn.out_proj.bias.data[:]
                module.self_attn = own_attn

    with torch.no_grad():
        merged_orig_output = predict_outputs(
            model,
            torch.cat((train_input, train_input)),
            batch_label.float(),
            num_classes,
            ensemble_configurations,
        )

    necessary_repeats = int(np.ceil(n_samples / len(train_xs)))
    inputs_repeated = batch_input.repeat(
        necessary_repeats, *((1,) * (len(batch_input.shape) - 1))
    )
    labels_repeated = batch_label.repeat(
        necessary_repeats, *((1,) * (len(batch_label.shape) - 1))
    )

    syn_X = inputs_repeated[:n_samples].cuda().detach().clone().requires_grad_(True)
    syn_y = (
        labels_repeated[:n_samples].float().cuda().detach().clone().requires_grad_(True)
    )

    if weight_synthetic_points:
        seq_attn_alphas = (
            torch.zeros_like(syn_X[:, 0, 0]).cuda().detach().requires_grad_(True)
        )

    # Ensure we have at least two classes
    included_classes = np.unique(train_ys[:n_samples])
    if len(included_classes) == 1:
        included_class = included_classes[0]
        ind_other_class = np.flatnonzero(train_ys != included_class)[0]
        syn_X.data[-1] = batch_input[ind_other_class].cuda().data[:]
        syn_y.data[-1] = batch_label[ind_other_class].float().cuda().data[:]

    params_to_optim = [dict(params=[syn_X, syn_y], lr=1e-2)]
    if weight_synthetic_points:
        params_to_optim.append(dict(params=[seq_attn_alphas], lr=3e-2))

    opt_syn_X_y = torch.optim.AdamW(params_to_optim, weight_decay=0)

    for i_epoch in trange(n_epochs):
        if weight_synthetic_points:
            softmaxed_seq_attn_mask = torch.nn.functional.softmax(seq_attn_alphas, dim=0)
            for module in transformed_model.transformer_encoder.modules():
                if hasattr(module, "seq_attention_mask"):
                    module.seq_attention_mask = softmaxed_seq_attn_mask
                    module.avg_attentions = []
        merged_output = predict_outputs(
            transformed_model,
            torch.cat((syn_X, train_input)),
            syn_y,
            num_classes,
            ensemble_configurations,
        )
        kl_div = kl_divergence(merged_orig_output, merged_output)
        opt_syn_X_y.zero_grad(set_to_none=True)
        kl_div.backward()
        opt_syn_X_y.step()
        opt_syn_X_y.zero_grad(set_to_none=True)
        acc = torch.mean(1.0 * (merged_output.argmax(dim=1).cpu() == train_ys)).item()
        if i_epoch % max(1, n_epochs // 10) == 0:
            if weight_synthetic_points:
                with torch.no_grad():
                    softmaxed_seq_attn_mask = torch.nn.functional.softmax(
                        seq_attn_alphas, dim=0
                    )
                    for module in transformed_model.transformer_encoder.modules():
                        if hasattr(module, "seq_attention_mask"):
                            module.seq_attention_mask = softmaxed_seq_attn_mask
                            module.avg_attentions = []
            merged_output = predict_outputs(
                transformed_model,
                torch.cat((syn_X, test_input)),
                syn_y,
                num_classes,
                ensemble_configurations,
            )
            test_acc = torch.mean(
                1.0 * (merged_output.argmax(dim=1).cpu() == test_ys)
            ).item()
            print(f"KL Divergence: {kl_div.item():.2E}")
            print(f"Train Accuracy: {acc:.1%}")
            print(f"Test Accuracy: {test_acc:.1%}")

    results = dict(test_acc=test_acc, train_acc=acc, kl_div=kl_div.item())
    np.save(
        os.path.join(output_dir, "syn_X.npy"),
        syn_X.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(output_dir, "syn_y.npy"),
        syn_y.detach().cpu().numpy(),
    )

    return results
