import itertools
import logging
import os
import random
import time

import numpy as np
import torch

from ..datasets.load_data import (
    load_from_all_datasets,
    load_from_well_described_datasets,
)
from ..scripts.tabular_metrics import auc_metric
from ..scripts.transformer_prediction_interface import TabPFNClassifier
from ..utils import (
    normalize_data,
    remove_outliers,
    normalize_by_used_features_f,
    torch_nanmean,
    torch_nanstd,
)
from sklearn.preprocessing import PowerTransformer
from copy import deepcopy
from tabpfn.masked_multihead_attention import MaskedMultiHeadAttention

log = logging.getLogger(__name__)


def generate_ensemble_configurations(
    differentiable_hps_as_style,
    style,
    seed,
    device,
    softmax_temperature,
    preprocess_transform,
    num_features,
    feature_shift_decoder,
    num_classes,
    multiclass_decoder,
    N_ensemble_configurations,
):
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
        torch.randperm(num_features) if feature_shift_decoder else [0]
    )
    class_shift_configurations = (
        torch.randperm(num_classes) if multiclass_decoder == "permutation" else [0]
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
    return ensemble_configurations


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
    feature_mean,
    feature_std,
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
            eval_xs_, yeo_params, feature_mean, feature_std = preprocess_input(
                eval_xs_,
                eval_ys,
                preprocess_transform=preprocess_transform_configuration,
                max_features=max_features,
                yeo_johnson_params=yeo_params,
                normalize_with_test=normalize_with_test,
                eval_position=eval_position,
                normalize_with_sqrt=normalize_with_sqrt,
                normalize_to_ranking=normalize_to_ranking,
                mean=feature_mean,
                std=feature_std,
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
    return inputs, labels, yeo_params_per_transform, feature_mean, feature_std


def normalize_data_return_mean_std(data, mean=None, std=None, normalize_positions=-1):
    if mean is None:
        assert std is None
        if normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], dim=0)
            std = torch_nanstd(data[:normalize_positions], dim=0) + 0.000001
        else:
            mean = torch_nanmean(data, dim=0)
            std = torch_nanstd(data, dim=0) + 0.000001
    assert mean is not None
    assert std is not None
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data, mean, std


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
    mean,
    std,
):
    import warnings

    orig_device = eval_xs.device
    if eval_xs.shape[1] > 1:
        raise Exception("Transforms only allow one batch dim - TODO")


    # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
    eval_xs, mean, std = normalize_data_return_mean_std(
        eval_xs,
        mean=mean,
        std=std,
        normalize_positions=-1 if normalize_with_test else eval_position,
    )

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

    # Removing empty features
    eval_xs = eval_xs[:, 0, :]
    if yeo_johnson_params is None:
        # otherwise don't remove, probably looking at synthetic data
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

    return eval_xs.to(orig_device), yeo_johnson_params, mean, std


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
    n_samples_per_class,
    proxy_labels,
    output_dir,
    debug,
    n_epochs,
    weight_synthetic_points,
    backprop_preproc,
    N_ensemble_configurations,
    data_collection,
    synthesize_targets,
    zero_nonexistent_features,
    init_syn_random,
    n_samples,
    sample_features_prob,
):
    tqdm = lambda x: x
    trange = range

    style = None
    inference_mode = False
    preprocess_transform = "mix"
    normalize_with_test = False
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
    classifier = TabPFNClassifier(device="cuda")

    model = classifier.model[2]
    from sklearn.utils.validation import check_array

    device = "cuda"

    # Input validation
    eval_xs = np.concatenate([train_xs, test_xs], axis=0)
    eval_xs = torch.tensor(eval_xs, device=device).float().unsqueeze(1)
    eval_ys = torch.tensor(train_ys).float().unsqueeze(1)
    eval_position = len(train_xs)

    num_classes = len(torch.unique(eval_ys))
    num_features = eval_xs.shape[2]
    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)

    model.to(device);

    ensemble_configurations = generate_ensemble_configurations(
        differentiable_hps_as_style,
        style,
        seed,
        device,
        softmax_temperature,
        preprocess_transform,
        num_features,
        feature_shift_decoder,
        num_classes,
        multiclass_decoder,
        N_ensemble_configurations,
    )

    # fix first ensemble configuration to be without preprocessor

    ensemble_configurations[0] = (
        (0, 0),
        'none',
        0,
    )

    import itertools

    # if N_ensemble_configurations == 1:
    #    ensemble_configurations = [default_ensemble_config]

    inputs, labels, yeo_params, feature_mean, feature_std = get_inputs_for_ensemble(
        eval_xs,
        eval_ys,
        ensemble_configurations,
        style,
        num_classes=num_classes,
        max_features=max_features,
        extend_features=extend_features,
        normalize_with_test=normalize_with_test,
        eval_position=eval_position,
        normalize_with_sqrt=normalize_with_sqrt,
        normalize_to_ranking=normalize_to_ranking,
        yeo_params_per_transform=None,
        feature_mean=None,
        feature_std=None,
    )

    assert len(inputs) == 1
    batch_input = inputs[0]
    batch_label = labels[0]
    train_input = batch_input.cuda()[: len(train_xs)]
    test_input = batch_input.cuda()[len(train_xs):]

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
    if backprop_preproc:
        train_x_for_syn, _, _ = normalize_data_return_mean_std(
            eval_xs[:len(train_xs)], mean=feature_mean, std=feature_std)
        train_y_for_syn = eval_ys[:len(train_xs)]
    else:
        train_x_for_syn = train_input
        train_y_for_syn = batch_label

    # https://stackoverflow.com/a/37414115

    
    if n_samples_per_class is not None:
        syn_Xs = []
        syn_ys = []
        for y in torch.unique(train_ys):
            this_class_x = train_x_for_syn[train_ys == y]
            this_class_y = train_y_for_syn[train_ys == y]
            necessary_repeats = int(np.ceil(n_samples_per_class / len(this_class_x)))
            x_repeated = this_class_x.repeat(
                necessary_repeats, *((1,) * (len(this_class_x.shape) - 1))
            )
            y_repeated = this_class_y.repeat(
                necessary_repeats, *((1,) * (len(this_class_y.shape) - 1))
            )
            syn_Xs.append(x_repeated[:n_samples_per_class])
            syn_ys.append(y_repeated[:n_samples_per_class])
    else:
        assert n_samples is not None
        syn_X = train_x_for_syn[:n_samples]
        syn_y = train_y_for_syn[:n_samples]
        included_classes = torch.unique(syn_y)
        if len(included_classes) == 1:
            ind_other_class = torch.nonzero(train_y_for_syn != included_classes[0])[0,0].item()
            syn_X = torch.cat((syn_X[:-1], train_x_for_syn[ind_other_class:ind_other_class+1]))
            syn_y = torch.cat((syn_y[:-1], train_y_for_syn[ind_other_class:ind_other_class+1]))
        syn_Xs = [syn_X]
        syn_ys = [syn_y]
    syn_X = torch.cat(syn_Xs).cuda().detach().clone().requires_grad_(True)
    syn_y = torch.cat(syn_ys).cuda().detach().clone().requires_grad_(True)
    if init_syn_random:
        syn_X.data[:] = (torch.randn_like(syn_X) * 0.05).data[:]
    if weight_synthetic_points:
        seq_attn_alphas = (
            torch.zeros_like(syn_X[:, 0, 0]).cuda().detach().requires_grad_(True)
        )

    # Ensure we have at least two features
    if backprop_preproc:
        for i_col in range(syn_X.shape[2]):
            if len(torch.unique(syn_X[:, 0, i_col])) < 2:
                syn_X.data[-1, 0, i_col] += 0.1

    params_to_optim = [dict(params=[syn_X, ], lr=1e-2)]
    if synthesize_targets:
        params_to_optim.append(dict(params=[syn_y], lr=1e-2))
    if weight_synthetic_points:
        params_to_optim.append(dict(params=[seq_attn_alphas], lr=3e-2))

    opt_syn_X_y = torch.optim.AdamW(params_to_optim, weight_decay=0)

    for i_epoch in trange(n_epochs):
        if weight_synthetic_points:
            softmaxed_seq_attn_mask = torch.nn.functional.softmax(
                seq_attn_alphas, dim=0
            )

            for module in transformed_model.transformer_encoder.modules():
                if hasattr(module, "seq_attention_mask"):
                    module.seq_attention_mask = softmaxed_seq_attn_mask
                    module.avg_attentions = []
        if backprop_preproc:
            syn_X_preproced, syn_y_preproced, _, _, _ = get_inputs_for_ensemble(
                syn_X,
                syn_y,
                ensemble_configurations,
                style,
                num_classes=num_classes,
                max_features=max_features,
                extend_features=extend_features,
                normalize_with_test=normalize_with_test,
                eval_position=eval_position,
                normalize_with_sqrt=normalize_with_sqrt,
                normalize_to_ranking=normalize_to_ranking,
                yeo_params_per_transform=yeo_params,
                # Assume normalized already
                feature_mean=0,
                feature_std=1,
            )

            syn_X_preproced = syn_X_preproced[0]
            syn_y_preproced = syn_y_preproced[0]
        else:
            syn_X_preproced = syn_X
            syn_y_preproced = syn_y
        if sample_features_prob > 0:
            mask = torch.bernoulli(
                    torch.ones_like(train_input) * (1-sample_features_prob),
                )
            if i_epoch == n_epochs -1 :
                # last batch just take normal data, also to get correct train auc
                mask = torch.ones_like(train_input)
            i_permute = torch.randint(
                train_input.shape[0], train_input.shape, device="cuda"
            )
            this_test_inputs = (
                mask * train_input
                + (1 - mask) * train_input.gather(dim=0, index=i_permute)
            )
            with torch.no_grad():
                merged_orig_output = predict_outputs(
                    model,
                    torch.cat((train_input, this_test_inputs)),
                    batch_label.float(),
                    num_classes,
                    ensemble_configurations,
                )
        else:
            this_test_inputs = train_input
        merged_output = predict_outputs(
            transformed_model,
            torch.cat((syn_X_preproced, this_test_inputs)),
            syn_y_preproced,
            num_classes,
            ensemble_configurations,
        )
        if proxy_labels == "tabpfn":
            kl_div = kl_divergence(merged_orig_output, merged_output)
        else:
            assert proxy_labels == "train"
            kl_div = torch.nn.functional.cross_entropy(
                merged_output, eval_ys.squeeze().type(torch.int64))
        opt_syn_X_y.zero_grad(set_to_none=True)
        kl_div.backward()
        opt_syn_X_y.step()
        if zero_nonexistent_features:
            syn_X.data[:, :, num_features:] = 0  # do not optimize nonexisting features
        opt_syn_X_y.zero_grad(set_to_none=True)
        acc = torch.mean(1.0 * (merged_output.argmax(dim=1).cpu() == train_ys)).item()
        auc = auc_metric(
            train_ys.cpu().numpy(),
            torch.softmax(merged_output, dim=1).detach().cpu().numpy(),
            multi_class="ovo",
        )

        # Ensure we have at least two features
        if backprop_preproc:
            for i_col in range(syn_X.shape[2]):
                if len(torch.unique(syn_X[:, 0, i_col])) < 2:
                    syn_X.data[-1, 0, i_col] += 0.1

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

            with torch.no_grad():
                if backprop_preproc:
                    syn_X_preproced, syn_y_preproced, _, _, _ = get_inputs_for_ensemble(
                        syn_X,
                        syn_y,
                        ensemble_configurations,
                        style,
                        num_classes=num_classes,
                        max_features=max_features,
                        extend_features=extend_features,
                        normalize_with_test=normalize_with_test,
                        eval_position=eval_position,
                        normalize_with_sqrt=normalize_with_sqrt,
                        normalize_to_ranking=normalize_to_ranking,
                        yeo_params_per_transform=yeo_params,
                        feature_mean=0,
                        feature_std=1,
                    )
                    syn_X_preproced = syn_X_preproced[0]
                    syn_y_preproced = syn_y_preproced[0]
                else:
                    syn_X_preproced = syn_X
                    syn_y_preproced = syn_y
                merged_output = predict_outputs(
                    transformed_model,
                    torch.cat((syn_X_preproced, test_input)),
                    syn_y_preproced,
                    num_classes,
                    ensemble_configurations,
                )
            test_acc = torch.mean(
                1.0 * (merged_output.argmax(dim=1).cpu() == test_ys)
            ).item()
            test_auc = auc_metric(
                test_ys.cpu().numpy(),
                torch.softmax(merged_output, dim=1).cpu().numpy(),
                multi_class="ovo",
            )
            print(f"KL Divergence:  {kl_div.item():.2E}")
            print(f"Train Accuracy: {acc:.1%}")
            print(f"Train AUC:      {auc:.1%}")
            print(f"Test Accuracy:  {test_acc:.1%}")
            print(f"Test AUC:       {test_auc:.1%}")

    results = dict(
        test_acc=test_acc,
        test_auc=test_auc,
        train_auc=auc,
        train_acc=acc,
        kl_div=kl_div.item(),
    )
    np.save(
        os.path.join(output_dir, "syn_X.npy"),
        syn_X.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(output_dir, "syn_y.npy"),
        syn_y.detach().cpu().numpy(),
    )

    return results
