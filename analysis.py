#!/usr/bin/env python3

import os
from typing import List

from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, chi2
import yaml


with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

DIR_IN = os.path.join("out", "bench")
models: List[str] = [
    (m["name"], m.get("quantizations", config["quantizations"]), m.get("prompt_types", config["prompt_types"]))
    for m in config["models"]
]

model_list = []
for model, quantizations, prompt_types in models:
    dir_in_m: str = os.path.join(DIR_IN, model)
    for quant in quantizations:
        dir_in_mq: str = os.path.join(dir_in_m, quant)
        with open(os.path.join(dir_in_mq, "model.yml")) as f:
            props: dict = yaml.safe_load(f)
            file_size_gib: float = props["file_size"] / 1024 ** 3

        labels = dict()
        for dataset in config["datasets"]:
            dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
            if not os.path.exists(dir_in_mqd):
                continue
            labels[dataset] = np.load(os.path.join(dir_in_mqd, "labels.npy"))
        pred = dict()
        for prompt_type in prompt_types:
            for dataset in config["datasets"]:
                name: str = f"{dataset}-{prompt_type}"
                dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
                if not os.path.exists(dir_in_mqd):
                    continue
                pred[name] = np.load(os.path.join(dir_in_mqd, f"pred-{prompt_type}.npy"))
                if "mmlu" in dataset:
                    pred[name] = np.argmax(pred[name], axis=1)
        model_list.append(dict(name=f"{model}-{quant}", labels=labels, pred=pred, file_size_gib=file_size_gib))

model_scores = []
for dataset in config["datasets"]:
    for prompt_type in config["prompt_types"]:
        name: str = f"{dataset}-{prompt_type}"
        print(f"====== {name} ======")

        ncorrect = np.zeros(len(model_list))
        ntest = None
        for i, model_i in enumerate(model_list):
            if ntest is None:
                ntest = model_i["labels"][dataset].shape[0]
            else:
                assert model_i["labels"][dataset].shape[0] == ntest
            ncorrect[i] = np.sum(model_i["pred"][name] == model_i["labels"][dataset])
            labels_bad: int = np.sum(model_i["labels"][dataset] == 123456789)
            print(f"{model_i['name']}: {int(ncorrect[i])}/{ntest}={100*ncorrect[i]/ntest:.2f}% labels_bad={labels_bad}")
        print()

        floor: float = 0.25 if "mmlu" in dataset else 0.0
        model_scores.append(dict(name=name, ncorrect=ncorrect, ntest=ntest, floor=floor))

WR_ERR: float = 0.01


def decompile_pars(pars, unc=False):
    pars = np.asarray(pars)
    assert pars.ndim == 1
    assert pars.shape[0] == 2 * len(model_scores) + len(model_list) - 2
    scales = pars[:len(model_scores)-1]
    elos_datasets = pars[len(model_scores)-1:2*len(model_scores)-1]
    elos_models = pars[2*len(model_scores)-1:]
    if unc:
        scales = np.concatenate([scales, [np.sqrt(np.sum(np.square(scales)))]])
        elos_models = np.concatenate([elos_models, [np.sqrt(np.sum(np.square(elos_models)))]])
    else:
        scales = np.concatenate([scales, [(scales.shape[0]+1)*400 - np.sum(scales)]])
        elos_models = np.concatenate([elos_models, [(elos_models.shape[0]+1)*1500 - np.sum(elos_models)]])
    return (scales, elos_datasets, elos_models)


def get_winrate(elo_self: float, elo_other: float, scale: float, floor: float) -> float:
    winrate = 1 / (1 + 10 ** ((elo_other - elo_self) / scale))
    winrate = floor + (1.0 - floor) * winrate
    return winrate


def get_nll(pars: np.ndarray) -> float:
    scales, elos_datasets, elos_models = decompile_pars(pars)

    nll = 0.0
    for i, ms_i in enumerate(model_scores):
        wr = get_winrate(elos_models, elos_datasets[i], scales[i], ms_i["floor"])
        err = np.sqrt(wr * (1.0 - wr) / ms_i["ntest"] + WR_ERR ** 2)
        residuals = wr - ms_i["ncorrect"] / ms_i["ntest"]
        nll += np.sum(np.square(residuals / err))
        # nll -= 2.0 * np.sum(binom.logpmf(k=ms_i["ncorrect"], n=ms_i["ntest"], p=wr))
    return nll


starting_scales = 400 * np.ones(len(model_scores) - 1)
starting_elos = 1500 * np.ones(len(model_scores) + len(model_list) - 1)
starting_pars = np.concatenate([starting_scales, starting_elos])
print(f"Pre-fit cost: {get_nll(starting_pars):.2f}")

m = Minuit(get_nll, starting_pars)
m.errordef = 1.0
for i in range(len(starting_scales)):
    m.limits[i] = (10, 700)
    # m.fixed[i] = True
m.migrad()
m.hesse()

print(f"Post-fit cost: {m.fval:.2f}")

nll_sat = m.fval
ndf = -len(starting_pars)
for ms in model_scores:
    # nll_sat += 2.0 * np.sum(binom.logpmf(k=ms["ncorrect"], n=ms["ntest"], p=ms["ncorrect"]/ms["ntest"]))
    ndf += ms["ncorrect"].shape[0]
print(f"NLL sat / NDF: {nll_sat:.2f}/{ndf} = {nll_sat/ndf:.2f}")
chi2_prob = 1.0 - chi2.cdf(nll_sat, ndf)
print(f"chi2 probability: {100*chi2_prob:.2f}%")
print()

final_scales, final_elos_datasets, final_elos_models = decompile_pars(m.values)
final_scales_unc, final_elos_datasets_unc, final_elos_models_unc = decompile_pars(m.errors, unc=True)

dataset_elo_scale_unc = zip(model_scores, final_elos_datasets, final_scales, final_elos_datasets_unc, final_scales_unc)
model_elo_unc = sorted(zip(model_list, final_elos_models, final_elos_models_unc), key=lambda meu: meu[1], reverse=True)

for dataset, elo, scale, elo_unc, scale_unc in dataset_elo_scale_unc:
    print(f"{dataset['name']}: elo={elo:.2f}+-{elo_unc:.2f} scale={scale:.2f}+-{scale_unc:.2f}")
print()

for model, elo, unc in model_elo_unc:
    print(f"{model['name']}: {elo:.2f}+-{unc:.2f} @ {model['file_size_gib']:.2f} GiB")
print()

num_within_sigma = [0, 0, 0]
num_total = 0
for i, ms_i in enumerate(model_scores):
    wr_data = ms_i["ncorrect"] / ms_i["ntest"]
    wr_data_unc = np.sqrt(wr_data * (1.0 - wr_data) / wr_data.shape[0])
    wr_elo = get_winrate(final_elos_models, final_elos_datasets[i], final_scales[i], ms_i["floor"])
    abs_diffs = np.abs(wr_data - wr_elo)
    for j, abs_diff_j in enumerate(abs_diffs):
        if abs_diff_j <= 1*wr_data_unc[j]:
            num_within_sigma[0] += 1
        if abs_diff_j <= 2*wr_data_unc[j]:
            num_within_sigma[1] += 1
        if abs_diff_j <= 3*wr_data_unc[j]:
            num_within_sigma[2] += 1
        num_total += 1
        if abs_diff_j > wr_data_unc[j]:
            print(f"dataset={ms_i['name']} model={model_list[j]['name']}: wr_elo={100*wr_elo[j]:.4f}% wr_data={100*wr_data[j]:.4f}%")
print(f"Within 1 sigma: {100*num_within_sigma[0]/num_total:.2f}%")
print(f"Within 2 sigma: {100*num_within_sigma[1]/num_total:.2f}%")
print(f"Within 3 sigma: {100*num_within_sigma[2]/num_total:.2f}%")

for fed, fs, ms in zip(final_elos_datasets, final_scales, model_scores):
    plt.figure()

    x_plot = np.linspace(1000, 2000, 201)
    plt.plot(x_plot, get_winrate(x_plot, fed, fs, ms["floor"]))

    wr_data = ms["ncorrect"] / ms["ntest"]
    wr_data_unc = np.sqrt(wr_data * (1.0 - wr_data) / ms["ntest"] + WR_ERR ** 2)
    plt.errorbar(final_elos_models, wr_data, wr_data_unc, final_elos_models_unc, marker=".", linestyle="none")

    plt.title(ms["name"])
    plt.xlabel("Elo")
    plt.ylabel("Winrate vs. benchmark")
    plt.savefig(f"{ms['name']}.png", dpi=240)

# for model in models:
#     for cot in [False, True]:
#         file_sizes_gib: List[float] = []
#         elos: List[float] = []
#         elos_unc: List[float] = []
#         for quant in quantizations:
#             name: str = f"{model}-{quant}-cot{1 if cot else 0}"

#             for meu in model_elo_unc:
#                 if meu[0]["name"] != name:
#                     continue
#                 file_sizes_gib.append(meu[0]["file_size_gib"])
#                 elos.append(meu[1])
#                 elos_unc.append(meu[2])
#         if not file_sizes_gib:
#             continue
#         plt.errorbar(file_sizes_gib, elos, elos_unc, marker="o", linestyle="none", label=f"{model}-cot{1 if cot else 0}")

# plt.title(", ".join(config["datasets"]))
# plt.xlabel("Model file size [GiB]")
# plt.ylabel("Elo score")
# # plt.xscale("log")
# plt.legend()

# plt.savefig("elo_vs_size.png", dpi=240)
