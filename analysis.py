#!/usr/bin/env python3

import os
from typing import List

from adjustText import adjust_text
from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, chi2
from tabulate import tabulate
import yaml


plt.rcParams["figure.figsize"] = (14, 11)
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

DIR_IN = os.path.join("out", "bench")
DIR_OUT = os.path.join("out", "analysis")
os.makedirs(DIR_OUT, exist_ok=True)

models: List[str] = [
    (m["name"], m.get("quantizations", config["quantizations"]), m.get("prompt_types", config["prompt_types"]))
    for m in config["models"]
]

model_list = []
for model, quantizations, prompt_types in models:
    dir_in_m: str = os.path.join(DIR_IN, model)
    for quant in quantizations:
        dir_in_mq: str = os.path.join(dir_in_m, quant)
        path_meta: str = os.path.join(dir_in_mq, "model.yml")
        if not os.path.exists(path_meta):
            continue
        with open(path_meta) as f:
            props: dict = yaml.safe_load(f)
            file_size_gib: float = props["file_size"] / 1024 ** 3

        all_files_existent: bool = True
        labels = dict()
        for dataset in config["datasets"]:
            dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
            path_labels: str = os.path.join(dir_in_mqd, "labels.npy")
            if not os.path.exists(path_labels):
                all_files_existent = False
                break
            labels[dataset] = np.load(path_labels)
        pred = dict()
        for prompt_type in prompt_types:
            for dataset in config["datasets"]:
                name: str = f"{dataset}-{prompt_type}"
                dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
                path_pred: str = os.path.join(dir_in_mqd, f"pred-{prompt_type}.npy")
                if not os.path.exists(path_pred):
                    all_files_existent = False
                    break
                pred[name] = np.load(path_pred)
                if dataset in ["mmlu_test", "gpqa_main", "mmlu-pro_test"]:
                    pred[name] = np.argmax(pred[name], axis=1)
        if all_files_existent:
            model_list.append(dict(name=f"{model}-{quant}", labels=labels, pred=pred, file_size_gib=file_size_gib))

model_scores = []
for dataset in config["datasets"]:
    for prompt_type in config["prompt_types"]:
        name: str = f"{dataset}-{prompt_type}"
        print(f"## {name}")
        print()

        if dataset in ["mmlu_test", "gpqa_main"]:
            floor: float = 0.25
        elif dataset == "gsm8k_test":
            floor: float = 0.00
        elif dataset == "mmlu-pro_test":
            floor: float = 0.10
        else:
            assert False

        rows: list = []
        ncorrect = np.zeros(len(model_list), dtype=np.int64)
        ntest = None
        for i, model_i in enumerate(model_list):
            if ntest is None:
                ntest = model_i["labels"][dataset].shape[0]
            else:
                assert model_i["labels"][dataset].shape[0] == ntest
            ncorrect[i] = np.sum(model_i["pred"][name] == model_i["labels"][dataset])
            rows.append([model_i["name"], model_i["file_size_gib"], f"{ncorrect[i]}/{ntest}", ncorrect[i]/ntest])
        rows = sorted(rows, key=lambda r: r[1], reverse=True)
        for r1 in rows:
            pareto_frontier: bool = True
            for r2 in rows:
                if r2[1] < r1[1] and r2[3] > r1[3]:
                    pareto_frontier = False
                    break
            r1.append(pareto_frontier)
        plt.figure()
        file_sizes_gib = np.array([r[1] for r in rows if r[4]])
        win_rates = np.array([r[3] for r in rows if r[4]])
        win_rates_unc = np.sqrt(win_rates * (1.0 - win_rates) / ntest)
        plt.errorbar(file_sizes_gib, win_rates, win_rates_unc, marker=".")
        file_sizes_gib = np.array([r[1] for r in rows if not r[4]])
        win_rates = np.array([r[3] for r in rows if not r[4]])
        win_rates_unc = np.sqrt(win_rates * (1.0 - win_rates) / ntest)
        plt.errorbar(file_sizes_gib, win_rates, win_rates_unc, marker=".", linestyle="none")
        if floor != 0.0:
            plt.hlines(floor, 0, 50, colors="black", linestyles=":")
        texts = [plt.text(r[1], r[3], r[0]) for r in rows]
        plt.xlim(0, 50)
        plt.ylim(0, 1)
        adjust_text(texts, [r[1] for r in rows], [r[3] for r in rows],
            arrowprops=dict(arrowstyle="->", color="lightgray"), expand=(1.35, 2.3),
            force_text=(0.4, 0.8), force_explode=(0.4, 1.0), ensure_inside_axes=True, max_move=100)
        plt.title(name)
        plt.xlabel("Model file size [GiB]")
        plt.ylabel("Model winrate vs. benchmark")
        plt.savefig(os.path.join(DIR_OUT, f"{name}-filesize-winrate.png"), dpi=240)

        for r1 in rows:
            r1[1] = f"{r1[1]:.2f}"
            r1[3] = f"= {100*r1[3]:.2f}%"
            r1[4] = "Yes" if r1[4] else "No"
        print(tabulate(rows, headers=["Model", "File size [GiB]", "Correct answers", "", "Pareto frontier?"], tablefmt="github"))
        print()

        model_scores.append(dict(name=name, ncorrect=ncorrect, ntest=ntest, floor=floor))


def decompile_pars(pars, cov_mat=None):
    pars = np.asarray(pars)
    assert pars.ndim == 1
    assert pars.shape[0] == 2 * len(model_scores) + len(model_list) - 2
    scales = pars[:len(model_scores)-1]
    elos_datasets = pars[len(model_scores)-1:2*len(model_scores)-1]
    elos_models = pars[2*len(model_scores)-1:]
    if cov_mat is not None:
        cov_mat = np.asarray(cov_mat)
        cov_mat_s = cov_mat[:len(model_scores)-1, :len(model_scores)-1]
        scales = np.concatenate(
            [scales, [np.sqrt(np.ones_like(scales) @ cov_mat_s @ np.ones_like(scales))]])
        cov_mat_em = cov_mat[2*len(model_scores)-1:, 2*len(model_scores)-1:]
        elos_models = np.concatenate(
            [elos_models, [np.sqrt(np.ones_like(elos_models) @ cov_mat_em @ np.ones_like(elos_models))]])
    else:
        scales = np.concatenate([scales, [(scales.shape[0]+1)*400 - np.sum(scales)]])
        elos_models = np.concatenate([elos_models, [(elos_models.shape[0]+1)*1500 - np.sum(elos_models)]])
    return (scales, elos_datasets, elos_models)


def get_winrate(elo_self: float, elo_other: float, scale: float, floor: float) -> float:
    winrate = 1 / (1 + 10 ** ((elo_other - elo_self) / scale))
    winrate = floor + (1.0 - floor) * winrate
    return winrate


def get_nll(pars: np.ndarray, wr_err: float) -> float:
    scales, elos_datasets, elos_models = decompile_pars(pars)

    nll = 0.0
    for i, ms_i in enumerate(model_scores):
        wr = get_winrate(elos_models, elos_datasets[i], scales[i], ms_i["floor"])
        err = np.sqrt(wr * (1.0 - wr) / ms_i["ntest"] + wr_err ** 2)
        residuals = wr - ms_i["ncorrect"] / ms_i["ntest"]
        nll += np.sum(np.square(residuals / err))
        # nll -= 2.0 * np.sum(binom.logpmf(k=ms_i["ncorrect"], n=ms_i["ntest"], p=wr))
    return nll


starting_scales = 400 * np.ones(len(model_scores) - 1)
starting_elos = 1500 * np.ones(len(model_scores) + len(model_list) - 1)
starting_pars = np.concatenate([starting_scales, starting_elos])
print(f"Pre-fit cost: {get_nll(starting_pars, 0.0):.2f}")

ndf = -len(starting_pars)
for ms in model_scores:
    ndf += ms["ncorrect"].shape[0]


def get_minuit(wr_err: float):
    def func(pars):
        return get_nll(pars, wr_err)
    m = Minuit(func, starting_pars)
    m.errordef = 1.0
    for i in range(len(starting_scales)):
        m.limits[i] = (10, 1000)
        # m.fixed[i] = True
    m.migrad()
    m.hesse()
    return m


tol = 1e-4
wr_err_low = 0.00
wr_err_high = 0.01
m_low = get_minuit(wr_err_low)
if m_low.fval <= ndf + tol:
    wr_err_final = wr_err_low
    m_final = m_low
else:
    m_high = get_minuit(wr_err_high)
    i = 0
    while m_high.fval > ndf:
        wr_err_low = wr_err_high
        wr_err_high += 0.01
        m_low = m_high
        m_high = get_minuit(wr_err_high)
        i += 1
        assert i <= 10
    wr_err_test = (wr_err_low + wr_err_high) / 2
    m_test = get_minuit(wr_err_test)

    i = 0
    while abs(m_test.fval - ndf) > tol and i < 10:
        if m_test.fval > ndf:
            wr_err_low = wr_err_test
            m_low = m_test
        else:
            wr_err_high = wr_err_test
            m_high = m_test
        wr_err_test = (wr_err_low + wr_err_high) / 2
        m_test = get_minuit(wr_err_test)
        i += 1
    wr_err_final = wr_err_low
    m_final = m_test

print(f"wr_err_final={100*wr_err_final:.2f}%")

nll_sat = m_final.fval
# for ms in model_scores:
    # nll_sat += 2.0 * np.sum(binom.logpmf(k=ms["ncorrect"], n=ms["ntest"], p=ms["ncorrect"]/ms["ntest"]))
print(f"chi2 / NDF: {nll_sat:.2f}/{ndf} = {nll_sat/ndf if ndf > 0 else np.nan:.2f}")
chi2_prob = 1.0 - chi2.cdf(nll_sat, ndf)
print(f"chi2 probability: {100*chi2_prob:.2f}%")
print()

print(f"Post-fit cost: {m_final.fval:.2f}")
print()

final_scales, final_elos_datasets, final_elos_models = decompile_pars(m_final.values)
final_scales_unc, final_elos_datasets_unc, final_elos_models_unc = decompile_pars(m_final.errors, m_final.covariance)

dataset_elo_scale_unc = sorted(
    zip(model_scores, final_elos_datasets, final_scales, final_elos_datasets_unc, final_scales_unc),
    key=lambda mesu: mesu[1], reverse=True
)
model_elo_unc = sorted(
    zip(model_list, final_elos_models, final_elos_models_unc),
    key=lambda meu: meu[1], reverse=True
)

rows = []
for dataset, elo, scale, elo_unc, scale_unc in dataset_elo_scale_unc:
    rows.append([dataset["name"], f"{elo:.2f}±{elo_unc:.2f}", f"{scale:.2f}±{scale_unc:.2f}"])
print(f"## Final Dataset Elo Scores")
print()
print(tabulate(rows, headers=["Dataset", "Elo score", "Scale"], tablefmt="github"))
print()

rows = []
for model, elo, unc in model_elo_unc:
    pareto_frontier: bool = True
    for model2, elo2, _ in model_elo_unc:
        if model2["file_size_gib"] < model["file_size_gib"] and elo2 > elo:
            pareto_frontier = False
            break
    rows.append([model["name"], f"{model['file_size_gib']:.2f}", f"{elo:.2f}±{unc:.2f}", "Yes" if pareto_frontier else "No"])
print(f"## Final Model Elo Scores")
print()
print(tabulate(rows, headers=["Model", "File Size [GiB]", "Elo score", "Pareto Frontier?"], tablefmt="github"))
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
    y_plot = get_winrate(x_plot, fed, fs, ms["floor"])
    plt.plot(x_plot, y_plot)
    plt.fill_between(x_plot, y_plot - wr_err_final, y_plot + wr_err_final, alpha=0.2)

    wr_data = ms["ncorrect"] / ms["ntest"]
    wr_data_unc = np.sqrt(wr_data * (1.0 - wr_data) / ms["ntest"])
    plt.errorbar(final_elos_models, wr_data, wr_data_unc, marker=".", linestyle="none")
    if ms["floor"] != 0.0:
        plt.hlines(ms["floor"], x_plot[0], x_plot[-1], colors="black", linestyles=":")
    texts = [plt.text(fem, wrd, m["name"]) for fem, wrd, m in zip(final_elos_models, wr_data, model_list)]
    plt.xlim(x_plot[0], x_plot[-1])
    plt.ylim(0, 1)
    adjust_text(texts, final_elos_models, wr_data,
        arrowprops=dict(arrowstyle="->", color="lightgray"), expand=(1.35, 2.3),
        force_explode=(0.4, 1.0), ensure_inside_axes=True, max_move=100)

    plt.title(ms["name"])
    plt.xlabel("Elo")
    plt.ylabel("Model winrate vs. benchmark")
    plt.savefig(os.path.join(DIR_OUT, f"{ms['name']}-elo-winrate.png"), dpi=240)

plot_data = []
for i, model_i in enumerate(model_list):
    plot_data.append(dict(model=model_i, elo=final_elos_models[i], elo_unc=final_elos_models_unc[i]))
for pd1 in plot_data:
    pareto_frontier = True
    for pd2 in plot_data:
        if pd2["elo"] > pd1["elo"] and pd2["model"]["file_size_gib"] < pd1["model"]["file_size_gib"]:
            pareto_frontier = False
            break
    pd1["pareto_frontier"] = pareto_frontier
plot_data = sorted(plot_data, key=lambda pd: pd["model"]["file_size_gib"])

plt.figure()
plt.errorbar(
    [pd["model"]["file_size_gib"] for pd in plot_data if pd["pareto_frontier"]],
    [pd["elo"] for pd in plot_data if pd["pareto_frontier"]],
    [pd["elo_unc"] for pd in plot_data if pd["pareto_frontier"]],
    marker=".",
)
plt.errorbar(
    [pd["model"]["file_size_gib"] for pd in plot_data if not pd["pareto_frontier"]],
    [pd["elo"] for pd in plot_data if not pd["pareto_frontier"]],
    [pd["elo_unc"] for pd in plot_data if not pd["pareto_frontier"]],
    marker=".",
    linestyle="none"
)
for pd in plot_data:
    texts = [plt.text(pd["model"]["file_size_gib"], pd["elo"], pd["model"]["name"]) for pd in plot_data]
    adjust_text(texts, [pd["model"]["file_size_gib"] for pd in plot_data], [pd["elo"] for pd in plot_data],
        arrowprops=dict(arrowstyle="->", color="lightgray"), expand=(1.35, 2.3),
        force_explode=(0.4, 1.0), ensure_inside_axes=True, max_move=100)
plt.xlabel("Model file size [GiB]")
plt.ylabel("Model elo score")
plt.savefig(os.path.join(DIR_OUT, "filesize-elo.png"), dpi=240)
