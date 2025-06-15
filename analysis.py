#!/usr/bin/env python3

import os
from typing import Iterable

from adjustText import adjust_text
from iminuit import Minuit
from inspect import Parameter, Signature
from kafe2 import CustomFit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from tabulate import tabulate

from elo_hellm.benchmark import get_benchmark
from elo_hellm.config import Config


config = Config("config.yml")

plt.rcParams["figure.figsize"] = (14, 11)
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

DIR_OUT = os.path.join("out", "analysis")
os.makedirs(DIR_OUT, exist_ok=True)

results: dict[tuple[str, str], dict] = dict()
for model in config.models:
    for model_scores in model.datasets:
        for prompt_type in model.prompt_types:
            benchmark = get_benchmark(model_scores, prompt_type)
            results[(model.name, benchmark.database_name())] = np.array(benchmark.get_results(model.name), dtype=np.int64)


class BenchmarkModelScores:
    name: str
    ncorrect: np.ndarray
    ntest: int
    floor: float

    def __init__(self, name: str, ncorrect: Iterable[int], ntest: int, floor: float):
        self.name = name
        self.ncorrect = np.asarray(ncorrect)
        self.ntest = ntest
        self.floor = floor


ncorrect_total = 0
ntest_total = 0
model_scores = []
for dataset in config.datasets:
    for prompt_type in config.prompt_types:
        benchmark = get_benchmark(dataset, prompt_type)
        name: str = benchmark.database_name()
        print(f"## {name}")
        print()

        rows: list[list] = []
        ncorrect = np.zeros(len(config.models), dtype=np.int64)
        ntest = None
        for i, model_i in enumerate(config.models):
            labels, pred = results[(model_i.name, name)]
            if ntest is None:
                ntest = labels.shape[0]
            else:
                assert labels.shape[0] == ntest
            ncorrect[i] = np.sum(pred == labels)
            rows.append([model_i.name, model_i.file_size / 1024 ** 3, f"{ncorrect[i]}/{ntest}", ncorrect[i]/ntest])
        rows = sorted(rows, key=lambda r: r[1], reverse=True)
        for r1 in rows:
            pareto_frontier: bool = True
            for r2 in rows:
                if r2[1] < r1[1] and r2[3] > r1[3]:
                    pareto_frontier = False
                    break
            r1.append(pareto_frontier)
        ncorrect_total += np.sum(ncorrect)
        ntest_total += ncorrect.shape[0] * ntest

        plt.figure()
        file_sizes_gib = np.array([r[1] for r in rows if r[4]])
        win_rates = np.array([r[3] for r in rows if r[4]])
        win_rates_unc = np.sqrt(win_rates * (1.0 - win_rates) / ntest)
        plt.errorbar(file_sizes_gib, win_rates, win_rates_unc, marker=".")
        file_sizes_gib = np.array([r[1] for r in rows if not r[4]])
        win_rates = np.array([r[3] for r in rows if not r[4]])
        win_rates_unc = np.sqrt(win_rates * (1.0 - win_rates) / ntest)
        plt.errorbar(file_sizes_gib, win_rates, win_rates_unc, marker=".", linestyle="none")
        if benchmark.score_rng != 0.0:
            plt.hlines(benchmark.score_rng, 0, 50, colors="black", linestyles=":")
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

        model_scores.append(BenchmarkModelScores(name, ncorrect, ntest, benchmark.score_rng))
print(f"Total correct answers: {ncorrect_total}/{ntest_total}")
print()


def decompile_pars(pars, cov_mat=None):
    pars = np.asarray(pars)
    assert pars.ndim == 1
    assert pars.shape[0] == 2 * len(model_scores) + len(config.models) - 2
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
        wr = get_winrate(elos_models, elos_datasets[i], scales[i], ms_i.floor)
        err = np.sqrt(wr * (1.0 - wr) / ms_i.ntest + wr_err ** 2)
        residuals = wr - ms_i.ncorrect / ms_i.ntest
        nll += np.sum(np.square(residuals / err))
        # nll -= 2.0 * np.sum(binom.logpmf(k=ms_i["ncorrect"], n=ms_i["ntest"], p=wr))
    if scales[-1] < 0:
        nll += 1e6 * scales[-1] ** 2
    return nll


starting_scales = 400 * np.ones(len(model_scores) - 1)
starting_elos = 1500 * np.ones(len(model_scores) + len(config.models) - 1)
starting_pars = np.concatenate([starting_scales, starting_elos])
print(f"Pre-fit cost: {get_nll(starting_pars, 0.0):.2f}")

ndf = -len(starting_pars)
for ms in model_scores:
    ndf += ms.ncorrect.shape[0]


def get_par_name(s: str):
    return s.replace("-", "_").replace(".", "")


def get_fit(wr_err: float) -> CustomFit:
    def func(*pars):
        return get_nll(pars, wr_err)
    pars: list[Parameter] = []
    for ms in model_scores[:-1]:
        pars.append(Parameter(name=f"scale_{get_par_name(ms.name)}", kind=Parameter.POSITIONAL_OR_KEYWORD, default=400))
    for ms in model_scores:
        pars.append(Parameter(name=f"elo_{get_par_name(ms.name)}", kind=Parameter.POSITIONAL_OR_KEYWORD, default=1500))
    for model in config.models[:-1]:
        pars.append(Parameter(name=f"elo_{get_par_name(model.name)}", kind=Parameter.POSITIONAL_OR_KEYWORD, default=1500))
    func.__signature__ = Signature(pars)

    fit = CustomFit(func)
    for ms in model_scores[:-1]:
        fit.limit_parameter(f"scale_{ms.name}", lower=1e-6)
    fit.do_fit()
    return fit


tol = 1e-4
wr_err_low = 0.00
wr_err_high = 0.01
fit_low = get_fit(wr_err_low)
if fit_low.cost_function_value <= ndf + tol:
    wr_err_final = wr_err_low
    fit_final = fit_low
else:
    fit_high = get_fit(wr_err_high)
    i = 0
    while fit_high.cost_function_value > ndf:
        wr_err_low = wr_err_high
        wr_err_high += 0.01
        fit_low = fit_high
        fit_high = get_fit(wr_err_high)
        i += 1
        assert i <= 10
    wr_err_test = (wr_err_low + wr_err_high) / 2
    fit_test = get_fit(wr_err_test)

    i = 0
    while abs(fit_test.cost_function_value - ndf) > tol and i < 10:
        if fit_test.cost_function_value > ndf:
            wr_err_low = wr_err_test
            fit_low = fit_test
        else:
            wr_err_high = wr_err_test
            fit_high = fit_test
        wr_err_test = (wr_err_low + wr_err_high) / 2
        fit_test = get_fit(wr_err_test)
        i += 1
    wr_err_final = wr_err_test
    fit_final = fit_test

fit_final.report(asymmetric_parameter_errors=True)
print(f"wr_err_final={100*wr_err_final:.2f}%")

nll_sat = fit_final.cost_function_value
# for ms in model_scores:
    # nll_sat += 2.0 * np.sum(binom.logpmf(k=ms.ncorrect, n=ms.ntest, p=ms.ncorrect/ms.ntest))
print(f"chi2 / NDF: {nll_sat:.2f}/{ndf} = {nll_sat/ndf if ndf > 0 else np.nan:.2f}")
chi2_prob = 1.0 - chi2.cdf(nll_sat, ndf)
print(f"chi2 probability: {100*chi2_prob:.2f}%")
print()

final_scales, final_elos_datasets, final_elos_models = decompile_pars(fit_final.paramter_values)
final_scales_unc, final_elos_datasets_unc, final_elos_models_unc = decompile_pars(
    fit_final.parameter_errors, fit_final.parameter_cov_mat)

ms_elo_scale_unc = sorted(
    zip(model_scores, final_elos_datasets, final_scales, final_elos_datasets_unc, final_scales_unc),
    key=lambda mesu: mesu[1], reverse=True
)
model_elo_unc = sorted(
    zip(config.models, final_elos_models, final_elos_models_unc),
    key=lambda meu: meu[1], reverse=True
)

rows = []
for ms, elo, scale, elo_unc, scale_unc in ms_elo_scale_unc:
    rows.append([ms.name, f"{elo:.2f}±{elo_unc:.2f}", f"{scale:.2f}±{scale_unc:.2f}"])
print("## Final Dataset Elo Scores")
print()
print(tabulate(rows, headers=["Dataset", "Elo score", "Scale"], tablefmt="github"))
print()

rows = []
for model, elo, unc in model_elo_unc:
    pareto_frontier: bool = True
    for model2, elo2, _ in model_elo_unc:
        if model2.file_size < model.file_size and elo2 > elo:
            pareto_frontier = False
            break
    rows.append([model.name, f"{model.file_size/1024**3:.2f}", f"{elo:.2f}±{unc:.2f}", "Yes" if pareto_frontier else "No"])
print("## Final Model Elo Scores")
print()
print(tabulate(rows, headers=["Model", "File Size [GiB]", "Elo score", "Pareto Frontier?"], tablefmt="github"))
print()

num_within_sigma = [0, 0, 0]
num_total = 0
for i, ms_i in enumerate(model_scores):
    wr_data = ms_i.ncorrect / ms_i.ntest
    wr_data_unc = np.sqrt(wr_data * (1.0 - wr_data) / wr_data.shape[0])
    wr_elo = get_winrate(final_elos_models, final_elos_datasets[i], final_scales[i], ms_i.floor)
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
            print(f"dataset={ms_i.name} model={config.models[j].name}: wr_elo={100*wr_elo[j]:.4f}% wr_data={100*wr_data[j]:.4f}%")
print(f"Within 1 sigma: {100*num_within_sigma[0]/num_total:.2f}%")
print(f"Within 2 sigma: {100*num_within_sigma[1]/num_total:.2f}%")
print(f"Within 3 sigma: {100*num_within_sigma[2]/num_total:.2f}%")

for fed, fs, ms in zip(final_elos_datasets, final_scales, model_scores):
    plt.figure()

    x_plot = np.linspace(1000, 2000, 201)
    y_plot = get_winrate(x_plot, fed, fs, ms.floor)
    plt.plot(x_plot, y_plot)
    plt.fill_between(x_plot, y_plot - wr_err_final, y_plot + wr_err_final, alpha=0.2)

    wr_data = ms.ncorrect / ms.ntest
    wr_data_unc = np.sqrt(wr_data * (1.0 - wr_data) / ms.ntest)
    plt.errorbar(final_elos_models, wr_data, wr_data_unc, marker=".", linestyle="none")
    if ms.floor != 0.0:
        plt.hlines(ms.floor, x_plot[0], x_plot[-1], colors="black", linestyles=":")
    texts = [plt.text(fem, wrd, m.name) for fem, wrd, m in zip(final_elos_models, wr_data, config.models)]
    plt.xlim(x_plot[0], x_plot[-1])
    plt.ylim(0, 1)
    adjust_text(texts, final_elos_models, wr_data,
        arrowprops=dict(arrowstyle="->", color="lightgray"), expand=(1.35, 2.3),
        force_explode=(0.4, 1.0), ensure_inside_axes=True, max_move=100)

    plt.title(ms.name)
    plt.xlabel("Elo")
    plt.ylabel("Model winrate vs. benchmark")
    plt.savefig(os.path.join(DIR_OUT, f"{ms.name}-elo-winrate.png"), dpi=240)

plot_data = []
for i, model_i in enumerate(config.models):
    plot_data.append(dict(model=model_i, elo=final_elos_models[i], elo_unc=final_elos_models_unc[i]))
for pd1 in plot_data:
    pareto_frontier = True
    for pd2 in plot_data:
        if pd2["elo"] > pd1["elo"] and pd2["model"].file_size < pd1["model"].file_size:
            pareto_frontier = False
            break
    pd1["pareto_frontier"] = pareto_frontier
plot_data = sorted(plot_data, key=lambda pd: pd["model"].file_size)

plt.figure()
plt.errorbar(
    [pd["model"].file_size/1024**3 for pd in plot_data if pd["pareto_frontier"]],
    [pd["elo"] for pd in plot_data if pd["pareto_frontier"]],
    [pd["elo_unc"] for pd in plot_data if pd["pareto_frontier"]],
    marker=".",
)
plt.errorbar(
    [pd["model"].file_size/1024**3 for pd in plot_data if not pd["pareto_frontier"]],
    [pd["elo"] for pd in plot_data if not pd["pareto_frontier"]],
    [pd["elo_unc"] for pd in plot_data if not pd["pareto_frontier"]],
    marker=".",
    linestyle="none"
)
for pd in plot_data:
    texts = [plt.text(pd["model"].file_size/1024**3, pd["elo"], pd["model"].name) for pd in plot_data]
    adjust_text(texts, [pd["model"].file_size/1024**3 for pd in plot_data], [pd["elo"] for pd in plot_data],
        arrowprops=dict(arrowstyle="->", color="lightgray"), expand=(1.35, 2.3),
        force_explode=(0.4, 1.0), ensure_inside_axes=True, max_move=100)
plt.xlabel("Model file size [GiB]")
plt.ylabel("Model elo score")
plt.savefig(os.path.join(DIR_OUT, "filesize-elo.png"), dpi=240)
