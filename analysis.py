#!/usr/bin/env python3

import os
from typing import List

from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import yaml


with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

DIR_IN = os.path.join("out", "bench")
models: List[str] = [m["name"] for m in config["models"]]


def sample_min_p(predictions: np.ndarray, min_p: float) -> np.ndarray:
    max_p = np.max(predictions, axis=1)
    filter_array = predictions >= np.reshape(max_p * min_p, max_p.shape + (1,))
    predictions = predictions * filter_array
    sum_p = np.sum(predictions, axis=1)
    predictions /= np.reshape(sum_p, max_p.shape + (1,))
    return predictions


for model in models:
    dir_in_m: str = os.path.join(DIR_IN, model)
    quantizations: List[str] = sorted(os.listdir(dir_in_m))
    for quant in quantizations:
        dir_in_mq: str = os.path.join(dir_in_m, quant)
        print(f"====== {model}-{quant} ======")
        for dataset in config["datasets"]:
            dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
            if not os.path.exists(dir_in_mqd):
                continue
            labels = np.load(os.path.join(dir_in_mqd, "labels.npy"))

            for cot in [False, True]:
                path_pred: str = os.path.join(dir_in_mqd, f"pred-cot{1 if cot else 0}.npy")
                if not os.path.exists(path_pred):
                    continue
                predictions = np.load(path_pred)
                assert predictions.shape[0] == labels.shape[0]

                # predictions = sample_min_p(predictions, 0.9)

                if predictions.ndim == 1:
                    greedy_correct_mean = np.mean(predictions == labels)
                    greedy_correct_unc = np.sqrt(greedy_correct_mean * (1.0 - greedy_correct_mean) / predictions.shape[0])
                    print(f"{dataset}, cot={cot}, greedy correct: {100*greedy_correct_mean:.2f}+-{100*greedy_correct_unc:.2f}%")
                elif predictions.ndim == 2:
                    probs_label = predictions[np.arange(predictions.shape[0]), labels]
                    confidence_mean = np.mean(probs_label)
                    confidence_unc = np.std(probs_label) / np.sqrt(probs_label.shape[0])
                    print(f"{dataset}, cot={cot}, confidence: {100*confidence_mean:.2f}+-{100*confidence_unc:.2f}%")

                    greedy_correct_mean = np.mean(np.argmax(predictions, axis=1) == labels)
                    greedy_correct_unc = np.sqrt(greedy_correct_mean * (1.0 - greedy_correct_mean) / predictions.shape[0])
                    print(f"{dataset}, cot={cot}, greedy correct: {100*greedy_correct_mean:.2f}+-{100*greedy_correct_unc:.2f}%")
                else:
                    assert False

        print()


model_list = []
for model in models:
    dir_in_m: str = os.path.join(DIR_IN, model)
    for quant in quantizations:
        dir_in_mq: str = os.path.join(dir_in_m, quant)
        with open(os.path.join(dir_in_mq, "model.yml")) as f:
            props: dict = yaml.safe_load(f)
            file_size_gib: float = props["file_size"] / 1024 ** 3

        labels = np.zeros((0,))
        for dataset in config["datasets"]:
            dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
            if not os.path.exists(dir_in_mqd):
                continue
            labels = np.concatenate([labels, np.load(os.path.join(dir_in_mqd, "labels.npy"))])
        for cot in [False, True]:
            if cot and config["skip_cot"]:
                continue
            name: str = f"{model}-{quant}-cot{1 if cot else 0}"
            pred = np.zeros((0,))
            for dataset in config["datasets"]:
                dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
                if not os.path.exists(dir_in_mqd):
                    continue
                pred_part = np.load(os.path.join(dir_in_mqd, f"pred-cot{1 if cot else 0}.npy"))
                if "mmlu" in dataset:
                    pred = np.concatenate([pred, np.argmax(pred_part, axis=1)])
                elif "gsm8k" in dataset:
                    pred = np.concatenate([pred, pred_part])
                else:
                    assert False
            model_list.append(dict(name=name, labels=labels, pred=pred, file_size_gib=file_size_gib))


def get_winrate(elo_self: float, elo_other: float) -> float:
    return 1 / (1 + 10 ** ((elo_other - elo_self) / 400))


def get_num_wins(pred, pred_other, labels):
    num_draws: int = np.sum((pred == labels) == (pred_other == labels))
    num_wins: int = num_draws // 2 + np.sum(np.logical_and(pred == labels, pred_other != labels))
    return num_wins


def get_nll(elos: np.ndarray) -> float:
    assert elos.ndim == 1
    assert elos.shape[0] == len(model_list) - 1
    elos = np.concatenate([elos, [len(model_list)*1500 - np.sum(elos)]])
    assert elos.shape[0] == len(model_list)

    nll = 0.0
    for i in range(elos.shape[0]):
        for j in range(i):
            labels = model_list[i]["labels"]
            assert np.all(labels == model_list[j]["labels"])

            pred_i = model_list[i]["pred"]
            pred_j = model_list[j]["pred"]

            nll -= binom.logpmf(k=get_num_wins(pred_i, pred_j, labels), n=labels.shape[0], p=get_winrate(elos[i], elos[j]))
    return nll


starting_elos = 1500 * np.ones(len(model_list) - 1)
print(f"Pre-fit cost: {get_nll(starting_elos):.2f}")

m = Minuit(get_nll, starting_elos)
m.errordef = 0.5
m.migrad()
m.hesse()

print(f"Post-fit cost: {m.fval:.2f}")

final_elos = np.array(m.values)
final_elos = np.concatenate([final_elos, [1500 * len(model_list) - np.sum(final_elos)]])

final_elos_unc = np.array(m.errors)
final_elos_unc = np.concatenate([final_elos_unc, [np.sqrt(np.sum(np.square(final_elos_unc)))]])

model_elo_unc = sorted(zip(model_list, final_elos, final_elos_unc), key=lambda meu: meu[1], reverse=True)

for model, elo, unc in model_elo_unc:
    print(f"{model['name']}: {elo:.2f}+-{unc:.2f} @ {model['file_size_gib']:.2f} GiB")

print()

max_wr_diff: float = 0.0
for i in range(len(model_elo_unc)):
    meui = model_elo_unc[i]
    for j in range(i):
        meuj = model_elo_unc[j]
        wr_elo: float = get_winrate(meui[1], meuj[1])
        num_wins: int = get_num_wins(meui[0]["pred"], meuj[0]["pred"], meui[0]["labels"])
        wr_data: float = num_wins / meui[0]["labels"].shape[0]
        wr_diff: float = np.abs(wr_elo - wr_data)
        if wr_diff > max_wr_diff:
            max_wr_diff_info = f"{meui[0]['name']} <-> {meuj[0]['name']}: wr_elo={100*wr_elo:.2f}% wr_data={100*wr_data:.2f}%"
            max_wr_diff = wr_diff

print(max_wr_diff_info)

for model in models:
    for cot in [False, True]:
        file_sizes_gib: List[float] = []
        elos: List[float] = []
        elos_unc: List[float] = []
        for quant in quantizations:
            name: str = f"{model}-{quant}-cot{1 if cot else 0}"

            for meu in model_elo_unc:
                if meu[0]["name"] != name:
                    continue
                file_sizes_gib.append(meu[0]["file_size_gib"])
                elos.append(meu[1])
                elos_unc.append(meu[2])
        if not file_sizes_gib:
            continue
        plt.errorbar(file_sizes_gib, elos, elos_unc, marker="o", linestyle="none", label=f"{model}-cot{1 if cot else 0}")

plt.title(", ".join(config["datasets"]))
plt.xlabel("Model file size [GiB]")
plt.ylabel("Elo score")
plt.legend()

plt.savefig("elo_vs_size.png", dpi=240)
