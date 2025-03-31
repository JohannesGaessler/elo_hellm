#!/usr/bin/env python3

import os
from typing import List

from iminuit import Minuit
import numpy as np
from scipy.stats import binom

DIR_IN = os.path.join("out", "bench")
DATASETS: List[str] = ["mmlu_test"]

models: List[str] = sorted(os.listdir(DIR_IN))


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
        for dataset in DATASETS:
            dir_in_mqd: str = os.path.join(dir_in_mq, dataset)
            labels = np.load(os.path.join(dir_in_mqd, "labels.npy"))

            for cot in [False, True]:
                predictions = np.load(os.path.join(dir_in_mqd, f"pred-cot{1 if cot else 0}.npy"))
                assert predictions.shape[0] == labels.shape[0]
                assert predictions.shape[1] == 4

                # predictions = sample_min_p(predictions, 0.9)

                probs_label = predictions[np.arange(predictions.shape[0]), labels]
                print(f"====== {model}-{quant} {dataset} cot={cot} ======")

                confidence_mean = np.mean(probs_label)
                confidence_unc = np.std(probs_label) / np.sqrt(probs_label.shape[0])
                print(f"Confidence: {100*confidence_mean:.2f}+-{100*confidence_unc:.2f}%")

                greedy_correct_mean = np.mean(np.argmax(predictions, axis=1) == labels)
                greedy_correct_unc = np.sqrt(greedy_correct_mean * (1.0 - greedy_correct_mean) / predictions.shape[0])
                print(f"Greedy correct: {100*greedy_correct_mean:.2f}+-{100*greedy_correct_unc:.2f}%")

                print()


model_list = []
for model in models:
    dir_in_m: str = os.path.join(DIR_IN, model)
    for quant in quantizations:
        dir_in_mq: str = os.path.join(dir_in_m, quant)
        dir_in_mqd: str = os.path.join(dir_in_mq, "mmlu_test")
        labels = np.load(os.path.join(dir_in_mqd, "labels.npy"))
        for cot in [False, True]:
            name: str = f"{model}-{quant}-cot{1 if cot else 0}"
            pred = np.load(os.path.join(dir_in_mqd, f"pred-cot{1 if cot else 0}.npy"))
            model_list.append(dict(name=name, labels=labels, pred=pred))


def get_winrate(elo_self: float, elo_other: float) -> float:
    return 1 / (1 + 10 ** ((elo_other - elo_self) / 400))


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

            pred_i = np.argmax(model_list[i]["pred"], axis=1)
            pred_j = np.argmax(model_list[j]["pred"], axis=1)

            num_wins: int = np.sum(np.logical_and(pred_i == labels, pred_j != labels))
            num_draws: int = np.sum((pred_i == labels) == (pred_j == labels))

            nll -= binom.logpmf(k=num_wins + num_draws // 2, n=labels.shape[0], p=get_winrate(elos[i], elos[j]))
    return nll


starting_elos = 1500 * np.ones(len(model_list) - 1)
print(get_nll(starting_elos))

m = Minuit(get_nll, starting_elos)
m.errordef = 0.5
m.migrad()
m.hesse()

final_elos = np.array(m.values)
final_elos = np.concatenate([final_elos, [1500 * len(model_list) - np.sum(final_elos)]])

final_elos_unc = np.array(m.errors)
final_elos_unc = np.concatenate([final_elos_unc, [np.sqrt(np.sum(np.square(final_elos_unc)))]])

for model, elo, unc in zip(model_list, final_elos, final_elos_unc):
    print(f"{model['name']}: {elo:.2f}+-{unc:.2f}")
