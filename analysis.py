#!/usr/bin/env python3

import os
from typing import List

import numpy as np

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
            predictions = np.load(os.path.join(dir_in_mqd, "predictions.npy"))
            labels = np.load(os.path.join(dir_in_mqd, "labels.npy"))
            assert predictions.shape[0] == labels.shape[0]
            assert predictions.shape[1] == 4

            # predictions = sample_min_p(predictions, 0.9)

            probs_label = predictions[np.arange(predictions.shape[0]), labels]
            print(f"====== {model}-{quant} {dataset} ======")

            confidence_mean = np.mean(probs_label)
            confidence_unc = np.std(probs_label) / np.sqrt(probs_label.shape[0])
            print(f"Confidence: {100*confidence_mean:.2f}+-{100*confidence_unc:.2f}%")

            greedy_correct_mean = np.mean(np.argmax(predictions, axis=1) == labels)
            greedy_correct_unc = np.sqrt(greedy_correct_mean * (1.0 - greedy_correct_mean) / predictions.shape[0])
            print(f"Greedy correct: {100*greedy_correct_mean:.2f}+-{100*greedy_correct_unc:.2f}%")

            print()
