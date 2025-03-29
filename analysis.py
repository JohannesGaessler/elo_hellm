#!/usr/bin/env python3

import json
from time import sleep
from typing import List

import datasets
import numpy as np

results: List[dict] = []

with open("results.jsonl", "r") as f:
    for line in f.readlines():
        results.append(json.loads(line))

mmlu = datasets.load_dataset("cais/mmlu", "all")
labels = np.array([ex["answer"] for ex in mmlu["test"]])

assert len(results) == labels.shape[0]

predictions = np.zeros(labels.shape + (4,))
for i in range(predictions.shape[0]):
    for j in range(predictions.shape[1]):
        token_info: dict = results[i]["completion_probabilities"][0]["top_probs"][j]
        if token_info["token"] == "a":
            predictions[i, 0] = token_info["prob"]
        elif token_info["token"] == "b":
            predictions[i, 1] = token_info["prob"]
        elif token_info["token"] == "c":
            predictions[i, 2] = token_info["prob"]
        elif token_info["token"] == "d":
            predictions[i, 3] = token_info["prob"]
        else:
            assert False

probs_label = predictions[np.arange(predictions.shape[0]), labels]
print(np.mean(probs_label))
print(np.mean(np.argmax(predictions, axis=1) == labels))
