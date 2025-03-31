#!/usr/bin/env python3

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml


with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

DIR_IN = os.path.join("out", "bench")
models: List[str] = [m["name"] for m in config["models"]]


for dataset in config["datasets"]:
    for cot in [False, True]:
        if cot and config["skip_cot"]:
            continue
        name: str = f"{dataset}-cot{1 if cot else 0}"

        plt.figure()

        for model in models:
            file_sizes = []
            scores = []
            for quant in config["quantizations"]:
                dir_in_mq: str = os.path.join(DIR_IN, model, quant)
                dir_in_mqd: str = os.path.join(dir_in_mq, dataset)

                with open(os.path.join(dir_in_mq, "model.yml")) as f:
                    props: dict = yaml.safe_load(f)
                    file_size: float = props["file_size"] / 1024 ** 3
                labels = np.load(os.path.join(dir_in_mqd, "labels.npy"))
                pred = np.load(os.path.join(dir_in_mqd, f"pred-cot{1 if cot else 0}.npy"))
                if "mmlu" in dataset:
                    pred = np.argmax(pred, axis=1)

                score = np.mean(pred == labels)

                if quant == "f16":
                    file_size_f16 = file_size
                    score_f16 = score
                file_sizes.append(file_size)
                scores.append(score)
            file_sizes = np.array(file_sizes)
            scores = np.array(scores)
            bpw = 16 * file_sizes / file_size_f16
            plt.scatter(bpw, 100*scores, label=model)

        plt.xlabel("BPW")
        plt.ylabel("Benchmark % correct")
        plt.title(name)
        plt.legend()
        plt.savefig(f"{name}.png", dpi=240)
