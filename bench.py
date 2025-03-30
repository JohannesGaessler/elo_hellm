#!/usr/bin/env python3

import json
import os
from time import sleep
from typing import Dict, List

import datasets
import numpy as np
import requests
import subprocess
from tqdm.contrib.concurrent import process_map
import yaml

with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

PARALLEL = 8
CTX_SIZE = 4096
PORT = 1337

SERVER_ADDRESS = f"http://localhost:{PORT}"

mmlu = datasets.load_dataset("cais/mmlu", "all")

TEMPLATE_PROMPT_MMLU = """{question}

Which of the following answers is correct?
{choices_block}"""

TEMPLATE_RESPONSE_MMLU = """The correct answer is ("""

LETTERS = ["a", "b", "c", "d"]
GRAMMAR = "root ::= [abcd]"
# GRAMMAR = "root ::= [abc]"
# GRAMMAR = "root ::= [ABC]"
# GRAMMAR = "root ::= \"a\""


def process_mmlu(example: dict) -> List[float]:
    question: str = example["question"]
    choices: List[str] = example["choices"]
    answer: int = example["answer"]

    assert type(question) is str
    assert type(choices) is list
    assert type(answer) is int

    choices = [f"({LETTERS[i]}): {choice_i}" for i, choice_i in enumerate(choices)]
    choices_block: str = "\n".join(choices)

    messages: List[str] = [
        dict(role="user", content=TEMPLATE_PROMPT_MMLU.format(question=question, choices_block=choices_block)),
    ]
    response = requests.post(
        f"{SERVER_ADDRESS}/apply-template",
        json=dict(messages=messages)
    )
    if response.status_code != 200:
        server_process.terminate()
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    prompt: str = json.loads(response.text)["prompt"]
    prompt += TEMPLATE_RESPONSE_MMLU

    response = requests.post(
        f"{SERVER_ADDRESS}/completion",
        json=dict(
            prompt=prompt,
            grammar=GRAMMAR,
            n_predict=1,
            n_probs=4,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            post_sampling_probs=True,
        )
    )
    if response.status_code != 200:
        server_process.terminate()
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")

    response_dict: dict = json.loads(response.text)
    predictions = [-1.0] * 4
    for i in range(4):
        token_info: dict = response_dict["completion_probabilities"][0]["top_probs"][i]
        if token_info["token"] == "a":
            predictions[0] = token_info["prob"]
        elif token_info["token"] == "b":
            predictions[1] = token_info["prob"]
        elif token_info["token"] == "c":
            predictions[2] = token_info["prob"]
        elif token_info["token"] == "d":
            predictions[3] = token_info["prob"]
        else:
            assert False
    return predictions


for model in config["models"]:
    name: str = model["name"]
    quantizations: List[str] = model.get("quantizations", config["quantizations"])
    filepath_template: str = model.get("filepath", config["filepath"])

    for quant in quantizations:
        filepath: str = filepath_template.format(name=name, quantization=quant)
        dir_out: str = os.path.join("out", "bench", name, quant)
        os.makedirs(dir_out, exist_ok=True)

        popen_args: List[str] = [
            config["server_binary"],
            "--n-gpu-layers", "999",
            "--parallel", str(PARALLEL),
            "--ctx-size", str(PARALLEL * CTX_SIZE),
            "--model", filepath,
            "--port", str(PORT),
        ]
        popen_env: Dict[str, str] = dict(
            CUDA_VISIBLE_DEVICES="0",
        )
        with open(os.path.join(dir_out, "srvr_out.log"), "w") as fstdout, open(os.path.join(dir_out, "srvr_err.log"), "w") as fstderr:
            server_process = subprocess.Popen(popen_args, env=popen_env, stdout=fstdout, stderr=fstderr)
            while True:
                try:
                    sleep(1.0)
                    response = requests.get(f"{SERVER_ADDRESS}/health")
                    if response.status_code == 200:
                        break
                except requests.ConnectionError:
                    pass

            os.makedirs(os.path.join(dir_out, "mmlu"), exist_ok=True)
            predictions = process_map(process_mmlu, list(mmlu["test"]), max_workers=PARALLEL, chunksize=10)
            np.save(os.path.join(dir_out, "mmlu", "predictions.npy"), predictions)

            server_process.terminate()
