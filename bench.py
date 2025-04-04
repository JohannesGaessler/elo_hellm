#!/usr/bin/env python3

from copy import deepcopy
import json
from multiprocessing import Pool
import os
from time import sleep, time
import threading
from typing import Dict, List

import datasets
import numpy as np
import requests
import subprocess
import yaml

with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

model_list: List[dict] = []
for model in config["models"]:
    name: str = model["name"]
    quantizations: List[str] = model.get("quantizations", config["quantizations"])
    path_model_template: str = model.get("path_model", config["path_model"])
    path_imatrix_template: str = model.get("path_imatrix", config["path_imatrix"])
    skip_cot: bool = model.get("skip_cot", config["skip_cot"])

    for quant in quantizations:
        path_model: str = path_model_template.format(name=name, quantization=quant)
        dir_out: str = os.path.join("out", "bench", name, quant)
        os.makedirs(dir_out, exist_ok=True)

        env: Dict[str, str] = dict(
            CUDA_VISIBLE_DEVICES="0",
        )

        if not os.path.exists(path_model):
            assert quant != "f16"
            path_model_f16: str = path_model_template.format(name=name, quantization="f16")
            path_imatrix: str = path_imatrix_template.format(name=name)
            assert os.path.exists(path_model_f16)
            print(f"Model {path_model} does not exist, quantizing from f16...")
            with open(os.path.join(dir_out, f"quantize.out"), "w") as fout, open(os.path.join(dir_out, f"quantize.err"), "w") as ferr:
                returncode = subprocess.run(
                    [config["path_quantize"], "--imatrix", path_imatrix, path_model_f16, path_model, quant, "8"],
                    stdout=fout,
                    stderr=ferr,
                    env=env,
                ).returncode
                assert returncode == 0

        file_size: int = os.path.getsize(path_model)

        model_list.append(dict(name=name, quantization=quant, path_model=path_model, file_size=file_size, skip_cot=skip_cot))

model_list = sorted(model_list, key=lambda m: m["file_size"], reverse=True)

PARALLEL = 8
CTX_SIZE = 4096

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

TEMPLATE_SERVER_ADDRESS = "http://localhost:{port}"

TEMPLATE_PROMPT_MULTIPLE_CHOICE = """{question}

Which of the following answers is correct?
{choices_block}"""

TEMPLATE_RESPONSE_MULTIPLE_CHOICE = """The correct answer is ("""

TEMPLATE_PROMPT_MULTIPLE_CHOICE_COT_1 = """{question}

Which of the following answers is correct?
{choices_block}

Think about the problem step-by-step."""

TEMPLATE_PROMPT_COT_2 = """Please enter your final answer."""

TEMPLATE_RESPONSE_MULTIPLE_CHOICE_COT_2 = """My final answer is ("""


def get_choices_block(choices: List[str]) -> str:
    choices = [f"({letter}): {choice}" for letter, choice in zip(LETTERS, choices)]
    return "\n".join(choices)


def get_grammar_multiple_choice(num_choices: int) -> str:
    return f"root ::= [{''.join(LETTERS[:num_choices])}]"


TEMPLATE_RESPONSE_MATH = """The correct answer is """
TEMPLATE_RESPONSE_MATH_COT_2 = """My final answer is """
TEMPLATE_GRAMMAR_MATH = "root ::= [0-9]+.*"

dataset_list = []

if "mmlu_test" in config["datasets"]:
    print("Loading MMLU...")
    mmlu = datasets.load_dataset("cais/mmlu", "all")
    # dataset_list.append(dict(name="mmlu_val", type="multiple_choice", data=mmlu["validation"]))
    dataset_list.append(dict(name="mmlu_test", type="multiple_choice", data=mmlu["test"]))

if "gsm8k_test" in config["datasets"]:
    print("Loading GSM8K...")
    gsm8k = datasets.load_dataset("openai/gsm8k", "main")
    # dataset_list.append(dict(name="gsm8k_train", type="math", data=gsm8k["train"]))
    dataset_list.append(dict(name="gsm8k_test", type="math", data=gsm8k["test"]))


def process_multiple_choice(example: dict) -> List[float]:
    server_address = example["server_address"]
    cot: bool = example["cot"]
    question: str = example["question"]
    choices: List[str] = example["choices"]

    assert type(question) is str
    assert type(choices) is list

    num_choices: int = len(choices)
    choices_block: str = get_choices_block(choices)

    if cot:
        messages: List[str] = [
            dict(role="user", content=TEMPLATE_PROMPT_MULTIPLE_CHOICE_COT_1.format(question=question, choices_block=choices_block)),
        ]
        response = requests.post(
            f"{server_address}/apply-template",
            json=dict(messages=messages)
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        prompt: str = json.loads(response.text)["prompt"]

        response = requests.post(
            f"{server_address}/completion",
            json=dict(
                prompt=prompt,
                n_predict=CTX_SIZE // 2,
                temperature=0.0,
            )
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        response_dict: dict = json.loads(response.text)
        messages.append(dict(role="assistant", content=response_dict["content"]))
        messages.append(dict(role="user", content=TEMPLATE_PROMPT_COT_2))

        response = requests.post(
            f"{server_address}/apply-template",
            json=dict(messages=messages)
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        prompt: str = json.loads(response.text)["prompt"]
        prompt += TEMPLATE_RESPONSE_MULTIPLE_CHOICE_COT_2
    else:
        messages: List[str] = [
            dict(role="user", content=TEMPLATE_PROMPT_MULTIPLE_CHOICE.format(question=question, choices_block=choices_block)),
        ]
        response = requests.post(
            f"{server_address}/apply-template",
            json=dict(messages=messages)
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        prompt: str = json.loads(response.text)["prompt"]
        prompt += TEMPLATE_RESPONSE_MULTIPLE_CHOICE

    response = requests.post(
        f"{server_address}/completion",
        json=dict(
            prompt=prompt,
            grammar=get_grammar_multiple_choice(num_choices),
            n_predict=1,
            n_probs=num_choices,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            post_sampling_probs=True,
        )
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")

    response_dict: dict = json.loads(response.text)
    predictions = [-1.0] * num_choices
    for i in range(num_choices):
        token_info: dict = response_dict["completion_probabilities"][0]["top_probs"][i]
        index_token: int = LETTERS.index(token_info["token"])
        predictions[index_token] = token_info["prob"]
    return predictions


def process_math(example: dict) -> List[float]:
    server_address = example["server_address"]
    cot: bool = example["cot"]
    question: str = example["question"]

    assert type(question) is str

    messages: List[str] = [
        dict(role="user", content=question),
    ]
    response = requests.post(
        f"{server_address}/apply-template",
        json=dict(messages=messages)
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    prompt: str = json.loads(response.text)["prompt"]

    if cot:
        response = requests.post(
            f"{server_address}/completion",
            json=dict(
                prompt=prompt,
                n_predict=CTX_SIZE // 2,
                temperature=0.0,
            )
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        response_dict: dict = json.loads(response.text)
        messages.append(dict(role="assistant", content=response_dict["content"]))
        messages.append(dict(role="user", content=TEMPLATE_PROMPT_COT_2))

        response = requests.post(
            f"{server_address}/apply-template",
            json=dict(messages=messages)
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        prompt: str = json.loads(response.text)["prompt"]
        prompt += TEMPLATE_RESPONSE_MATH_COT_2
    else:
        prompt += TEMPLATE_RESPONSE_MATH

    response = requests.post(
        f"{server_address}/completion",
        json=dict(
            prompt=prompt,
            grammar=TEMPLATE_GRAMMAR_MATH,
            n_predict=10,
            temperature=0.0,
        )
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")

    response_dict: dict = json.loads(response.text)
    content: str = response_dict["content"]
    try:
        index_period: int = content.find(".")
    except ValueError:
        index_period: int = len(content)
    try:
        index_space: int = content.find(".")
    except ValueError:
        index_space: int = len(content)
    try:
        prediction: int = int(content[:min(index_period, index_space)])
    except ValueError:
        prediction: int = 123456789
    return prediction


def process_model(model, gpu_id: int):
    port = 1337 + gpu_id
    server_address = TEMPLATE_SERVER_ADDRESS.format(port=port)

    name: str = model["name"]
    quant: str = model["quantization"]
    path_model: str = model["path_model"]
    skip_cot: bool = model["skip_cot"]

    dir_out: str = os.path.join("out", "bench", name, quant)
    os.makedirs(dir_out, exist_ok=True)

    all_targets_existent: bool = True
    for ds in dataset_list:
        ds_name = ds["name"]
        for cot in [False, True]:
            if skip_cot and cot:
                continue

            target: str = os.path.join(dir_out, ds_name, f"pred-cot{1 if cot else 0}.npy")
            if not os.path.exists(target):
                all_targets_existent = False
                break
    if all_targets_existent:
        return

    with open(os.path.join(dir_out, "model.yml"), "w") as f:
        yaml.safe_dump(model, f)

    env: Dict[str, str] = dict(
        CUDA_VISIBLE_DEVICES=str(gpu_id),
    )

    popen_args: List[str] = [
        config["path_server"],
        "--flash-attn",
        "--n-gpu-layers", "999",
        "--parallel", str(PARALLEL),
        "--ctx-size", str(PARALLEL * CTX_SIZE),
        "--model", path_model,
        "--port", str(port),
    ]
    with open(os.path.join(dir_out, f"server.out"), "w") as fout, open(os.path.join(dir_out, f"server.err"), "w") as ferr:
        server_process = subprocess.Popen(popen_args, env=env, stdout=fout, stderr=ferr)
        while True:
            try:
                sleep(1.0)
                response = requests.get(f"{server_address}/health")
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass

        for ds in dataset_list:
            ds_name = ds["name"]
            ds_type = ds["type"]

            os.makedirs(os.path.join(dir_out, ds_name), exist_ok=True)

            if ds_type == "multiple_choice":
                labels = np.array([ex["answer"] for ex in ds["data"]])
            elif ds_type == "math":
                labels = np.array([int(ex["answer"].split()[-1].replace(",", "")) for ex in ds["data"]])
            else:
                assert False
            # labels = labels[:1000]
            np.save(os.path.join(dir_out, ds_name, "labels.npy"), labels)

            for cot in [False, True]:
                if skip_cot and cot:
                    continue

                target: str = os.path.join(dir_out, ds_name, f"pred-cot{1 if cot else 0}.npy")
                if os.path.exists(target):
                    continue

                data_modded = []
                for ex in ds["data"]:
                    ex_copy = deepcopy(ex)
                    ex_copy["server_address"] = server_address
                    ex_copy["cot"] = cot
                    data_modded.append(ex_copy)
                # data_modded = data_modded[:1000]

                print(f"Start: {name}-{quant}, {ds_name}, cot={cot}, gpu_id={gpu_id}, server_address={server_address}")
                t0 = time()
                with Pool(PARALLEL) as pool:
                    if ds_type == "multiple_choice":
                        predictions = pool.map(process_multiple_choice, data_modded, chunksize=10)
                    elif ds_type == "math":
                        predictions = pool.map(process_math, data_modded, chunksize=10)
                    else:
                        assert False
                print(f"Done: {name}-{quant}, {ds_name}, cot={cot}, time={time() - t0:.2f}s")
                np.save(target, predictions)

        server_process.terminate()


lock = threading.Lock()


def thread_target(gpu_id: int):
    while True:
        lock.acquire()
        if not model_list:
            lock.release()
            break
        model = model_list.pop(0)
        lock.release()
        process_model(model, gpu_id)


NUM_GPUS = 6
threads = []

for i in range(1, NUM_GPUS):
    t = threading.Thread(target=thread_target, args=[i])
    t.start()
    threads.append(t)

thread_target(0)

for i in range(1, NUM_GPUS):
    threads[i - 1].join()
