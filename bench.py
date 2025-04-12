#!/usr/bin/env python3

from copy import deepcopy
import json
import os
import random
from time import sleep, time
from typing import Dict, List

import datasets
import numpy as np
import requests
import subprocess
from tqdm.contrib.concurrent import process_map
import yaml


with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

model_list: List[dict] = []
for model in config["models"]:
    name: str = model["name"]
    quantizations: List[str] = model.get("quantizations", config["quantizations"])
    path_model_template: str = model.get("path_model", config["path_model"])
    path_imatrix_template: str = model.get("path_imatrix", config["path_imatrix"])
    parallel: int = 1 if config["debug"] else model.get("parallel", config["parallel"])
    gpus_per_job: int = model.get("gpus_per_job", config["gpus_per_job"])
    prompt_types: List[str] = model.get("prompt_types", config["prompt_types"])

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
                    [config["path_quantize"], "--imatrix", path_imatrix, path_model_f16, path_model, quant, "32"],
                    stdout=fout,
                    stderr=ferr,
                    env=env,
                ).returncode
                assert returncode == 0

        file_size: int = os.path.getsize(path_model)

        model_list.append(dict(
            name=name,
            quantization=quant,
            path_model=path_model,
            file_size=file_size,
            parallel=parallel,
            gpus_per_job=gpus_per_job,
            prompt_types=prompt_types,
        ))

model_list = sorted(model_list, key=lambda m: m["file_size"], reverse=False)

CTX_SIZE = 4096

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]

TEMPLATE_SERVER_ADDRESS = "http://localhost:{port}"
TEMPLATE_PROMPT_MULTIPLE_CHOICE = """{question}

Which of the following answers is correct?
{choices_block}"""
TEMPLATE_RESPONSE_MULTIPLE_CHOICE_INSTANT = """The correct answer is ("""
TEMPLATE_PROMPT_NORMAL_2 = """Please enter your final answer."""
TEMPLATE_RESPONSE_MULTIPLE_CHOICE_NORMAL_2 = """My final answer is ("""


def get_choices_block(choices: List[str]) -> str:
    choices = [f"({letter}): {choice}" for letter, choice in zip(LETTERS, choices)]
    return "\n".join(choices)


def get_grammar_multiple_choice(num_choices: int) -> str:
    return f"root ::= [{''.join(LETTERS[:num_choices])}]"


TEMPLATE_RESPONSE_MATH = """The correct answer is """
TEMPLATE_RESPONSE_MATH_NORMAL_2 = """My final answer is """
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

if "gpqa_main" in config["datasets"]:
    print("Loading GPQA...")
    random.seed(123456)
    gpqa_raw = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
    data = []
    for ex in gpqa_raw:
        if ex["Extra Revised Question"] is not None:
            question = ex["Extra Revised Question"]
            choice_correct = ex["Extra Revised Correct Answer"]
            choices_wrong = [
                ex["Extra Revised Incorrect Answer 1"],
                ex["Extra Revised Incorrect Answer 2"],
                ex["Extra Revised Incorrect Answer 3"],
            ]
        else:
            question = ex["Question"]
            choice_correct = ex["Correct Answer"]
            choices_wrong = [
                ex["Incorrect Answer 1"],
                ex["Incorrect Answer 2"],
                ex["Incorrect Answer 3"],
            ]
        choices = [choice_correct] + choices_wrong
        random.shuffle(choices)
        answer = choices.index(choice_correct)

        data.append(dict(question=question, choices=choices, answer=answer))
    dataset_list.append(dict(name="gpqa_main", type="multiple_choice", data=data))

if "mmlu-pro_test" in config["datasets"]:
    print("Loading MMLU-Pro...")
    mmlu_pro_raw = datasets.load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    data = [dict(question=ex["question"], choices=ex["options"], answer=ex["answer_index"]) for ex in mmlu_pro_raw]
    data = list(filter(lambda d: len(d["choices"]) == 10, data))
    dataset_list.append(dict(name="mmlu-pro_test", type="multiple_choice", data=data))


def process_multiple_choice(example: dict) -> List[float]:
    name: str = example["name"]
    server_address = example["server_address"]
    prompt_type: str = example["prompt_type"]
    question: str = example["question"]
    choices: List[str] = example["choices"]

    assert type(question) is str
    assert type(choices) is list

    num_choices: int = len(choices)
    choices_block: str = get_choices_block(choices)

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

    if prompt_type != "instant":
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
        messages.append(dict(role="user", content=TEMPLATE_PROMPT_NORMAL_2))

        response = requests.post(
            f"{server_address}/apply-template",
            json=dict(messages=messages)
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        prompt: str = json.loads(response.text)["prompt"]
        prompt += TEMPLATE_RESPONSE_MULTIPLE_CHOICE_NORMAL_2

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
    if config["debug"]:
        print(f"====== {name} ======")
        print(prompt)
        print(f"predictions={predictions}")
        print()
    return predictions


def process_math(example: dict) -> List[float]:
    name: str = example["name"]
    server_address = example["server_address"]
    prompt_type: str = example["prompt_type"]
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

    if prompt_type != "instant":
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
        messages.append(dict(role="user", content=TEMPLATE_PROMPT_NORMAL_2))

        response = requests.post(
            f"{server_address}/apply-template",
            json=dict(messages=messages)
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        prompt: str = json.loads(response.text)["prompt"]
        prompt += TEMPLATE_RESPONSE_MATH_NORMAL_2
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
    content: str = response_dict["content"].replace(",", "")
    index = 0
    prediction = 123456789
    while index < len(content):
        try:
            index += 1
            prediction = int(content[:index])
        except ValueError:
            break
    if prediction == 123456789 or config["debug"]:
        print(f"====== {name} ======")
        print(prompt)
        print(f"prediction={content} -> {prediction}")
        print()
    return prediction


def process_model(model):
    NUM_GPUS: int = 6

    name: str = model["name"]
    quant: str = model["quantization"]
    path_model: str = model["path_model"]
    parallel: int = model["parallel"]
    gpus_per_job: int = model["gpus_per_job"]
    prompt_types: List[str] = model["prompt_types"]

    dir_out: str = os.path.join("out", "bench", name, quant)
    os.makedirs(dir_out, exist_ok=True)

    all_targets_existent: bool = True
    for ds in dataset_list:
        ds_name = ds["name"]
        for prompt_type in prompt_types:
            target: str = os.path.join(dir_out, ds_name, f"pred-{prompt_type}.npy")
            if not os.path.exists(target):
                all_targets_existent = False
                break
    if all_targets_existent:
        return

    with open(os.path.join(dir_out, "model.yml"), "w") as f:
        yaml.safe_dump(model, f)

    servers = []
    for i in range(0, NUM_GPUS, gpus_per_job):
        port = 1337 + i
        address = TEMPLATE_SERVER_ADDRESS.format(port=port)

        env: Dict[str, str] = dict(
            CUDA_VISIBLE_DEVICES=",".join([str(j) for j in range(i, i+gpus_per_job)]),
        )

        popen_args: List[str] = [
            config["path_server"],
            "--flash-attn",
            "--n-gpu-layers", "999",
            "--parallel", str(parallel),
            "--ctx-size", str(parallel * CTX_SIZE),
            "--model", path_model,
            "--port", str(port),
        ]
        fout = open(os.path.join(dir_out, f"server_{i}.out"), "w")
        ferr = open(os.path.join(dir_out, f"server_{i}.err"), "w")
        process = subprocess.Popen(popen_args, env=env, stdout=fout, stderr=ferr)
        servers.append(dict(process=process, address=address, fout=fout, ferr=ferr))
    for server in servers:
        while True:
            try:
                sleep(1.0)
                exit_code = server["process"].poll()
                if exit_code is not None:
                    raise RuntimeError(f"llama.cpp server for {name}-{quant} exited unexpectedly with exit code {exit_code}")
                response = requests.get(f"{server['address']}/health")
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass

    for ds in dataset_list:
        ds_name = ds["name"]
        ds_type = ds["type"]
        ds_data = list(ds["data"])  # Slicing returns a different types if you don't cast to list.
        imax: int = config["max_examples_per_dataset"]
        if imax >= 0:
            ds_data = ds_data[:imax]

        os.makedirs(os.path.join(dir_out, ds_name), exist_ok=True)

        if ds_type == "multiple_choice":
            labels = np.array([ex["answer"] for ex in ds_data], dtype=np.int64)
        elif ds_type == "math":
            labels = np.array(
                [int(ex["answer"].split()[-1].replace(",", "")) for ex in ds_data],
                dtype=np.int64
            )
        else:
            assert False
        np.save(os.path.join(dir_out, ds_name, "labels.npy"), labels)

        for prompt_type in prompt_types:
            target: str = os.path.join(dir_out, ds_name, f"pred-{prompt_type}.npy")
            if os.path.exists(target):
                continue

            data_modded = []
            for i, ex_i in enumerate(ds_data):
                ex_copy = deepcopy(ex_i)
                ex_copy["name"] = f"{name}-{quant}-{ds_name}-{prompt_type}"
                ex_copy["server_address"] = servers[i % len(servers)]["address"]
                ex_copy["prompt_type"] = prompt_type
                data_modded.append(ex_copy)

            t0 = time()
            print(f"Start: {name}-{quant}, {ds_name}, {prompt_type}")
            max_workers: int = 2 * len(servers) * parallel
            chunksize: int = 1
            if ds_type == "multiple_choice":
                predictions = process_map(process_multiple_choice, data_modded, max_workers=max_workers, chunksize=chunksize)
                predictions = np.array(predictions, dtype=np.float64)
            elif ds_type == "math":
                predictions = process_map(process_math, data_modded, max_workers=max_workers, chunksize=chunksize)
                predictions = np.array(predictions, dtype=np.int64)
            else:
                assert False
            print(f"Done: {name}-{quant}, {ds_name}, {prompt_type}, time={time() - t0:.2f}s")
            np.save(target, predictions)

    for server in servers:
        server["process"].terminate()
        server["fout"].close()
        server["ferr"].close()
    for server in servers:
        server["process"].wait()


for model in model_list:
    process_model(model)
