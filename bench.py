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

jobs: List[dict] = []
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

        for dataset in dataset_list:
            for prompt_type in prompt_types:
                jobs.append(dict(
                    name=name,
                    quantization=quant,
                    path_model=path_model,
                    file_size=file_size,
                    parallel=parallel,
                    gpus_per_job=gpus_per_job,
                    dataset=dataset,
                    prompt_type=prompt_type,
                ))

jobs = sorted(jobs, key=lambda job: len(job["dataset"]["data"]) * job["file_size"], reverse=True)

CTX_SIZE = 4096

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

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


def process_job(job, gpu_ids: List[int]):
    port = 1337 + gpu_ids[0]
    server_address = TEMPLATE_SERVER_ADDRESS.format(port=port)

    name: str = job["name"]
    quant: str = job["quantization"]
    path_model: str = job["path_model"]
    parallel: int = job["parallel"]
    dataset: dict = job["dataset"]
    prompt_type: str = job["prompt_type"]

    dir_out: str = os.path.join("out", "bench", name, quant, dataset["name"], prompt_type)
    os.makedirs(dir_out, exist_ok=True)

    target: str = os.path.join(dir_out, f"pred.npy")
    if os.path.exists(target):
        return

    with open(os.path.join(dir_out, "job.yml"), "w") as f:
        yaml.safe_dump(job, f)

    if dataset["type"] == "multiple_choice":
        labels = np.array([ex["answer"] for ex in dataset["data"]], dtype=np.int64)
    elif dataset["type"] == "math":
        labels = np.array([int(ex["answer"].split()[-1].replace(",", "")) for ex in dataset["data"]], dtype=np.int64)
    else:
        assert False
    if config["max_examples_per_dataset"] >= 0:
        labels = labels[:config["max_examples_per_dataset"]]
    np.save(os.path.join(dir_out, "labels.npy"), labels)

    env: Dict[str, str] = dict(
        CUDA_VISIBLE_DEVICES=",".join([str(g) for g in gpu_ids]),
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

        data_modded = []
        for ex in dataset["data"]:
            ex_copy = deepcopy(ex)
            ex_copy["name"] = f"{name}-{quant}-{dataset['name']}-{prompt_type}"
            ex_copy["server_address"] = server_address
            ex_copy["prompt_type"] = prompt_type
            data_modded.append(ex_copy)
        if config["max_examples_per_dataset"] >= 0:
            data_modded = data_modded[:config["max_examples_per_dataset"]]

        print(f"Start: {name}-{quant}, {dataset['name']}, {prompt_type}, gpu_ids={gpu_ids}")
        t0 = time()
        with Pool(parallel) as pool:
            chunksize: int = 1 if config["debug"] else min(10, max(1, len(data_modded) // (4*parallel)))
            if dataset['type'] == "multiple_choice":
                predictions = pool.map(process_multiple_choice, data_modded, chunksize=chunksize)
                predictions = np.array(predictions, dtype=np.float64)
            elif dataset['type'] == "math":
                predictions = pool.map(process_math, data_modded, chunksize=chunksize)
                predictions = np.array(predictions, dtype=np.int64)
            else:
                assert False
        print(f"Done: {name}-{quant}, {dataset['name']}, {prompt_type}, time={time() - t0:.2f}s")
        np.save(target, predictions)

        server_process.terminate()


lock = threading.Lock()


def find_job(num_gpus: int):
    ret = None
    lock.acquire()
    for i in range(len(jobs)):
        if jobs[i]["gpus_per_job"] == num_gpus:
            ret = jobs.pop(i)
            break
    lock.release()
    return ret


def process_jobs(gpu_ids: List[int]):
    job = find_job(len(gpu_ids))
    while job is not None:
        process_job(job, gpu_ids)
        job = find_job(len(gpu_ids))


gpu_ids: List[int] = list(range(6))
threads_2 = []

process_jobs(gpu_ids)
process_jobs(gpu_ids[:5])

threads_2.append(threading.Thread(target=process_jobs, args=[gpu_ids[4:]]))
threads_2[-1].start()
process_jobs(gpu_ids[:4])

threads_2.append(threading.Thread(target=process_jobs, args=[gpu_ids[2:4]]))
threads_2[-1].start()
process_jobs(gpu_ids[:2])

for thread in threads_2:
    thread.join()

thread_3 = threading.Thread(target=process_jobs, args=[gpu_ids[3:]])
thread_3.start()
process_jobs(gpu_ids[:3])
thread_3.join()

threads_1 = []
for i in range(1, 6):
    threads_1.append(threading.Thread(target=process_jobs, args=[gpu_ids[i:i+1]]))
    threads_1[-1].start()
process_jobs(gpu_ids[:1])

for thread in threads_1:
    thread.join()
