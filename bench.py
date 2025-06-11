#!/usr/bin/env python3

import json
import os
from time import sleep, time
from typing import Dict, Optional, List

import requests
import subprocess
from tqdm.contrib.concurrent import process_map

from elo_hellm.benchmark import Benchmark, get_benchmark
from elo_hellm.config import Config, Model


TEMPLATE_SERVER_ADDRESS = "http://localhost:{port}"

config = Config("config.yml")


def get_servers(model: Model) -> List[dict]:
    os.makedirs("logs", exist_ok=True)

    servers = []
    for i in range(0, config.num_gpus, model.gpus_per_job):
        port = 1337 + i
        address = TEMPLATE_SERVER_ADDRESS.format(port=port)

        env: Dict[str, str] = dict(
            CUDA_VISIBLE_DEVICES=",".join([str(j) for j in range(i, i+model.gpus_per_job)]),
        )

        popen_args: List[str] = [
            config.path_server,
            "--flash-attn",
            "--n-gpu-layers", "999",
            "--parallel", str(model.parallel),
            "--ctx-size", str(model.parallel * config.ctx_size),
            "--model", model.path,
            "--port", str(port),
            "--swa-full",  # FIXME
        ]
        if config.debug:
            popen_args.append("--verbose")
        fout = open(os.path.join("logs", f"{model.name}_{i}.log"), "w")
        process = subprocess.Popen(popen_args, env=env, stdout=fout, stderr=subprocess.STDOUT)
        servers.append(dict(process=process, address=address, fout=fout))
        if config.debug:
            break
    for server in servers:
        while True:
            try:
                sleep(1.0)
                exit_code = server["process"].poll()
                if exit_code is not None:
                    raise RuntimeError(f"llama.cpp server for {model.name} exited unexpectedly with exit code {exit_code}")
                response = requests.get(f"{server['address']}/health")
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
    return servers


def get_completion(data: dict) -> str:
    server_address: str = data["server_address"]
    npredict: int = data["npredict"]
    grammar: Optional[str] = data["grammar"]

    response = requests.post(
        f"{server_address}/apply-template",
        json=dict(messages=data["messages"])
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    prompt: str = json.loads(response.text)["prompt"]
    prompt += data["prompt_suffix"]

    json_dict = dict(
        prompt=prompt,
        n_predict=npredict,
        temperature=0.0,
    )
    if grammar is not None:
        json_dict["grammar"] = grammar
    response = requests.post(f"{server_address}/completion", json=json_dict)
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    response_dict: dict = json.loads(response.text)
    return response_dict["content"]


def process_model(model: Model):
    servers = []
    try:
        for ds in model.datasets:
            for prompt_type in model.prompt_types:
                benchmark: Benchmark = get_benchmark(ds, prompt_type)
                for turn in range(benchmark.nturns()):
                    data = benchmark.get_input_data(model.name, turn)
                    if not data:
                        continue
                    if not servers:
                        servers = get_servers(model)
                    for i, di in enumerate(data):
                        di["server_address"] = servers[i % len(servers)]["address"]

                    t0 = time()
                    print(f"Start: {model.name}, {benchmark.database_name()}, turn={turn}")
                    max_workers: int = 2 * len(servers) * model.parallel
                    chunksize: int = 1
                    completions = process_map(get_completion, data, max_workers=max_workers, chunksize=chunksize)
                    for d, c in zip(data, completions):
                        d["completion"] = c
                    benchmark.update_database(model.name, data)
                    print(f"Done: {model.name}, {benchmark.database_name()}, turn={turn}, time={time() - t0:.2f}s")
    finally:
        for server in servers:
            server["process"].terminate()
            server["fout"].close()
        for server in servers:
            server["process"].wait()


for model in config.models:
    process_model(model)
