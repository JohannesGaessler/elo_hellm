#!/usr/bin/env python3

import os

import yaml


class Model:
    base_name: str
    quantization: str
    name: str
    path: str
    datasets: list[str]
    prompt_types: list[str]
    parallel: int
    gpus_per_job: int
    file_size: int

    def __init__(self, base_name: str, quantization: str, path: str, datasets: list[str], prompt_types: list[str], parallel: int, gpus_per_job: int):
        assert type(base_name) is str
        self.base_name = base_name
        assert type(quantization) is str
        self.quantization = quantization
        assert type(path) is str
        self.path = path
        assert type(datasets) is list
        self.datasets = datasets
        assert type(prompt_types) is list
        self.prompt_types = prompt_types
        assert type(parallel) is int
        self.parallel = parallel
        assert type(gpus_per_job) is int
        self.gpus_per_job = gpus_per_job

        self.name = f"{self.base_name}-{self.quantization}"
        self.file_size = os.path.getsize(self.path)


class Config:
    debug: bool
    max_examples_per_dataset: int
    path_server: str
    ctx_size: int
    num_gpus: int
    path_model: str
    datasets: list[str]
    prompt_types: list[str]
    quantizations: list[str]
    parallel: int
    gpus_per_job: int
    models: list[Model]

    def __init__(self, path: str):
        with open(path) as f:
            config: dict = yaml.safe_load(f)
        self.debug = config.get("debug", False)
        assert type(self.debug) is bool
        self.max_examples_per_dataset = config.get("max_examples_per_dataset", -1)
        assert type(self.max_examples_per_dataset) is int
        self.path_server = config.get("path_server")
        assert type(self.path_server) is str
        self.ctx_size = config.get("ctx_size", 4096)
        assert type(self.ctx_size) is int
        self.num_gpus = config.get("num_gpus", 1)
        assert type(self.num_gpus) is int
        self.path_model = config.get("path_model")
        assert type(self.path_model) is str
        self.datasets = config.get("datasets")
        assert type(self.datasets) is list
        self.prompt_types = config.get("prompt_types")
        assert type(self.prompt_types) is list
        self.quantizations = config.get("quantizations")
        assert type(self.quantizations) is list
        self.parallel = config.get("parallel", 8)
        assert type(self.parallel) is int
        self.gpus_per_job = config.get("gpus_per_job", 1)
        assert type(self.gpus_per_job) is int

        models = config.get("models")
        assert type(models) is list

        self.models = []
        for m in models:
            assert type(m) is dict
            base_name: str = m.get("base_name")
            quantizations: list[str] = m.get("quantizations", self.quantizations)
            for q in quantizations:
                self.models.append(Model(
                    base_name=base_name,
                    quantization=q,
                    path=m.get("path_model", self.path_model).format(name=base_name, quantization=q),
                    prompt_types=m.get("prompt_types", self.prompt_types),
                    datasets=m.get("datasets", self.datasets),
                    parallel=m.get("parallel", self.parallel),
                    gpus_per_job=m.get("gpus_per_job", self.gpus_per_job),
                ))
        self.models = sorted(self.models, key=lambda m: m.file_size, reverse=False)
