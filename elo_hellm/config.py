#!/usr/bin/env python3

import os

import yaml


class Model:
    name: str
    path: str
    datasets: list[str]
    prompt_types: list[str]
    parallel: int
    gpus_per_server: int
    file_size: int

    def __init__(self, name: str, path: str, datasets: list[str], prompt_types: list[str], parallel: int, gpus_per_server: int):
        assert type(name) is str
        self.name = name
        assert type(path) is str
        self.path = path
        assert type(datasets) is list
        self.datasets = datasets
        assert type(prompt_types) is list
        self.prompt_types = prompt_types
        assert type(parallel) is int
        self.parallel = parallel
        assert type(gpus_per_server) is int
        self.gpus_per_server = gpus_per_server

        self.file_size = os.path.getsize(self.path)


class Config:
    debug: bool
    max_examples_per_dataset: int
    path_server: str
    stockfish_path: str
    stockfish_threads: int
    stockfish_hash: int
    ctx_size: int
    num_gpus: int
    model_dir: str
    datasets: list[str]
    prompt_types: list[str]
    parallel: int
    gpus_per_server: int
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
        self.stockfish_path = config.get("stockfish_path", "/usr/bin/stockfish")
        assert type(self.stockfish_path) is str
        self.stockfish_threads = config.get("stockfish_threads", 8)
        assert type(self.stockfish_threads) is int
        self.stockfish_hash = config.get("stockfish_hash", 1024)
        assert type(self.stockfish_hash) is int
        self.ctx_size = config.get("ctx_size", 4096)
        assert type(self.ctx_size) is int
        self.num_gpus = config.get("num_gpus", 1)
        assert type(self.num_gpus) is int
        self.model_dir = config.get("model_dir")
        assert type(self.model_dir) is str
        self.datasets = config.get("datasets")
        assert type(self.datasets) is list
        self.prompt_types = config.get("prompt_types")
        assert type(self.prompt_types) is list
        self.parallel = config.get("parallel", 8)
        assert type(self.parallel) is int
        self.gpus_per_server = config.get("gpus_per_server", 1)
        assert type(self.gpus_per_server) is int

        models = config.get("models")
        assert type(models) is list

        self.models = []
        for m in models:
            assert type(m) is dict
            name: str = m.get("name")
            self.models.append(Model(
                name=name,
                path=os.path.join(m.get("model_dir", self.model_dir), name),
                prompt_types=m.get("prompt_types", self.prompt_types),
                datasets=m.get("datasets", self.datasets),
                parallel=m.get("parallel", self.parallel),
                gpus_per_server=m.get("gpus_per_server", self.gpus_per_server),
            ))
        self.models = sorted(self.models, key=lambda m: m.file_size, reverse=False)
