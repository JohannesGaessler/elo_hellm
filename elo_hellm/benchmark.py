#!/usr/bin/env python3

from abc import ABC, abstractmethod
from copy import deepcopy
import os
import random
import sqlite3
from typing import Optional

import datasets
import yaml


with open("config.yml") as f:
    config: dict = yaml.safe_load(f)

path_db: str = os.path.join("results.sqlite")
connection: sqlite3.Connection = None
cursor: sqlite3.Cursor = None


def get_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    global connection, cursor
    if cursor is None:
        connection = sqlite3.connect(path_db)
        cursor = connection.cursor()
    return connection, cursor


datasets_raw: dict = dict()


def get_dataset_raw(name: str) -> dict:
    if name not in datasets_raw:
        print(f"Loading {name}...")
        if name == "gpqa":
            datasets_raw[name] = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main")
        elif name == "gsm8k":
            datasets_raw[name] = datasets.load_dataset("openai/gsm8k", "main")
        elif name == "mmlu":
            datasets_raw[name] = datasets.load_dataset("cais/mmlu", "all")
        elif name == "mmlu_pro":
            datasets_raw[name] = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
        else:
            assert False
    return datasets_raw[name]


datasets_usable: dict = dict()


def get_dataset(name: str) -> list[dict]:
    if name not in datasets_usable:
        if name == "gpqa_main":
            random.seed(123456)
            gpqa_raw = get_dataset_raw("gpqa")["train"]
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
        elif name == "gsm8k_test":
            data = list(get_dataset_raw("gsm8k")["test"])
            for d in data:
                d["answer"] = d["answer"].split()[-1].replace(",", "")
        elif name == "mmlu_test":
            data = list(get_dataset_raw("mmlu")["test"])
        elif name == "mmlu_pro_test":
            mmlu_pro_raw = get_dataset_raw("mmlu_pro")["test"]
            data = [dict(question=ex["question"], choices=ex["options"], answer=ex["answer_index"]) for ex in mmlu_pro_raw]
            data = list(filter(lambda d: len(d["choices"]) == 10, data))
        else:
            assert False
        for i, data_i in enumerate(data):
            data_i["iex"] = i
        if config["max_examples_per_dataset"] >= 0:
            data = data[:config["max_examples_per_dataset"]]
        datasets_usable[name] = data
    return deepcopy(datasets_usable[name])


class Benchmark(ABC):
    name: str
    prompt_type: str
    npredict_last: int
    score_rng: float

    def __init__(self, name: str, prompt_type: str):
        self.name = name
        assert prompt_type in ["normal", "instant"]
        self.prompt_type = prompt_type

        connection, cursor = get_db()
        columns_types: list[str] = [f"{c} {t}" for (c, t) in zip(self.database_columns(), self.database_types())]
        sql: str = f"CREATE TABLE IF NOT EXISTS {self.database_name()}({', '.join(columns_types)});"
        cursor.execute(sql)
        connection.commit()

    def nturns(self) -> int:
        if self.prompt_type == "instant":
            return 1
        if self.prompt_type == "normal":
            return 2

    def database_name(self) -> str:
        return f"{self.name}_{self.prompt_type}"

    def database_columns(self) -> list[str]:
        return ["model", "iex", "pred", "turn"] + [f"gen{i}" for i in range(self.nturns())]

    def database_types(self) -> list[str]:
        return ["TEXT", "INTEGER", "INTEGER", "INTEGER"] + ["TEXT"] * self.nturns()

    def get_input_data(self, model: str, turn: int) -> list[dict]:
        data = get_dataset(self.name)
        cursor: sqlite3.Cursor = get_db()[1]
        nturns: int = self.nturns()

        if turn == 0:
            sql: str = f"SELECT iex FROM {self.database_name()} WHERE model = ? AND turn != ?;"
            query: list[tuple[int]] = cursor.execute(sql, [model, turn]).fetchall()
            indices_done: list[int] = [q[0] for q in query]
            data_turn = list(filter(lambda d: d["iex"] not in indices_done, data))
            for dt in data_turn:
                dt["turn"] = turn
                dt["prompt_type"] = self.prompt_type
                dt["npredict"] = self.npredict_last if turn + 1 == nturns else 2048  # FIXME
                self.add_message_data(dt)
            return data_turn

        columns: list[str] = ["iex"] + [f"gen{i}" for i in range(turn)]
        sql: str = f"SELECT {', '.join(columns)} FROM {self.database_name()} WHERE model = ? AND iex < ? AND turn = ? ORDER BY iex;"
        query = cursor.execute(sql, [model, len(data), turn]).fetchall()

        data_turn = []
        for q in query:
            iex: int = q[0]
            dti: dict = data[iex]
            for i in range(turn):
                dti[f"gen{i}"] = q[1 + i]
            data_turn.append(dti)
        for dt in data_turn:
            dt["turn"] = turn
            dt["prompt_type"] = self.prompt_type
            dt["npredict"] = self.npredict_last if turn + 1 == nturns else 2048  # FIXME
            self.add_message_data(dt)
        return data_turn

    @staticmethod
    @abstractmethod
    def add_message_data(data: dict) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_prediction(completion: str) -> int:
        pass

    def update_database(self, model: str, data: list[dict]):
        connection, cursor = get_db()
        name: str = self.database_name()
        nturns: int = self.nturns()
        for d in data:
            turn: int = d["turn"]
            completion: str = d["completion"]
            pred: str = "NULL" if turn + 1 < nturns else str(self.get_prediction(completion))

            if turn == 0:
                values: list[str] = [model, str(d["iex"]), pred, str(turn + 1), completion]
                sql: str = f"INSERT INTO {name} (model, iex, pred, turn, gen0) VALUES ({', '.join(['?']*len(values))});"
                cursor.execute(sql, values)
            else:
                sql: str = (f"UPDATE {name} SET pred=?, turn=?, gen{turn}=? WHERE model=? AND iex=?;")
                cursor.execute(sql, [pred, turn + 1, completion, model, d["iex"]])
        connection.commit()

    def get_results(self, model: str):
        cursor: sqlite3.Cursor = get_db()[1]

        nturns: int = self.nturns()
        data: list[dict] = self.get_input_data(model, nturns)
        sql: str = (f"SELECT iex, pred FROM {self.database_name()} "
            f"WHERE model = ? AND iex < ? AND turn = ? ORDER BY iex;")
        query = cursor.execute(sql, [model, len(data), nturns])
        labels = []
        pred = []
        for q in query:
            labels.append(data[q[0]]["answer"])
            pred.append(q[1])
        return labels, pred


LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]


class BenchmarkMultipleChoice(Benchmark):
    nchoices: int

    def __init__(self, name: str, prompt_type: str):
        super().__init__(name, prompt_type)
        self.npredict_last = 1
        if name == "gpqa_main":
            self.nchoices = 5
        elif name == "mmlu_test":
            self.nchoices = 4
        elif name == "mmlu_pro_test":
            self.nchoices = 10
        else:
            assert False
        self.score_rng = 1.0 / self.nchoices

    @staticmethod
    def add_message_data(data: dict) -> None:
        turn: int = data["turn"]
        prompt_type: str = data["prompt_type"]

        messages: list[dict] = []
        prompt_suffix = ""
        grammar: Optional[str] = None

        choices = [f"({letter}): {choice}" for letter, choice in zip(LETTERS, data["choices"])]
        choices_block = "\n".join(choices)
        messages.append(dict(role="user", content=f"""{data['question']}

Which of the following answers is correct?
{choices_block}"""))

        if prompt_type == "instant":
            prompt_suffix: str = "The correct answer is ("
            grammar = f"root ::= [{''.join(LETTERS[:len(choices)])}]"
        elif prompt_type == "normal":
            if turn == 0:
                pass
            elif turn == 1:
                messages.append(dict(role="assistant", content=data["gen0"]))
                messages.append(dict(role="user", content="Please enter your final answer."))
                prompt_suffix: str = "My final answer is ("
                grammar = f"root ::= [{''.join(LETTERS[:len(choices)])}]"
        data["messages"] = messages
        data["prompt_suffix"] = prompt_suffix
        data["grammar"] = grammar

    @staticmethod
    def get_prediction(completion: str) -> int:
        return LETTERS.index(completion[:1])


class BenchmarkMath(Benchmark):
    def __init__(self, name: str, prompt_type: str):
        super().__init__(name, prompt_type)
        self.npredict_last = 10
        self.score_rng = 0.0

    @staticmethod
    def add_message_data(data: dict) -> None:
        turn: int = data["turn"]
        prompt_type: str = data["prompt_type"]

        messages: list[dict] = []
        prompt_suffix = ""
        grammar: Optional[str] = None

        messages.append(dict(role="user", content=data["question"]))

        if prompt_type == "instant":
            prompt_suffix: str = "The correct answer is "
            grammar = "root ::= [0-9]+.*"
        elif prompt_type == "normal":
            if turn == 0:
                pass
            elif turn == 1:
                messages.append(dict(role="assistant", content=data["gen0"]))
                messages.append(dict(role="user", content="Please enter your final answer."))
                prompt_suffix: str = "My final answer is "
                grammar = "root ::= [0-9]+.*"
        data["messages"] = messages
        data["prompt_suffix"] = prompt_suffix
        data["grammar"] = grammar

    @staticmethod
    def get_prediction(completion: str) -> int:
        completion = completion.replace(",", "")
        index = 0
        pred = 123456789
        while index < len(completion):
            try:
                index += 1
                pred = int(completion[:index])
            except ValueError:
                break
        return pred


benchmarks: dict[tuple[str, str], Benchmark] = dict()


def get_benchmark(dataset: str, prompt_type: str) -> Benchmark:
    if (dataset, prompt_type) not in benchmarks:
        if dataset in ["gsm8k_test"]:
            benchmark = BenchmarkMath(dataset, prompt_type)
        elif dataset in ["gpqa_main", "mmlu_test", "mmlu_pro_test"]:
            benchmark = BenchmarkMultipleChoice(dataset, prompt_type)
        else:
            assert False
        benchmarks[(dataset, prompt_type)] = benchmark
    return benchmarks[(dataset, prompt_type)]
