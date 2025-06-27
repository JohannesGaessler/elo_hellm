#!/usr/bin/env python3

from abc import ABC, abstractmethod
from copy import deepcopy
import json
import os
import random
import sqlite3
from typing import Optional

import chess
import datasets
from stockfish import Stockfish

from elo_hellm.config import Config

config: Config = Config("config.yml")

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
        elif name == "chess960":
            print("Generating chess960 initial states...")
            data = []
            for i in range(960):
                data.append(dict(state0=chess.Board.from_chess960_pos(i).fen()))
        else:
            assert False
        for i, data_i in enumerate(data):
            data_i["iex"] = i
        if config.max_examples_per_dataset >= 0:
            data = data[:config.max_examples_per_dataset]
        datasets_usable[name] = data
    return deepcopy(datasets_usable[name])


class Benchmark(ABC):
    name: str
    prompt_type: str
    turn: int
    npredict_last: int
    score_rng: float

    def __init__(self, name: str, prompt_type: str, turn: int):
        self.name = name
        assert prompt_type in ["normal", "instant"]
        self.prompt_type = prompt_type
        self.turn = turn

        connection, cursor = get_db()
        columns_types: list[str] = [f"{c} {t}" for (c, t) in zip(self.database_columns(), self.database_types())]
        sql: str = f"CREATE TABLE IF NOT EXISTS {self.database_name()}({', '.join(columns_types)});"
        cursor.execute(sql)
        connection.commit()

    def n_gens(self) -> int:
        if self.prompt_type == "instant":
            return 1
        if self.prompt_type == "normal":
            return 2

    def database_name(self) -> str:
        return f"{self.name}_{self.prompt_type}"

    def database_columns(self) -> list[str]:
        return ["model", "iex", "pred", "turn", "i_gen"] + [f"gen{i}" for i in range(self.n_gens())]

    def database_types(self) -> list[str]:
        return ["TEXT", "INTEGER", "INTEGER", "INTEGER", "INTEGER"] + ["TEXT"] * self.n_gens()

    def get_input_data(self, model: str, i_gen: int) -> list[dict]:
        data = get_dataset(self.name)
        database_name: str = self.database_name
        connection, cursor = get_db()
        n_gens: int = self.n_gens()
        assert i_gen < n_gens

        if i_gen == 0:
            sql: str = f"SELECT iex FROM {database_name} WHERE model = ? AND i_gen != ?;"
            query: list[tuple[int]] = cursor.execute(sql, [model, i_gen]).fetchall()
            indices_done: list[int] = [q[0] for q in query]
            data = list(filter(lambda d: d["iex"] not in indices_done, data))
            for dt in data:
                dt["turn"] = self.turn
                dt["i_gen"] = i_gen
                dt["database_name"] = database_name
                dt["prompt_type"] = self.prompt_type
                dt["npredict"] = self.npredict_last if i_gen + 1 == n_gens else 2048  # FIXME
                dt["add_message_data"] = self.add_message_data
            return data

        columns: list[str] = ["iex"] + [f"gen{i}" for i in range(i_gen)]
        sql: str = f"SELECT {', '.join(columns)} FROM {database_name} WHERE model = ? AND iex < ? AND i_gen = ? ORDER BY iex;"
        query = cursor.execute(sql, [model, len(data), i_gen]).fetchall()

        data = []
        for q in query:
            iex: int = q[0]
            dti: dict = data[iex]
            for i in range(i_gen):
                dti[f"gen{i}"] = q[1 + i]
            data.append(dti)
        for dt in data:
            dt["turn"] = self.turn
            dt["i_gen"] = i_gen
            dt["database_name"] = database_name
            dt["prompt_type"] = self.prompt_type
            dt["npredict"] = self.npredict_last if i_gen + 1 == n_gens else 2048  # FIXME
            dt["add_message_data"] = self.add_message_data
        return data

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
        n_gens: int = self.n_gens()
        for d in data:
            i_gen: int = d["i_gen"]
            completion: str = d["completion"]
            pred: str = "NULL" if i_gen + 1 < n_gens else str(self.get_prediction(completion))

            if i_gen == 0:
                values: list[str] = [model, str(d["iex"]), pred, str(i_gen + 1), completion]
                sql: str = f"INSERT INTO {name} (model, iex, pred, i_gen, gen0) VALUES ({', '.join(['?']*len(values))});"
                cursor.execute(sql, values)
            else:
                sql: str = (f"UPDATE {name} SET pred=?, i_gen=?, gen{i_gen}=? WHERE model=? AND iex=?;")
                cursor.execute(sql, [pred, i_gen + 1, completion, model, d["iex"]])
        connection.commit()

    def get_results(self, model: str):
        cursor: sqlite3.Cursor = get_db()[1]

        n_gens: int = self.n_gens()
        data: list[dict] = self.get_input_data(model, n_gens)
        sql: str = (f"SELECT iex, pred FROM {self.database_name()} "
            f"WHERE model = ? AND iex < ? AND i_gen = ? ORDER BY iex;")
        query = cursor.execute(sql, [model, len(data), n_gens])
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
        super().__init__(name, prompt_type, turn=0)
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
        i_gen: int = data["i_gen"]
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
        elif prompt_type == "normal" and i_gen == 1:
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
        super().__init__(name, prompt_type, turn=0)
        self.npredict_last = 10
        self.score_rng = 0.0

    @staticmethod
    def add_message_data(data: dict) -> None:
        i_gen: int = data["i_gen"]
        prompt_type: str = data["prompt_type"]

        messages: list[dict] = []
        prompt_suffix = ""
        grammar: Optional[str] = None

        messages.append(dict(role="user", content=data["question"]))

        if prompt_type == "instant":
            prompt_suffix: str = "The correct answer is "
            grammar = "root ::= [0-9]+.*"
        elif prompt_type == "normal" and i_gen == 1:
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


class BenchmarkChess960(Benchmark):
    nchoices: int = 10

    def __init__(self, prompt_type: str, turn: int):
        super().__init__("chess960", prompt_type, turn)
        self.npredict_last = 1
        self.score_rng = 1.0 / self.nchoices

        connection, cursor = get_db()
        sql: str = "CREATE TABLE IF NOT EXISTS stockfish_cache(fen TEXT, moves TEXT NOT NULL, PRIMARY KEY (fen));"
        cursor.execute(sql)
        connection.commit()

    @staticmethod
    def add_random_moves(moves: list[dict], iex: int, i_gen: int, turn: int) -> None:
        assert len(moves) <= BenchmarkChess960.nchoices
        if len(moves) == BenchmarkChess960.nchoices:
            return
        random.seed(12345678 + 10000*iex + 10*turn + i_gen)

        LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h"]
        NUMBERS = ["1", "2", "3", "4", "5", "6", "7", "8"]

        while len(moves) < BenchmarkChess960.nchoices:
            random_move: str = random.choice(LETTERS) + random.choice(NUMBERS) + random.choice(LETTERS) + random.choice(NUMBERS)
            if random_move[:2] == random_move[2:]:
                continue
            for m in moves:
                if m["Move"] == random_move:
                    continue
            moves.append(dict(Move=random_move, illegal=True))

    @staticmethod
    def move_to_key(move: dict) -> int:
        illegal: bool = move.get("illegal", False)
        if illegal:
            return -1000000
        mate: Optional[int] = move["Mate"]
        if mate is not None:
            if mate > 0:
                return 1000 + mate
            else:
                return -1000 - mate
        return move["Centipawn"]

    @staticmethod
    def add_message_data(data: dict) -> None:
        local_data = data["local_data"]
        iex: int = data["iex"]
        i_gen: int = data["i_gen"]
        turn: int = data["turn"]
        database_name: str = data["database_name"]
        prompt_type: str = data["prompt_type"]
        state0: str = data["state0"]

        if not hasattr(local_data, "stockfish"):
            local_data.stockfish = Stockfish(path=config.stockfish_path, parameters=dict(
                Threads=config.stockfish_threads, Hash=config.stockfish_hash, UCI_Chess960="true"))
        if not hasattr(local_data, "connection"):
            local_data.connection = sqlite3.connect(path_db)
            local_data.cursor = local_data.connection.cursor()
        local_data.stockfish.set_fen_position(state0)

        if turn > 0:
            sql: str = f"SELECT pred FROM {database_name} WHERE turn < ? ORDER BY turn;"
            query: list = local_data.cursor.execute(sql, [turn]).fetchall()
            assert len(query) == turn
            preds: list[int] = [q[0] for q in query]

            for i in range(turn):
                state: str = local_data.stockfish.get_fen_position()
                sql: str = "SELECT moves FROM stockfish_cache WHERE fen=?;"
                query: list = local_data.cursor.execute(sql, [state]).fetchall()
                assert len(query) == 1
                moves: list[dict] = json.loads(query[0][0])
                local_data.stockfish.make_moves_from_current_position([moves[preds[i]]["Move"]])

        state: str = local_data.stockfish.get_fen_position()
        sql: str = "SELECT moves FROM stockfish_cache WHERE fen=?;"
        query: list = local_data.cursor.execute(sql, [state]).fetchall()

        if query:
            assert len(query) == 1
            moves: list[dict] = json.loads(query[0][0])
        else:
            assert local_data.stockfish.is_fen_valid(state)
            local_data.stockfish.set_fen_position(state)
            moves: list[dict] = local_data.stockfish.get_top_moves(BenchmarkChess960.nchoices)
            moves = sorted(moves, key=BenchmarkChess960.move_to_key, reverse=True)
            BenchmarkChess960.add_random_moves(moves, iex, turn)

            sql: str = "INSERT INTO stockfish_cache VALUES (?, ?);"
            local_data.cursor.execute(sql, [state, json.dumps(moves)])
            local_data.connection.commit()
        permutation = [i for i in range(BenchmarkChess960.nchoices)]
        random.seed(123456 + 1000*iex + turn)
        random.shuffle(permutation)

        data["label"] = permutation.index(0)
        data["moves"] = [moves[permutation[i]] for i in permutation]

        active_player: str = "White" if turn % 2 == 0 else "Black"

        messages: list[dict] = []
        prompt_suffix = ""
        grammar: Optional[str] = None

        choices = [f"({letter}): {move['Move']}" for letter, move in zip(LETTERS, moves)]
        choices_block = "\n".join(choices)
        messages.append(dict(role="user", content=f"""Consider the following game of chess in Forsyth–Edwards Notation:

{state}

Which of the following moves is the best one for {active_player} to take?
{choices_block}"""))

        assert i_gen == 0 and prompt_type == "instant"
        prompt_suffix: str = f"The best move for {active_player} to take is ("
        grammar = f"root ::= [{''.join(LETTERS[:len(choices)])}]"

        data["messages"] = messages
        data["prompt_suffix"] = prompt_suffix
        data["grammar"] = grammar

    @staticmethod
    def get_prediction(completion: str) -> int:
        return LETTERS.index(completion[:1])

    def update_database(self, model: str, data: list[dict]):
        connection, cursor = get_db()
        name: str = self.database_name()
        for d in data:
            turn: int = d["turn"]
            completion: str = d["completion"]
            pred: int = self.get_prediction(completion)
            move_uci: str = d["moves"][pred]["Move"]
            move: chess.Move = chess.Move.from_uci(move_uci)

            board = chess.Board(d[f"state{turn}"])
            if board.is_legal(move):
                board.push(chess.Move.from_uci(move_uci))
            else:
                moves: list[dict] = d["moves"]
                legal_moves: list[dict] = list(filter(lambda m: not m.get("illegal", False), moves))
                assert legal_moves
                legal_moves = sorted(legal_moves, key=BenchmarkChess960.move_to_key)
                worst_legal_move_uci: str = legal_moves[-1]["Move"]
                board.push(chess.Move.from_uci(worst_legal_move_uci))
            state_next: str = board.fen()

            if turn == 0:
                values: list[str] = [model, str(d["iex"]), str(turn + 1), d["label"], completion, str(pred), state_next]
                sql: str = f"INSERT INTO {name} (model, iex, turn, label0, gen0, pred0, state1) VALUES ({', '.join(['?']*len(values))});"
                cursor.execute(sql, values)
            else:
                sql: str = (f"UPDATE {name} SET turn=?, label{turn}=?, gen{turn}=?, pred{turn}=?, state{turn + 1}=? "
                    "WHERE model=? AND iex=?;")
                cursor.execute(sql, [turn + 1, d["label"], completion, pred, state_next, model, d["iex"]])
        connection.commit()

    def get_results(self, model: str):
        cursor: sqlite3.Cursor = get_db()[1]

        nturns: int = self.nturns()
        data: list[dict] = self.get_input_data(model, nturns)
        labels_preds: list[str] = [f"label{i}, pred{i}" for i in range(nturns)]
        sql: str = (f"SELECT {', '.join(labels_preds)} FROM {self.database_name()} "
            f"WHERE model = ? AND iex < ? AND turn = ? ORDER BY iex;")
        query = cursor.execute(sql, [model, len(data), nturns])
        labels = []
        pred = []
        for q in query:
            for i in range(nturns):
                labels.append(q[2*i + 0])
                pred.append(q[2*i + 1])
        return labels, pred


benchmarks: dict[tuple[str, str], Benchmark] = dict()


def get_benchmark(dataset: str, prompt_type: str, turn: int) -> Benchmark:
    key: tuple = (dataset, prompt_type, turn)
    if key not in benchmarks:
        if dataset in ["gsm8k_test"]:
            assert turn == 0
            benchmark = BenchmarkMath(dataset, prompt_type)
        elif dataset in ["gpqa_main", "mmlu_test", "mmlu_pro_test"]:
            assert turn == 0
            benchmark = BenchmarkMultipleChoice(dataset, prompt_type)
        elif dataset == "chess960":
            benchmark = BenchmarkChess960(prompt_type, turn)
        else:
            assert False
        benchmarks[key] = benchmark
    return benchmarks[key]
