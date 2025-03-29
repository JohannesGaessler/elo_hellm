#!/usr/bin/env python3

from typing import List

import datasets
import requests
from tqdm import tqdm

# SERVER_ADDRESS = "http://johannes-romed82t-00:8357"
SERVER_ADDRESS = "http://localhost:8080"

response = requests.get(f"{SERVER_ADDRESS}/health")
assert response.status_code == 200

mmlu = datasets.load_dataset("cais/mmlu", "all")

TEMPLATE_MMLU = """Question: {question}

{choices_block}

Solution: The correct answer is ("""

LETTERS = ["a", "b", "c", "d"]
GRAMMAR = "root ::= [abcd]"
# GRAMMAR = "root ::= [abc]"
# GRAMMAR = "root ::= [ABC]"
# GRAMMAR = "root ::= \"a\""

jsonl_lines: List[str] = []

for i, ex in tqdm(list(enumerate(mmlu["test"]))):
    question: str = ex["question"]
    choices: List[str] = ex["choices"]
    answer: int = ex["answer"]

    assert type(question) is str
    assert type(choices) is list
    assert type(answer) is int

    choices = [f"({LETTERS[i]}): {choice_i}" for i, choice_i in enumerate(choices)]
    choices_block: str = "\n".join(choices)

    prompt: str = TEMPLATE_MMLU.format(question=question, choices_block=choices_block)

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
    jsonl_lines.append(f"{response.text}\n")

with open("results.jsonl", "w") as f:
    f.writelines(jsonl_lines)
