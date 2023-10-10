<div align="center">
<img src="https://raw.githubusercontent.com/google/langfun/main/docs/_static/logo.svg" width="520px" alt="logo"></img>
</div>

# Langfun

[![PyPI version](https://badge.fury.io/py/langfun.svg)](https://badge.fury.io/py/langfun)
[![codecov](https://codecov.io/gh/google/langfun/branch/main/graph/badge.svg)](https://codecov.io/gh/google/langfun)
![pytest](https://github.com/google/langfun/actions/workflows/ci.yaml/badge.svg)

[**Installation**](#install) | [**Getting started**](#hello-world)

## What is Langfun

Langfun is a Python library that aims to make language models (LM) fun
to work with. Its design enables a programming model that flows naturally,
resembling the human thought process. It emphasizes the reuse and combination of
language pieces to form prompts, thereby accelerating innovation. In contrast to
other LM frameworks, which feed program-generated data into the LM, langfun
takes a distinct approach: It starts with natural language, allowing for
seamless interactions between language and program logic, and concludes with
natural language and optional structured output. Consequently, langfun can
aptly be described as Language as functions, capturing the core of its
methodology.

## Install

```
pip install langfun
```

Or install nightly build with:

```
pip install langfun --pre
```

## Hello World

```python
import langfun as lf
import pyglove as pg

question = (
    'Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and '
    'bakes muffins for her friends every day with four. She sells the remainder at the '
    'farmers\' market daily for $2 per fresh duck egg. '
    'How much in dollars does she make every day at the farmers\' market?')

class Step(pg.Object):
  description: str
  step_output: float

class Solution(pg.Object):
  steps: list[Step]
  final_answer: int

r = lf.query(question, Solution, lm=lf.llms.Gpt35())
print(r)
```
Output:
```
Solution(
  steps = [
    0 : Step(
      description = 'Janet has 16 eggs - 3 eggs for breakfast = 13 eggs left',
      step_output = 13.0
    ),
    1 : Step(
      description = 'Janet has 13 eggs - 4 eggs for muffins = 9 eggs left',
      step_output = 9.0
    ),
    2 : Step(
      description = 'Janet makes 9 eggs * $2 per egg = $18 at the farmers market',
      step_output = 18.0
    )
  ],
  final_answer = 18
)
```

*Disclaimer: this is not an officially supported Google product.*
