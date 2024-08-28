<div align="center">
<img src="https://raw.githubusercontent.com/google/langfun/main/docs/_static/logo.svg" width="520px" alt="logo"></img>
</div>

# Langfun

[![PyPI version](https://badge.fury.io/py/langfun.svg)](https://badge.fury.io/py/langfun)
[![codecov](https://codecov.io/gh/google/langfun/branch/main/graph/badge.svg)](https://codecov.io/gh/google/langfun)
![pytest](https://github.com/google/langfun/actions/workflows/ci.yaml/badge.svg)

[**Installation**](#install) | [**Getting started**](#hello-langfun) | [**Tutorial**](https://colab.research.google.com/github/google/langfun/blob/main/docs/notebooks/langfun101.ipynb)

## Introduction

Langfun is a [PyGlove](https://github.com/google/pyglove) powered library that
aims to *make language models (LM) fun to work with*. Its central principle is
to enable seamless integration between natural language and programming by
treating language as functions. Through the introduction of *Object-Oriented Prompting*, 
Langfun empowers users to prompt LLMs using objects and types, offering enhanced
control and simplifying agent development.

To unlock the magic of Langfun, you can start with
[Langfun 101](https://colab.research.google.com/github/google/langfun/blob/main/docs/notebooks/langfun101.ipynb). Notably, Langfun is compatible with popular LLMs such as Gemini, GPT,
Claude, all without the need for additional fine-tuning.

## Why Langfun?

Langfun is *powerful and scalable*:

*   Seamless integration between natural language and computer programs.
*   Modular prompts, which allows a natural blend of texts and modalities;
*   Efficient for both request-based workflows and batch jobs;
*   A powerful eval framework that thrives dimension explosions.

Langfun is *simple and elegant*:

*   An intuitive programming model, graspable in 5 minutes;
*   Plug-and-play into any Python codebase, making an immediate difference;
*   Comprehensive LLMs under a unified API: Gemini, GPT, Claude, Llama3, and more.
*   Designed for agile developement: offering intellisense, easy debugging, with minimal overhead;

## Hello, Langfun

```python
import langfun as lf
import pyglove as pg

from IPython import display

class Item(pg.Object):
  name: str
  color: str

class ImageDescription(pg.Object):
  items: list[Item]

image = lf.Image.from_uri('https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Solar_system.jpg/1646px-Solar_system.jpg')
display.display(image)

desc = lf.query(
    'Describe objects in {{my_image}} from top to bottom.',
    ImageDescription,
    lm=lf.llms.Gpt4o(api_key='<your-openai-api-key>'),
    my_image=image,
)
print(desc)
```
*Output:*

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Solar_system.jpg/1646px-Solar_system.jpg" width="520px" alt="my_image"></img>

```
ImageDescription(
  items = [
    0 : Item(
      name = 'Mercury',
      color = 'Gray'
    ),
    1 : Item(
      name = 'Venus',
      color = 'Yellow'
    ),
    2 : Item(
      name = 'Earth',
      color = 'Blue and white'
    ),
    3 : Item(
      name = 'Moon',
      color = 'Gray'
    ),
    4 : Item(
      name = 'Mars',
      color = 'Red'
    ),
    5 : Item(
      name = 'Jupiter',
      color = 'Brown and white'
    ),
    6 : Item(
      name = 'Saturn',
      color = 'Yellowish-brown with rings'
    ),
    7 : Item(
      name = 'Uranus',
      color = 'Light blue'
    ),
    8 : Item(
      name = 'Neptune',
      color = 'Dark blue'
    )
  ]
)
```
See [Langfun 101](https://colab.research.google.com/github/google/langfun/blob/main/docs/notebooks/langfun101.ipynb) for more examples.

## Install

Langfun offers a range of features through [Extras](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-extras), allowing users to install only what they need. The minimal installation of Langfun requires only [PyGlove](https://github.com/google/pyglove), [Jinja2](https://github.com/pallets/jinja/), and [requests](https://github.com/psf/requests). To install Langfun with its minimal dependencies, use:

```
pip install langfun
```

For a complete installation with all dependencies, use:

```
pip install langfun[all]
```

To install a nightly build, include the `--pre` flag, like this:

```
pip install langfun[all] --pre
```

If you want to customize your installation, you can select specific features using package names like `langfun[X1, X2, ..., Xn]`, where `Xi` corresponds to a tag from the list below:

| Tag                 |  Description                             |
| ------------------- | ---------------------------------------- |
| all                 | All Langfun features.                    |
| llm                 | All supported LLMs.                      |
| llm-google          | All supported Google-powered LLMs.       |
| llm-google-vertexai | LLMs powered by Google Cloud VertexAI    |
| llm-google-genai    | LLMs powered by Google Generative AI API |
| llm-openai          | LLMs powered by OpenAI                   |
| mime                | All MIME supports.                       |
| mime-auto           | Automatic MIME type detection.           |
| mime-docx           | DocX format support.                     |
| mime-pil            | Image support for PIL.                   |
| mime-xlsx           | XlsX format support.                     |
| ui                  | UI enhancements                          |


For example, to install a nightly build that includes Google-powered LLMs, full modality support, and UI enhancements, use:
```
pip install langfun[llm-google,mime,ui] --pre
```

*Disclaimer: this is not an officially supported Google product.*
