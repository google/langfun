# Copyright 2024 The Langfun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experiment patching for Langfun evaluations."""

import inspect
from typing import Union
import langfun.core as lf
from langfun.core import llms as lf_llms
from langfun.core.eval import base
import pyglove as pg


#
# Program-based patchers.
#


def patch_member(cls, key, value, parent_key: str | None = None):
  """Patches a member of a class."""

  def _rebind_fn(k, v, p):
    if (
        isinstance(p, cls)
        and k.key == key
        and (parent_key is None or (p and p.sym_path.key == parent_key))
    ):
      if inspect.isfunction(value):
        return value(k, v, p)
      return value
    return v

  return _rebind_fn


def patch_lm(lm: Union[lf.LanguageModel, pg.hyper.OneOf]):  # pylint: disable=redefined-outer-name
  """Patches the LLM of evaluations."""
  return patch_member(base.Evaluable, "lm", lm)


def patch_parsing_lm(lm: Union[lf.LanguageModel, pg.hyper.OneOf]):  # pylint: disable=redefined-outer-name
  """Patches the parsing LLM of evaluations."""
  return patch_member(base.Evaluable, "parsing_lm", lm)


def patch_schema_fn(schema_fn: Union[pg.Functor, pg.hyper.OneOf]):
  """Patches the schema_fn of evaluations."""
  return patch_member(base.Evaluable, "schema_fn", schema_fn)


def patch_prompt(prompt: Union[str, lf.Template, pg.hyper.OneOf]):
  """Patches the prompt of evaluations."""
  return patch_member(base.Evaluable, "prompt", prompt)


def patch_inputs(inputs: Union[pg.Functor, pg.hyper.OneOf]):
  """Patches the inputs used in evaluations."""
  return patch_member(base.Evaluable, "inputs", inputs)


def patch_additional_args(**kwargs):
  """Patches additional_args."""

  def value_fn(k, unused_v, p):
    # We infer the symbolic value for the old args, as it might be a
    # contextual attribute referring to its containing object.
    old_args = p.sym_inferred(k.key)
    if old_args:
      old_args = dict(old_args)
      old_args.update(kwargs)
      return old_args
    return kwargs

  return patch_member(base.Evaluable, "additional_args", value_fn)


#
# String-based patching.
#

_NAMED_MODELS = {
    # GPT models.
    "gpt35turbo": lf_llms.Gpt35Turbo,
    "gpt35turbo16k": lf_llms.Gpt35Turbo16K,
    "gpt4": lf_llms.Gpt4,
    "gpt4turbo": lf_llms.Gpt4Turbo,
    # Anthropic models.
    "haiku": lf_llms.Claude3Haiku,
    "claude3haiku": lf_llms.Claude3Haiku,
    "opus": lf_llms.Claude3Opus,
    "claude3opus": lf_llms.Claude3Opus,
    "sonnet": lf_llms.Claude3Sonnet,
    "claude3sonnet": lf_llms.Claude3Opus,
}


def model_by_name(name: str) -> lf.LanguageModel:
  """Gets model by name."""
  name = name.strip().lower()
  if name in _NAMED_MODELS:
    return _NAMED_MODELS[name]()
  raise ValueError(f"Unknown model name: {name}")


@pg.patcher(auto_typing=True)
def lm(unused_eval, models: list[str]):
  """Patch the LM used for benchmarking."""
  return patch_lm(pg.oneof([model_by_name(name) for name in models]))


@pg.patcher(auto_typing=True)
def temperature(unused_eval, value: float):
  """Patch the temperature used for benchmarking."""
  return patch_member(lf.LMSamplingOptions, "temperature", value)


@pg.patcher(auto_typing=True)
def max_tokens(unused_eval, value: int | None):
  """Patch the temperature used for benchmarking."""
  return patch_member(lf.LMSamplingOptions, "max_tokens", value)
