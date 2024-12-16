# Copyright 2023 The Langfun Authors
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
"""Tokenize the prompt for `lf.query`."""

from typing import Any, Type, Union

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import querying
from langfun.core.structured import schema as schema_lib
import pyglove as pg


def tokenize(
    prompt: Union[str, pg.Symbolic] | list[str | pg.Symbolic],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ] = None,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    **kwargs,
) -> list[tuple[str | bytes, int]]:
  """Tokenize the prompt for `lf.query`.

  Args:
    prompt: The prompt(s) based on which each completion will be scored.
    schema: The schema as the output type. If None, it will be inferred from
      the completions.
    lm: The language model used for scoring.
    examples: Fewshot exemplars used together with the prompt in getting the
      completions.
    protocol: The protocol for formulating the prompt based on objects.
    **kwargs: Keyword arguments that are referred by the prompt.

  Returns:
    A list of (text, token_id) tuples.
  """
  input_message = querying.query_prompt(
      prompt,
      schema,
      examples=examples,
      protocol=protocol,
      **kwargs,
  )
  if lm is None:
    lm_override = lf.get_contextual_override('lm')
    if lm_override is None:
      raise ValueError('`lm` must be specified or provided from `lf.context`.')
    lm = lm_override.value

  return lm.tokenize(input_message)
