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
    prompt: Union[str, pg.Symbolic, list[str | pg.Symbolic]],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ] = None,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    protocol: str = 'python',
    **kwargs,
) -> list[tuple[str | bytes, int]]:
  """Renders a prompt and tokenizes it using a language model.

  `lf.tokenize` first renders a prompt based on the provided `prompt`,
  `schema`, and `examples`, similar to `lf.query`, and then uses the
  specified language model (`lm`) to tokenize the resulting message.
  This is useful for understanding how a prompt is seen by the model or
  for estimating token counts before sending requests.

  **Example:**

  ```python
  import langfun as lf
  tokens = lf.tokenize('Hello world!', lm=lf.llms.Gpt4())
  print(tokens)
  # Output might look like: [('Hello', 15339), (' world', 1917), ('!', 0)]
  ```

  Args:
    prompt: The prompt to render and tokenize. Can be a string, `pg.Symbolic`,
      or `lf.Template`.
    schema: The schema for formatting the prompt, if `prompt` is structured or
      if schema-based formatting is needed.
    lm: The language model to use for tokenization.
    examples: Few-shot examples to include in the rendered prompt.
    protocol: The protocol for formulating the prompt based on objects.
    **kwargs: Keyword arguments that are referred by the prompt.

  Returns:
    A list of (token_str, token_id) tuples representing the tokenized prompt.
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


async def atokenize(
    prompt: Union[str, pg.Symbolic] | list[str | pg.Symbolic],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ] = None,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    protocol: str = 'python',
    **kwargs,
) -> list[tuple[str | bytes, int]]:
  """Async version of `lf.tokenize`."""
  # TODO(daiyip): implement native async tokenization.
  return await lf.invoke_async(
      tokenize,
      prompt,
      schema,
      lm=lm,
      examples=examples,
      protocol=protocol,
      **kwargs,
  )
