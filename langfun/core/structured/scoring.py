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
"""Scoring the output objects based on their inputs."""

from typing import Any, Type, Union

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import prompting
from langfun.core.structured import schema as schema_lib
import pyglove as pg


def score(
    prompt: Union[str, pg.Symbolic],
    completions: list[str | pg.Symbolic],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ] = None,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    **kwargs,
) -> list[float]:
  """Scores the outputs based on the prompt."""
  if not completions:
    raise ValueError('`completions` must not be empty.')

  if schema is None:
    for c in completions:
      if schema is None:
        schema = type(c)
      elif schema is not type(c):
        raise ValueError(
            '`schema` cannot be inferred from completions of different types: '
            f'{[type(c) for c in completions]}.'
        )

  input_message = prompting.query(
      prompt,
      schema,
      examples=examples,
      protocol=protocol,
      skip_lm=True,
      returns_message=True,
      **kwargs,
  )
  if lm is None:
    lm_override = lf.get_contextual_override('lm')
    if lm_override is None:
      raise ValueError('`lm` must be specified or provided from `lf.context`.')
    lm = lm_override.value

  results = lm.score(
      input_message,
      [
          mapping.MappingExample.value_repr(
              c, protocol=protocol, compact=False, verbose=False
          )
          for c in completions
      ],
  )
  return [r.score for r in results]
