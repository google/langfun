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
    prompt: Union[str, pg.Symbolic] | list[str | pg.Symbolic],
    completions: list[str | pg.Symbolic],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ] = None,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    return_scoring_results: bool = False,
    **kwargs,
) -> list[float] | list[lf.LMScoringResult]:
  """Scores the outputs based on the prompt.

  Examples:
    ```
    # Example 1: Scoring text output based on the user prompt.
    scores = lf.score('{{x}} + {{y}} =', ['1', '2', '3'], lm=lm, x=1, y=2)
    assert len(scores) == 3

    # Example 2: Scoring int output based on the formulated OOP prompt.
    scores = lf.score('1 + 1 =', [1, 2, 3], lm=lm)
    assert len(scores) == 3

    class Answer(pg.Object):
      result: int

    # Example 3: Scoring object output based on the formulated OOP prompt.
    scores = lf.score('1 + 1 =', [Answer(1), Answer(2), Answer(3)], lm=lm)
    assert len(scores) == 3

    # Example 4: Scoring object field value based on the formulated OOP prompt
    # and the generated tokens before the first `pg.oneof`.
    scores = lf.score('1 + 1 =', [Answer(pg.oneof([1, 2, 3]))], lm=lm)
    assert len(scores) == 3

    # Example 5: Scoring multiple prompt/completion pairs.
    scores = lf.score(
        ['1 + 1=', '2 + 3='],
        ['2', '4'],
        lm=lm
    )
    assert len(scores) == 2
    ```

  Args:
    prompt: The prompt(s) based on which each completion will be scored.
    completions: A list of strings or symbolic objects as the output.
    schema: The schema as the output type. If None, it will be inferred from
      the completions.
    lm: The language model used for scoring.
    examples: Fewshot exemplars used together with the prompt in getting the
      completions.
    protocol: The protocol for formulating the prompt based on objects.
    return_scoring_results: If True, returns a list of `lf.LMScoringResult`,
      otherwise returns a list of floats as the scores of each completion.
    **kwargs: Keyword arguments that are referred by the prompt.

  Returns:
    A list of floats or `lf.LMScoringResult` as the score of each completion.
  """
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

  if isinstance(prompt, list):
    prompts = []
    for p in prompt:
      prompts.append(
          prompting.query_prompt(
              p,
              schema,
              examples=examples,
              protocol=protocol,
              **kwargs,
          )
      )
    input_message = prompts
  else:
    input_message = prompting.query_prompt(
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

  completion_reprs = []
  for c in completions:
    if isinstance(c, mapping.MappingError):
      completion_reprs.append(c.lm_response)
    else:
      rep = mapping.MappingExample.value_repr(
          c, protocol=protocol, compact=False, verbose=False
      )

      # NOTE(daiyip): supporting scenario of scoring object field with
      # `pg.oneof`.
      oneof_pos = rep.find('OneOf(')
      if oneof_pos == -1:
        completion_reprs.append(rep)
      else:
        assert protocol == 'python', protocol
        if isinstance(input_message, list):
          raise ValueError(
              'Scoring on object fields using `pg.oneof` must share the '
              f'same prompt. Encountered: {prompt}'
          )
        input_message.text += '\n' + rep[:oneof_pos]
        oneof = _get_first_oneof(c)
        for v in oneof.candidates:
          completion_reprs.append(
              pg.format(
                  v,
                  python_format=True,
                  compact=False,
                  verbose=False,
                  root_indent=oneof.sym_path.depth
              )
          )

  results = lm.score(
      input_message,
      completion_reprs,
  )
  if return_scoring_results:
    return results
  return [r.score for r in results]


def _get_first_oneof(value: Any) -> pg.hyper.OneOf:
  """Gets the first pg.oneof from a symbolic object."""
  oneofs = []
  def select_oneofs(k, v, p):
    del k, p
    if isinstance(v, pg.hyper.OneOf):
      oneofs.append(v)
      return pg.TraverseAction.CONTINUE
    return pg.TraverseAction.ENTER
  pg.traverse(value, select_oneofs)
  assert oneofs
  return oneofs[0]
