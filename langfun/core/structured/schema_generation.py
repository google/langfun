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
"""LLM-based class generation."""

import typing
from typing import Any, Type
import langfun.core as lf
from langfun.core.coding.python import correction
from langfun.core.structured import mapping
import pyglove as pg


class GenerateClass(mapping.Mapping):
  """Python class generation."""

  input_title = 'GENERATION_CONTEXT'
  context_title = 'CLASS_NAME'
  output_title = 'OUTPUT_CLASS'
  permission = pg.coding.CodePermission.ALL

  preamble: lf.Template = lf.Template("""
      Help generate a class based on the last {{ context_title }} and {{ input_title }}.

      Instructions:
      - Use `Object` as the base class for all generated classes
      - Create auxillary classes for composition if needed.
      - Use Python type annotation for declaraing fields:
        (e.g. bool, str, int, float, Optional[str], List[int], Union[str, int])
      - Do not use types that need import.
      - Avoid self-referential types. e.g:
        ```
        class Node(Object):
          children: list[Node]
        ```
      - Do not generate methods.
      """)

  def parse_result(self, lm_output: lf.Message) -> Type[Any]:
    output_vars, final_code = correction.run_with_correction(
        lm_output.text,
        global_vars=self.allowed_annotation_types,
        sandbox=False,
        max_attempts=self.autofix,
        lm=self.autofix_lm,
        returns_code=True,
        outputs_intermediate=True,
    )
    class_name = self.context
    cls = output_vars.get(class_name, None)
    if cls is None:
      raise pg.coding.CodeError(
          final_code,
          TypeError(f'Class {class_name} is absent from LLM output.'),
      )
    return cls

  @property
  def allowed_annotation_types(self):
    return dict(
        pg=pg,
        Any=typing.Any,
        Object=pg.Object,
        List=typing.List,
        Dict=typing.Tuple,
        Tuple=typing.Tuple,
        Sequence=typing.Sequence,
        Optional=typing.Optional,
        Union=typing.Union,
    )


def generate_class(
    name: str,
    prompt: str | pg.Symbolic,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    returns_message: bool = False,
    skip_lm: bool = False,
    **kwargs,
) -> Type[Any] | lf.Message:
  """Generate a class with specified name based on the prompt.

  Example:
    ```
    trip_cls = lf.classgen(
        'Trip',
        'A trip plan to visit {{ city }}, city='San Francisco',
        lm=lf.llms.GeminiPro()
    )
    ```

  Args:
    name: Class name to be generated.
    prompt: A str (may contain {{}} as template) as natural language input, or a
      `pg.Symbolic` object as structured input as prompt to LLM.
    lm: The language model to use. If not specified, the language model from
      `lf.context` context manager will be used.
    examples: An optional list of fewshot examples for helping class generation.
      If None, a default single shot example will be used. Use
      `lf.structured.classgen_example` to generate example.
    returns_message: If True, returns `lf.Message` as the output, instead of
      returning the structured `message.result`.
    skip_lm: If True, returns the rendered prompt as a UserMessage object.
      otherwise return the LLM response based on the rendered prompt.
    **kwargs: Template variables passed to `prompt` and keyword arguments passed
      to `lf.structured.GenerateClass`.

  Returns:
    Generated class.

  Raises:
    CodeError: if generation failed.
  """
  if isinstance(prompt, str):
    prompt = lf.Template(prompt, **kwargs)
  elif isinstance(prompt, lf.Template):
    prompt = prompt.rebind(**kwargs, raise_on_no_change=False)

  if isinstance(prompt, lf.Template):
    prompt = prompt.render(lm=lm)

  call_kwargs = dict(skip_lm=skip_lm)
  if lm is not None:
    call_kwargs['lm'] = lm
  message = GenerateClass(
      input=prompt,
      context=name,
      examples=examples or default_classgen_examples(),
      **kwargs,
  )(**call_kwargs)
  return message if returns_message else message.result


def classgen_example(
    prompt: str | pg.Symbolic, cls: Type[Any]
) -> mapping.MappingExample:
  """Creates a class generation example."""
  if isinstance(prompt, lf.Template):
    prompt = prompt.render()
  return mapping.MappingExample(
      input=prompt,
      context=cls.__name__,
      output=cls,
  )


def default_classgen_examples() -> list[mapping.MappingExample]:
  """Default examples for class generation."""

  class Step(pg.Object):
    description: str
    output: float

  class Solution(pg.Object):
    steps: list[Step]  # pytype: disable=invalid-annotation
    result: float

  return [
      classgen_example(
          'How to evaluate an arithmetic expression?',
          Solution,
      )
  ]
