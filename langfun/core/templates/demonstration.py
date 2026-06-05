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
"""Demonstrations."""

from typing import Annotated

import langfun.core as lf
from langfun.core.templates.completion import Completion


@lf.use_init_args(['examples', 'description'])
class Demonstration(lf.LangFunc):
  """Base class for demonstration.

  {{ description }}
  {% for example in examples %}
    Example {{loop.index}}:
  {{ example.render().text | indent(4, True) }}
  {% endfor %}
  """

  examples: Annotated[
      list[Completion],
      (
          'A list of completions as few-shot examples.'
      )
  ]

  description: Annotated[
      lf.LangFunc,
      (
          'A leading description of the demonstrations.'
      )
  ] = lf.LangFunc('Here are the examples for demonstrating this:')

  def __bool__(self):
    return bool(self.examples)

  def __call__(self, *args, **kwargs):
    raise ValueError(f'{self.__class__.__name__} is not callable.')
