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
"""Helper classes and functions for evaluation tests."""

from langfun.core import language_model
from langfun.core import llms
from langfun.core import message as message_lib
from langfun.core import structured

from langfun.core.eval.v2 import evaluation as evaluation_lib
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2 import metrics as metrics_lib

import pyglove as pg

Example = example_lib.Example
Suite = experiment_lib.Suite
Evaluation = evaluation_lib.Evaluation
RunId = experiment_lib.RunId
Run = experiment_lib.Run


@pg.functor()
def test_inputs(num_examples: int | None = 10):
  if num_examples is None:
    num_examples = 20
  return [
      pg.Dict(x=i, y=i ** 2, groundtruth=i + i ** 2)
      for i in range(num_examples)
  ]


class TestLLM(llms.Fake):
  """Test language model."""

  offset: int = 0

  def _response_from(self, prompt: message_lib.Message) -> message_lib.Message:
    return message_lib.AIMessage(
        str(prompt.metadata.x + prompt.metadata.y + self.offset)
    )

  @property
  def resource_id(self) -> str:
    return f'test_llm:{self.offset}'


class TestEvaluation(Evaluation):
  """Test evaluation class."""
  inputs = test_inputs()
  metrics = [metrics_lib.Match()]
  lm: language_model.LanguageModel = TestLLM()

  def process(self, v):
    if v.x == 5:
      raise ValueError('x should not be 5')
    return structured.query(
        '{{x}} + {{y}} = ?', int, lm=self.lm, x=v.x, y=v.y,
        metadata_x=v.x, metadata_y=v.y
    )


def test_experiment():
  """Returns a test experiment."""
  return Suite([
      TestEvaluation(lm=TestLLM(offset=0)),
      TestEvaluation(lm=TestLLM(offset=pg.oneof(range(5)))),
  ])
