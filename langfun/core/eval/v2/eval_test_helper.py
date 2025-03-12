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

  def process(self, example):
    v = example.input
    if v.x == 5:
      raise ValueError('x should not be 5')
    return structured.query(
        '{{x}} + {{y}} = ?', int, lm=self.lm, x=v.x, y=v.y,
        metadata_x=v.x, metadata_y=v.y
    )


class BadJsonConvertible(pg.Object):

  def to_json(self, *args, **kwargs):
    raise ValueError('Cannot convert to JSON.')


class TestEvaluationWithExampleCheckpointingError(TestEvaluation):
  """Test evaluation class with bad example checkpointing."""
  inputs = test_inputs()
  metrics = [metrics_lib.Match()]

  def process(self, example):
    return 1, dict(
        x=BadJsonConvertible()
    )


class BadHtmlConvertible(pg.Object, pg.views.HtmlTreeView.Extension):

  def _html_tree_view(self, *args, **kwargs):
    raise ValueError('Cannot render HTML.')


class TestEvaluationWithExampleHtmlGenerationError(Evaluation):
  """Test evaluation class with bad example HTML generation."""
  inputs = test_inputs()
  metrics = [metrics_lib.Match()]

  def process(self, example):
    return 1, dict(
        x=BadHtmlConvertible()
    )


class TestEvaluationWithIndexHtmlGenerationError(TestEvaluation):
  """Test evaluation class with bad index HTML generation."""

  def _html_tree_view(self, *args, **kwargs):
    raise ValueError('Cannot render HTML.')


def test_experiment():
  """Returns a test experiment."""
  return Suite([
      TestEvaluation(lm=TestLLM(offset=0)),
      TestEvaluation(lm=TestLLM(offset=pg.oneof(range(5)))),
  ])


def test_experiment_with_example_checkpointing_error():
  """Returns a test experiment with example checkpointing error."""
  return TestEvaluationWithExampleCheckpointingError()


def test_experiment_with_example_html_generation_error():
  """Returns a test experiment with bad example HTML."""
  return TestEvaluationWithExampleHtmlGenerationError()


def test_experiment_with_index_html_generation_error():
  """Returns a test experiment with bad index HTML."""
  return TestEvaluationWithIndexHtmlGenerationError()
