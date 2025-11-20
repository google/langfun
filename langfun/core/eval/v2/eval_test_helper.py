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

import threading
import time

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

  __test__ = False

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

  __test__ = False

  def process(self, example):
    v = example.input
    if v.x == 5:
      raise ValueError('x should not be 5')
    return structured.query(
        '{{x}} + {{y}} = ?', int, lm=self.lm, x=v.x, y=v.y,
        metadata_x=v.x, metadata_y=v.y
    )


class BadJsonConvertible(pg.Object):

  def sym_jsonify(self, *args, **kwargs):
    raise ValueError('Cannot convert to JSON.')


class TestEvaluationWithExampleCheckpointingError(TestEvaluation):
  """Test evaluation class with bad example checkpointing."""
  inputs = test_inputs()
  metrics = [metrics_lib.Match()]

  __test__ = False

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

  __test__ = False

  def process(self, example):
    return 1, dict(
        x=BadHtmlConvertible()
    )


class TestEvaluationWithIndexHtmlGenerationError(TestEvaluation):
  """Test evaluation class with bad index HTML generation."""

  __test__ = False

  def _html_tree_view(self, *args, **kwargs):
    raise ValueError('Cannot render HTML.')


def test_evaluation(offset: int | pg.hyper.OneOf = 0):
  """Returns a test evaluation."""
  return TestEvaluation(lm=TestLLM(offset=offset))


def test_experiment():
  """Returns a test experiment."""
  return Suite([
      test_evaluation(),
      test_evaluation(pg.oneof(range(5))),
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


class TestPlugin(experiment_lib.Plugin):
  """Plugin for testing."""

  started_experiments: list[experiment_lib.Experiment] = []
  completed_experiments: list[experiment_lib.Experiment] = []
  skipped_experiments: list[experiment_lib.Experiment] = []
  started_example_ids: list[int] = []
  completed_example_ids: list[int] = []
  start_time: float | None = None
  complete_time: float | None = None

  __test__ = False

  def _on_bound(self):
    super()._on_bound()
    self._lock = threading.Lock()

  def on_run_start(
      self,
      runner: experiment_lib.Runner,
      root: experiment_lib.Experiment
  ) -> None:
    del root
    with pg.notify_on_change(False), pg.allow_writable_accessors(True):
      self.start_time = time.time()

  def on_run_complete(
      self,
      runner: experiment_lib.Runner,
      root: experiment_lib.Experiment
  ) -> None:
    del root
    with pg.notify_on_change(False), pg.allow_writable_accessors(True):
      self.complete_time = time.time()

  def on_experiment_start(
      self,
      runner: experiment_lib.Runner,
      experiment: experiment_lib.Experiment
  ) -> None:
    del runner
    with pg.notify_on_change(False), self._lock:
      self.started_experiments.append(pg.Ref(experiment))

  def on_experiment_skipped(
      self,
      runner: experiment_lib.Runner,
      experiment: experiment_lib.Experiment
  ) -> None:
    del runner
    with pg.notify_on_change(False), self._lock:
      self.skipped_experiments.append(pg.Ref(experiment))

  def on_experiment_complete(
      self,
      runner: experiment_lib.Runner,
      experiment: experiment_lib.Experiment
  ) -> None:
    del runner
    with pg.notify_on_change(False), self._lock:
      self.completed_experiments.append(pg.Ref(experiment))

  def on_example_start(
      self,
      runner: experiment_lib.Runner,
      experiment: experiment_lib.Experiment,
      example: Example
  ) -> None:
    del runner, experiment
    with pg.notify_on_change(False), self._lock:
      self.started_example_ids.append(example.id)

  def on_example_complete(
      self,
      runner: experiment_lib.Runner,
      experiment: experiment_lib.Experiment,
      example: Example
  ) -> None:
    del runner, experiment
    with pg.notify_on_change(False), self._lock:
      self.completed_example_ids.append(example.id)
