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
"""Sequential runner."""

from typing import Any, Callable, Iterator
from langfun.core.eval.v2.runners import base


class SequentialRunner(base.RunnerBase):
  """A runner that executes evaluations and examples sequentially.

  The sequential runner executes all evaluations and their examples in the
  calling thread. Background tasks are also run sequentially, which makes it
  easier to debug as exceptions from background tasks will be raised
  immediately.
  """

  NAME = 'sequential'

  def background_run(
      self, func: Callable[..., Any], *args: Any, **kwargs: Any
  ) -> None:
    """Runs the function with the IO pool."""
    func(*args, **kwargs)

  def _run(self, evaluations: list[base.Evaluation]) -> None:
    """Runs the experiment in sequence."""
    for e in evaluations:
      self.run_evaluation(e)

  def _evaluate_items(
      self, evaluation: base.Evaluation, items: Iterator[base.Example]
  ) -> None:
    """Runs the evaluation items in sequence."""
    for item in items:
      self.evaluate_item(evaluation, item)
