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
"""Reporting evaluation results."""

import time
from typing import Annotated

from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib

Runner = experiment_lib.Runner
Experiment = experiment_lib.Experiment
Example = example_lib.Example


_SUMMARY_FILE = 'summary.html'
_EVALULATION_DETAIL_FILE = 'index.html'


class HtmlReporter(experiment_lib.Plugin):
  """Plugin for periodically generating HTML reports for the experiment."""

  summary_interval: Annotated[
      int,
      'The interval of writing summary in seconds.'
  ] = 60

  experiment_report_interval: Annotated[
      int,
      'The interval of writing report for inidividual experiments in seconds.'
  ] = 60

  def _on_bound(self):
    super()._on_bound()
    self._last_summary_time = 0
    self._last_experiment_report_time = {}

  def on_run_start(
      self,
      runner: Runner,
      root: Experiment
  ) -> None:
    self._maybe_update_summary(runner)
    self._last_experiment_report_time = {leaf.id: 0 for leaf in root.leaf_nodes}

  def on_run_complete(
      self,
      runner: Runner,
      root: Experiment
  ) -> None:
    self._maybe_update_summary(runner, force=True)

  def on_experiment_start(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    if experiment.is_leaf:
      self._maybe_update_experiment_html(runner, experiment)

  def on_experiment_complete(
      self, runner: Runner, experiment: Experiment
  ):
    if experiment.is_leaf:
      self._maybe_update_experiment_html(runner, experiment, force=True)

  def on_example_complete(
      self, runner: Runner, experiment: Experiment, example: Example
  ):
    self._save_example_html(runner, experiment, example)
    self._maybe_update_experiment_html(runner, experiment)
    self._maybe_update_summary(runner)

  def _maybe_update_summary(self, runner: Runner, force: bool = False) -> None:
    """Maybe update the summary of current run."""
    run = runner.current_run
    def _summary():
      run.experiment.to_html(
          collapse_level=None,
          extra_flags=dict(
              current_run=run, interactive=False, card_view=True,
          )
      ).save(
          run.output_path_for(run.experiment, _SUMMARY_FILE)
      )

    if force or (time.time() - self._last_summary_time > self.summary_interval):
      runner.background_run(_summary)
      self._last_summary_time = time.time()

  def _maybe_update_experiment_html(
      self, runner: Runner, experiment: Experiment, force: bool = False
  ) -> None:
    def _save():
      html = experiment.to_html(
          collapse_level=None,
          extra_flags=dict(
              current_run=runner.current_run,
              interactive=False,
              card_view=False,
          ),
      )
      html.save(
          runner.current_run.output_path_for(
              experiment, _EVALULATION_DETAIL_FILE
          )
      )
    if force or (
        time.time() - self._last_experiment_report_time[experiment.id]
        > self.experiment_report_interval
    ):
      runner.background_run(_save)
      self._last_experiment_report_time[experiment.id] = time.time()

  def _save_example_html(
      self, runner: Runner, experiment: Experiment, example: Example
  ) -> None:
    """Saves the example."""
    def _save():
      html = example.to_html(
          collapse_level=None,
          enable_summary_tooltip=False,
          extra_flags=dict(
              # For properly rendering the next link.
              num_examples=getattr(experiment, 'num_examples', None)
          ),
      )
      html.save(
          runner.current_run.output_path_for(
              experiment, f'{example.id}.html'
          )
      )
    runner.background_run(_save)
