# Copyright 2025 The Langfun Authors
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
"""Beam-based evaluation runner.

BeamRunner is a runner that uses Apache Beam to run evaluations in parallel.
It is useful for running evaluations with a large number of examples and/or when
each example is costly to evaluate and can be parallelized.

BeamRunner supports plugins as all other runners do, with the following caveats:

1. Checkpointer plugins are ignored, as BeamRunner performs its own per-example
   checkpointing.

2. Per-example plugins are executed in the Beam worker to maximize throughput,
   while all non-per-example plugins are executed in the main process, which
   collects the results from the workers. Since it might be expensive to
   deserialize `Example.metadata` for complex evaluations, the main process
   does not deserialize `Example.metadata` from the workers. If you need to
   to access `Example.metadata` in your plugin, consider make it a per-example
   plugin (which only implements `on_example_start` and/or
   `on_example_complete`)

To use it, simply create a `lf.eval.Suite` or `lf.eval.Evaluation`
and run it with `lf.eval.run(runner='beam')` and passing in an additional
`beam_runner` argument.
"""

import concurrent.futures
import dataclasses
import functools
import hashlib
import os
import random
import threading
import time
from typing import Annotated, Any, Iterator

from langfun.core.eval.v2 import checkpointing
from langfun.core.eval.v2 import evaluation as evaluation_lib
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2.runners import base

import pyglove as pg

try:
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  import apache_beam as beam
  from apache_beam.options import pipeline_options
  # pytype: enable=import-error
  # pylint: enable=g-import-not-at-top
except ImportError:
  beam = None
  pipeline_options = None


if beam is not None:
  class _EvaluateFn(beam.DoFn):
    """Beam DoFn for evaluating examples."""

    def __init__(
        self,
        runner_str: str,
        ckpt_format: str,
        concurrent_startup_delay: tuple[int, int] | None = None,
    ):
      self._runner_str = runner_str
      self._ckpt_format = ckpt_format
      self._concurrent_startup_delay = concurrent_startup_delay

    def setup(self):
      if self._concurrent_startup_delay is not None:
        time.sleep(random.randint(*self._concurrent_startup_delay))
      self._runner = pg.from_json_str(self._runner_str)
      assert isinstance(self._runner, LeafNodeRunner)
      self._runner.setup()
      self._output_dir = self._runner.current_run.output_dir(
          self._runner.current_run.experiment
      )

    def teardown(self):
      assert self._runner is not None
      self._runner.teardown()

    def process(self, example: tuple[int, str]) -> Iterator[str]:
      """Evaluates an example and writes the checkpoint file.

      Args:
        example: A tuple of (example_id, example_json).

      Yields:
        The path to the checkpoint file.
      """
      example_id, example_json = example
      ckpt_file = os.path.join(
          self._output_dir, f'checkpoint_{example_id}.{self._ckpt_format}'
      )
      if pg.io.path_exists(ckpt_file):
        yield ckpt_file

      # Process one example.
      example = self._runner.process(pg.from_json_str(example_json))

      # Perform atomic checkpointing.
      tmp_ckpt_file = os.path.join(
          self._output_dir, f'tmp_checkpoint_{example_id}.{self._ckpt_format}'
      )
      example_json_str = pg.to_json_str(example)
      with pg.io.open_sequence(tmp_ckpt_file, 'w') as f:
        f.add(example_json_str)
      pg.io.rename(tmp_ckpt_file, ckpt_file)

      # Write the MD5 digest of the example so we know if the example has been
      # processed multiple times.
      digest = hashlib.md5(example_json_str.encode()).hexdigest()[:8]
      pg.io.writefile(
          os.path.join(self._output_dir, f'{example_id}.{digest}.md5'),
          digest
      )
      yield ckpt_file

  @dataclasses.dataclass
  class _EvaluationCollector:
    """Per-evaluation component for collecting evaluated examples."""

    runner: base.RunnerBase
    evaluation: evaluation_lib.Evaluation
    example_ids: set[int]
    ckpt_format: str
    max_collector_threads: int = 128

    def __post_init__(self):
      self._monitor_thread = None
      self._example_ids_to_be_collected = set(self.example_ids)
      self._example_ids_being_collected = set()
      self._collector_pool = concurrent.futures.ThreadPoolExecutor(
          max_workers=self.max_collector_threads
      )
      self._error = None

    @functools.cached_property
    def output_dir(self) -> str:
      """Returns the output directory of the evaluation."""
      return self.runner.current_run.output_dir(self.evaluation)

    def start(self):
      self._monitor_thread = threading.Thread(target=self._monitor_loop)
      self._monitor_thread.start()

    def join(self):
      if self._monitor_thread:
        self._monitor_thread.join()
      if self._error is not None:   # pytype: disable=attribute-error
        raise self._error           # pytype: disable=attribute-error

    def _monitor_loop(self):
      """Monitors the checkpoint files and collects evaluated examples."""
      try:
        ckpt_file_pattern = os.path.join(
            self.output_dir, f'checkpoint_*.{self.ckpt_format}'
        )
        while (
            not self._error
            and (self._example_ids_to_be_collected
                 or self._example_ids_being_collected)
        ):
          if self._example_ids_to_be_collected:
            for filepath in pg.io.glob(ckpt_file_pattern):
              example_id = int(
                  os.path.basename(filepath).split('.')[0].split('_')[-1]
              )
              if example_id in self._example_ids_to_be_collected:
                # Remove example ID from the set to avoid duplicate processing.
                self._example_ids_to_be_collected.remove(example_id)
                self._example_ids_being_collected.add(example_id)
                self._collector_pool.submit(
                    self._collect_ckpt, filepath, example_id
                )
                pg.logging.info(
                    '[%s] Collecting example %d from %s...',
                    self.evaluation.id,
                    example_id,
                    filepath,
                )
          time.sleep(1)

        if not self._error:
          self.runner.on_experiment_complete(self.evaluation)
      except BaseException as e:  # pylint: disable=broad-except
        self._error = e
        pg.logging.error(
            '[%s] Collector failed with error: %s',
            self.evaluation.id,
            pg.ErrorInfo.from_exception(e)
        )

    def _collect_ckpt(self, ckpt_filepath: str, example_id: int):
      """Collects examples from a checkpoint file."""
      try:
        loaded_examples = self.evaluation.state.load(
            ckpt_filepath,
            example_input_by_id=self.evaluation.example_input_by_id,
            load_example_metadata=False
        )
        assert len(loaded_examples) == 1, loaded_examples
        example = loaded_examples[0]
      except BaseException as e:  # pylint: disable=broad-except
        error_info = pg.ErrorInfo.from_exception(e)
        pg.logging.error(
            '[%s] Failed to collect example %d: %s',
            self.evaluation.id,
            example_id,
            error_info
        )
        example = example_lib.Example(
            id=example_id,
            input=self.evaluation.example_input_by_id(example_id),
            error=error_info,
        )

      # This will skip processing but still allow metrics to be collected.
      # `process` will never be called for evaluation, thus we do not
      # need to setup/teardown evaluation.
      example = self.evaluation.evaluate(
          example, reevaluate_upon_previous_errors=False
      )
      example.newly_processed = True
      pg.logging.info(
          '[%s] Successfully collected example %d.',
          self.evaluation.id,
          example_id,
      )
      self.runner.on_example_complete(self.evaluation, example)
      self._example_ids_being_collected.remove(example_id)

else:
  _EvaluateFn = None   # pylint: disable=invalid-name
  _EvaluationCollector = None  # pylint: disable=invalid-name


class LeafNodeRunner(base.RunnerBase):
  """A runner that runs in a DoFn worker."""

  NAME = '__beam_leaf_node_runner__'
  progress_tracker = None
  max_background_threads = 0

  def _on_bound(self):
    super()._on_bound()
    for plugin in self.plugins:
      if not plugin.is_per_example():
        raise ValueError(
            'Only per-example plugins are supported in LeafNodeRunner. '
            f'Encountered: {plugin!r}'
        )
    if not isinstance(self.current_run.experiment, evaluation_lib.Evaluation):
      raise ValueError(
          'The experiment must be a leaf evaluation in LeafNodeRunner. '
          f'Encountered: {self.current_run.experiment!r}'
      )

  def setup(self):
    self.current_run.experiment.setup()

  def teardown(self):
    self.current_run.experiment.teardown()

  def process(self, example: example_lib.Example) -> example_lib.Example:
    """Processes one example."""
    for plugin in self.plugins:
      plugin.on_example_start(self, self.current_run.experiment, example)
    example = self.current_run.experiment.evaluate(example)
    for plugin in self.plugins:
      plugin.on_example_complete(self, self.current_run.experiment, example)
    return example

  def _run(self, evaluations: list[evaluation_lib.Evaluation]) -> None:
    """Runs the experiment in sequence."""
    raise NotImplementedError('Not needed in leaf node runner.')

  def _evaluate_items(
      self,
      evaluation: evaluation_lib.Evaluation,
      items: Iterator[example_lib.Example]
  ) -> None:
    """Evaluates the items of an evaluation."""
    raise NotImplementedError('Not needed in leaf node runner.')


class BeamRunner(base.RunnerBase):
  """Beam runner for Langfun evaluations.

  NOTE: This runner depends on Apache Beam, which needs to be installed
  separately.
  """

  NAME = 'beam'

  beam_runner: Annotated[
      Any | None,
      'The beam runner to use. If None, the direct runner will be used.'
  ] = None

  beam_pipeline_options: Annotated[
      dict[str, Any],
      'Beam pipeline options.'
  ] = {}

  ckpt_format: Annotated[
      str,
      'The file extension of the checkpoint files.'
  ] = 'jsonl'

  max_collector_threads: Annotated[
      int,
      'The maximum number of threads to collect examples per evaluation.'
  ] = 128

  concurrent_startup_delay: Annotated[
      tuple[int, int] | None,
      (
          'A range of seconds to delay the initial evaluation of each thread '
          'in the thread pool, helping to prevent a burst in LLM QPS at '
          'startup. If set to None, no delay will be applied.'
      )
  ] = None

  def _on_bound(self):
    if beam is None:
      raise ValueError(
          'Apache Beam is not installed. '
          'Please run `pip install apache-beam` to install beam.'
      )
    if self.current_run.use_cache != 'no':
      raise ValueError(
          'Cache is not supported in BeamRunner. '
          f'Encountered: {self.current_run.use_cache}'
      )
    host_plugins = []
    per_example_plugins = []
    for plugin in self.plugins:
      if isinstance(plugin, checkpointing.Checkpointer):
        pg.logging.warning(
            'Built-in checkpointing is enabled on BeamRunner. '
            f'Ignoring checkpointer: {plugin!r}.'
        )
      elif plugin.is_per_example():
        per_example_plugins.append(pg.Ref(plugin))
      else:
        host_plugins.append(pg.Ref(plugin))

    self.rebind(
        plugins=host_plugins,
        skip_notification=True,
        raise_on_no_change=False
    )
    self._per_example_plugins = per_example_plugins
    super()._on_bound()

  def _run(self, evaluations: list[evaluation_lib.Evaluation]) -> None:
    """Run evaluations using Beam."""
    assert beam is not None
    assert pipeline_options is not None
    collectors = []

    with beam.Pipeline(
        runner=self.beam_runner or beam.runners.DirectRunner(),
        options=pipeline_options.PipelineOptions(**self.beam_pipeline_options)
    ) as pipeline:
      for i, evaluation in enumerate(evaluations):
        example_ids = self.current_run.example_ids
        if example_ids is None:
          example_ids = range(1, evaluation.num_examples + 1)
        inputs = [
            example_lib.Example(id=i, input=evaluation.example_input_by_id(i))
            for i in example_ids
        ]
        if self.current_run.shuffle_inputs:
          random.shuffle(inputs)

        leaf_node_runner = LeafNodeRunner(
            current_run=self.current_run.clone(
                override=dict(
                    experiment=evaluation,
                    raise_if_has_error=False,
                )
            ),
            plugins=self._per_example_plugins,
        )
        _ = (
            pipeline
            | f'Input-{evaluation.id}-{i}' >> beam.Create(
                [(x.id, pg.to_json_str(x)) for x in inputs]
            )
            | f'Evaluate-{evaluation.id}-{i}'
            >> beam.ParDo(
                _EvaluateFn(
                    pg.to_json_str(leaf_node_runner),
                    ckpt_format=self.ckpt_format,
                    concurrent_startup_delay=self.concurrent_startup_delay,
                )
            )
        )
        self.on_experiment_start(evaluation)
        # This is not precise, but we at least notify example start.
        for example in inputs:
          self.on_example_start(evaluation, example)
        collector = _EvaluationCollector(
            self,
            evaluation,
            example_ids=set(example_ids),
            ckpt_format=self.ckpt_format,
            max_collector_threads=self.max_collector_threads,
        )
        collector.start()
        collectors.append(collector)

    for collector in collectors:
      collector.join()

  def _evaluate_items(
      self,
      evaluation: evaluation_lib.Evaluation,
      items: Iterator[example_lib.Example]
  ) -> None:
    """Evaluates the items of an evaluation."""
    raise NotImplementedError('Not needed in beam runner.')
