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
"""Base class for Langfun evaluation tasks."""

import abc
import datetime
import functools
import threading
import time

from typing import Annotated, Any, Callable, Iterable
import langfun.core as lf
import langfun.core.coding as lf_coding

from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2 import metric_values as metric_values_lib
from langfun.core.eval.v2 import metrics as metrics_lib

import pyglove as pg


class Evaluation(experiment_lib.Experiment):
  """Evaluation.

  An evaluation can be a leaf node or a container of other evaluations,
  depending on whether the current evaluation object is configured with
  any `pg.oneof`.

  For example, `MyEval(lm=pg.oneof([lf.llms.Gpt4(), lf.llms.Gemini1_5Pro()]))`
  is a container of two sub-experiments, one for each LLM. In such case, the
  evaluation object with `pg.oneof` is called a hyper evaluation, which
  represents a search space of evaluations, and each sub-evaluation is called
  a leaf evaluation, which will perform the actual evaluation.
  """

  inputs: Annotated[
      pg.Functor,
      'A functor that returns a list of inputs.'
  ]

  metrics: Annotated[
      list[metrics_lib.Metric],
      'The metrics to be evaluated.'
  ]

  max_workers: Annotated[
      int,
      'The maximum number of workers to use for the evaluation.'
  ] = 32

  def _on_bound(self):
    # Invalidate cached properties.
    self.__dict__.pop('is_leaf', None)
    self.__dict__.pop('children', None)
    super()._on_bound()
    self._log_entries = []
    self._log_lock = threading.Lock()

  #
  # Handling evaluation hierarchy (materialized vs. hyper evaluations).
  #

  @functools.cached_property
  def is_leaf(self) -> bool:
    """Returns whether the task is a leaf."""
    return self.is_deterministic

  @functools.cached_property
  def children(self) -> list['Evaluation']:
    """Returns the children tasks."""
    if self.is_leaf:
      return []
    children = []
    for i, child in enumerate(pg.iter(self)):
      child.sym_setparent(self)
      child.sym_setpath(self.sym_path + 'children' + i)
      children.append(child)
    return children

  #
  # Handling evaluation inputs.
  #

  @functools.cached_property
  def example_inputs(self) -> Iterable[Any]:
    """Returns the examples from the inputs."""
    return self.inputs()

  def example_input_by_id(self, example_id: int) -> Any:
    """Returns the example from the inputs by ID."""
    assert example_id <= len(self.example_inputs), example_id
    return self._example_input_by_id[example_id]

  @functools.cached_property
  def _example_input_by_id(self) -> dict[int, Any]:
    """Returns the examples from the inputs by ID."""
    return {i + 1: v for i, v in enumerate(self.example_inputs)}

  @property
  def num_examples(self) -> int:
    """Returns the number of examples from the inputs."""
    # NOTE(daiyip): setting `num_examples` of the input functor allows fast
    # retrieval of number of examples without iterating the whole dataset.
    num_examples = getattr(self.inputs, 'num_examples', None)
    if not isinstance(num_examples, int):
      it = self.example_inputs
      if hasattr(it, '__len__'):
        num_examples = len(it)
      else:
        num_examples = len(list(it))
    return num_examples

  #
  # Evaluation logics.
  #

  @abc.abstractmethod
  def process(
      self,
      example: example_lib.Example
  ) -> Any | tuple[Any, dict[str, Any]]:
    """Processes a single example from the evaluation set.

    Users should override this method to implement the evaluation logic.

    Args:
      example: An example object to process. `example.input` is an object
        returned from `Evaluable.inputs`.

    Returns:
      A processed output. Or a tuple of (output, metadata).
      The output will be used for computing the metrics, and the metadata will
      be included in the evaluation HTML view.
    """

  def evaluate(
      self,
      example: example_lib.Example | int,
      raise_if_has_error: bool = False,
  ) -> example_lib.Example:
    """Evaluates a single example input.

    Args:
      example: An example ID or an example object with ID.
      raise_if_has_error: Whether to raise an error if the example has error.

    Returns:
      The evaluated example with the output and metric metadata populated.
    """
    if isinstance(example, int):
      example = example_lib.Example(id=example)
    assert isinstance(example, example_lib.Example), example

    if pg.MISSING_VALUE == example.input:
      example.input = self.example_input_by_id(example.id)

    checkpointed = self._state.ckpt_example(example.id)

    with pg.timeit('evaluate') as timeit, lf.track_usages() as usage_summary:
      if checkpointed is None or checkpointed.has_error:
        if checkpointed is None:
          self.info(
              f'Example {example.id} is being processed for the first time '
              'as no prior run is found. '
          )
        else:
          self.info(
              f'Example {example.id} is being reprocessed as prior run '
              f'contains error: {checkpointed.error}'
          )
        example.start_time = time.time()
        self._state.update(example, in_progress=True)
        self._process(example, raise_if_has_error=raise_if_has_error)
      else:
        self.info(
            f'Example {example.id} skipped processing as prior run '
            'is available and error free.'
        )
        example.start_time = checkpointed.start_time
        self._state.update(example, in_progress=True)

        # Use the output and metadata obtained from the previous processing.
        example.output = checkpointed.output
        example.metadata = checkpointed.metadata
        example.newly_processed = False

        # For previously processed examples, we merge previous usages as
        # cached, so the usage summary will account previous usages, but as
        # cached.
        assert checkpointed.usage_summary is not None
        usage_summary.merge(checkpointed.usage_summary, as_cached=True)

      # Recompute the metrics and metadata for the example even its processed
      # output and metadata were from the cache.
      # NOTE(daiyip): It's possible that metrics could use LLMs, so we need to
      # track the usage of the metrics separately.
      with pg.timeit('metric'):
        self.info(f'Starting metric computation for example {example.id}.')
        metric_metadata = {}
        for metric in self.metrics:
          metric_metadata.update(metric.audit(example))
        example.metric_metadata = metric_metadata
        self.info(f'Completed metric computation for example {example.id}.')

    # For previously processed examples, we keep the execution status for the
    # processing step.
    execution_status = dict(example.execution_status or {})
    execution_status.update(timeit.status())

    example.execution_status = execution_status
    example.usage_summary = usage_summary
    if example.newly_processed:
      example.end_time = time.time()

    self._state.update(example, in_progress=False)
    return example

  def _process(
      self,
      example: example_lib.Example,
      raise_if_has_error: bool = False
  ) -> None:
    """Processes a single example."""
    with (
        pg.notify_on_change(False),
        pg.allow_writable_accessors(True),
        # NOTE(daiyip): set the `input` symbol of the globals to None, so
        # LLM generated code with calls to `input` will raise an error, thus
        # not blocking the evaluation.
        lf_coding.context(input=None),
    ):
      try:
        with pg.timeit('process'):
          output = self.process(example)
        if (isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[1], dict)):
          output, metadata = output
        else:
          metadata = {}
        example.output = output
        example.metadata = metadata
      except BaseException as e:  # pylint: disable=broad-except
        if raise_if_has_error:
          raise
        example.error = pg.ErrorInfo.from_exception(e)

  #
  # Handling evaluation scheduling.
  #

  def resource_ids(self) -> set[str]:
    """Returns a set of resource IDs required by this evaluation.

    Resource IDs are used to by the runner to determine which evaluations can
    be run in parallel. Evaluations using the same resource key will be run
    sequentially.

    Returns:
      A unique string representing the resource required.
    """
    return {
        v.resource_id for _, v in self.sym_init_args.items()
        if isinstance(v, lf.LanguageModel)
    }

  #
  # Handling evaluation state.
  #

  @property
  def state(self) -> 'EvaluationState':
    """Returns the state of the evaluation."""
    return self._state

  def load_state(
      self,
      state_file: str,
      *,
      load_example_metadata: bool = True,
      filter: Callable[[example_lib.Example], bool] | None = None,  # pylint: disable=redefined-builtin
      raise_if_not_exist: bool = False
  ) -> None:
    """Loads saved state from a sequence IO file."""
    if pg.io.path_exists(state_file):
      self._state.load(
          state_file,
          example_input_by_id=self.example_input_by_id,
          load_example_metadata=load_example_metadata,
          filter=filter,
      )
    elif raise_if_not_exist:
      raise ValueError(f'State file {state_file} does not exist.')

  def _reset(self) -> None:
    """Resets the state of the evaluation."""
    super()._reset()
    if self.is_leaf:
      # Create a new state for the leaf evaluation.
      self._state = EvaluationState()
      for metric in self.metrics:
        metric.reset()

  #
  # Evaluation-level logging.
  #

  def _log(self, log_func, level: lf.logging.LogLevel, message: str, **kwargs):
    # Write to external logging system.
    log_message = message
    if self.id not in log_message:
      log_message = f'{self.id}: {log_message}'
    if kwargs:
      log_message = f'{log_message} (metadata: {kwargs!r})'
    log_func(log_message)

    # Add to experiment log history.
    log_entry = lf.logging.LogEntry(
        level=level,
        time=datetime.datetime.now(),
        message=message,
        metadata=kwargs,
    )
    with self._log_lock:
      self._log_entries.append(log_entry)

  def debug(self, message: str, **kwargs):
    """Logs a debug message to the session."""
    self._log(pg.logging.debug, 'debug', message, **kwargs)

  def info(self, message: str, **kwargs):
    """Logs an info message to the session."""
    self._log(pg.logging.info, 'info', message, **kwargs)

  def warning(self, message: str, **kwargs):
    """Logs a warning message to the session."""
    self._log(pg.logging.warning, 'warning', message, **kwargs)

  def error(self, message: str, **kwargs):
    """Logs an error message to the session."""
    self._log(pg.logging.error, 'error', message, **kwargs)

  def fatal(self, message: str, **kwargs):
    """Logs a fatal message to the session."""
    # We use error level for fatal message, which does not trigger assertion.
    self._log(pg.logging.error, 'fatal', message, **kwargs)

  #
  # HTML views.
  #

  def _html_tree_view_content(
      self, *, view, extra_flags: dict[str, Any] | None, **kwargs
  ):
    if not self.is_leaf:
      return super()._html_tree_view_content(
          view=view, extra_flags=extra_flags, **kwargs
      )

    extra_flags = extra_flags or {}
    run = extra_flags.pop('current_run', None)
    if extra_flags.pop('card_view', True):
      return self._summary_card_view(
          extra_flags.get('interactive', True), run
      )
    assert run is not None
    return self._details_view(run)

  def _parameter_badge(self, key, value) -> pg.Html.WritableTypes:
    """Renders a badge for a parameter."""
    face_value = pg.format(
        value,
        compact=True,
        python_format=True,
        hide_default_values=True,
        use_inferred=True
    )
    short_text = face_value
    if len(face_value) > 40:
      short_text = f'{type(value).__name__}(...)'
    label = f'{key.split(".")[-1]}: {short_text}'
    tooltip = f'{key}: {face_value}'
    return pg.views.html.controls.Badge(
        text=label,
        tooltip=tooltip,
        css_classes=['parameter'],
        interactive=False,
    )

  def _summary_card_view(
      self,
      interactive: bool = True,
      run: experiment_lib.Run | None = None,
  ) -> pg.Html.WritableTypes:
    """Renders the summary card view of the evaluation."""
    del run
    return pg.Html(
        pg.Html.element(
            'div',
            [
                pg.views.html.controls.LabelGroup([
                    self._parameter_badge(k, v)
                    for k, v in self.non_default_values(
                        flatten=True
                    ).items()
                ], css_classes=['parameter-group']),
                pg.Html.element(
                    'div',
                    [
                        m.to_html(
                            extra_flags=dict(
                                interactive=interactive,
                            )
                        )
                        for m in self.metrics
                    ],
                    css_classes=['metric-group'],
                ),
            ],
            css_classes=['badge-groups'],
        )
    )

  def _details_view(
      self, run: experiment_lib.Run
  ) -> pg.Html:
    """Renders the details view of the evaluation."""

    def _title():
      additional_links = []
      return pg.Html.element(
          'div',
          [
              pg.views.html.controls.LabelGroup(
                  [
                      pg.views.html.controls.Label(
                          'Summary',
                          link=run.output_link('summary.html'),
                          css_classes=['summary-link'],
                      ),
                      '|',
                      pg.views.html.controls.Label(
                          'Directory',
                          link=self.output_link(run, ''),
                          css_classes=['dir-link'],
                      ),
                  ] + additional_links,
                  css_classes=['experiment-links'],
              ),
              pg.views.html.controls.Label(
                  self.id,
                  css_classes=['experiment-id'],
              ),
              self.progress.to_html(
                  extra_flags=dict(interactive=False),
              ),
              self.usage_summary.to_html(
                  extra_flags=dict(as_badge=True, interactive=False),
              ),
          ]
      )

    def _parameter_badges():
      """Renders a tab group for a metric (group)."""
      return pg.views.html.controls.LabelGroup(
          [
              self._parameter_badge(k, v)
              for k, v in self.non_default_values(flatten=True).items()
          ],
          css_classes=['parameter-group'],
      )

    def _definition_tab() -> pg.views.html.controls.Tab:
      """Renders a tab for the definition of the evaluation."""
      return pg.views.html.controls.Tab(
          label='Definition',
          content=pg.Html.element(
              'div',
              [
                  pg.views.html.controls.Label(
                      pg.format(
                          self,
                          compact=False,
                          verbose=False,
                          use_inferred=True,
                          hide_frozen=True,
                          exclude_keys=set(['progress', 'usage_summary'])
                      ),
                      css_classes=['eval-definition'],
                  ),
              ]
          )
      )

    def _in_progress_tab() -> pg.views.html.controls.Tab | None:
      """Renders a tab for the in progress examples."""
      if not self.state.in_progress_examples:
        return None
      return pg.views.html.controls.Tab(
          label='In Progress',
          content=pg.Html.element(
              'div', [
                  self._in_progress_view(
                      list(self.state.in_progress_examples.values())
                  )
              ]
          )
      )

    def _metric_tab(metric: metrics_lib.Metric) -> pg.views.html.controls.Tab:
      """Renders a tab for a metric (group)."""
      return pg.views.html.controls.Tab(
          label=f'Metric: {metric.name}',
          content=pg.Html.element(
              'div',
              [
                  metric.to_html(
                      extra_flags=dict(
                          interactive=False,
                      )
                  ),
                  pg.views.html.controls.TabControl(
                      tabs=[
                          _metric_value_tab(mv)
                          for mv in metric.values()
                      ]
                  )
              ]
          )
      )

    def _metric_value_tab(
        metric_value: metric_values_lib.MetricValue
    ) -> pg.views.html.controls.Tab:
      """Renders the example links for a metric value."""
      return pg.views.html.controls.Tab(
          label=metric_value.sym_path.key,
          content=pg.Html.element(
              'div',
              [
                  pg.views.html.controls.Label(
                      str(dp.example_id),
                      link=self.output_link(run, f'{dp.example_id}.html'),
                      target='example-view',
                      css_classes=['example-link'],
                  )
                  for dp in sorted(
                      metric_value.data_points, key=lambda dp: dp.example_id
                  )
              ]
          )
      )

    def _logs_tab() -> pg.views.html.controls.Tab:
      """Renders a tab for the logs of the evaluation."""
      with self._log_lock:
        log_history = '\n'.join(str(l) for l in self._log_entries)
      return pg.views.html.controls.Tab(
          label='Logs',
          content=pg.Html.element(
              'div',
              [
                  pg.Html.element(
                      'textarea',
                      [pg.Html.escape(log_history)],
                      readonly=True,
                      css_classes=['logs-textarea'],
                  )
              ]
          )
      )

    def _main_tabs() -> pg.Html:
      return pg.Html.element(
          'div',
          [
              pg.views.html.controls.TabControl(
                  [
                      _definition_tab(),
                      [_metric_tab(m) for m in self.metrics],
                      _in_progress_tab(),
                      _logs_tab(),
                  ],
                  selected=1,
              )
          ],
      )

    return pg.Html.element(
        'div',
        [
            _title(),
            _parameter_badges(),
            _main_tabs(),
            pg.Html.element(
                'iframe', [],
                name='example-view',
                src='about:blank',
                title='Example view.',
                css_classes=['example-view'],
            ),
        ],
        css_classes=['eval-details'],
    )

  def _in_progress_view(
      self, in_progress_examples: list[example_lib.Example]
  ) -> pg.Html:
    """Renders a HTML view for the in-progress examples."""
    current_time = time.time()
    logs = [f'(Total {len(in_progress_examples)} examples in progress)']
    for example in in_progress_examples:
      if example.newly_processed:
        logs.append(
            f'Example {example.id}: In progress for '
            f'{current_time - example.start_time:.2f} seconds.'
        )
      else:
        logs.append(f'Example {example.id}: Recomputing metrics...')
    return pg.Html.element(
        'textarea',
        [pg.Html.escape('\n'.join(logs))],
        readonly=True,
        css_classes=['logs-textarea'],
    )

  def _html_tree_view_config(self) -> dict[str, Any]:
    return dict(
        css_classes=['eval-card'] if self.is_leaf else None
    )

  def _html_tree_view_css_styles(self) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        details.eval-card {
          display: inline-block;
          border: 0px;
          box-shadow: rgba(0, 0, 0, 0.16) 0px 1px 4px;
          margin: 15px;
        }
        .eval-card details {
          border: 0px;
        }
        .badge-groups {
          font-weight: normal;
          padding: 5px;
        }
        .parameter-group {
          display: inline-grid;
          grid-template-rows: auto auto;
          border: 0px;
          margin-right: 10px;
        }
        .parameter.badge {
          margin: 2px;
        }
        .metric-group {
          display: inline-grid;
          grid-template-rows: auto auto;
        }
        .eval-details .progress-bar > .shade {
          visibility: hidden;
          width: 0px;
          margin: 0px;
        }
        .eval-details .progress-label {
          font-size: 16px;
          background-color: #eee;
        }
        .eval-details .progress-time {
          font-size: 16px;
          color: dodgerblue;
          background-color: #eee;
          margin-right: 10px;
        }
        .eval-details .usage-summary.badge {
          color: orange;
          font-size: 16px;
          background-color: #eee;
        }
        .eval-details .experiment-links {
          display: block;
          border: 0px;
          margin: 0px;
        }
        .eval-details .tab-control {
          width: 100%;
        }
        .eval-details .tab-button {
          font-size: large;
        }
        .experiment-links .label {
          color: revert;
          margin: 0px;
          padding: 2px;
        }
        .eval-details .experiment-id {
          font-size: 2.0em;
          font-weight: bold;
          display: block;
        }
        .eval-details .parameter-group {
          display: inline-block;
          padding: 5px;
        }
        .eval-definition {
          white-space: pre;
          background-color: #eee;
          padding: 15px;
        }
        .eval-details .metric-container {
          display: block;
          padding: 15px 0px;
        }
        .example-link {
          color: revert;
        }
        .example-view {
          border: 0px;
          width:100%;
          height:100%;
        }
        .logs-textarea {
          width: 100%;
          height: 500px;
          padding: 5px;
          border: 1px solid #DDD;
          background-color: #EEE;
          resize: vertical;
        }
        """
    ]


class EvaluationState:
  """Evaluation state."""

  class ExampleStatus(pg.Object):
    """Example state."""
    evaluated: Annotated[
        bool,
        'Whether the example is evaluated.'
    ] = False

    in_progress: Annotated[
        bool,
        (
            'Whether the example is in progress. '
        )
    ] = False

    newly_processed: Annotated[
        bool,
        (
            'Whether the example is newly processed. '
            'Applicable only when evaluated is True.'
        )
    ] = False

    has_error: Annotated[
        bool,
        (
            'Whether the example has error. '
            'Applicable only when evaluated is True.'
        )
    ] = False

  def __init__(self):
    super().__init__()
    self._ckpt_examples: dict[int, example_lib.Example] = {}
    self._evaluation_status: dict[
        int, EvaluationState.ExampleStatus
    ] = {}
    self._in_progress_examples: dict[int, example_lib.Example] = {}

  def load(
      self,
      state_file: str,
      *,
      example_input_by_id: Callable[[int], Any] | None = None,
      load_example_metadata: bool | Callable[
          [example_lib.Example], bool] = True,
      filter: Callable[[example_lib.Example], bool] | None = None,  # pylint: disable=redefined-builtin
  ) -> None:
    """Loads the state from the example sequence file."""
    for example in example_lib.Example.iter_ckpts(
        state_file,
        example_input_by_id=example_input_by_id,
        load_example_metadata=load_example_metadata,
    ):
      if filter is not None and not filter(example):
        continue
      example.newly_processed = False
      self._ckpt_examples[example.id] = example

  @property
  def evaluation_status(self) -> dict[int, ExampleStatus]:
    """Returns the evaluation status of the examples."""
    return self._evaluation_status

  @property
  def in_progress_examples(self) -> dict[int, example_lib.Example]:
    """Returns the in-progress examples."""
    return self._in_progress_examples

  @property
  def ckpt_examples(self) -> dict[int, example_lib.Example]:
    """Returns the unevaluated examples from checkpoints."""
    return self._ckpt_examples

  def ckpt_example(self, example_id: int) -> example_lib.Example | None:
    """Returns the unevaluated example from checkpoints for a given ID."""
    return self._ckpt_examples.get(example_id)

  def get_status(self, example_id: int) -> ExampleStatus:
    """Returns the evaluation status of the example."""
    return self._evaluation_status.get(
        example_id, EvaluationState.ExampleStatus()
    )

  def update(self, example: example_lib.Example, in_progress: bool) -> None:
    """Updates the state with the given example."""
    self._update_status(example, in_progress)

    if in_progress:
      self._in_progress_examples[example.id] = example
    else:
      self._in_progress_examples.pop(example.id, None)
      # Processed examples will be removed once it's done.
      self._ckpt_examples.pop(example.id, None)

  def _update_status(
      self,
      example: example_lib.Example,
      in_progress: bool
  ) -> None:
    """Updates the evaluation status of the example."""
    self._evaluation_status[example.id] = (
        EvaluationState.ExampleStatus(
            evaluated=example.output != pg.MISSING_VALUE,
            in_progress=in_progress,
            newly_processed=example.newly_processed,
            has_error=example.has_error,
        )
    )

# Register the logging functions to skip the source of the logging functions.
pg.logging.register_frame_to_skip([
    Evaluation._log,    # pylint: disable=protected-access
    Evaluation.debug,
    Evaluation.info,
    Evaluation.warning,
    Evaluation.error,
    Evaluation.fatal
])
