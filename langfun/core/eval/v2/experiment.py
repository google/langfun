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
"""Evaluation experiment."""

import abc
import datetime
import functools
import hashlib
import inspect
import os
import re
from typing import Annotated, Any, Callable, Literal, Optional

import langfun.core as lf
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import progress as progress_lib
import pyglove as pg


class Experiment(lf.Component, pg.views.HtmlTreeView.Extension):
  """Evaluation Experiment.

  # Experiment Structure.

  An evaluation experiment is structured as a tree of evaluation tasks, where
  each task is represented as a node in the tree. Leaf tasks are instances of
  `Evaluation` with concrete hyper-parameter values. Nodes such as `Suite` and
  `Evaluation` that utilize `pg.oneof` are non-leaf tasks, as they represent
  multiple configurations. Leaf tasks can be retrieved using property
  `leaf_nodes`, while non-leaf tasks can be retrieved using property
  `nonleaf_nodes`. An experiment without any leaf tasks is considered
  empty.

  For example:

    ```
    Suite(
        MyEvaluation1(
            lm=pg.oneof([lm1, lm2]),
        ),
        Suite(
            MyEvaluation2(
                lm=lm1,
            ),
            MyEvaluation3(
                lm=lm2,
            ),
        )
    )
    ```

  In this example:
  - The two `Suite` nodes and the `MyEvaluation1` node (with pg.oneof) are
    non-leaf nodes, as they contain leaf tasks.
  - There are four leaf nodes. Two leaf nodes under `MyEvaluation1`, which
    correspond to `MyEvaluation1` instances with `lm1` and `lm2` as
    hyper-parameters respectively. The objects of `MyEvaluation2` and
    `MyEvaluation3` are also leaf nodes as they have specific hyper-parameter
    values.

  # Running an Experiment

  To run an experiment, users can call `Experiment.run`. This will execute the
  experiment using a specified `Runner` instance (e.g., 'parallel' or
  'sequential'). Progress and results will be periodically written to HTML
  files. Users can also assign an id to each run, which will identify the output
  directory of that run.

  By default, the experiment will resume from the latest run under the root
  directory (using the ID 'latest'). Users can specify 'new' to start a fresh
  run or provide a specific run ID (typically in the format %Y%m%d_%<number>).
  Additionally, when initiating a new run, users may specify a `warm_start_from`
  directory to restore the experiment’s state from a previous run.

  Examples:

    ```
    root_dir = '/path/to/experiment/root'

    # Resume the latest experiment run, or start a new run if none exists.
    experiment.run(root_dir)

    # Equivalent to:
    experiment.run(root_dir, 'latest')

    # Start a new, clean run.
    experiment.run(root_dir, 'new')

    # Start a new run with a warm start from the another run located at
    # '/path/to/another/run' (e.g. /my_expreriment/run_20241031_1).
    experiment.run(root_dir, 'new', warm_start_from='/path/to/another/run')

    # Resume run '20241031_1', re-running failed examples and recomputing
    # metrics as needed.
    experiment.run(root_dir, '20241031_1')

    # Reprocess the previous run located in 'run_20241031_1'.
    experiment.run(root_dir, '20241031_1', reprocess=True)
    ```

  # Experiment Registration and Lookup

  Experiments can be registered by setting a class-level NAME attribute.
  Users can then retrieve a registered experiment using Experiment.find(name).

  For example:

  ```
  class MyEval(lf.eval.v2.Evaluation):
    NAME = 'my_eval'

  class MyEvalVariation1(MyEval):
    NAME = 'my_eval/gemini'
    lm = pg.oneof([lf.llms.GeminiPro(), lf.llms.GeminiFlash(), ...])

  class MyEvalVariation2(MyEval):
    NAME = 'my_eval/openai'
    lm = pg.oneof([lf.llms.Gpt4o(), lf.llms.Gpt4Turbo(), ...])

  # Run all experiments with "gemini" in their name.
  experiment = Experiment.find('.*/gemini')
  experiment.run()

  # Run all experiments with "my_eval" in their name.
  experiment = Experiment.find('my_eval.*')
  experiment.run()
  ```

  # Checkpointing

  Experiments support checkpointing, which is enabled by default. It allows 
  users to resume their experiments from a saved state. When an experiment runs,
  it creates a new directory for that run and saves the current state to a
  checkpoint file. If the experiment is interrupted or fails, users can resume
  it by specifying the 'id' or 'warm_start_from' argument (shown above) to
  seamlessly continue from previously saved state without starting over.

  # Monitoring and Reporting

  Evaluations can take considerable time to complete, so Langfun provides
  several tools to monitor progress. Progress bars display the status of each
  evaluation: HTML-based progress bars update in real time within Colab
  notebooks, while text-based progress bars appear in the terminal using tqdm.

  Additionally, Langfun generates HTML files at regular intervals to provide
  progress updates and detailed evaluation results. These files are saved in
  the evaluation's output directory, organized as follows:

   root_dir>                # Root directory of the experiment.
     |_ <run_id>            # Root directory of current run.
         |_ summary.html    # Summary of the run. Updated every 60 seconds.
         |_ <experiment_cls>      # Directory of a particular experiment type.
             |_ <experiment_hash> # Directory of a particular experiment config.
             |_ index.html        # Experiment report. Updated every 60 seconds.
             |_ 1.html            # Detailed evaluation output of example 1.
             |_ 2.html            # Detailed evaluation output of example 2.
             |_ ...

  # Experiment Plugins

  Experiment can be extended by plugins. Plugins can listen to the events of
  experiment execution and produce additional outputs. For example, a plugin
  can be added to an experiment to generate additional metrics or to save
  additional data to a database. More details will be added in the future.
  """

  #
  # Class-level functionalities.
  #

  # An global unique str as a well-known name for an experiment,
  # which can be retrieved by `Experiment.find(name)`. If None, the experiment
  # does not have a well-known name, thus users need to create the experiment
  # by constructing it explicitly.
  NAME = None

  # Global registry for experiment classes with GLOBAL_ID.
  _NAME_TO_CLASS = {}

  def __init_subclass__(cls):
    super().__init_subclass__()

    if inspect.isabstract(cls):
      return

    if cls.NAME is not None:
      cls._NAME_TO_CLASS[cls.NAME] = cls

  @classmethod
  def find(cls, pattern: str) -> 'Experiment':
    """Finds an experiment by global name.

    Args:
      pattern: A regular expression to match the global names of registered 
        experiments.

    Returns:
      An experiment object. If multiple experiments are found, a
        `Suite` of matched experiments will be returned. If no experiment is
        found, an empty `Suite` will be returned.
    """
    if pattern in cls._NAME_TO_CLASS:
      return cls._NAME_TO_CLASS[pattern]()
    regex = re.compile(pattern)
    selected = []
    for cls_name, exp_cls in cls._NAME_TO_CLASS.items():
      if regex.match(cls_name):
        selected.append(exp_cls())
    return selected[0] if len(selected) == 1 else Suite(selected)

  #
  # Instance-level functionalities.
  #

  progress: Annotated[
      progress_lib.Progress,
      'The progress of the experiment.'
  ] = progress_lib.Progress()

  usage_summary: Annotated[
      lf.UsageSummary,
      'The usage summary of the experiment.'
  ] = lf.UsageSummary()

  plugins: Annotated[
      list['Plugin'],
      (
          'Plugins for current experiment, which can listen to the events '
          'of experiment execution and produce additional outputs.'
      )
  ] = []

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('hash', None)
    self.__dict__.pop('dir', None)
    self._reset()

  #
  # Identity of an experiment.
  #

  @property
  def id(self) -> str:
    """Returns the ID for this evaluaton."""
    return f'{self.__class__.__name__}@{self.hash}'

  def definition(self, hide_default_values: bool = True) -> str:
    """Returns the definition of the experiment."""
    return self.format(
        compact=False,
        hide_default_values=hide_default_values,
        use_inferred=True,
        exclude_keys=('progress', 'usage_summary')
    )

  @functools.cached_property
  def hash(self) -> str:
    """A 8-byte MD5 hash computed from experiment identity."""
    identity = self.format(
        compact=True, hide_default_values=True, use_inferred=True,
        exclude_keys=('plugins', 'progress', 'usage_summary')
    )
    return hashlib.md5(identity.encode()).hexdigest()[:8]

  @classmethod
  def link(cls, path: str) -> str:
    return f'file://{path}'

  #
  # Hierarchy of an experiment tree.
  #

  @property
  @abc.abstractmethod
  def children(self) -> list['Experiment']:
    """Returns the child experiments."""

  @property
  @abc.abstractmethod
  def is_leaf(self) -> bool:
    """Returns whether the experiment is a leaf node."""

  def empty(self) -> bool:
    """Returns whether the experiment is empty."""
    return not self.leaf_nodes

  @functools.cached_property
  def nodes(self) -> list['Experiment']:
    """Returns all the experiment nodes in the subtree (including self)."""
    nodes = [self]
    for child in self.children:
      nodes.extend(child.nodes)
    return nodes

  @functools.cached_property
  def leaf_nodes(self) -> list['Experiment']:
    """Returns the leaf nodes.

    The leaf-nodes of an experiment are evaluable objects that has materilized
    hyper-parameters.
    """
    if self.is_leaf:
      return [self]

    nodes = []
    for child in self.children:
      nodes.extend(child.leaf_nodes)
    return nodes

  @functools.cached_property
  def nonleaf_nodes(self) -> list['Experiment']:
    """Returns the non-leaf nodes."""
    if self.is_leaf:
      return []
    nodes = [self]
    for child in self.children:
      nodes.extend(child.nonleaf_nodes)
    return nodes

  @functools.cached_property
  def parent(self) -> Optional['Experiment']:
    """Returns the parent experiment."""
    parent = self.sym_parent
    while parent is not None and not isinstance(parent, Experiment):
      parent = parent.sym_parent
    return parent

  def get(self, evaluation_id: str) -> Optional['Experiment']:
    """Returns the experiment by ID."""
    for leaf in self.leaf_nodes:
      if leaf.id == evaluation_id:
        return leaf
    return None

  #
  # Mutable states during evaluaton.
  #

  def reset(self) -> None:
    """Resets the experiment for a new run."""
    self.progress.reset()
    self.rebind(
        usage_summary=lf.UsageSummary(),
        skip_notification=True,
        raise_on_no_change=False
    )
    if self.is_leaf:
      self._reset()
    else:
      for child in self.children:
        child.reset()

  def _reset(self) -> None:
    """Subclass could override."""

  #
  # Helper methods for running the evaluation without explicitly creating the
  # runner.
  #

  def run(
      self,
      root_dir: str,
      id: str | None = None,   # pylint: disable=redefined-builtin
      *,
      runner: str = 'parallel',
      warm_start_from: str | None = None,
      filter: Callable[['Experiment'], bool] | None = None,   # pylint: disable=redefined-builtin
      example_ids: list[int] | None = None,
      raise_if_has_error: bool = False,
      reprocess: bool | list[int] = False,
      generate_example_html: Literal['new', 'all', 'no'] | list[int] = 'new',
      process_timeout: int | None = None,
      use_cache: Literal['global', 'per_dataset', 'no'] = 'per_dataset',
      note: str | None = None,
      tags: list[str] | None = None,
      plugins: list['Plugin'] | None = None,
      **kwargs
  ) -> 'Run':
    """Runs the experiment.

    Examples:
      # Start a new run under root_dir.
      experiment.run(root_dir, 'new')

      # Continue the latest experiment run.
      experiment.run(root_dir, 'latest')

      # Continue the latest experiment run or start a new run if it does not
      # exist.
      experiment.run(root_dir)

      # Start a new run and warm start from another run's directory
      # '/path/to/another/run_20241031_1/'.
      experiment.run(
          root_dir, 'new',
          warm_start_from='/path/to/another/run_20241031_1/'
      )

      # Reprocess previous run under sub-dir 'run_20241031_1'.
      experiment.run(root_dir, '20241031_1', reprocess=True)

    Args:
      root_dir: The root of the output directory of the experiment.
      id: The ID of the current run. It can be None, a special keyword 'latest'
        or 'new', or a datetime string in format `%Y%m%d%_%` (e.g. 20241031_1).
        If None, it will use the latest run ID under the root directory or
        create a new run based on the current time if no previous run exists.
        If `latest`, it will use the latest run ID under the root directory.
        If `new`, it will create a new run ID based on the current time.
      runner: The runner to use. If None, it will use the default runner for
        the experiment.
      warm_start_from: The ID of the previous run to warm start from. If None,
        it will continue the experiment identified by `id` from where it left
        off. Otherwise, it will create a new experiment run by warming start.
      filter: A filter function to decide whether an experiment should be run
        or not.
      example_ids: The example IDs to run. If None, it will run all examples.
      raise_if_has_error: If True, it will raise an error if any example fails.
        Otherwise, it will continue and report the error in the output.
      reprocess: A boolean or a list of example IDs. If boolean, it indicates
        that whether all the examples to be evaluated will be reprocessed,
        meaning that existing checkpoints will be ignored. If a list of
        example IDs, it indicates that only the specified examples will be
        reprocessed.
      generate_example_html: Among 'new', 'all', 'no' or a list of example IDs.
        If 'new', generate HTML files for all newly processed examples, and
          keep/copy existing HTML files for unchanged examples.
        If 'all', generate HTML files for all examples.
        If 'no', do not generate HTML files for any examples.
        If a list of example IDs, generate HTML files for the specified
        examples.
      process_timeout: The timeout in seconds for each process. If None, it
        will use the default timeout for the runner.
      use_cache: Whether to use LLM cache for the experiment.
        If `global`, it will use a global cache shared by all experiments.
        If `per_dataset`, it will use a cache dedicated for each dataset.
        If `no`, it will not use any cache.
      note: The note for the current run.
      tags: The tags for the current run.
      plugins: Runner plugins to use.
      **kwargs: Additional kwargs to pass to the runner.

    Returns:
      The current run.
    """
    if plugins is not None:
      kwargs['plugins'] = plugins
    runner = Runner.create(
        runner,
        current_run=Run(
            root_dir=root_dir,
            experiment=pg.Ref(self),
            id=RunId.from_id(id, root_dir),
            warm_start_from=warm_start_from,
            filter=filter,
            example_ids=example_ids,
            raise_if_has_error=raise_if_has_error,
            reprocess=reprocess,
            generate_example_html=generate_example_html,
            use_cache=use_cache,
            process_timeout=process_timeout,
            note=note,
            tags=tags or [],
        ),
        **kwargs
    )
    runner.run()
    return runner.current_run

  def run_preconfigured(
      self,
      root_dir: str | None = None,
      id: str | None = None,   # pylint: disable=redefined-builtin
      **kwargs
  ) -> 'Run':
    """Runs the experiment with pre-configured kwargs from `cls.RUN_ARGS`.

    This helper method allows users to config running arguments as a part of
    the class.

    Args:
      root_dir: root directory of the experiment.
      id: ID of the current run.
      **kwargs: Keyword arguments to override the RUN_CONFIG.

    Returns:
      The current run.
    """
    run_config = getattr(self, 'RUN_ARGS', {})
    run_config.update(kwargs)
    if root_dir is not None:
      run_config['root_dir'] = root_dir
    if id is not None:
      run_config['id'] = id
    return self.run(**run_config)

  #
  # HTML views.
  #

  def output_link(
      self,
      run: Optional['Run'], relative_path: str
  ) -> str | None:
    """Returns the output path of the experiment."""
    if run is None:
      return None
    return self.link(run.output_path_for(self, relative_path))

  def _html_tree_view_summary_title(
      self,
      current_run: Optional['Run'] = None,
      interactive: bool = True,
  ):
    title, link, dir_link = self.id, None, None
    if current_run is not None:
      dir_link = self.output_link(current_run, '')
      if self.is_leaf:
        link = self.output_link(current_run, 'index.html')
      elif self.parent is None:
        title = str(current_run.id)
        link = self.output_link(current_run, 'summary.html')
    return pg.Html.element(
        'div',
        [
            # Experiment ID.
            pg.views.html.controls.Label(
                title,
                link=link,
                tooltip=pg.format(  # pytype: disable=wrong-arg-types
                    self,
                    verbose=False,
                    use_inferred=True,
                    hide_default_values=True,
                    exclude_keys=(
                        'root_dir', 'plugins', 'progress', 'usage_summary'
                    ),
                ),
                css_classes=['experiment-name'],
            ),
            # Experiment directory (if root or leaf).
            pg.views.html.controls.Label(  # pylint: disable=g-long-ternary
                '[dir]',
                link=dir_link,
                css_classes=['experiment-dir'],
            ) if dir_link is not None else None,
            # Progress bar.
            self.progress.to_html(
                extra_flags=dict(interactive=interactive),
            ),
            # Usage summary,
            self.usage_summary.to_html(
                extra_flags=dict(as_badge=True, interactive=interactive)
            ),
        ],
        css_classes=['experiment-summary']
    )

  def _html_tree_view_summary(
      self,
      *,
      view,
      name: str | None = None,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ):
    extra_flags = extra_flags or {}
    if not extra_flags.get('card_view', True):
      return None

    kwargs.pop('title', None)
    kwargs.pop('enable_key_tooltip', None)
    kwargs.pop('enable_summary_tooltip', None)
    return view.summary(
        self,
        name=name if self.is_leaf else None,
        title=self._html_tree_view_summary_title(
            extra_flags.get('current_run', None),
            extra_flags.get('interactive', True)
        ),
        enable_key_tooltip=False,
        enable_summary_tooltip=False,
        **kwargs
    )

  def _html_tree_view_content(
      self,
      *,
      view,
      collapse_level: int | None = 1,
      extra_flags: dict[str, Any],
      **kwargs):
    return pg.Html.element(
        'div',
        [
            c.to_html(
                collapse_level=view.get_collapse_level(
                    (collapse_level, -1), 0
                ),
                name=f'#{i + 1}',
                extra_flags=extra_flags,
                **view.get_passthrough_kwargs(**kwargs)
            )
            for i, c in enumerate(self.children)
        ],
    )

  def _html_tree_view_css_styles(self) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .experiment-summary {
            display: inline-block;
            font-weight: normal;
        }
        .experiment-name {
            font-weight: bold;
        }
        .experiment-dir.label {
            color: revert;
            margin-left: 0px;
            padding: 2px;
        }
        .usage-summary-badge {
          margin-left: 10px;
        }
        body {
          font: normal 16px "Roboto","Noto",sans-serif;
        }
        """
    ]


@pg.use_init_args(['children'])
class Suite(Experiment):
  """A suite of evaluations."""

  children: Annotated[
      list[Experiment], 'A list of child experiments.'
  ] = []

  @property
  def is_leaf(self) -> bool:
    """Returns whether the task is a leaf."""
    return False


class RunId(pg.Object):
  """Structured repreesentation a experiment run ID."""
  date: datetime.date
  number: int

  _REGEX = re.compile(r'^(\d{8})_(\d+)$')

  def dirname(self, root_dir: str | None = None) -> str:
    """Returns the directory name of the run ID."""
    dir_name = f'run_{self}'
    if root_dir is None:
      return dir_name
    return os.path.join(root_dir, dir_name)

  def __str__(self) -> str:
    """Returns the string representation of the run ID."""
    return f'{self.date.strftime("%Y%m%d")}_{self.number}'

  def __lt__(self, other: 'RunId') -> bool:
    """Returns whether the run ID is less than the other."""
    return self.date < other.date or (
        self.date == other.date and self.number < other.number
    )

  def _le__(self, other: 'RunId') -> bool:
    """Returns whether the run ID is less than or equal to the other."""
    return self == other or self < other

  def __gt__(self, other: 'RunId') -> bool:
    """Returns whether the run ID is greater than the other."""
    return other < self

  def __ge__(self, other: 'RunId') -> bool:
    """Returns whether the run ID is greater than or equal to the other."""
    return self == other or self > other

  def next(self) -> 'RunId':
    """Returns the next run ID."""
    return RunId(
        date=self.date,
        number=self.number + 1,
    )

  @classmethod
  def from_dirname(cls, dirname: str) -> Optional['RunId']:
    """Creates a run ID from the directory name."""
    if not dirname.startswith('run_'):
      return None
    run_id_str = dirname.removeprefix('run_')
    if cls.is_valid(run_id_str):
      return cls.from_id(run_id_str)
    return None

  @classmethod
  def is_valid(cls, run_id: str) -> bool:
    """Returns whether the run ID is valid."""
    return run_id in ('latest', 'new') or bool(cls._REGEX.match(run_id))

  @classmethod
  def from_id(
      cls,
      run_id: str | None,
      root_dir: str | None = None
  ) -> 'RunId':
    """Creates a run ID from the string representation."""
    if run_id is not None and not cls.is_valid(run_id):
      raise ValueError(
          f'`run_id` must be one of `latest`, `new` and a '
          f'datetime string in format `%Y%m%d%_<number>` (e.g. 20240101_1). '
          f'Encountered: {run_id!r}.'
      )
    if run_id in (None, 'latest', 'new'):
      if root_dir is None:
        raise ValueError(
            '`root_dir` must be provided for `latest` or `new` run ID.'
        )
      if run_id == 'latest':
        run_id = cls.get_latest(root_dir)
        if run_id is None:
          raise ValueError(
              f'There are no previous runs under the root directory: '
              f'{root_dir}. Consider running the experiment using `new` as id.'
          )
        return run_id
      if run_id == 'new':
        return cls.new(root_dir)
      return cls.get_latest(root_dir) or cls.new()

    assert run_id is not None
    date_str, number_str = run_id.split('_')
    return cls(
        date=datetime.datetime.strptime(date_str, '%Y%m%d').date(),
        number=int(number_str),
    )

  @classmethod
  def get_latest(cls, root_dir: str) -> Optional['RunId']:
    """Returns the latest run ID under the root directory."""
    if not pg.io.isdir(root_dir):
      return None
    run_ids = [
        RunId.from_dirname(dirname)
        for dirname in pg.io.listdir(root_dir)
    ]
    run_ids = [run_id for run_id in run_ids if run_id is not None]
    if not run_ids:
      return None
    return max(run_ids)

  @classmethod
  def new(cls, root_dir: str | None = None) -> 'RunId':
    """Creates a new run ID."""
    latest = None if root_dir is None else cls.get_latest(root_dir)
    if latest is not None and latest.date == datetime.date.today():
      return latest.next()
    return cls(
        date=datetime.date.today(),
        number=1,
    )


class Run(pg.Object, pg.views.html.HtmlTreeView.Extension):
  """A run of an experiment."""

  root_dir: Annotated[
      str,
      'The root of the output directory of the experiment.'
  ]

  id: Annotated[
      RunId,
      (
          'The ID of the current run.'
      )
  ]

  experiment: Annotated[
      Experiment,
      'The root experiment to run.'
  ]

  warm_start_from: Annotated[
      str | None,
      (
          'The directory for a previous run to warm start from.'
      )
  ] = None

  example_ids: Annotated[
      list[int] | None,
      (
          'The example IDs to run. If None, it will run all examples. '
          'Though '
      )
  ] = None

  raise_if_has_error: Annotated[
      bool,
      (
          'If True, it will raise an error if any example fails.'
      )
  ] = False

  note: Annotated[
      str | None,
      'The user note for the current run.'
  ] = None

  tags: Annotated[
      list[str],
      'The user tags for the current run.'
  ] = []

  reprocess: Annotated[
      bool | list[int],
      (
          'If True, it will reprocess all examples under the current '
          'run directory. If a list of integers, examples of the given IDS '
          'will be reprocessed.'
      )
  ] = False

  generate_example_html: Annotated[
      Literal['new', 'all', 'no'] | list[int],
      (
          'If "new", generate HTML files for all newly processed examples, '
          'and keep/copy existing HTML files for unchanged examples. '
          'If "all", generate HTML files for all examples. '
          'If "no", do not generate HTML files for any examples. '
          'If a list of example IDs, generate HTML files for the specified '
          'examples.'
      )
  ] = 'new'

  filter: Annotated[
      Callable[[Experiment], bool] | None,
      'A filter to decide whether a leaf experiment should be run or not.'
  ] = None

  process_timeout: Annotated[
      int | None,
      'Timeout for each evaluation example.'
  ] = None

  use_cache: Annotated[
      Literal['global', 'per_dataset', 'no'],
      (
          'The cache policy for the runner. If `global`, the runner will use '
          'the cache for all evaluations. If `per_dataset`, the runner will '
          'use the cache for each evaluation. If `no`, the runner will not '
          'use the cache.'
      )
  ] = 'per_dataset'

  @property
  def output_root(self) -> str:
    """Returns the root directory of the experiment."""
    return self.id.dirname(self.root_dir)

  @property
  def input_root(self) -> str:
    """Returns the input root d."""
    return self.warm_start_from if self.warm_start_from else self.output_root

  def output_dir(self, experiment: Experiment) -> str:
    """Returns the output directory of the experiment."""
    if experiment.is_leaf:
      return os.path.join(self.output_root, experiment.id.replace('@', '/'))
    return self.output_root

  def input_dir(self, experiment: Experiment) -> str:
    """Returns the input directory of the experiment."""
    if experiment.is_leaf:
      return os.path.join(self.input_root, experiment.id.replace('@', '/'))
    return self.input_root

  def input_path_for(self, experiment: Experiment, relative_path: str) -> str:
    """Returns the input path for the experiment."""
    return os.path.join(self.input_dir(experiment), relative_path)

  def output_path_for(self, experiment: Experiment, relative_path: str) -> str:
    """Returns the output path for the experiment."""
    return os.path.join(self.output_dir(experiment), relative_path)

  def examples_to_evaluate(self, experiment: Experiment) -> set[int]:
    """Returns the example IDs to evaluate."""
    if not experiment.is_leaf:
      return set()
    return set(
        self.example_ids if self.example_ids else
        range(1, experiment.num_examples + 1)
    )

  def examples_to_reprocess(self, experiment: Experiment) -> set[int]:
    """Returns the example IDs to reprocess per request."""
    if not self.reprocess:
      return set()
    reprocess_ids = self.examples_to_evaluate(experiment)
    if isinstance(self.reprocess, list):
      reprocess_ids &= set(self.reprocess)
    return reprocess_ids

  def examples_to_load(self, experiment: Experiment) -> set[int]:
    """Returns the example IDs to load from checkpoint files.."""
    load_ids = self.examples_to_evaluate(experiment)
    if isinstance(self.generate_example_html, list):
      load_ids |= set(self.generate_example_html)
    load_ids -= self.examples_to_reprocess(experiment)
    return load_ids

  def examples_to_load_metadata(self, experiment: Experiment) -> set[int]:
    """Returns the example IDs to load the metadata."""
    load_metadata_ids = set()
    if isinstance(self.generate_example_html, list):
      load_metadata_ids = set(self.generate_example_html)
    elif self.generate_example_html == 'all':
      load_metadata_ids = self.examples_to_evaluate(experiment)
    load_metadata_ids -= self.examples_to_reprocess(experiment)
    return load_metadata_ids


class Runner(pg.Object):
  """Interface for experiment runner."""

  # Class-level variable for registering the runner.
  NAME = None

  _REGISTRY = {}

  current_run: Annotated[
      Run,
      'The current run.'
  ]

  plugins: Annotated[
      list['Plugin'],
      'The plugins for the runner.'
  ] = []

  def __init_subclass__(cls):
    super().__init_subclass__()
    if inspect.isabstract(cls):
      return
    if cls.NAME is None:
      raise ValueError(
          'Runner class must define a NAME constant. '
          'Please use the same constant in the runner class.'
      )
    cls._REGISTRY[cls.NAME] = cls

  @abc.abstractmethod
  def run(self) -> None:
    """Runs a evaluation task."""

  @classmethod
  def create(cls, runner: str, **kwargs) -> 'Runner':
    """Creates a runner instance by ID and kwargs."""
    return cls._REGISTRY[runner](**kwargs)


class Plugin(lf.Component):
  """Base class for experiment plugins."""

  def on_run_start(
      self,
      runner: Runner,
      root: Experiment
  ) -> None:
    """Called when a runner is started."""

  def on_run_complete(
      self,
      runner: Runner,
      root: Experiment
  ) -> None:
    """Called when a runner is complete."""

  def on_run_abort(
      self,
      runner: Runner,
      root: Experiment,
      error: BaseException,
  ) -> None:
    """Called when a runner is aborted."""

  def on_experiment_start(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    """Called when an evaluation is started."""

  def on_experiment_skipped(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    """Called when an experiment (both leaf and non-leaf) is skipped."""

  def on_experiment_complete(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    """Called when an experiment (both leaf and non-leaf) is complete."""

  def on_experiment_abort(
      self,
      runner: Runner,
      experiment: Experiment,
      error: BaseException,
  ) -> None:
    """Called when an experiment (both leaf and non-leaf) is aborted."""

  def on_example_start(
      self,
      runner: Runner,
      experiment: Experiment,
      example: example_lib.Example
  ) -> None:
    """Called when an example is about to be evaluated."""

  def on_example_complete(
      self,
      runner: Runner,
      experiment: Experiment,
      example: example_lib.Example
  ) -> None:
    """Called when an example is evaluated."""
