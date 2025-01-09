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
"""Base classes for langfun evaluation."""

import abc
import collections
import dataclasses
import functools
import hashlib
import inspect
import io
import os
import re
import threading
import time
import types
from typing import Annotated, Any, Callable, Iterator, Literal, Optional, Sequence, Type, Union

import langfun.core as lf
import langfun.core.coding as lf_coding
from langfun.core.llms.cache import in_memory
import langfun.core.structured as lf_structured
import pyglove as pg


class Evaluable(lf.Component):
  """Base class for evaluation and suite."""

  EXPERIMENT_JSON = 'experiment.json'
  RESULT_JSON = 'result.json'
  OOP_FAILURES_JSON = 'oop_failures.json'
  NON_OOP_FAILURES_JSON = 'non_oop_failures.json'
  INDEX_HTML = 'index.html'
  SUMMARY_HTML = 'summary.html'

  root_dir: Annotated[
      str | None,
      (
          'The root directory for all evaluables under an evaluation suite'
          "A child evaluable's dir will be relative to its parent dir."
      ),
  ] = lf.contextual(default=None)

  report_precision: Annotated[
      int, 'Number of decimals when reporting precision.'
  ] = lf.contextual(default=1)

  @property
  @abc.abstractmethod
  def id(self) -> str:
    """Returns the ID of the task.

    Returns:
      Evaluation task ID. Different evaluation task should have their unique
      task IDs, for each task will be stored in sub-directoreis identified by
      their IDs. For suites, the ID could be an empty string as they will not
      produce sub-directories
    """

  @property
  def dir(self) -> str | None:
    """Returns the directory for saving results and details."""
    if self.root_dir is None:
      return None
    return os.path.join(self.root_dir, self.id)

  @classmethod
  def link(cls, path: str) -> str:
    return f'file://{path}'

  @property
  def index_link(self) -> str | None:
    """Returns the index page."""
    if self.dir is None:
      return None
    return self.link(os.path.join(self.dir, Evaluable.INDEX_HTML))

  def summary(self, pivot_field: str = 'lm') -> 'Summary':
    """Returns a summary for all child evaluations.."""
    return Summary([pg.Ref(x) for x in self.leaf_nodes], pivot_field)

  @property
  def summary_link(self) -> str | None:
    """Returns the summary page."""
    if self.root_dir is None:
      return None
    return self.link(os.path.join(self.root_dir, Evaluable.SUMMARY_HTML))

  def _on_bound(self):
    super()._on_bound()
    self._reset()
    self._dryrun_output = None
    # Invalidate cached properties.
    self.__dict__.pop('leaf_nodes', None)
    self.__dict__.pop('nonleaf_nodes', None)

  def _reset(self):
    """Reset evaluation state."""
    self._result = None

  @property
  def dryrun_output(self) -> lf.Message | None:
    """Returns the dryrun output in lf.Message."""
    return self._dryrun_output

  @property
  def result(self) -> pg.Dict | None:
    """Returns the evaluation result in pg.Dict."""
    return self._result

  def dryrun(
      self,
      *,
      filter: Callable[['Evaluable'], bool] | None = None,  # pylint: disable=redefined-builtin
      example: Any = None,
      debug: bool | lf.LMDebugMode = False,
      verbose: bool = True,
      **kwargs,
  ) -> None:
    """Dry run on a single input and fill the `dryrun_output`.

    Args:
      filter: An optional filter to decide whether a leaf evaluation should be
        run or not.
      example: An optional example to dry run against. If None, the first
        example from the inputs will be used.
      debug: Debug flag for LM call.
      verbose: If True, verbose dry-run information will be printed.
      **kwargs: Keyword arguments that will be passed through to leaf
        evaluations.
    """
    if self.is_leaf:
      self._dryrun(example=example, debug=debug, verbose=verbose, **kwargs)
    else:
      filter = filter or (lambda x: True)
      for i, leaf in enumerate(self.leaf_nodes):
        if verbose and i > 0:
          lf.console.write('')
          lf.console.write('-' * 80)

        if filter(leaf):
          lf.console.write(
              f'#{i + 1} DRY-RUNNING {leaf.id}...',
              color='green',
              styles=['bold'],
          )
          leaf.dryrun(
              example=example, debug=debug, verbose=verbose, **kwargs)
        elif verbose:
          lf.console.write(
              f'SKIPPING #{i + 1} {leaf.id} BY FILTER.',
              color='yellow',
              styles=['bold'],
          )

  @abc.abstractmethod
  def _dryrun(
      self,
      *,
      example: Any,
      debug: bool | lf.LMDebugMode,
      verbose: bool
  ) -> None:
    """Dry run on a single input and fill the `dryrun_output`."""

  @property
  def is_leaf(self) -> bool:
    return isinstance(self, Evaluation) and not self.children

  @functools.cached_property
  def leaf_nodes(self) -> list['Evaluation']:
    """Returns the leaf nodes for evaluation."""
    if isinstance(self, Evaluation) and not self.children:
      return [self]

    nodes = []
    for child in self.children:
      nodes.extend(child.leaf_nodes)
    return nodes

  @functools.cached_property
  def nonleaf_nodes(self) -> list['Evaluable']:
    """Returns the non-leaf nodes."""
    nodes = []
    for child in self.children:
      nodes.extend(child.nonleaf_nodes)
    if not self.is_leaf:
      nodes.append(self)
    return nodes

  def run(
      self,
      *,
      filter: Callable[['Evaluable'], bool] | None = None,  # pylint: disable=redefined-builtin
      start: int = 0,
      end: int | None = None,
      rerun: bool = False,
      save: bool = True,
      debug: bool | lf.LMDebugMode = False,
      dryrun: bool = False,
      verbose: bool = False,
      show_progress: bool | int = True,
      label: str | None = None,
      summary: bool = True,
      pivot_field: str = 'lm',
      from_root: bool = True,
      timeout: int | None = None,
      **kwargs,
  ) -> Union['Summary', pg.Dict]:
    """Run the evaluation, which fills and returns the result."""
    # Internal usage logging.

    if dryrun:
      self.dryrun(filter=filter, verbose=False, debug=debug)

    summary = self.summary(pivot_field) if from_root and summary else None
    should_save = bool(save and self.dir)

    if self.is_leaf:
      if isinstance(show_progress, bool):
        if show_progress:
          progress_bar = lf.concurrent.ProgressBar.install(
              label=label, total=self.num_examples, color='blue')
        else:
          progress_bar = None
      else:
        progress_bar = show_progress

      run_status = 'FIRST_RUN'
      if self.dir and pg.io.path_exists(
          os.path.join(self.dir, Evaluable.EXPERIMENT_JSON)
      ):
        if show_progress:
          lf.concurrent.ProgressBar.update(
              progress_bar, status='LOADING SAVED RESULTS...', color='yellow'
          )
        if self.try_load_result():
          run_status = 'CACHED'

      if rerun and self.result:
        self._result = None
        run_status = 'RERUN'

      if self.result:
        if show_progress:
          lf.concurrent.ProgressBar.update(
              progress_bar, delta=self.num_examples
          )
      else:
        self._run(
            start=start,
            end=end,
            debug=debug,
            dryrun=dryrun,
            verbose=verbose,
            progress_bar=progress_bar,
            label=label,
            timeout=timeout,
            **kwargs,
        )

        if should_save:
          if show_progress:
            lf.concurrent.ProgressBar.update(
                progress_bar, status='SAVING RESULTS...', color='yellow'
            )

          # Save evaluation results.
          self.save()

          # Save summary if present.
          if summary:
            summary.save(os.path.join(self.root_dir, Evaluable.SUMMARY_HTML))

      if show_progress:
        lf.concurrent.ProgressBar.update(
            progress_bar,
            status=self._completion_status(run_status),
            color='green',
        )
    else:
      assert from_root
      summary_lock = threading.Lock()
      def _run_group(arg: tuple[int, list[_LeafNode]]) -> None:
        overview_bar, leaf_group = arg
        for leaf in leaf_group:
          if leaf.enabled:
            leaf.node.run(
                start=start,
                end=end,
                rerun=rerun,
                save=save,
                debug=debug,
                dryrun=False,
                verbose=verbose,
                show_progress=leaf.progress_bar,
                summary=False,
                from_root=False,
                **kwargs,
            )
            if should_save and summary:
              with summary_lock:
                summary.save(
                    os.path.join(self.root_dir, Evaluable.SUMMARY_HTML)
                )

          # Signal sub-eval complete by setting the color green.
          lf.concurrent.ProgressBar.uninstall(leaf.progress_bar)
          lf.concurrent.ProgressBar.update(overview_bar, 1, {
              'LastCompleted': leaf.node.id
          })

      # NOTE(daiyip): Run leaf nodes grouped by model resource id. This allows
      # evaluations using the same resource to run sequentially, which favors
      # completing evaluations over running evaluations sparsely.
      filter = filter or (lambda x: True)
      leaf_nodes: list[_LeafNode] = []
      leaf_groups: dict[str, list[_LeafNode]] = collections.defaultdict(list)

      for i, leaf in enumerate(self.leaf_nodes):
        node = _LeafNode(index=i + 1, node=leaf, enabled=filter(leaf))
        leaf_groups[leaf.lm.resource_id].append(node)
        leaf_nodes.append(node)

      if leaf_groups:
        # Install progress bars.
        overview_bar = lf.concurrent.ProgressBar.install(
            'All', len(self.leaf_nodes), color='Blue')
        for leaf in leaf_nodes:
          leaf.progress_bar = lf.concurrent.ProgressBar.install(
              f'[#{leaf.index} - {leaf.node.id}]',
              total=leaf.node.num_examples if leaf.enabled else 0,
              color='cyan' if leaf.enabled else 'yellow',
              status=None if leaf.enabled else 'SKIPPED.')

        # Run leaf groups in parallel.
        try:
          for _, _, _ in lf.concurrent_map(
              _run_group,
              [(overview_bar, group) for group in leaf_groups.values()],
              silence_on_errors=None,
              max_workers=len(leaf_groups)):
            pass

          # Save results for non-leaf nodes.
          lf.concurrent.ProgressBar.update(
              overview_bar,
              status='SAVING RESULTS...',
              color='yellow')

          for node in self.nonleaf_nodes:
            node._result = {c.id: c.result for c in node.leaf_nodes}  # pylint: disable=protected-access
            if should_save:
              node.save(result=False, report=False)

          if should_save and summary:
            lf.concurrent.ProgressBar.update(
                overview_bar, status='FINALIZING SUMMARY...'
            )

            summary.save(os.path.join(self.root_dir, Evaluable.SUMMARY_HTML))

            lf.console.write(
                f'({self.summary_link})',
                color='magenta',
                styles=['underline'],
            )

          # Signal all task completed by making the bar green.
          lf.concurrent.ProgressBar.update(
              overview_bar,
              status='COMPLETED',
              color='green')

        finally:
          # for leaf in leaf_nodes:
          #   lf.concurrent.ProgressBar.uninstall(leaf.progress_bar)
          lf.concurrent.ProgressBar.uninstall(overview_bar)
    return summary or self.result

  @abc.abstractmethod
  def _run(
      self,
      *,
      start: int,
      end: int,
      debug: bool | lf.LMDebugMode,
      dryrun: bool,
      verbose: bool,
      progress_bar: int | None,
      label: str | None,
      timeout: int | None = None,
      **kwargs,
  ) -> None:
    """Run the evaluate and fill `self.result`. Subclass to implement."""

  @abc.abstractmethod
  def _completion_status(self, run_status: str) -> str:
    """Returns the status for progress bar when evaluation completes."""

  @property
  @abc.abstractmethod
  def hash(self) -> str:
    """A 8-byte MD5 hash computed from experiment identity."""

  @functools.cached_property
  def parent(self) -> Optional['Evaluable']:
    """Parent evaluable."""
    parent = self.sym_parent
    while parent is not None and not isinstance(parent, Evaluable):
      parent = parent.sym_parent
    return parent

  def load_result(self) -> None:
    """Load saved evaluation results and details."""
    if self.dir is None:
      raise ValueError('`dir` must not be None.')

    with pg.catch_errors(FileNotFoundError):
      self._result = pg.load(os.path.join(self.dir, Evaluation.RESULT_JSON))

  def save(
      self, definition: bool = True, result: bool = True, report: bool = True
  ) -> None:
    # Save experiment definition.
    if definition:
      pg.save(self, os.path.join(self.dir, Evaluable.EXPERIMENT_JSON))

    # Save evaluation result.
    if result:
      pg.save(self.result, os.path.join(self.dir, Evaluation.RESULT_JSON))

  def _html(
      self,
      body_builders: list[Callable[[io.StringIO], None]],
      include_def: bool = False,
      include_cache_stats: bool = False,
  ) -> str:
    s = io.StringIO()
    s.write('<html><head><style>')
    self._render_styles(s)
    s.write(
        '</style></head><body>'
        f'<h1 style="color:blue;background-color:#DDDDDD">{self.id}</h1>'
    )
    self._render_navbar(s)
    for builder in body_builders:
      builder(s)
    if include_def:
      s.write(
          '<h2> Definition </h2>'
          '<div style="white-space:pre;padding:10px;color:#3254a8;'
          'background-color:#EEEEEE">'
          f'{self.format(compact=False)}</div>'
      )
    if include_cache_stats and self.is_deterministic:
      s.write(
          '<h2> Cache Stats </h2>'
          '<div style="white-space:pre;padding:10px;color:#3254a8;'
          f'background-color:#EEEEEE">{self.result.cache_stats}</div>'
      )
    s.write('</body></html>')
    return s.getvalue()

  def _render_styles(self, s: io.StringIO) -> None:
    s.write("""
        td {padding: 5px;}
        .header {font-weight: bold;}
        """)

  def _render_navbar(self, s: io.StringIO) -> None:
    links = [
        f'<a href="{self.summary_link}">Summary</a>',
        f'<a href="{self.index_link}">{self.id}</a>',
    ]
    for i, link in enumerate(links):
      s.write(link)
      if i != len(links) - 1:
        # Add a right triangle symbol.
        s.write(' &#9656 ')
    s.write(f' [<a href="{self.link(self.dir)}">Directory</a>]')

  def _render_index_page(self, s: io.StringIO) -> None:
    self._render_result(s)
    if self.dryrun_output is not None:
      self._render_dryrun_output(s)

  def _render_result(self, s: io.StringIO) -> None:
    s.write(
        '<h2> Result </h2>'
        '<table style="border:1px solid;"><tr class="header">'
    )
    if self.children:
      s.write('<td>ID</td>')

    self._render_result_header(s)
    s.write('<tr>')
    if self.children:
      for c in self.children:
        s.write(
            '<tr>'
            f'<td><a href={c.index_link}>{c.id}</a></td>'
        )
        c._render_result_row(s)  # pylint: disable=protected-access
        s.write('</tr>')
    else:
      s.write('<tr>')
      self._render_result_row(s)
      s.write('</tr>')
    s.write('</table>')

  def _render_result_header(self, s: io.StringIO) -> None:
    """Render result header."""

  def _render_result_row(self, s: io.StringIO) -> None:
    """Render result row."""

  def _render_dryrun_output(self, s: io.StringIO) -> None:
    s.write('<h2> Dry Run </h2>')
    self._render_message(self.dryrun_output, s)

  def _render_message(self, message: lf.Message, s: io.StringIO) -> None:
    s.write(
        message.to_html_str(
            extra_flags=dict(
                include_message_metadata=False,
                source_tag=['lm-input', 'lm-response'],
            )
        )
    )

  @classmethod
  def from_dir(
      cls, maybe_dir: str, load_result: bool = True
  ) -> Optional['Evaluable']:
    exp_json = os.path.join(maybe_dir, Evaluable.EXPERIMENT_JSON)
    if not pg.io.path_exists(exp_json):
      return None

    experiment: Evaluable = pg.load(exp_json)
    experiment.rebind(
        root_dir=os.path.abspath(os.path.join(maybe_dir, os.path.pardir))
    )
    if load_result:
      experiment.try_load_result()
    return experiment

  def try_load_result(self) -> bool:
    """Try load result."""
    if self.result is None:
      result_json = os.path.join(self.dir, Evaluable.RESULT_JSON)
      if pg.io.path_exists(result_json):
        self._result = pg.load(result_json)
        return True
    return False


@dataclasses.dataclass
class _LeafNode:
  """Information for leaf node execution."""
  index: int
  node: 'Evaluation'
  enabled: bool = True
  progress_bar: int | None = None


@pg.use_init_args(['children'])
class Suite(Evaluable):
  """Evaluation suite."""
  children: Annotated[list[Evaluable], 'Child evaluation sets or suites.']

  # Use empty ID as suite is just a container of child evaluations.
  id: str = ''

  __kwargs__: Annotated[
      Any,
      (
          'Wildcard keyword arguments for `__init__` that can be accessed from '
          'parent suite if the argument is absent from current evaluation set.'
      ),
  ]

  def _on_bound(self):
    super()._on_bound()
    overrides = {
        k: v for k, v in self.sym_init_args.items()
        if k not in ('id', 'children')
    }
    for child in self.children:
      child.rebind(overrides, notify_parents=False)
    self.__dict__.pop('hash', None)

  @functools.cached_property
  def hash(self) -> str:
    return hashlib.md5(
        ' '.join(sorted([c.hash for c in self.children])).encode()
    ).hexdigest()[:8]

  def _dryrun(self, *args, **kwargs) -> None:
    raise AssertionError('Shal not trigger.')

  def _run(self, *args, **kwargs) -> None:
    raise AssertionError('Shall not trigger.')

  def _completion_status(self, run_status: str) -> str:
    return f'COMPLETED({run_status})'


class Evaluation(Evaluable):
  """Base class for evaluation set."""

  inputs: pg.typing.Annotated[
      pg.typing.Functor(),
      (
          'A functor that returns a list of user-defined objects as the input '
          'examples. It could be inputs loaded from a JSON file via '
          '`lf.eval.inputs_from(path)`, from a Python coded list via '
          '`lf.eval.as_inputs(values)` or a user-defined functor that '
          'generates input objects at runtime.'
      ),
  ]

  method: Annotated[
      Literal['call', 'query', 'complete'], 'Method for symbolic prompting.'
  ] = lf.contextual(default='query')

  prompt: Annotated[
      lf.Template,
      (
          'Template for rendering the template. Example object could be '
          'accessed via `example`.'
      ),
  ] = lf.contextual()

  schema_fn: pg.typing.Annotated[
      pg.typing.Functor().noneable(),
      (
          'A functor that returns a type annotation that will be converted to '
          '`lf.Schema`, or a tuple of (annotation, fewshot examples). '
          'For "call" method, it could be None, indicating that the raw '
          'response from the LM will be used as the output, and the fewshot '
          'examples will be used for parsing. For "query" and "complete", it '
          'must be provided, and the fewshot examples will be used directly '
          'for prompting. Here are the example code on how the '
          'functors should be defined:'
          + inspect.cleandoc("""
              ```
              @pg.functor()
              def solution():
                class Solution(pg.Object):
                  final_answer: int
                return Solution

              @pg.functor()
              def solution_with_fewshot_examples():
                class Solution(pg.Object):
                  final_answer: int
                return Solution, [
                    lf.structured.MappingExample(
                        input='Compute 1 + 2',
                        output=Solution(3),
                        schema=Solution)
                ]
              ```
              """)
      ),
  ] = lf.contextual()

  lm: Annotated[lf.LanguageModel, 'Language model to use for evaluation.'] = (
      lf.contextual()
  )

  parsing_lm: Annotated[
      lf.LanguageModel | None,
      (
          'Language model for parsing. Applicable only when method is set'
          'to `call`. If None, `lm` will also be used for parsing. '
      ),
  ] = lf.contextual(default=None)

  completion_prompt_field: Annotated[
      str | None,
      (
          'A str field that will be automatically added to the class of the '
          'input object for `lf.complete`. If None, no field will be added to '
          'the class, instead the prompt will be passed as the first argument '
          'of the input object to complete. Applicable only when `method` is '
          'set to `complete`.'
      ),
  ] = lf.contextual(default=None)

  autofix: Annotated[
      int,
      (
          'The number of attempts for auto fix, which allows LLMs to correct '
          'generated code for the output structure. If 0, autofix will be '
          'disabled.'
      ),
  ] = lf.contextual(default=0)

  autofix_lm: Annotated[
      lf.LanguageModel | None,
      (
          'Language model for autofix. If None, `lm` will also be used for '
          'autofix.'
      ),
  ] = lf.contextual(default=None)

  additional_args: Annotated[
      dict[str, Any] | None,
      'Additional kwargs that will be passed to `self.process`',
  ] = lf.contextual(default=None)

  use_cache: Annotated[bool, 'If True, LM cache will be enabled.'] = (
      lf.contextual(default=True)
  )

  max_workers: Annotated[
      int, 'Max workers to run the evaluation in parallel.'
  ] = 32

  # Constants.
  CACHE_JSON = 'cache.json'
  OOP_FAILURES_HTML = 'oop_failures.html'
  NON_OOP_FAILURES_HTML = 'non_oop_failures.html'

  @functools.cached_property
  def hash(self) -> str:
    """Returns the semantic-based hash of the evaluation."""
    if self.is_deterministic:
      identity = pg.format(self._identifiers(), compact=True)
    else:
      identity = ' '.join(sorted([c.hash for c in self.children]))
    return hashlib.md5(identity.encode()).hexdigest()[:8]

  def _identifiers(self) -> dict[str, Any]:
    parsing_model = None
    if self.method == 'call':
      parsing_model = self.parsing_lm or self.lm
    return {
        'task': self.__class__.__name__,
        'model': self.lm.model_id,
        'parsing_model': getattr(parsing_model, 'model_id', None),
        'sampling_options': self.lm.sampling_options,
        'inputs': self.inputs,
        'prompt': self.prompt,
        'method': self.method,
        'schema_fn': self.schema_fn,
        # Make unspecified additional_args (None) and empty additional_args
        # ({}) to produce the same hash.
        'additional_args': self.additional_args or {},
    }

  @functools.cached_property
  def examples(self):
    """Returns examples for evaluation."""
    kwargs = {}
    # Allow inputs to be dependent on current evaluation.
    if 'evaluation' in self.inputs.__signature__.arg_names:
      kwargs['evaluation'] = self
    return self.inputs(**kwargs)

  @property
  def num_examples(self) -> int:
    """Returns the number of examples for evaluation."""
    return len(self.examples)

  @property
  def num_completed(self) -> int:
    """Returns the number of completed examples."""
    return self._num_completed

  @property
  def complete_rate(self) -> float:
    """Returns the complete rate."""
    return self.num_completed / self.num_examples

  #
  # Properties on failures.
  #

  @property
  def failures(self) -> list[tuple[Any, Exception]]:
    """Returns the failed examples and their errors."""
    return self._failures

  @property
  def num_failures(self) -> int:
    """Returns the number of failed examples."""
    return len(self.failures)

  @functools.cached_property
  def failure_breakdown(self) -> dict[str, int]:
    """Returns the breakdown of failures."""
    breakdown = collections.defaultdict(int)
    for _, error in self.failures:
      breakdown[_error_key(error)] += 1
    sorted_items = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
    return pg.Dict({x[0]: x[1] for x in sorted_items})

  @property
  def failure_rate(self) -> float:
    """Returns the failure rate in range [0, 1]."""
    if self.num_completed == 0:
      return 0.0
    return self.num_failures / self.num_completed

  @functools.cached_property
  def oop_failures(self) -> list[tuple[Any, lf_structured.MappingError]]:
    """Returns the OOP failures."""
    return [item for item in self.failures
            if isinstance(item[1], lf_structured.MappingError)]

  @property
  def num_oop_failures(self) -> int:
    """Returns the number of OOP failures."""
    return len(self.oop_failures)

  @property
  def oop_failure_rate(self) -> float:
    """Returns the OOP failure rate in range [0, 1]."""
    if self.num_completed == 0:
      return 0.0
    return self.num_oop_failures / self.num_completed

  @functools.cached_property
  def non_oop_failures(self) -> list[tuple[Any, Exception]]:
    """Returns the OOP failures."""
    return [item for item in self.failures
            if not isinstance(item[1], lf_structured.MappingError)]

  @property
  def num_non_oop_failures(self) -> int:
    """Returns the number of non-OOP failures."""
    return len(self.non_oop_failures)

  @property
  def non_oop_failure_rate(self) -> float:
    """Returns the non-OOP failure rate in range [0, 1]."""
    if self.num_completed == 0:
      return 0.0
    return self.num_non_oop_failures / self.num_completed

  #
  # Properties on usage.
  #

  @property
  def has_usage(self) -> bool:
    """Returns True if token usage is enabled."""
    return self._num_usages > 0

  @property
  def average_prompt_tokens(self) -> int:
    """Returns the average prompt tokens."""
    if not self.has_usage:
      return 0
    return self._total_prompt_tokens // self._num_usages

  @property
  def average_completion_tokens(self) -> int:
    """Returns the average completion tokens."""
    if not self.has_usage:
      return 0
    return self._total_completion_tokens // self._num_usages

  @property
  def average_total_tokens(self) -> int:
    """Returns the average total tokens."""
    return self.average_prompt_tokens + self.average_completion_tokens

  @functools.cached_property
  def schema(self) -> lf_structured.Schema | None:
    """Schema."""
    if self.schema_fn is None:
      return None

    schema = self._call_schema_fn()
    fewshot_examples = None
    if isinstance(schema, tuple):
      schema, fewshot_examples = schema
    self.__dict__['fewshot_examples'] = (
        self._maybe_adjust_examples_for_completion(fewshot_examples))
    return self._formalize_schema(schema)

  @functools.cached_property
  def fewshot_examples(self) -> list[lf.structured.MappingExample] | None:
    """Fewshot examples."""
    if self.schema_fn is None:
      return None

    schema = self._call_schema_fn()
    fewshot_examples = None
    if isinstance(schema, tuple):
      schema, fewshot_examples = schema
    self.__dict__['schema'] = self._formalize_schema(schema)
    return self._maybe_adjust_examples_for_completion(fewshot_examples)

  def _call_schema_fn(self):
    kwargs = {}
    # Allow schema to be a function based on current evaluation.
    if 'evaluation' in self.schema_fn.__signature__.arg_names:
      kwargs['evaluation'] = self
    return self.schema_fn(**kwargs)

  def _formalize_schema(self, annotation) -> lf_structured.Schema | None:
    """Formalizes schema from annotation."""
    if annotation in (str, None):
      return None
    if self.method == 'complete':
      if not hasattr(annotation, '__schema__'):
        raise TypeError(
            'The annotation returned by `schema_fn` must be a `pg.Object` '
            'subclassclass to be used for `lf.complete`. '
            'Encountered: {annotation!r}.'
        )
      self._maybe_adjust_schema_for_completion(annotation)
    schema = lf_structured.Schema.from_value(annotation)
    # NOTE(daiyip): add references to the dependent classes of the returned type
    # to prevent unused subclasses get garbage collected by Python.
    setattr(schema, '__dependencies__', schema.class_dependencies())
    return schema

  def _maybe_adjust_schema_for_completion(self, cls):
    if (self.completion_prompt_field is None
        or self.completion_prompt_field in cls.__schema__):
      return

    fields = list(cls.__schema__.values())
    fields.insert(0, (self.completion_prompt_field, pg.typing.Str()))
    cls.update_schema(fields, extend=False)

  def _maybe_adjust_examples_for_completion(
      self,
      fewshot_examples: list[lf.structured.MappingExample] | None
      ) -> list[lf.structured.MappingExample] | None:
    if (not fewshot_examples
        or self.completion_prompt_field is None
        or self.method != 'complete'):
      return fewshot_examples

    completion_examples = []
    for ex in fewshot_examples:
      example_cls = ex.output.__class__
      self._maybe_adjust_schema_for_completion(example_cls)
      ex = lf.structured.MappingExample(
          context=ex.context,
          input=example_cls.partial(ex.input),
          output=example_cls(ex.input, **ex.output.sym_init_args),
      )
      completion_examples.append(ex)
    return completion_examples

  @property
  def id(self) -> str:
    """Returns the ID of this evaluation."""
    id_prefix = self.__class__.__name__
    if not self.is_deterministic:
      return id_prefix
    return f'{id_prefix}@{self.hash}'

  @functools.cached_property
  def children(self) -> list['Evaluation']:
    """Returns the trials as child evaluations if this evaluation is a space."""
    if self.is_deterministic:
      return []
    children = []
    for i, child in enumerate(pg.iter(self)):
      child.sym_setparent(self)
      child.sym_setpath(self.sym_path + f'children[{i}]')
      children.append(child)
    return children

  @functools.cached_property
  def cache(self) -> lf.LMCache | None:
    """Returns LM cache to use."""
    if not self.use_cache:
      return None

    cache_file = None
    if self.dir:
      cache_file = os.path.join(self.dir, Evaluation.CACHE_JSON)
    return in_memory.InMemory(cache_file)

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('hash', None)
    self.__dict__.pop('children', None)
    self.__dict__.pop('examples', None)
    self.__dict__.pop('schema', None)
    self.__dict__.pop('fewshot_examples', None)
    self.__dict__.pop('cache', None)

  def _reset(self):
    super()._reset()
    self._failures = []
    self._num_completed = 0

    self._total_prompt_tokens = 0
    self._total_completion_tokens = 0
    self._num_usages = 0
    self.__dict__.pop('oop_failures', None)
    self.__dict__.pop('non_oop_failures', None)

  @property
  def oop_failures_link(self) -> str | None:
    """Returns the link to the OOP failures page."""
    if self.dir is None:
      return None
    return self.link(os.path.join(self.dir, Evaluation.OOP_FAILURES_HTML))

  @property
  def non_oop_failures_link(self) -> str | None:
    """Returns the link to then non-OOP failures page."""
    if self.dir is None:
      return None
    return self.link(os.path.join(self.dir, Evaluation.NON_OOP_FAILURES_HTML))

  def _dryrun(
      self,
      *,
      example: Any,
      debug: bool | lf.LMDebugMode,
      verbose: bool,
      **kwargs,
  ) -> None:
    # We make a copy to avoid pollute the state of current object.
    copy: Evaluation = self.clone()

    # Set the example for dryrun.
    example = example or copy.examples[0]
    copy.__dict__['examples'] = [example]

    # We set the symbolic parent of the cloned to access contextual information
    # when needed.
    copy.sym_setparent(self.sym_parent)

    # Process the input.
    if verbose:
      lf.console.write(
          str(example),
          title='INPUT',
          color='green',
      )

    error, output_message = None, None

    try:
      with lf.use_settings(debug=debug):
        output_message = copy.process(example, **(self.additional_args or {}))
        self.process_output(example, output_message)

        if self.schema is None:
          output = output_message.text
        else:
          output = output_message.result

      if verbose:
        lf.console.write('')
        lf.console.write(
            str(output),
            title='OUTPUT',
            color='blue',
        )
    except lf_structured.MappingError as e:
      lf.console.write('')
      lf.console.write(
          str(e),
          title='ERROR',
          color='red',
      )
      error = e

    copy.audit(1, example, output_message, error, dryrun=True)
    result = copy.finalize()

    if verbose:
      lf.console.write('')
      lf.console.write(
          str(result),
          title='RESULT',
          color='magenta',
      )
    self._dryrun_output = output_message
    return

  def _run(
      self,
      *,
      start: int = 0,
      end: int | None,
      debug: bool | lf.LMDebugMode,
      verbose: bool,
      progress_bar: int | None,
      label: str | None,
      timeout: int | None = None,
      **kwargs,
  ) -> None:
    # Setup examples.
    # Reset examples so it could be read from the input functor.
    self.__dict__.pop('examples', None)

    if end is None:
      end = len(self.examples)
    examples = self.examples[start:end]

    # Process examples.
    with lf.use_settings(debug=debug, cache=self.cache):
      self._reset()

      def _process(idx_and_example: Any):
        # NOTE(daiyip): set the `input` symbol of the globals to None, so LLM
        # generated code with calls to `input` will raise an error, thus not
        # blocking the evaluation.
        _, example = idx_and_example
        with lf_coding.context(input=None):
          output_message = self.process(example, **(self.additional_args or {}))
          self.process_output(example, output_message)
          return output_message

      try:
        for (idx, example), message, error in lf.concurrent_map(
            _process,
            enumerate(examples),
            max_workers=self.max_workers,
            show_progress=progress_bar or False,
            status_fn=self._status,
            timeout=timeout,
        ):
          if error is not None:
            message = (
                error.lm_response
                if isinstance(error, lf_structured.MappingError)
                else None
            )
          self.audit(idx + 1, example, message, error)
      finally:
        # Save cache upon completion or interruption.
        if self.dir and self.cache:
          self.cache.save()

    # Summarize result.
    self._result = self.finalize()
    if verbose:
      lf.console.write(
          str(self.result),
          title=f'RESULT ON {self.id}',
          color='magenta',
      )

  def call_postprocess(self, lm_response: str) -> str:
    """Post-process for `lf.call`. Subclass can override."""
    return lm_response

  def process(self, example: Any, **kwargs) -> lf.Message:
    """Process an example and returns its output."""
    prompt = lf.Template.from_value(self.prompt, example=example)
    if self.method == 'call':
      return lf_structured.call(
          prompt,
          self.schema,
          lm=self.lm,
          parsing_lm=self.parsing_lm,
          parsing_examples=self.fewshot_examples,
          response_postprocess=self.call_postprocess,
          autofix=self.autofix,
          autofix_lm=self.autofix_lm,
          returns_message=True,
          **kwargs,
      )
    elif self.method == 'query':
      return lf_structured.query(
          prompt,
          self.schema,
          lm=self.lm,
          examples=self.fewshot_examples,
          autofix=self.autofix,
          autofix_lm=self.autofix_lm,
          returns_message=True,
          **kwargs,
      )
    else:
      assert self.method == 'complete', self.method
      assert isinstance(self.schema.spec, pg.typing.Object), self.schema
      # TODO(daiyip): Currently multi-modal inputs within the prompt for
      # completion is not supported.
      input_value = self.schema.spec.cls.partial(prompt.render().text)
      return lf_structured.complete(
          input_value,
          lm=self.lm,
          examples=self.fewshot_examples,
          autofix=self.autofix,
          autofix_lm=self.autofix_lm,
          returns_message=True,
          **kwargs,
      )

  def process_output(self, example: Any, output: lf.Message) -> None:
    """Process the output for an example.

    Subclasses can override this method to generate and attach additional
    metadata for debugging purpose. For example, draw bounding boxes on the
    input image based on LLM predicted boxes and attach to output_message's
    metadata.

    Example:

      class BoundingBoxEval(lf.eval.Matching):
        ...
        def process_output(example, output):
          output.metadata.image_with_bbox = draw_bboxes(
              example.image, output.result)

    Args:
      example: User input.
      output: LLM's output message. Users could attach additional
        information to the message, which will be shown in debugging
    """
    del example, output

  def _status(self, progress: lf.concurrent.Progress) -> dict[str, Any]:
    status = {'Model': self.lm.model_id}
    status.update(self._eval_status(progress))

    if progress.last_error is not None:
      status['LastError'] = progress.last_error_str()
    if progress.timeit_summary:
      status['TimeIt'] = progress.timeit_summary_str()
    return status

  def _eval_status(self, progress: lf.concurrent.Progress) -> dict[str, Any]:
    return {
        'Succeeded': '%s (%d/%d)' % (
            self._format_rate(progress.success_rate),
            progress.succeeded,
            progress.completed,
        ),
        'Failed': '%s (%d/%d)' % (
            self._format_rate(progress.failure_rate),
            progress.failed,
            progress.completed,
        ),
    }

  def _completion_status(self, run_status: str) -> str:
    assert self.result is not None
    m = self.result.metrics
    return (
        'COMPLETED(%s): Successes=%s(%d/%d) Failures=%s (%d/%d)'
        % (
            run_status,
            self._format_rate(1 - m.failure_rate),
            m.total - m.failures,
            m.total,
            self._format_rate(m.failure_rate),
            m.failures,
            m.total,
        )
    )

  def finalize(self) -> pg.Dict:
    """Finalizes the evaluation result."""
    if self.cache is not None:
      cache_stats = dict(
          use_cache=True,
          num_queries=self.cache.stats.num_queries,
          num_hits=self.cache.stats.num_hits,
          num_updates=self.cache.stats.num_updates,
      )
    else:
      cache_stats = dict(use_cache=False)

    if self.has_usage:
      usage = pg.Dict(
          total_prompt_tokens=self._total_prompt_tokens,
          total_completion_tokens=self._total_completion_tokens,
          num_usages=self._num_usages,
          average_prompt_tokens=self.average_prompt_tokens,
          average_completion_tokens=self.average_completion_tokens,
          average_total_tokens=self.average_total_tokens,
      )
    else:
      usage = None

    result = pg.Dict(
        experiment_setup=pg.Dict(
            id=self.id,
            dir=self.dir,
            model=self.lm.model_id,
            prompt_template=pg.decolor(str(self.prompt)),
            method=self.method,
            schema_fn=str(self.schema_fn),
        ),
        cache_stats=cache_stats,
        metrics=pg.Dict(
            total=self.num_completed,
            failures=self.num_failures,
            failure_rate=self.failure_rate,
            oop_failures=self.num_oop_failures,
            oop_failure_rate=self.oop_failure_rate,
            non_oop_failures=self.num_non_oop_failures,
            non_oop_failure_rate=self.non_oop_failure_rate,
            failure_breakdown=self.failure_breakdown,
        ),
        usage=usage,
    )
    return result

  def summary_card(self) -> str:
    """Returns summary card in HTML."""
    s = io.StringIO()
    definition = _html_repr(self, compact=False, escape=True)
    s.write('<div><table><tr><td>')
    self._render_link(
        s,
        definition,
        self.hash,
        '',
        lambda: self.link(self.dir),
    )
    if self.result is None:
      s.write(
          '</td></tr><tr><td>'
          '<span style="color: gray">(IN-PROGRESS...)</span>'
      )
    else:
      if self.dir:
        s.write(f' &nbsp;[<a href="{self.link(self.dir)}">dir</a>]')
      s.write('</td></tr><tr><td>')
      self._render_summary_metrics(s)

      # Summarize average usage.
      if self.result.usage:
        self._render_summary_usage(s)

    s.write('</td></tr></table></div>')
    return s.getvalue()

  def _render_summary_usage(self, s: io.StringIO) -> None:
    """Renders usage in HTML."""
    usage = self.result.usage
    total = usage.total_prompt_tokens + usage.total_completion_tokens
    s.write(
        '&nbsp;<a title="'
        f'# of usages: {usage.num_usages}&#013;'
        f'total prompt: {usage.total_prompt_tokens}&#013;'
        f'total response: {usage.total_completion_tokens}&#013;'
        f'avg prompt: {usage.average_prompt_tokens}&#013;'
        f'avg response: {usage.average_completion_tokens}'
        f'" style="color:gray">({total} tokens)</a>'
    )

  def _render_link(self,
                   s: io.StringIO,
                   title: str,
                   text: str,
                   style: str,
                   url_fn: Callable[[], str]) -> None:
    """Renders a link in HTML."""
    s.write(
        f'<a target="_blank" title="{title}" style="{style}"'
    )
    if self.dir:
      s.write(f' href="{url_fn()}"')
    s.write(f'>{text}</a>')

  def _render_summary_metrics(self, s: io.StringIO) -> None:
    """Renders metrics in HTML."""
    assert self.result is not None
    m = self.result.metrics

    # OOP failures.
    oop_failure_title = f'OOP failures ({m.oop_failures}/{m.total})'
    if m.oop_failures:
      oop_failure_title += '&#013;'
      for name, count in m.failure_breakdown.items():
        if name.startswith('MappingError'):
          oop_failure_title += '&#013;%s: %s (%d/%d)' % (
              name.removeprefix('MappingError.'),
              self._format_rate(count / m.total),
              count,
              m.total,
          )

    extra_style = ''
    if m.oop_failure_rate > 0.1 and m.oop_failures > 3:
      extra_style = ';font-weight:bold'
    self._render_link(
        s,
        oop_failure_title,
        self._format_rate(m.oop_failure_rate),
        f'color:magenta{extra_style}',
        lambda: self.oop_failures_link,
    )
    s.write(' | ')

    # Non-OOP failures.
    non_oop_failure_title = f'Non-OOP failures ({m.non_oop_failures}/{m.total})'
    if m.non_oop_failures:
      non_oop_failure_title += '&#013;'
      for name, count in m.failure_breakdown.items():
        if not name.startswith('MappingError'):
          non_oop_failure_title += '&#013;%s: %s (%d/%d)' % (
              name,
              self._format_rate(count / m.total),
              count,
              m.total,
          )

    extra_style = ';font-weight:bold' if m.non_oop_failures > 0 else ''
    self._render_link(
        s,
        non_oop_failure_title,
        self._format_rate(m.non_oop_failure_rate),
        f'color:red{extra_style}',
        lambda: self.non_oop_failures_link,
    )

  def _format_rate(self, rate: float) -> str:
    """Formats a rate."""
    return f'%.{self.report_precision}f%% ' % (rate * 100)

  def audit(
      self,
      example_idx: int,
      example: Any,
      message: lf.Message | None,
      error: Exception | None = None,
      dryrun: bool = False,
  ) -> None:
    """Audits the example against the output. Subclasses should override.

    Args:
      example_idx: 1-based index of the example in its dataset.
      example: The input object.
      message: The entire message returned by the LM, which could be used to
        trace the LM input, response and parsed structure. If error is raised
        before LLM could return a response, None will be its value.
      error: The exception during processing the example.
      dryrun: Whether or not audition takes place during dryrun.
    """
    if error is not None:
      self._failures.append((example, error))

      # Invalid cache of num_oop_failures.
      self.__dict__.pop('oop_failures', None)
      self.__dict__.pop('non_oop_failures', None)
      self.__dict__.pop('failure_breakdown', None)

      if isinstance(error, lf_structured.MappingError):
        message = error.lm_response
    else:
      assert message is not None
      output = message.text if self.schema is None else message.result
      self.audit_processed(example_idx, example, output, message, dryrun=dryrun)

    # Audit usage.
    if message is not None:
      self.audit_usage(message, dryrun=dryrun)
    self._num_completed += 1

  def audit_usage(self, message: lf.Message, dryrun: bool = False) -> None:
    del dryrun
    for m in message.trace():
      usage = m.metadata.get('usage', None)
      if usage:
        self._total_prompt_tokens += usage.prompt_tokens
        self._total_completion_tokens += usage.completion_tokens
        self._num_usages += 1

  def audit_processed(
      self, example_idx: int, example: Any, output: Any, message: lf.Message,
      dryrun: bool = False
  ) -> None:
    """Audits a successfully processed example. Subclass should override."""

  def save(
      self, definition: bool = True, result: bool = True, report: bool = True
  ) -> None:
    """Save evaluation details."""
    super().save(definition, result, report)

    if report:
      # Save index page.
      pg.save(
          self._html(
              [self._render_index_page],
              include_def=True,
              include_cache_stats=True,
          ),
          os.path.join(self.dir, Evaluable.INDEX_HTML),
          file_format='txt',
      )

      # Save failures.
      pg.save(
          [
              pg.Dict(input=input, error=_format_error(error))
              for input, error in self.oop_failures
          ],
          os.path.join(self.dir, Evaluation.OOP_FAILURES_JSON),
      )
      pg.save(
          self._html([self._render_result, self._render_oop_failures]),
          os.path.join(self.dir, Evaluation.OOP_FAILURES_HTML),
          file_format='txt',
      )
      pg.save(
          [
              pg.Dict(input=input, error=_format_error(error))
              for input, error in self.non_oop_failures
          ],
          os.path.join(self.dir, Evaluation.NON_OOP_FAILURES_JSON),
      )
      pg.save(
          self._html([self._render_result, self._render_non_oop_failures]),
          os.path.join(self.dir, Evaluation.NON_OOP_FAILURES_HTML),
          file_format='txt',
      )

  def _render_result_header(self, s: io.StringIO) -> None:
    s.write(
        '<td>Method</td>'
        '<td>Inputs</td>'
        '<td>Model</td>'
        '<td>Prompt</td>'
        '<td>Schema</td>'
        '<td>Additional Args</td>'
    )
    if self.result.usage:
      s.write('<td>Usage</td>')
    s.write('<td>OOP Failures</td>')
    s.write('<td>Non-OOP Failures</td>')

  def _render_result_row(self, s: io.StringIO) -> None:
    s.write(
        f'<td style="color:{self._method_color}">{self.method}</td>'
        f'<td style="color:darkgray">{_html_repr(self.inputs)}</td>'
        f'<td style="color:#494a5c">{_html_repr(self.lm)}</td>'
    )
    # Prompt.
    prompt_title = _html_repr(self.prompt.template_str, escape=True)
    s.write(
        f'<td title="{prompt_title}"'
        f'style="color:darkgray">{_html_repr(self.prompt)}</td>'
    )
    # Schema.
    schema_title = self.schema.schema_str('python') if self.schema else None
    s.write(
        '<td style="color:purple" '
        f'title="{schema_title}">'
        f'{_html_repr(self.schema_fn)}</td>'
    )
    s.write(
        '<td style="color:purple" '
        f'{_html_repr(self.additional_args, compact=False)}</td>'
    )
    # Usage.
    if self.result.usage:
      s.write('<td>')
      self._render_summary_usage(s)
      s.write('</td>')

    # OOP failures.
    s.write(
        '<td><span style="color:magenta">%s</span>%s</td>'
        % (
            self._format_rate(self.oop_failure_rate),
            '<a href="%s">(%d/%d)</a>'
            % (self.oop_failures_link,
               self.num_oop_failures,
               self.num_completed),
        )
    )
    # Non-OOP failures.
    s.write(
        '<td><span style="color:red">%s</span>%s</td>'
        % (
            self._format_rate(self.non_oop_failure_rate),
            '<a href="%s">(%d/%d)</a>'
            % (self.non_oop_failures_link,
               self.num_non_oop_failures,
               self.num_completed),
        )
    )

  @property
  def _method_color(self) -> str:
    """Returns the color for rendering method."""
    if self.method == 'call':
      return 'gray'
    elif self.method == 'query':
      return 'blue'
    else:
      return 'cyan'

  def _render_oop_failures(self, s: io.StringIO) -> None:
    self._render_failures(s, '^MappingError.*', error_color='magenta')

  def _render_non_oop_failures(self, s: io.StringIO) -> None:
    self._render_failures(s, '^(?!MappingError).*', error_color='red')

  def _render_failures(
      self, s: io.StringIO, error_regex: str, error_color: str) -> None:
    """Formats the failed cases into html."""
    # Failure summary.
    s.write(
        '<h2> Error Summary </h2>'
        '<div style="white-space:pre">\n'
        '<table style="border:1px solid">'
        '<tr class="header"><td>Error type</td><td>Stats</td></tr>'
    )
    error_regex = re.compile(error_regex)
    if self.result.metrics.failure_breakdown:
      for name, count in self.result.metrics.failure_breakdown.items():
        if not error_regex.match(name):
          continue

        link = f'<a href="#{name}">{name}</a>'
        error_rate = self._format_rate(count / self.result.metrics.total)
        stats = (f'<span style="color:{error_color}">{error_rate} '
                 f'({count}/{self.result.metrics.total})</span>')
        s.write(f'<tr><td>{link}</td><td>{stats})</td></tr>')
    s.write(
        '</table></div>'
        '<h2> Failed Cases </h2>'
        '<div style="white-space:pre">'
    )
    # Failure details by error type.
    failures_by_error = collections.defaultdict(list)
    for example, error in self.failures:
      error_name = _error_key(error)
      if error_regex.match(error_name):
        failures_by_error[error_name].append((example, error))

    for error_key, failures in failures_by_error.items():
      s.write(
          f'<h3 id="{error_key}"><a href="#{error_key}">{error_key}</a> '
          f'(count={len(failures)})</h3>'
          '<table style="border:1px solid">'
          '<tr class="header"><td>No.</td><td>Input</td>'
          '<td>LM invocation</td><td>Error</td></tr>'
      )
      for i, (example, error) in enumerate(failures):
        lm_response = None
        if isinstance(error, lf.structured.MappingError):
          lm_response = error.lm_response
          error = error.cause

        bgcolor = 'white' if i % 2 == 0 else '#DDDDDD'
        s.write(f'<tr style="background-color: {bgcolor}"><td>{i + 1}</td>')
        s.write('<td style="color:green;white-space:pre-wrap">')
        s.write(pg.format(example, verbose=False))
        s.write('</td><td>')
        if lm_response is not None:
          self._render_message(lm_response, s)
        s.write(f'</td><td style="color:{error_color};white-space:pre">')
        s.write(_format_error(error))
        s.write('</td></tr>')
      s.write('</table>')
    s.write('</div>')

  @classmethod
  def visualize(cls, evaluations: list['Evaluation']) -> str | None:
    """Visualize the a list of evaluations of this task in HTML."""
    del evaluations
    return None


@pg.functor()
def inputs_from(path: str | list[str], **kwargs) -> list[Any]:
  """A functor that returns a list of user-defined objects as eval inputs."""
  if isinstance(path, str):
    if path.endswith('.json'):
      return pg.load(path)
    elif path.endswith('.jsonl'):
      return list(iter(pg.open_jsonl(path)))
    elif path.endswith('.csv'):
      import pandas as pd  # pylint: disable=g-import-not-at-top
      dataset_df = pd.read_csv(path, **kwargs)
      dataset = []
      for i in range(dataset_df.shape[0]):
        row = {}
        for col in dataset_df.columns:
          row[col] = dataset_df.iloc[i][col]
        dataset.append(row)
      return dataset
    else:
      raise ValueError(f'Unsupported file format: {path}')
  examples = []
  for p in path:
    examples.extend(pg.load(p))
  return examples


@pg.functor()
def as_inputs(examples: list[Any]) -> list[Any]:
  """User provided examples as eval inputs."""
  return examples


def load(eval_dir: str) -> Evaluation:
  """Loads evaluation from a directory."""
  return pg.load(os.path.join(eval_dir, Evaluable.EXPERIMENT_JSON))


#
# Continuous summarization for evaluations.
#


class Summary(pg.Object):
  """Summary for a list of evaluations."""

  evaluations: Annotated[
      list[Evaluation], 'Evaluations included in current summary.'
  ]

  pivot_field: Annotated[
      str, 'Filed name for pivoting the table for summary.'
  ] = 'lm'

  def tasks(self) -> list[Type[Evaluation]]:
    """All tasks in the summary."""
    return list(set([e.__class__ for e in self.evaluations]))

  def __len__(self):
    return len(self.evaluations)

  @property
  def all_completed(self) -> bool:
    """Returns True if all evaluations are completed."""
    return all(e.result is not None for e in self.evaluations)

  def select(
      self,
      task: Type[Evaluation] | tuple[Type[Evaluation], ...] = Evaluation,
      lm: Union[
          lf.LanguageModel,
          Type[lf.LanguageModel],
          tuple[lf.LanguageModel | Type[lf.LanguageModel], ...],
      ] = lf.LanguageModel,
      method: Union[str, tuple[str, ...], None] = None,
      schema_fn: Union[pg.Functor, tuple[pg.Functor, ...], None] = None,
      completed: bool | None = None,
      pivot_field: str | None = None,
  ) -> 'Summary':
    """Creates a summary by selecting evaluations with conditions."""

    def _match_lm(lm, evaluation):
      if isinstance(lm, lf.LanguageModel):
        return evaluation.lm.model_id == lm.model_id
      elif inspect.isclass(lm) and issubclass(lm, lf.LanguageModel):
        return isinstance(evaluation.lm, lm)
      elif isinstance(lm, tuple):
        return any(_match_lm(x, evaluation) for x in lm)
      return False

    def _match_method(method, evaluation):
      if method is None:
        return True
      if isinstance(method, str):
        return method == evaluation.method
      elif isinstance(method, tuple):
        return evaluation.method in method
      return False

    def _match_schema(schema_fn, evaluation):
      if schema_fn is None:
        return True
      if isinstance(schema_fn, pg.Functor):
        return pg.eq(schema_fn, evaluation.schema_fn)
      elif isinstance(schema_fn, tuple):
        return any(_match_schema(x, evaluation) for x in schema_fn)
      return False

    def _match_completed(completed, evaluation):
      if completed is None:
        return True
      elif completed:
        return evaluation.result is not None
      else:
        return evaluation.result is None

    selected = [
        pg.Ref(e)
        for e in self.evaluations
        if (
            isinstance(e, task)
            and _match_lm(lm, e)
            and _match_method(method, e)
            and _match_schema(schema_fn, e)
            and _match_completed(completed, e)
        )
    ]
    return Summary(
        evaluations=selected, pivot_field=pivot_field or self.pivot_field
    )

  class Table(pg.Object):
    """A pivot table for view evaluations."""

    class Row(pg.Object):
      descriptor: dict[str, Any]
      data: list[Evaluation | None]

    rows: list[Row]
    cols: list[Any]  # Present values for the pivot field.
    pivot_field: str
    descriptor_keys: list[str]

    def html(self) -> str:
      s = io.StringIO()
      s.write('<table style="border:1px solid;">')
      s.write('<tr style="font-weight: bold;">')
      for h in self.descriptor_keys:
        s.write(f'<td style="padding: 5px">{h}</td>')
      for c in self.cols:
        s.write(
            f'<td style="padding: 5px">{self.pivot_field}={_html_repr(c)}</td>')
      s.write('</tr>')
      for i, row in enumerate(self.rows):
        bgcolor = 'white' if i % 2 == 1 else '#EEEEEE'
        s.write(f'<tr style="background-color: {bgcolor}">')

        for k in self.descriptor_keys:
          v = row.descriptor.get(k)
          s.write(f'<td style="padding: 5px">{_html_repr(v)}</td>')

        for e in row.data:
          s.write('<td style="padding: 5px;">')
          if e is None:
            s.write('<span style="color: gray">N/A<span>')
          else:
            s.write(e.summary_card())
          s.write('</td>')
        s.write('</tr>')
      s.write('</table>')
      return s.getvalue()

    def _repr_html_(self) -> str:
      return self.html()

    @classmethod
    def from_evaluations(
        cls, evaluations: list['Summary.Entry'], pivot_field: str = 'lm'
    ) -> 'Summary.Table':
      """Creates a table from a list of evaluations."""

      # Introduce a container for enabling symbolic comparison/hashing
      # for langfun components which disable symbolic comparison by default.
      class SymbolicComparable(pg.Object):
        value: Any

        def __lt__(self, other) -> bool:
          return pg.lt(self.value, other.value)

        def __gt__(self, other) -> bool:
          return pg.gt(self.value, other.value)

      # Figuring out the values for pivot field as columns.
      cols = set()
      nonpivot_values = collections.defaultdict(set)
      for e in evaluations:
        for k in e.sym_init_args:
          if pivot_field == k:
            cols.add(SymbolicComparable(e.sym_inferred(k)))
          elif k not in ('id', 'groundtruth'):
            nonpivot_values[k].add(SymbolicComparable(e.sym_inferred(k)))

      cols = sorted(cols)

      # Figure out the row descriptor keys by looking at fields whose values
      # have differences.
      descriptor_keys = set()
      for k, v in nonpivot_values.items():
        if len(v) > 1:
          descriptor_keys.add(k)

      descriptor_keys = sorted(descriptor_keys)
      groups = collections.defaultdict(dict)
      for e in evaluations:
        descriptor_values = tuple(
            [SymbolicComparable(e.sym_inferred(k)) for k in descriptor_keys]
        )
        groups[descriptor_values][
            SymbolicComparable(e.sym_inferred(pivot_field))
        ] = pg.Ref(e)
      rows = []
      for descriptor_values, pivoted_evaluations in groups.items():
        descriptor = {
            k: v.value for k, v in zip(descriptor_keys, descriptor_values)
        }
        data = pg.List([pivoted_evaluations.get(c) for c in cols])
        rows.append(Summary.Table.Row(descriptor=descriptor, data=data))
      return cls(
          cols=[c.value for c in cols],
          rows=rows,
          pivot_field=pivot_field,
          descriptor_keys=list(descriptor_keys),
      )

  def html(self, pivot_field: str | None = None) -> str:
    """Renders the summary in HTML."""
    pivot_field = pivot_field or self.pivot_field
    s = io.StringIO()
    s.write('<html><body>')
    for task in sorted(self.tasks(), key=lambda cls: cls.__name__):
      table_id = task.__name__.lower()
      evaluations = self.select(task=task).evaluations
      table = Summary.Table.from_evaluations(evaluations, pivot_field)
      s.write('<div>')
      s.write(
          f'<a id="{table_id}" href="#{table_id}">'
          f'<h2>{task.__name__}</h2></a>'
      )

      # Allow users to plugin visualization code (e.g. matplot) in the summary
      # page.
      visual_part = task.visualize(evaluations)
      if visual_part:
        s.write(visual_part)

      s.write(f'<h4 style="color:gray">{len(evaluations)} experiments</h4>')
      s.write('<hr/>')
      s.write(table.html())
      s.write('</div>')
    s.write('</body></html>')
    return s.getvalue()

  def _repr_html_(self) -> str:
    return self.html()

  def json(
      self,
  ) -> dict[
      str,  # Task name
      list[pg.Dict],  # List of pg.Dict with `experiment` and `metrics`.
  ]:
    """Returns the JSON representation of the summary."""
    task_results = {}
    for task in sorted(self.tasks(), key=lambda cls: cls.__name__):
      results = []
      for entry in self.select(task=task).evaluations:
        results.append(
            pg.Dict(
                id=entry.id,
                experiment=entry,
                dir=entry.dir,
                metrics=entry.result.metrics if entry.result else None,
                usage=entry.result.usage if entry.result else None,
            )
        )
      task_results[task.__name__] = results
    return task_results

  def save(self, file: str, pivot_field: str | None = None) -> None:
    pg.save(self.html(pivot_field), file, file_format='txt')
    if file.endswith('.html'):
      json_file = file.replace('.html', '.json')
    else:
      json_file = os.path.join(file, '.json')
    pg.save(self.json(), json_file)

  @classmethod
  def from_dirs(
      cls,
      root_dir: str | Sequence[str],
      filter: Union[str, Sequence[str], None] = None,  # pylint: disable=redefined-builtin
  ) -> 'Summary':
    """Creates a summary from one or more root directories."""
    return cls(
        [
            x
            for x in lf.concurrent_execute(
                Evaluable.from_dir,
                [
                    os.path.join(root_dir, i)
                    for i in _iter_dirs(root_dir, filter)
                ],
            )
            if x is not None and x.is_leaf
        ]
    )

  class MonitorResult:
    """Async result for monitored summary."""

    def __init__(self, context, thread):
      self._context = context
      self._thread = thread

    @property
    def summary(self) -> Optional['Summary']:
      """Returns the most recent summary."""
      return self._context.summary

    @property
    def completed(self) -> bool:
      """Returns True if all evaluations have result."""
      return self._context.completed

    def stop(self) -> 'Summary':
      """Signal and wait the monitor thread to stop."""
      self._context.stopping = True
      return self.join()

    def join(self) -> 'Summary':
      """Waits the monitor thread to complete."""
      self._thread.join()
      summary = self.summary
      assert summary is not None
      return summary

  @classmethod
  def monitor_async(
      cls,
      root_dir: str | Sequence[str],
      save_as: str | None = None,
      filter: Union[str, Sequence[str], None] = None,  # pylint: disable=redefined-builtin
      pivot_field: str = 'lm',
      expect_new_dirs: bool = False,
      scan_interval: int = 60,
      refresh_when_stop: bool = True,
  ) -> MonitorResult:
    """Monitor one or more root directories and save summary in period."""
    context = pg.Dict(stopping=False, completed=False, summary=None)

    def _monitor():
      dir_to_eval = {}

      def load_evaluation(maybe_dir):
        e = dir_to_eval.get(maybe_dir)
        updated = False
        if not e:
          e = Evaluable.from_dir(maybe_dir, load_result=False)
          dir_to_eval[maybe_dir] = e
          updated = True

        if e and e.is_leaf:
          updated = e.try_load_result()
        return updated

      def refresh_summary():
        updated = any(
            lf.concurrent_execute(load_evaluation, _iter_dirs(root_dir, filter))
        )
        evaluations = [
            pg.Ref(e) for e in dir_to_eval.values() if e and e.is_leaf
        ]
        context.summary = Summary(
            evaluations=evaluations, pivot_field=pivot_field
        )
        if updated:
          if lf.console.under_notebook():
            lf.console.display(context.summary, clear=True)
          if save_as:
            context.summary.save(save_as)
        return context.summary.all_completed

      while not context.stopping:
        completed = refresh_summary()
        if not expect_new_dirs and completed:
          context.completed = True
          break

        # Be responsive to stopping signal.
        idle_start = time.time()
        while time.time() - idle_start < scan_interval:
          if context.stopping:
            break
          time.sleep(1)

      if context.stopping and refresh_when_stop:
        refresh_summary()

    thread = threading.Thread(target=_monitor)
    thread.start()
    return Summary.MonitorResult(context, thread)

  @classmethod
  def monitor(
      cls,
      root_dir: str | Sequence[str],
      save_as: str | None = None,
      filter: Union[str, Sequence[str], None] = None,  # pylint: disable=redefined-builtin
      pivot_field: str = 'lm',
      expect_new_dirs: bool = False,
  ) -> 'Summary':
    result = cls.monitor_async(
        root_dir,
        save_as,
        filter,
        pivot_field=pivot_field,
        expect_new_dirs=expect_new_dirs,
    )
    return result.join()


def _format_error(error: Exception):
  """Formats an error into a string."""
  return (f'({error.__class__.__name__}) ' + pg.decolor(str(error)))


def _error_key(error: Exception) -> str:
  """Returns the key for an error."""
  error_names = []
  while error is not None:
    error_names.append(error.__class__.__name__)
    error = getattr(error, 'cause', None)
  return '.'.join(error_names)


def _html_repr(value: Any, compact: bool = True, escape: bool = False) -> str:
  """Formats prompt in HTML."""
  if type(value) is lf.Template:  # pylint: disable=unidiomatic-typecheck
    return repr(value.template_str)
  s = pg.format(
      value, compact=compact, verbose=False, hide_default_values=True)
  if escape:
    s = s.replace('"', '&quot;')
  return s


def _iter_dirs(
    root_dir: str | Sequence[str],
    filter: Union[str, Sequence[str], None] = None,  # pylint: disable=redefined-builtin
) -> Iterator[str]:
  """Itererate sub-entries of one or more directories."""
  root_dirs = [root_dir] if isinstance(root_dir, str) else root_dir
  if filter is None:
    filters = []
  elif isinstance(filter, str):
    filters = [re.compile(filter)]
  else:
    filters = [re.compile(f) for f in filter]

  def accepts(entry):
    if not filters:
      return True
    for f in filters:
      if f.match(entry):
        return True
    return False

  for root_dir in root_dirs:
    for entry in pg.io.listdir(root_dir):
      if accepts(entry):
        yield os.path.join(root_dir, entry)


def monitor(
    root_dir: str | Sequence[str],
    save_as: str | None = None,
    filter: Union[str, Sequence[str], None] = None,  # pylint: disable=redefined-builtin
    pivot_field: str = 'lm',
    expect_new_dirs: bool = False,
) -> Summary:
  """Monitor one or more root directories for summary."""
  return Summary.monitor(
      root_dir,
      save_as,
      filter,
      pivot_field=pivot_field,
      expect_new_dirs=expect_new_dirs,
  )


def monitor_async(
    root_dir: str | Sequence[str],
    save_as: str | None = None,
    filter: Union[str, Sequence[str], None] = None,  # pylint: disable=redefined-builtin
    pivot_field: str = 'lm',
    expect_new_dirs: bool = False,
    scan_interval: int = 60,
    refresh_when_stop: bool = True,
) -> Summary.MonitorResult:
  """Asynchronorsly monitor one or more root directories for summary."""
  return Summary.monitor_async(
      root_dir,
      save_as,
      filter,
      pivot_field=pivot_field,
      expect_new_dirs=expect_new_dirs,
      scan_interval=scan_interval,
      refresh_when_stop=refresh_when_stop,
  )


#
# Named evaluations and experiments support.
#


class _NamedEvaluationRegistry:
  """Named evaluation registry."""

  def __init__(self):
    self._registry = {}

  def names(self) -> list[str]:
    """Returns all registered names."""
    return sorted(self._registry.keys())

  def get(self, name: str) -> list[Type[Evaluable]]:
    """Gets an evaluation by name."""
    matches = []
    if name in self._registry:
      matches.append(self._registry[name])
    else:
      regex = re.compile(name)
      for key, cls in self._registry.items():
        if regex.match(key):
          matches.append(cls)
    return matches

  def register(
      self,
      name: str,
      experiment_cls: Type[Evaluable],
  ):
    """Register an experiment class."""
    self._registry[name] = experiment_cls


_eval_registry = _NamedEvaluationRegistry()


def registered_names() -> list[str]:
  """Returns all registered names."""
  return _eval_registry.names()


def get_evaluations(evaluation: str | Evaluable) -> list[Evaluable]:
  """Gets an evaluation experiment by name."""
  if isinstance(evaluation, str):
    return [e() for e in _eval_registry.get(evaluation)]
  return [evaluation]


def register(name: str):
  """Decorator to create a named evaluation class."""

  def _register(func_or_cls: Type[Evaluation] | types.FunctionType):
    if inspect.isfunction(func_or_cls):
      e = func_or_cls()
      if not isinstance(e, Evaluable):
        raise TypeError(
            f'The return value of `{func_or_cls}` should be an instance of '
            '`lf.eval.Evaluable` subclass.'
        )

      class GeneratedSuite(Suite):
        # NOTE(daiyip): Delay serialization key registration for generated
        # class.
        auto_register = False
        children = e.children if isinstance(e, Suite) else [e]

      cls = GeneratedSuite
      cls.__name__ = func_or_cls.__name__
      cls.__doc__ = func_or_cls.__doc__
      cls.__qualname__ = func_or_cls.__qualname__
      cls.__module__ = getattr(func_or_cls, '__module__', 'wrapper')
      cls.register_for_deserialization(cls.__type_name__)

    elif issubclass(func_or_cls, Evaluable):
      cls = func_or_cls
    else:
      raise ValueError(f'Unsupported type: {type(func_or_cls)}')

    _eval_registry.register(name, cls)
    return cls

  return _register


def get(
    root_dir: str,
    evaluations: list[str | Evaluable],
    filter: Union[                    # pylint: disable=redefined-builtin
        str,                          # Regex to filter evaluation based on ID.
        Callable[[Evaluable], bool],  # Custom filter function.
        None                          # No filtering (Default).
    ] = None,                         # pylint: disable=bad-whitespace
    patches: list[Union[
        str,                                    # String-based PyGlove patcher.
        pg.patching.Patcher,                    # PyGlove patcher object.
        Callable[[pg.KeyPath, Any, Any], Any],  # PyGlove rebind function.
    ]] | None = None,                           # pylint: disable=bad-whitespace
) -> Suite:
  """Gets a suite from a list of patched evaluations.

  Args:
    root_dir: The root directory of the experiment.
    evaluations: A list of evaluations to be included in the suite.
    filter: A regular expression (str) for selecting sub-experiments of matched
      IDs, or a filter function to filter the evaluations.
    patches: A list of patches to be applied to the suite. Each element can be
      a string (for string-based patcher), a `pg.patching.Patcher` object, or
      a rebind function (e.g. `pg.rebind`). See `lf.eval.patch_*` for more
      details.

  Returns:
    A suite of selected `lf.eval.Evaluation` objects.
  """
  matches = []
  for e in evaluations:
    matches.extend(get_evaluations(e))

  if not matches:
    raise ValueError('No evaluations found.')

  suite = Suite(matches, root_dir=root_dir)
  if patches:
    suite = pg.patch(suite, patches)

  if isinstance(filter, str):
    regex = re.compile(filter)
    filter = lambda x: bool(regex.match(x.id))

  if filter:
    suite = Suite(
        [leaf for leaf in suite.leaf_nodes if filter(leaf)], root_dir=root_dir)
  return suite


def run(
    root_dir: str,
    evaluations: list[str | Evaluable],
    filter: Union[                    # pylint: disable=redefined-builtin
        str,                          # Regex to filter evaluation based on ID.
        Callable[[Evaluable], bool],  # Custom filter function.
        None                          # No filtering (Default).
    ] = None,                         # pylint: disable=bad-whitespace
    patches: list[Union[
        str,                                    # String-based PyGlove patcher.
        pg.patching.Patcher,                    # PyGlove patcher object.
        Callable[[pg.KeyPath, Any, Any], Any],  # PyGlove rebind function.
    ]] | None = None,                           # pylint: disable=bad-whitespace
    mode: Literal['run', 'rerun', 'dryrun', 'noop'] = 'run',
    debug: bool = False,
    print_definition: bool = False,
    **kwargs,
) -> Suite:
  """Run selected evaluations with patching.

  Args:
    root_dir: The root directory of the experiment.
    evaluations: A list of evaluations to be included in the suite.
    filter: A regular expression (str) for selecting sub-experiments of matched
      IDs, or a filter function to filter the evaluations.
    patches: A list of patches to be applied to the suite. Each element can be
      a string (for string-based patcher), a `pg.patching.Patcher` object, or
      a rebind function (e.g. `pg.rebind`). See `lf.eval.patch_*` for more
      details.
    mode: The mode to run the suite. "run" to run the suite, with reusing
      existing results if available; "rerun" to rerun all evaluations even if
      there are existing results; "dryrun" to dryrun the suite; and "noop"
      to do nothing.
    debug: Whether to run in debug mode.
    print_definition: Whether to print the experiment definition.
    **kwargs: Additional arguments to be passed to dryrun/run the suite.

  Returns:
    A suite of selected `lf.eval.Evaluation` objects.
  """
  suite = get(root_dir, evaluations, patches=patches, filter=filter)
  if print_definition:
    lf.console.write(
        pg.format(
            suite,
            compact=False,
            verbose=False,
            hide_default_values=True,
            python_format=True,
        ),
        title='[EXPERIMENT DEFINITION]',
        color='blue',
    )

  if mode == 'run':
    rerun = mode == 'rerun'
    suite.run(debug=debug, rerun=rerun, **kwargs)
  elif mode == 'dryrun':
    suite.dryrun(debug=debug, **kwargs)
  return suite
