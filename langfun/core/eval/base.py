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
from typing import Annotated, Any, Callable, Literal, Optional

import langfun.core as lf
from langfun.core.llms.cache import in_memory
import langfun.core.structured as lf_structured
import pyglove as pg


class Evaluable(lf.Component):
  """Base class for evaluation and suite."""

  EXPERIMENT_JSON = 'experiment.json'
  RESULT_JSON = 'result.json'
  FAILURES_JSON = 'failures.json'
  INDEX_HTML = 'index.html'

  id: Annotated[
      str,
      (
          'The ID of the evaluation, which should be unique across all '
          'evaluations.'
      ),
  ]

  root_dir: Annotated[
      str | None,
      (
          'The root directory for all evaluables under an evaluation suite'
          "A child evaluable's dir will be relative to its parent dir."
      ),
  ] = lf.contextual(default=None)

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
      save: bool = True,
      debug: bool | lf.LMDebugMode = False,
      dryrun: bool = False,
      verbose: bool = False,
      show_progress: bool | int = True,
      label: str | None = None,
      from_root: bool = True,
      **kwargs,
  ) -> pg.Dict:
    """Run the evaluation, which fills and returns the result."""
    if dryrun:
      self.dryrun(filter=filter, verbose=False, debug=debug)

    if self.is_leaf:
      if isinstance(show_progress, bool):
        if show_progress:
          progress_bar = lf.concurrent.ProgressBar.install(
              label=label, total=self.num_examples, color='blue')
        else:
          progress_bar = None
      else:
        progress_bar = show_progress

      self._run(
          start=start,
          end=end,
          debug=debug,
          dryrun=dryrun,
          verbose=verbose,
          progress_bar=progress_bar,
          label=label,
          **kwargs,
      )
      if save and self.dir:
        if show_progress:
          lf.concurrent.ProgressBar.update(
              progress_bar, postfix='SAVING RESULTS...', color='yellow')

        # Save evaluation results.
        self.save()

        if show_progress:
          lf.concurrent.ProgressBar.update(
              progress_bar,
              postfix=self._completion_status(),
              color='green')
    else:
      def _run_group(arg: tuple[int, list[_LeafNode]]) -> None:
        overview_bar, leaf_group = arg
        for leaf in leaf_group:
          if leaf.enabled:
            leaf.node.run(
                start=start,
                end=end,
                save=save,
                debug=debug,
                dryrun=False,
                verbose=verbose,
                show_progress=leaf.progress_bar,
                from_root=False,
                **kwargs,
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
              postfix=None if leaf.enabled else 'SKIPPED.')

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
              postfix='SAVING RESULTS...',
              color='yellow')

          for node in self.nonleaf_nodes:
            node._result = {c.id: c.result for c in node.children}  # pylint: disable=protected-access
            if save and self.dir:
              node.save()

          # Signal all task completed by making the bar green.
          lf.concurrent.ProgressBar.update(
              overview_bar,
              postfix='COMPLETED',
              color='green')

        finally:
          # for leaf in leaf_nodes:
          #   lf.concurrent.ProgressBar.uninstall(leaf.progress_bar)
          lf.concurrent.ProgressBar.uninstall(overview_bar)

    if from_root and save and self.dir:
      lf.console.write(
          f'({self.index_link})',
          color='magenta',
          styles=['underline'],
      )
    return self.result

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
      **kwargs,
  ) -> None:
    """Run the evaluate and fill `self.result`. Subclass to implement."""

  @abc.abstractmethod
  def _completion_status(self) -> str:
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

  def save(self):
    # Save experiment definition.
    pg.save(self, os.path.join(self.dir, Evaluable.EXPERIMENT_JSON))

    # Save evaluation result.
    pg.save(self.result, os.path.join(self.dir, Evaluation.RESULT_JSON))

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

  def _html(
      self,
      body_builders: list[Callable[[io.StringIO], None]],
      include_def: bool = False,
      include_cache_stats: bool = False,
  ) -> str:
    s = io.StringIO()
    s.write('<html><head><style>')
    self._render_styles(s)
    s.write('</style></head><body>')
    s.write(f'<h1 style="color:blue;background-color:#DDDDDD">{self.id}</h1>')
    self._render_navbar(s)
    for builder in body_builders:
      builder(s)
    if include_def:
      s.write('<h2> Definition </h2>')
      s.write('<div style="white-space:pre;padding:10px;color:#3254a8;')
      eval_def = self.format()
      s.write(f'background-color:#EEEEEE">{eval_def}</div>')
    if include_cache_stats and self.is_deterministic:
      s.write('<h2> Cache Stats </h2>')
      s.write('<div style="white-space:pre;padding:10px;color:#3254a8;')
      s.write(f'background-color:#EEEEEE">{self.result.cache_stats}</div>')
    s.write('</body></html>')
    return s.getvalue()

  def _render_styles(self, s: io.StringIO) -> None:
    s.write("""
        td {padding: 5px;}
        .header {font-weight: bold;}
        """)

  def _render_navbar(self, s: io.StringIO) -> None:
    current = self
    links = []
    while current is not None and current.index_link is not None:
      links.insert(0, f'<a href="{current.index_link}">{current.id}</a>')
      current = current.parent

    for i, link in enumerate(links):
      s.write(link)
      if i != len(links) - 1:
        # Add a right triangle symbol.
        s.write(' &#9656 ')

  def _render_index_page(self, s: io.StringIO) -> None:
    self._render_result(s)
    if self.dryrun_output is not None:
      self._render_dryrun_output(s)

  def _render_result(self, s: io.StringIO) -> None:
    s.write('<h2> Result </h2>')
    s.write('<table style="border:1px solid;"><tr class="header">')
    if self.children:
      s.write('<td>ID</td>')

    self._render_result_header(s)
    s.write('<tr>')
    if self.children:
      for c in self.children:
        s.write('<tr>')
        s.write(f'<td><a href={c.index_link}>{c.id}</a></td>')
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
    for m in message.trace():
      if 'lm-input' in m.tags:
        text_color = 'green'
      elif 'lm-response' in m.tags:
        text_color = 'blue'
      else:
        text_color = 'black'

      s.write(
          f'<div style="color: {text_color}; white-space: pre-wrap;'
          'padding: 10px; border: 1px solid; margin-top: 10px">'
      )
      s.write(m.text)
      if m.result is not None:
        s.write(
            '<div style="color: magenta; white-space: pre-wrap;'
            'padding: 10px; border: 1px solid; margin: 10px">'
        )
        s.write(pg.format(m.result))
        s.write('</div>')
      s.write('</div>')


@dataclasses.dataclass
class _LeafNode:
  """Information for leaf node execution."""
  index: int
  node: 'Evaluation'
  enabled: bool = True
  progress_bar: int | None = None


@pg.use_init_args(['id', 'children'])
class Suite(Evaluable):
  """Evaluation suite."""

  children: Annotated[list[Evaluable], 'Child evaluation sets or suites.']

  def _on_bound(self):
    super()._on_bound()
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

  def _completion_status(self) -> str:
    return 'COMPLETED'


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
  ]

  prompt: Annotated[
      lf.Template,
      (
          'Template for rendering the template. Example object could be '
          'accessed via `example`.'
      ),
  ]

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
          'functors should be defined:' + inspect.cleandoc("""
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
                        nl_text='Compute 1 + 2',
                        schema=Solution,
                        value=Solution(3))
                ]
              ```
              """)
      ),
  ]

  call_postprocess: Annotated[
      Callable[[str], str] | None,
      (
          'A callable object to post process the text response before '
          'parsing. Applicable only when `method` is set to \'call\'.'
      ),
  ] = None

  lm: Annotated[lf.LanguageModel, 'Language model to use for evaluation.']

  parsing_lm: Annotated[
      lf.LanguageModel | None,
      (
          'Language model for parsing. Applicable only when method is set'
          'to `call`. If None, `lm` will also be used for parsing. '
      ),
  ] = None

  completion_prompt_field: Annotated[
      str | None,
      (
          'A str field that will be automatically added to the class of the '
          'input object for `lf.complete`. If None, no field will be added to '
          'the class, instead the prompt will be passed as the first argument '
          'of the input object to complete. Applicable only when `method` is '
          'set to `complete`.'
      )
  ] = None

  autofix: Annotated[
      int,
      (
          'The number of attempts for auto fix, which allows LLMs to correct '
          'generated code for the output structure. If 0, autofix will be '
          'disabled.'
      ),
  ] = 3

  autofix_lm: Annotated[
      lf.LanguageModel | None,
      (
          'Language model for autofix. If None, `lm` will also be used for '
          'autofix.'
      ),
  ] = None

  use_cache: Annotated[bool, 'If True, LM cache will be enabled.'] = True

  max_workers: Annotated[
      int, 'Max workers to run the evaluation in parallel.'
  ] = 32

  # Constants.
  CACHE_JSON = 'cache.json'
  FAILURES_HTML = 'failures.html'

  @functools.cached_property
  def hash(self) -> str:
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
        'model': self.lm.model_id,
        'parsing_model': getattr(parsing_model, 'model_id', None),
        'sampling_options': self.lm.sampling_options,
        'inputs': self.inputs,
        'prompt': self.prompt.template_str,
        'fewshot_examples': self.fewshot_examples,
        'method': self.method,
        'schema_fn': self.schema_fn,
    }

  @functools.cached_property
  def examples(self):
    """Returns examples for evaluation."""
    return self.inputs()

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

  @property
  def failures(self) -> list[tuple[Any, Exception]]:
    """Returns the failed examples and their errors."""
    return self._failures

  @property
  def num_failures(self) -> int:
    """Returns the number of failed examples."""
    return len(self.failures)

  @property
  def failure_rate(self) -> float:
    """Returns the failure rate in range [0, 1]."""
    if self.num_completed == 0:
      return 0.0
    return self.num_failures / self.num_completed

  @functools.cached_property
  def schema(self) -> lf_structured.Schema | None:
    """Schema."""
    if self.schema_fn is None:
      return None

    schema = self.schema_fn()
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

    schema = self.schema_fn()
    fewshot_examples = None
    if isinstance(schema, tuple):
      schema, fewshot_examples = schema
    self.__dict__['schema'] = self._formalize_schema(schema)
    return self._maybe_adjust_examples_for_completion(fewshot_examples)

  def _formalize_schema(self, annotation) -> lf_structured.Schema:
    """Formalizes schema from annotation."""
    if self.method == 'complete':
      if not hasattr(annotation, '__schema__'):
        raise TypeError(
            'The annotation returned by `schema_fn` must be a `pg.Object` '
            'subclassclass to be used for `lf.complete`. '
            'Encountered: {annotation!r}.'
        )
      self._maybe_adjust_schema_for_completion(annotation)
    return lf_structured.Schema.from_value(annotation)

  def _maybe_adjust_schema_for_completion(self, cls):
    if (self.completion_prompt_field is None
        or self.completion_prompt_field in cls.__schema__):
      return

    fields = list(cls.__schema__.values())
    fields.insert(0, (self.completion_prompt_field, pg.typing.Str()))
    pg.symbolic.update_schema(cls, fields, extend=False)

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
      if ex.nl_context is not None:
        init_args = dict(ex.value.sym_init_args)
        example_cls = ex.value.__class__
        self._maybe_adjust_schema_for_completion(example_cls)
        ex = lf.structured.MappingExample(
            value=lf.structured.mapping.Pair(
                left=example_cls.partial(ex.nl_context),
                right=example_cls(ex.nl_context, **init_args),
            )
        )
      completion_examples.append(ex)
    return completion_examples

  @functools.cached_property
  def children(self) -> list['Evaluation']:
    """Returns the trials as child evaluations if this evaluation is a space."""
    if self.is_deterministic:
      return []
    children = []
    for i, child in enumerate(pg.iter(self)):
      child.rebind(id=f'{self.id}@{child.hash}', skip_notification=True)
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
    if self.method != 'call' and self.schema_fn is None:
      raise ValueError(
          f'`schema_fn` must be specified for method {self.method!r}')
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

  @property
  def failures_link(self) -> str | None:
    """Returns the link to the failures page."""
    if self.dir is None:
      return None
    return self.link(os.path.join(self.dir, Evaluation.FAILURES_HTML))

  def _dryrun(
      self,
      *,
      example: Any,
      debug: bool | lf.LMDebugMode,
      verbose: bool,
      **kwargs,
  ) -> None:
    # Set the example for dryrun.
    example = example or self.examples[0]

    # We make a copy to avoid pollute the state of current object.
    copy = self.clone()
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

    with lf.use_settings(debug=debug):
      output_message = copy.process(example)
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

    # Audit the result.
    copy.audit(example, output, output_message)
    result = copy.summarize()
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
      **kwargs,
  ) -> None:
    # Setup examples.
    if end is None:
      end = len(self.examples)
    examples = self.examples[start:end]

    # Process examples.
    with lf.use_settings(debug=debug, cache=self.cache):
      self._reset()

      try:
        for example, message, error in lf.concurrent_map(
            self.process,
            examples,
            max_workers=self.max_workers,
            show_progress=progress_bar or False,
            status_fn=self._status,
        ):
          if error is not None:
            self._failures.append((example, str(error)))
          else:
            output = message.text if self.schema is None else message.result
            self.audit(example, output, message)
          self._num_completed += 1
      finally:
        # Save cache upon completion or interruption.
        if self.dir and self.cache:
          self.cache.save()

    # Summarize result.
    self._result = self.summarize()
    if verbose:
      lf.console.write(
          str(self.result),
          title=f'RESULT ON {self.id}',
          color='magenta',
      )

  def process(self, example: Any, **kwargs) -> lf.Message:
    """Process an example and returns its output."""
    prompt = self.prompt.render(example=example).text
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
      input_value = self.schema.spec.cls.partial(prompt)
      return lf_structured.complete(
          input_value,
          lm=self.lm,
          examples=self.fewshot_examples,
          autofix=self.autofix,
          autofix_lm=self.autofix_lm,
          returns_message=True,
          **kwargs,
      )

  def _status(self, progress: lf.concurrent.Progress) -> dict[str, Any]:
    return {
        'Model': self.lm.model_id,
        'Succeeded': '%.2f%% (%d/%d)' % (
            progress.success_rate * 100,
            progress.succeeded,
            progress.completed,
        ),
        'Failed': '%.2f%% (%d/%d)' % (
            progress.failure_rate * 100,
            progress.failed,
            progress.completed,
        ),
    }

  def _completion_status(self) -> str:
    return 'COMPLETED: Successes=%.2f%% (%d/%d) Failures=%.2f%% (%d/%d)' % (
        (1 - self.failure_rate) * 100,
        self.num_completed - self.num_failures,
        self.num_completed,
        self.failure_rate * 100,
        self.num_failures,
        self.num_completed,
    )

  def summarize(self) -> pg.Dict:
    """Summarizes the evaluation result."""
    if self.cache:
      cache_stats = dict(
          use_cache=True,
          num_queries=self.cache.stats.num_queries,
          num_hits=self.cache.stats.num_hits,
          num_updates=self.cache.stats.num_updates,
      )
    else:
      cache_stats = dict(use_cache=False)
    result = pg.Dict(
        experiment_setup=pg.Dict(
            id=self.id,
            dir=self.dir,
            model=self.lm.model_id,
            prompt_template=lf.text_formatting.decolored(str(self.prompt)),
            method=self.method,
            schema_fn=str(self.schema_fn),
        ),
        cache_stats=cache_stats,
        metrics=pg.Dict(
            total=self.num_completed,
            failures=self.num_failures,
            failure_rate=self.failure_rate,
        ),
    )
    return result

  def audit(self, example: Any, output: Any, message: lf.Message) -> None:
    """Audits the example against the output. Subclasses should override.

    Args:
      example: The input object.
      output: The output from LM. For `lf.call`, if `schema_fn` is not provided,
        it will be the raw LM response string. Otherwise it will be the
        structured output from the LM.
      message: The entire message returned by the LM, which could be used to
        trace the LM input, response and parsed structure.
    """

  def save(self) -> None:
    """Save evaluation details."""
    super().save()

    # Save failures.
    pg.save(
        [
            pg.Dict(input=input, error=lf.text_formatting.decolored(str(error)))
            for input, error in self.failures
        ],
        os.path.join(self.dir, Evaluation.FAILURES_JSON),
    )
    pg.save(
        self._html([self._render_result, self._render_failures]),
        os.path.join(self.dir, Evaluation.FAILURES_HTML),
        file_format='txt',
    )

  def _render_result_header(self, s: io.StringIO) -> None:
    s.write('<td>Method</td>')
    s.write('<td>Inputs</td>')
    s.write('<td>Model</td>')
    s.write('<td>Prompt</td>')
    s.write('<td>Schema</td>')
    s.write('<td>Failures</td>')

  def _render_result_row(self, s: io.StringIO) -> None:
    def display_value(v, p):
      if isinstance(v, pg.hyper.OneOf):
        if not p:
          return v.candidates
        return [getattr(c, p) for c in v.candidates]
      return getattr(v, p) if p else v

    s.write('<td style="color:{self._method_color}">'
            f'{display_value(self.method, "")}</td>')
    s.write('<td style="color:darkgray">'
            f'{display_value(self.inputs, "")}</td>')
    s.write('<td style="color:#494a5c">'
            f'{display_value(self.lm, "model_id")}</td>')
    s.write('<td style="color:darkgray">'
            f'{display_value(self.prompt, "template_str")}</td>')

    if isinstance(self.schema_fn, pg.hyper.OneOf):
      title = '...'
      schema_fn = self.schema_fn.candidates
    else:
      title = self.schema.schema_str('python') if self.schema else None
      schema_fn = self.schema_fn
    s.write(
        '<td style="color:purple" '
        f'title="{title}">'
        f'{schema_fn}</td>'
    )
    s.write(
        '<td><span style="color:orange">%s</span>%s</td>'
        % (
            '%.2f%%' % (self.failure_rate * 100),
            '<a href="%s">(%d/%d)</a>'
            % (self.failures_link, self.num_failures, self.num_completed),
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

  def _render_failures(self, s: io.StringIO) -> None:
    """Formats the failed cases into html."""
    s.write('<h2> Failed Cases </h2>')
    s.write('<div style="white-space:pre">\n')
    s.write(
        '<table style="border:1px solid">'
        '<tr class="header"><td>No.</td><td>Input</td><td>Error</td></tr>'
    )

    for i, (example, error) in enumerate(self.failures):
      bgcolor = 'white' if i % 2 == 0 else '#DDDDDD'
      s.write(f'<tr style="background-color: {bgcolor}"><td>{i + 1}</td>')
      input_str = pg.format(example, verbose=False)
      s.write(f'<td style="color:green;white-space:pre-wrap">{input_str}</td>')
      error_str = lf.text_formatting.decolored(str(error))
      s.write(f'<td style="color:red;white-space:pre">{error_str}</td>')
      s.write('</tr>')
    s.write('</table></div>')


@pg.functor()
def inputs_from(path: str | list[str]) -> list[Any]:
  """A functor that returns a list of user-defined objects as eval inputs."""
  if isinstance(path, str):
    return pg.load(path)
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
