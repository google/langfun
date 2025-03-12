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
"""Base classes for agentic actions."""

import abc
import contextlib
import datetime
import time

import typing
from typing import Annotated, Any, ContextManager, Iterable, Iterator, Optional, Type, Union
import langfun.core as lf
from langfun.core import structured as lf_structured
import pyglove as pg


class Action(pg.Object):
  """Base class for agent actions."""

  def _on_bound(self):
    super()._on_bound()
    self._session = None
    self._result = None
    self._metadata = {}

  @property
  def session(self) -> Optional['Session']:
    """Returns the session started by this action."""
    return self._session

  @property
  def result(self) -> Any:
    """Returns the result of the action."""
    return self._result

  @property
  def metadata(self) -> dict[str, Any] | None:
    """Returns the metadata associated with the result from previous call."""
    return self._metadata

  def __call__(
      self,
      session: Optional['Session'] = None,
      *,
      show_progress: bool = True,
      **kwargs) -> Any:
    """Executes the action."""
    new_session = session is None
    if new_session:
      session = Session()
      if show_progress:
        lf.console.display(pg.view(session, name='agent_session'))

    with session.track_action(self):
      result = self.call(session=session, **kwargs)

      # For the top-level action, we store the session in the metadata.
      if new_session:
        self._session = session
      self._result = result
      self._metadata = session.current_action.metadata
      return self._result

  @abc.abstractmethod
  def call(self, session: 'Session', **kwargs) -> Any:
    """Calls the action.

    Args:
      session: The session to use for the action.
      **kwargs: Additional keyword arguments to pass to the action.

    Returns:
      The result of the action.
    """


# Type definition for traced item during execution.
TracedItem = Union[
    lf_structured.QueryInvocation,
    'ActionInvocation',
    'ExecutionTrace',
    # NOTE(daiyip): Consider remove log entry once we migrate existing agents.
    lf.logging.LogEntry,
]


class ExecutionTrace(pg.Object, pg.views.html.HtmlTreeView.Extension):
  """Trace of the execution of an action."""

  name: Annotated[
      str | None,
      (
          'The name of the execution trace. If None, the trace is unnamed, '
          'which is the case for the top-level trace of an action. An '
          'execution trace could have sub-traces, called phases, which are '
          'created and named by `session.phase()` context manager.'
      )
  ] = None

  start_time: Annotated[
      float | None,
      'The start time of the execution. If None, the execution is not started.'
  ] = None

  end_time: Annotated[
      float | None,
      'The end time of the execution. If None, the execution is not ended.'
  ] = None

  items: Annotated[
      list[TracedItem],
      'All tracked execution items in the sequence.'
  ] = []

  def _on_bound(self):
    super()._on_bound()
    self._usage_summary = lf.UsageSummary()
    for item in self.items:
      if hasattr(item, 'usage_summary'):
        self._usage_summary.merge(item.usage_summary)

    self._tab_control = None
    self._time_badge = None

  def start(self) -> None:
    assert self.start_time is None, 'Execution already started.'
    self.rebind(start_time=time.time(), skip_notification=True)
    if self._time_badge is not None:
      self._time_badge.update(
          'Starting',
          add_class=['starting'],
          remove_class=['not-started'],
      )

  def stop(self) -> None:
    assert self.end_time is None, 'Execution already stopped.'
    self.rebind(end_time=time.time(), skip_notification=True)
    if self._time_badge is not None:
      self._time_badge.update(
          f'{int(self.elapse)} seconds',
          tooltip=pg.format(self.execution_summary(), verbose=False),
          add_class=['finished'],
          remove_class=['running'],
      )

  def __len__(self) -> int:
    return len(self.items)

  @property
  def has_started(self) -> bool:
    return self.start_time is not None

  @property
  def has_stopped(self) -> bool:
    return self.end_time is not None

  @property
  def elapse(self) -> float:
    """Returns the elapsed time of the execution."""
    if self.start_time is None:
      return 0.0
    if self.end_time is None:
      return time.time() - self.start_time
    return self.end_time - self.start_time

  @property
  def queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns queries from the sequence."""
    return self._child_items(lf_structured.QueryInvocation)

  @property
  def actions(self) -> list['ActionInvocation']:
    """Returns action invocations from the sequence."""
    return self._child_items(ActionInvocation)

  @property
  def logs(self) -> list[lf.logging.LogEntry]:
    """Returns logs from the sequence."""
    return self._child_items(lf.logging.LogEntry)

  @property
  def all_queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns all queries from current trace and its child execution items."""
    return self._all_child_items(lf_structured.QueryInvocation)

  @property
  def all_logs(self) -> list[lf.logging.LogEntry]:
    """Returns all logs from current trace and its child execution items."""
    return self._all_child_items(lf.logging.LogEntry)

  def _child_items(self, item_cls: Type[Any]) -> list[Any]:
    child_items = []
    for item in self.items:
      if isinstance(item, item_cls):
        child_items.append(item)
      elif isinstance(item, ExecutionTrace):
        child_items.extend(item._child_items(item_cls))  # pylint: disable=protected-access
    return child_items

  def _all_child_items(self, item_cls: Type[Any]) -> list[Any]:
    child_items = []
    for item in self.items:
      if isinstance(item, item_cls):
        child_items.append(item)
      elif isinstance(item, ActionInvocation):
        child_items.extend(item.execution._all_child_items(item_cls))  # pylint: disable=protected-access
      elif isinstance(item, ExecutionTrace):
        child_items.extend(item._all_child_items(item_cls))  # pylint: disable=protected-access
    return child_items

  def append(self, item: TracedItem) -> None:
    """Appends an item to the sequence."""
    with pg.notify_on_change(False):
      self.items.append(item)

    if isinstance(item, lf_structured.QueryInvocation):
      current_invocation = self
      while current_invocation is not None:
        current_invocation.usage_summary.merge(item.usage_summary)
        current_invocation = typing.cast(
            ExecutionTrace,
            current_invocation.sym_ancestor(
                lambda x: isinstance(x, ExecutionTrace)
            )
        )

    if self._tab_control is not None:
      self._tab_control.append(self._execution_item_tab(item))

    if (self._time_badge is not None
        and not isinstance(item, lf.logging.LogEntry)):
      sub_task_label = self._execution_item_label(item)
      self._time_badge.update(
          text=sub_task_label.text,
          tooltip=sub_task_label.tooltip.content,
          add_class=['running'],
          remove_class=['not-started'],
      )

  def extend(self, items: Iterable[TracedItem]) -> None:
    """Extends the sequence with a list of items."""
    for item in items:
      self.append(item)

  @property
  def usage_summary(self) -> lf.UsageSummary:
    """Returns the usage summary of the action."""
    return self._usage_summary

  def execution_summary(self) -> dict[str, Any]:
    """Execution summary string."""
    return pg.Dict(
        num_queries=len(self.queries),
        execution_breakdown=[
            dict(
                action=action.action.__class__.__name__,
                usage=action.usage_summary.total,
                execution_time=action.execution.elapse,
            )
            for action in self.actions
        ]
    )

  #
  # HTML views.
  #

  def _html_tree_view_summary(
      self,
      *,
      name: str | None = None,
      extra_flags: dict[str, Any] | None = None,
      view: pg.views.html.HtmlTreeView, **kwargs
  ):
    return None

  def _execution_badge(self, interactive: bool = True):
    if not self.has_started:
      label = '(Not started)'
      tooltip = 'Execution not started.'
      css_class = 'not-started'
    elif not self.has_stopped:
      label = 'Starting'
      tooltip = 'Execution starting.'
      css_class = 'running'
    else:
      label = f'{int(self.elapse)} seconds'
      tooltip = pg.format(self.execution_summary(), verbose=False)
      css_class = 'finished'
    time_badge = pg.views.html.controls.Badge(
        label,
        tooltip=tooltip,
        css_classes=['execution-time', css_class],
        interactive=interactive,
    )
    if interactive:
      self._time_badge = time_badge
    return time_badge

  def _html_tree_view_content(
      self,
      *,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ):
    del kwargs
    extra_flags = extra_flags or {}
    interactive = extra_flags.get('interactive', True)
    if interactive or self.items:
      self._tab_control = pg.views.html.controls.TabControl(
          [self._execution_item_tab(item) for item in self.items],
          tab_position='left'
      )
      return self._tab_control.to_html()
    return '(no tracked items)'

  def _execution_item_tab(self, item: TracedItem) -> pg.views.html.controls.Tab:
    if isinstance(item, ActionInvocation):
      css_class = 'action'
    elif isinstance(item, lf_structured.QueryInvocation):
      css_class = 'query'
    elif isinstance(item, lf.logging.LogEntry):
      css_class = 'log'
    elif isinstance(item, ExecutionTrace):
      css_class = 'phase'
    else:
      raise ValueError(f'Unsupported item type: {type(item)}')

    return pg.views.html.controls.Tab(
        label=self._execution_item_label(item),
        content=pg.view(item),
        css_classes=[css_class]
    )

  def _execution_item_label(
      self, item: TracedItem
  ) -> pg.views.html.controls.Label:
    if isinstance(item, ActionInvocation):
      return pg.views.html.controls.Label(
          item.action.__class__.__name__,
          tooltip=pg.format(
              item.action,
              verbose=False,
              hide_default_values=True,
              max_str_len=80,
              max_bytes_len=20,
          ),
      )
    elif isinstance(item, lf_structured.QueryInvocation):
      schema_title = 'str'
      if item.schema:
        schema_title = lf_structured.annotation(item.schema.spec)
      return pg.views.html.controls.Label(
          schema_title,
          tooltip=(
              pg.format(
                  item.input,
                  verbose=False,
                  hide_default_values=True,
                  max_str_len=80,
                  max_bytes_len=20,
              )
          ),
      )
    elif isinstance(item, lf.logging.LogEntry):
      return pg.views.html.controls.Label(
          'Log',
          tooltip=item.message,
      )
    elif isinstance(item, ExecutionTrace):
      return pg.views.html.controls.Label(
          item.name or 'Phase',
          tooltip=f'Execution phase {item.name!r}.'
      )
    else:
      raise ValueError(f'Unsupported item type: {type(item)}')

  def _html_tree_view_css_styles(self) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .tab-button.action > ::before {
          content: "A";
          font-weight: bold;
          color: red;
          padding: 10px;
        }
        .tab-button.phase > ::before {
          content: "P";
          font-weight: bold;
          color: purple;
          padding: 10px;
        }
        .tab-button.query > ::before {
          content: "Q";
          font-weight: bold;
          color: orange;
          padding: 10px;
        }
        .tab-button.log > ::before {
          content: "L";
          font-weight: bold;
          color: green;
          padding: 10px;
        }
        .details.execution-trace, .details.action-invocation {
          border: 1px solid #eee;
        }
        .execution-trace-title {
          display: inline-block;
        }
        .badge.execution-time {
          margin-left: 4px;
          border-radius: 0px;
        }
        .execution-time.starting {
          background-color: ghostwhite;
          font-weight: normal;
        }
        .execution-time.running {
          background-color: ghostwhite;
          font-weight: normal;
        }
        .execution-time.finished {
          background-color: aliceblue;
          font-weight: bold;
        }
       """
    ]


class ActionInvocation(pg.Object, pg.views.html.HtmlTreeView.Extension):
  """A class for capturing the invocation of an action."""
  action: Action

  result: Annotated[
      Any,
      'The result of the action.'
  ] = None

  metadata: Annotated[
      dict[str, Any],
      'The metadata returned by the action.'
  ] = {}

  execution: Annotated[
      ExecutionTrace,
      'The execution sequence of the action.'
  ] = ExecutionTrace()

  # Allow symbolic assignment without `rebind`.
  allow_symbolic_assignment = True

  def _on_bound(self):
    super()._on_bound()
    self._current_phase = self.execution
    self._tab_control = None

  @property
  def current_phase(self) -> ExecutionTrace:
    """Returns the current execution phase."""
    return self._current_phase

  @contextlib.contextmanager
  def phase(self, name: str) -> Iterator[ExecutionTrace]:
    """Context manager for starting a new execution phase."""
    phase = ExecutionTrace(name=name)
    phase.start()
    parent_phase = self._current_phase
    self._current_phase.append(phase)
    self._current_phase = phase
    try:
      yield phase
    finally:
      phase.stop()
      self._current_phase = parent_phase

  @property
  def logs(self) -> list[lf.logging.LogEntry]:
    """Returns immediate child logs from execution sequence."""
    return self.execution.logs

  @property
  def actions(self) -> list['ActionInvocation']:
    """Returns immediate child action invocations."""
    return self.execution.actions

  @property
  def queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns immediate queries made by the action."""
    return self.execution.queries

  @property
  def all_queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns all queries made by the action and its child execution items."""
    return self.execution.all_queries

  @property
  def all_logs(self) -> list[lf.logging.LogEntry]:
    """Returns all logs made by the action and its child execution items."""
    return self.execution.all_logs

  @property
  def usage_summary(self) -> lf.UsageSummary:
    """Returns the usage summary of the action."""
    return self.execution.usage_summary

  def start(self) -> None:
    """Starts the execution of the action."""
    self.execution.start()

  def end(self, result: Any, metadata: dict[str, Any]) -> None:
    """Ends the execution of the action with result and metadata."""
    self.rebind(
        result=result,
        metadata=metadata,
        skip_notification=True,
        raise_on_no_change=False
    )
    self.execution.stop()
    if self._tab_control is not None:
      if self.metadata:
        self._tab_control.insert(
            1,
            pg.views.html.controls.Tab(
                'metadata',
                pg.view(
                    self.metadata,
                    collapse_level=None,
                    enable_summary_tooltip=False
                ),
                name='metadata',
            )
        )
      self._tab_control.insert(
          1,
          pg.views.html.controls.Tab(
              'result',
              pg.view(
                  self.result,
                  collapse_level=None,
                  enable_summary_tooltip=False
              ),
              name='result',
          ),
      )
      self._tab_control.select(['metadata', 'result'])

  #
  # HTML views.
  #

  def _html_tree_view_summary(
      self, *, view: pg.views.html.HtmlTreeView, **kwargs
  ):
    return None

  def _html_tree_view_content(
      self,
      *,
      view: pg.views.html.HtmlTreeView,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ):
    extra_flags = extra_flags or {}
    interactive = extra_flags.get('interactive', True)
    if (isinstance(self.action, RootAction)
        and self.execution.has_stopped
        and len(self.execution) == 1):
      return view.content(self.execution.items[0], extra_flags=extra_flags)

    tabs = []
    if not isinstance(self.action, RootAction):
      tabs.append(
          pg.views.html.controls.Tab(
              'action',
              view.render(  # pylint: disable=g-long-ternary
                  self.action,
                  collapse_level=None,
                  root_path=self.action.sym_path,
                  enable_summary_tooltip=False,
              ),
              name='action',
          )
      )
    if self.execution.has_stopped:
      tabs.append(
          pg.views.html.controls.Tab(
              'result',
              view.render(
                  self.result,
                  collapse_level=None,
                  enable_summary_tooltip=False
              ),
              name='result'
          )
      )
      if self.metadata:
        tabs.append(
            pg.views.html.controls.Tab(
                'metadata',
                view.render(
                    self.metadata,
                    collapse_level=None,
                    enable_summary_tooltip=False
                ),
                name='metadata'
            )
        )

    tabs.append(
        pg.views.html.controls.Tab(
            pg.Html.element(
                'span',
                [
                    'execution',
                    self.execution._execution_badge(interactive),  # pylint: disable=protected-access
                    (
                        self.usage_summary.to_html(  # pylint: disable=g-long-ternary
                            extra_flags=dict(as_badge=True)
                        )
                        if (interactive
                            or self.usage_summary.total.num_requests > 0)
                        else None
                    ),
                ],
                css_classes=['execution-tab-title']
            ),
            view.render(self.execution, extra_flags=extra_flags),
            name='execution',
        )
    )
    tab_control = pg.views.html.controls.TabControl(tabs)
    # Select the tab following a priority: metadata, result, action, execution.
    tab_control.select(['metadata', 'result', 'action', 'execution'])
    if interactive:
      self._tab_control = tab_control
    return tab_control

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .execution-tab-title {
          text-align: left;
        }
        .execution-tab-title .usage-summary.label {
          border-radius: 0px;
          font-weight: normal;
          color: #AAA;
        }
        """
    ]


class RootAction(Action):
  """A placeholder action for the root of the action tree."""

  def call(self, session: 'Session', **kwargs) -> Any:
    raise NotImplementedError('Shall not be called.')


class Session(pg.Object, pg.views.html.HtmlTreeView.Extension):
  """Session for performing an agentic task."""

  root: ActionInvocation = ActionInvocation(RootAction())
  id: Annotated[
      str | None,
      'An optional identifier for the sessin, which will be used for logging.'
  ] = None

  def _on_bound(self):
    super()._on_bound()
    self._current_action = self.root

  @property
  def final_result(self) -> Any:
    """Returns the final result of the session."""
    return self.root.result

  @property
  def current_action(self) -> ActionInvocation:
    """Returns the current invocation."""
    return self._current_action

  def add_metadata(self, **kwargs: Any) -> None:
    """Adds metadata to the current invocation."""
    with pg.notify_on_change(False):
      self._current_action.metadata.update(kwargs)

  def phase(self, name: str) -> ContextManager[ExecutionTrace]:
    """Context manager for starting a new execution phase."""
    return self.current_action.phase(name)

  @contextlib.contextmanager
  def track_action(self, action: Action) -> Iterator[ActionInvocation]:
    """Track the execution of an action."""
    if not self.root.execution.has_started:
      self.root.start()

    invocation = ActionInvocation(pg.maybe_ref(action))
    parent_action = self._current_action
    parent_action.current_phase.append(invocation)

    try:
      self._current_action = invocation
      # Start the execution of the current action.
      self._current_action.start()
      yield invocation
    finally:
      # Stop the execution of the current action.
      self._current_action.end(action.result, action.metadata)
      self._current_action = parent_action
      if parent_action is self.root:
        parent_action.end(
            result=action.result, metadata=action.metadata,
        )

  @contextlib.contextmanager
  def track_queries(
      self,
      phase: str | None = None
  ) -> Iterator[list[lf_structured.QueryInvocation]]:
    """Tracks `lf.query` made within the context.

    Args:
      phase: The name of a new phase to track the queries in. If not provided,
        the queries will be tracked in the parent phase.

    Yields:
      A list of `lf.QueryInvocation` objects, each for a single `lf.query`
      call.
    """
    with self.phase(phase) if phase else contextlib.nullcontext():
      with lf_structured.track_queries(include_child_scopes=False) as queries:
        try:
          yield queries
        finally:
          self._current_action.current_phase.extend(queries)

  def query(
      self,
      prompt: Union[str, lf.Template, Any],
      schema: Union[
          lf_structured.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
      ] = None,
      default: Any = lf.RAISE_IF_HAS_ERROR,
      *,
      lm: lf.LanguageModel | None = None,
      examples: list[lf_structured.MappingExample] | None = None,
      **kwargs
      ) -> Any:
    """Calls `lf.query` and associates it with the current invocation.

    The following code are equivalent:

      Code 1:
      ```
      session.query(...)
      ```

      Code 2:
      ```
      with session.track_queries() as queries:
        output = lf.query(...)
      ```
    The former is preferred when `lf.query` is directly called by the action.
    If `lf.query` is called by a function that does not have access to the
    session, the latter should be used.

    Args:
      prompt: The prompt to query.
      schema: The schema to use for the query.
      default: The default value to return if the query fails.
      lm: The language model to use for the query.
      examples: The examples to use for the query.
      **kwargs: Additional keyword arguments to pass to `lf.query`.
    
    Returns:
      The result of the query.
    """
    with self.track_queries():
      return lf_structured.query(
          prompt,
          schema=schema,
          default=default,
          lm=lm,
          examples=examples,
          **kwargs
      )

  def query_prompt(
      self,
      prompt: Union[str, lf.Template, Any],
      schema: Union[
          lf_structured.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
      ] = None,
      **kwargs,
  ) -> Any:
    """Calls `lf.query_prompt` and associates it with the current invocation.

    The following code are equivalent:

      Code 1:
      ```
      session.query_prompt(...)
      ```

      Code 2:
      ```
      with session.track_queries() as queries:
        output = lf.query_prompt(...)
      ```
    The former is preferred when `lf.query_prompt` is directly called by the
    action.
    If `lf.query_prompt` is called by a function that does not have access to
    the
    session, the latter should be used.

    Args:
      prompt: The prompt to query.
      schema: The schema to use for the query.
      **kwargs: Additional keyword arguments to pass to `lf.query_prompt`.

    Returns:
      The result of the query.
    """
    with self.track_queries():
      return lf_structured.query_prompt(prompt, schema=schema, **kwargs)

  def query_output(
      self,
      response: Union[str, lf.Template, Any],
      schema: Union[
          lf_structured.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
      ] = None,
      **kwargs,
  ) -> Any:
    """Calls `lf.query_output` and associates it with the current invocation.

    The following code are equivalent:

      Code 1:
      ```
      session.query_output(...)
      ```

      Code 2:
      ```
      with session.track_queries() as queries:
        output = lf.query_output(...)
      ```
    The former is preferred when `lf.query_output` is directly called by the
    action.
    If `lf.query_output` is called by a function that does not have access to
    the
    session, the latter should be used.

    Args:
      response: The response to query.
      schema: The schema to use for the query.
      **kwargs: Additional keyword arguments to pass to `lf.query_prompt`.

    Returns:
      The result of the query.
    """
    with self.track_queries():
      return lf_structured.query_output(response, schema=schema, **kwargs)

  def _log(self, level: lf.logging.LogLevel, message: str, **kwargs):
    self._current_action.current_phase.append(
        lf.logging.LogEntry(
            level=level,
            time=datetime.datetime.now(),
            message=message,
            metadata=kwargs,
        )
    )

  def debug(self, message: str, **kwargs):
    """Logs a debug message to the session."""
    self._log('debug', message, **kwargs)

  def info(self, message: str, **kwargs):
    """Logs an info message to the session."""
    self._log('info', message, **kwargs)

  def warning(self, message: str, **kwargs):
    """Logs a warning message to the session."""
    self._log('warning', message, **kwargs)

  def error(self, message: str, **kwargs):
    """Logs an error message to the session."""
    self._log('error', message, **kwargs)

  def fatal(self, message: str, **kwargs):
    """Logs a fatal message to the session."""
    self._log('fatal', message, **kwargs)

  def as_message(self) -> lf.AIMessage:
    """Returns the session as a message."""
    return lf.AIMessage(
        'Agentic task session.',
        result=self.root
    )

  #
  # HTML views.
  #

  def _html_tree_view_content(
      self,
      *,
      view: pg.views.html.HtmlTreeView,
      **kwargs
  ):
    return view.content(self.root, **kwargs)

  @classmethod
  def _html_tree_view_config(cls):
    config = super()._html_tree_view_config()
    config.update(
        enable_summary_tooltip=False,
        enable_key_tooltip=False,
    )
    return config
