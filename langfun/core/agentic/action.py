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
import collections
import contextlib
import dataclasses
import functools
import itertools
import threading
import time
import typing
from typing import Annotated, Any, Callable, Iterable, Iterator, Optional, Type, Union
import uuid
import langfun.core as lf
from langfun.core import structured as lf_structured
import pyglove as pg


class ActionError(Exception):   # pylint: disable=g-bad-exception-name
  """Base class for common action errors."""


class ActionTimeoutError(ActionError):
  """Raised when an action exceeds the max execution time."""


class Action(pg.Object):
  """Base class for agentic actions.

  An `Action` represents a single, executable step or task that an agent can
  perform, such as calling a tool, querying a language model, or returning a
  final answer. Actions are designed to be composable and trackable within a
  `Session`.

  # Developing Actions

  In Langfun, an `Action` is a class representing a task an agent can execute.
  To define custom actions, subclass `lf.agentic.Action` and implement the
  `call` method, which contains the logic for the action's execution.

  ```python
  class Calculate(lf.agentic.Action):
    expression: str

    def call(self, session: Session, *, lm: lf.LanguageModel, **kwargs):
      return session.query(expression, float, lm=lm)
  ```

  Key aspects of the `call` method:

  - `session` (First Argument): An `lf.Session` object required to make queries,
    perform logging, and add metadata to the action. It also tracks the
    execution of the action and its sub-actions.

    - Use `session.query(...)` to make calls to a Language Model.
    - Use `session.debug(...)`, `session.info(...)`, `session.warning(...)`,
      and `session.error(...)` for adding logs associated with the
      current action.
    - Use `session.add_metadata(...)` to associate custom metadata with
      the current action.

  - Keyword Arguments (e.g., lm): Arguments required for the action's execution
    (like a language model) should be defined as keyword arguments.

  - **kwargs: Include **kwargs to allow:

    - Users to pass additional arguments to child actions.
    - The action to gracefully handle extra arguments passed by parent actions.

  # Using Actions

  ## Creating Action objects
  Action objects can be instantiated in two primary ways:

  - Direct instantiation by Users:

    ```
    calculate_action = Calculate(expression='1 + 1')
    ```

  - Generation by Language Models (LLMs): LLMs can generate Action objects when
    provided with an "action space" (a schema defining possible actions). The
    LLM populates the action's attributes. User code can then invoke the
    generated action.

  ```python
  import pyglove as pg
  import langfun as lf

  # Define possible actions for the LLM
  class Search(lf.agentic.Action):
    query: str
    def call(self, session: lf.Session, *, lm: lf.LanguageModel, **kwargs):
      # Placeholder for actual search logic
      return f"Results for: {self.query}"

  class DirectAnswer(lf.agentic.Action):
    answer: str
    def call(self, session: lf.Session, *, lm: lf.LanguageModel, **kwargs):
      return self.answer

  # Define the schema for the LLM's output
  class NextStep(pg.Object):
    step_by_step_thoughts: list[str]
    next_action: Calculate | Search | DirectAnswer

  # Query the LLM to determine the next step
  next_step = lf.query(
      'What is the next step for {{question}}?',
      NextStep,
      question='why is the sky blue?'
  )
  # Execute the action chosen by the LLM
  result = next_step.next_action()
  print(result)
  ```

  ## Invoking Actions and Managing Sessions:

  When an action is called, the session argument (the first argument to call)
  is handled as follows:

  - Implicit Session Management: If no session is explicitly provided when
    calling an action, Langfun automatically creates and passes one.

    ```python
    calc = Calculate(expression='1 + 1')

    # A session is implicitly created and passed here.
    result = calc()
    print(result)

    # Access the implicitly created session.
    # print(calc.session)
    ```

  - Explicit Session Management: You can create and manage `lf.Session` objects
    explicitly. This is useful for customizing session identifiers or managing
    a shared context for multiple actions.

  ```python
  calc = Calculate(expression='1 + 1')

  # Explicitly create and pass a session.
  with lf.Session(id='my_agent_session') as session:
    result = calc(session=session)  # Pass the session explicitly
    print(result)
  ```

  ## Accessing Execution Trajectory:

  After an action is executed, the Session object holds a record of its
  execution, known as the trajectory. This includes queries made and any
  sub-actions performed.

  - To access all queries issued directly by the root action:

    ```python
    print(session.root.execution.queries)
    ```
  - To access all actions issued by the root action and any of its
    sub-actions (recursively):

    ```python
    print(session.root.execution.all_queries)
    ```
  - To access all child actions issued by the root action:

    ```python
    print(session.root.execution.actions)
    ```

  - To access all the actions in the sub-tree issued by the root action:

    ```python
    print(session.root.execution.all_actions)
    ```
  """

  def _on_bound(self):
    super()._on_bound()
    self._session = None
    self._invocation: ActionInvocation | None = None

    # NOTE(daiyip): Users could use `self._state` to keep track of the state
    # during the execution of the action.
    # Strictly speaking action state should better fit into
    # ActionInvocation, we make it a property of Action as it's usually
    # initialized before `__call__` is called.
    self._state: Any = None

  @property
  def session(self) -> Optional['Session']:
    """Returns the session started by this action."""
    return self._session

  @property
  def state(self) -> Any:
    """Returns the state of the action."""
    return self._state

  @property
  def result(self) -> Any:
    """Returns the result of the action."""
    return self._invocation.result if self._invocation else None

  @property
  def metadata(self) -> dict[str, Any] | None:
    """Returns the metadata associated with the result from previous call."""
    return self._invocation.metadata if self._invocation else None

  @property
  def invocation(self) -> Optional['ActionInvocation']:
    """Returns last invocation. None if the action is not executed."""
    return self._invocation

  async def acall(
      self,
      session: Optional['Session'] = None,
      *,
      max_execution_time: float | None = None,
      show_progress: bool = True,
      verbose: bool = False,
      **kwargs
  ) -> Any:
    """Async version of `__call__`."""
    # TODO(daiyip): implement native async calling.
    return await lf.invoke_async(
        self.__call__,
        session,
        max_execution_time=max_execution_time,
        show_progress=show_progress,
        verbose=verbose,
        **kwargs
    )

  def __call__(
      self,
      session: Optional['Session'] = None,
      *,
      max_execution_time: float | None = None,
      show_progress: bool = True,
      verbose: bool = False,
      **kwargs
  ) -> Any:
    """Executes the action.

    Args:
      session: The session to use for the action.
      max_execution_time: The max allowed execution time in seconds for the
        action. The effective `max_execution_time` is the smaller of the
        remaining time of the parent invocation and the `max_execution_time`
        for the current invocation. Since we are running action within the
        current thread, there is no guarantee that the action will be executed
        within this time limit, but we will try to stop the action as soon as
        the next operation on the session is called.
      show_progress: Whether to show the progress of the action.
      verbose: Whether to log the action execution.
      **kwargs: Additional keyword arguments to pass to the action.

    Returns:
      The result of the action.
    """
    if session is None:
      session = Session(verbose=verbose)
      session.start()

      if show_progress:
        lf.console.display(pg.view(session, name='agent_session'))

      # For the top-level action, we store the session in the metadata.
      self._session = session
    else:
      self._session = None

    with session.track_action(self, max_execution_time=max_execution_time):
      try:
        # Early terminate the action if the execution time is exceeded.
        session.check_execution_time()
        result = self.call(session=session, **kwargs)
        self._invocation.end(result)
      except BaseException as e:
        error = pg.ErrorInfo.from_exception(e)
        self._invocation.end(result=None, error=error)
        if self._session is not None:
          self._session.end(result=None, error=error)
        raise

    if self._session is not None:
      # Session is created by current action. Stop the session.
      self._session.end(result)
    return result

  @abc.abstractmethod
  def call(self, session: 'Session', **kwargs) -> Any:
    """Calls the action.

    Args:
      session: The session to use for the action.
      **kwargs: Additional keyword arguments to pass to the action.

    Returns:
      The result of the action.
    """


#
# Execution tracking.
#


class ExecutionUnit(pg.Object):
  """Base class for execution units in an agentic trajectory.

  An `ExecutionUnit` represents a logical step or container in the agent's
  execution flow. It serves as the common interface for top-level executable
  items.

  The concrete subclasses of `ExecutionUnit` are typically:
  * **`ActionInvocation`**: Represents a single, specific action executed by
    the agent.
  * **`ParallelExecutions`**: Represents a container for a group of
    `ExecutionUnits` that were executed concurrently.

  Users can retrieve the immediate child execution units from an `ExecutionUnit`
  object. To access the leaf nodes of the execution tree, use `all_actions`,
  `all_queries`, and `all_logs` instead.

  Each unit exposes a **`position`** property to reveal its specific location
  within the execution hierarchy (e.g., '1.2.3').

  Users could use **`parent_execution_unit`** to get the parent execution unit
  of the current execution unit.
  """

  @dataclasses.dataclass
  class Position:
    """The position of an executed unit under current session."""

    parent: Optional['ExecutionUnit.Position'] = None
    index: int = 0

    def indices(self) -> tuple[int, ...]:
      """Returns the indices from root to current execution unit."""
      # A deque is efficient for adding items to the front.
      path = collections.deque()
      current_pos = self

      # Traverse up from the current position to the root.
      while current_pos.parent is not None:
        path.appendleft(current_pos.index)
        current_pos = current_pos.parent

      path.appendleft(current_pos.index)
      return tuple(path)

    def to_str(
        self,
        *,
        index_base: int = 1,
        separator: str = '.',
        **kwargs
    ) -> str:
      """Returns a string description of the position."""
      # For root action, we return empty string as it's position descriptor.
      if self.parent is None:
        return ''
      parent_descriptor = self.parent.to_str(
          index_base=index_base,
          separator=separator,
          **kwargs
      )
      if not parent_descriptor:
        return f'{self.index + index_base}'
      return parent_descriptor + separator + f'{self.index + index_base}'

    def __repr__(self) -> str:
      return f'Position({", ".join(str(x) for x in self.indices())})'

    def __str__(self) -> str:
      return self.to_str()

    def __eq__(self, other: 'ExecutionUnit.Position') -> bool:
      if isinstance(other, ExecutionUnit.Position):
        return self.indices() == other.indices()
      if isinstance(other, tuple):
        return self.indices() == other
      if isinstance(other, str):
        return str(self) == other
      return False

    def __ne__(self, other: 'ExecutionUnit.Position') -> bool:
      return not self == other

    def __hash__(self) -> int:
      return hash(self.indices())

    def __lt__(self, other: 'ExecutionUnit.Position') -> bool:
      return self.indices() < other.indices()

    def __gt__(self, other: 'ExecutionUnit.Position') -> bool:
      return self.indices() > other.indices()

  def _on_parent_change(self, *args, **kwargs):
    super()._on_parent_change(*args, **kwargs)
    self.__dict__.pop('parent_execution_unit', None)
    self.__dict__.pop('position', None)

  @functools.cached_property
  def parent_execution_unit(self) -> Optional['ExecutionUnit']:
    """Returns the parent execution unit of the current execution unit."""
    parent_trace = self.sym_ancestor(lambda x: isinstance(x, ExecutionTrace))
    assert isinstance(parent_trace, ExecutionTrace), (
        'Execution unit is not associated with any `ExecutionTrace`: '
        f'{self}'
    )
    return parent_trace.parent_execution_unit

  @functools.cached_property
  def position(self) -> Position:
    """Returns the execution position of the action."""
    parent_trace = self.sym_ancestor(lambda x: isinstance(x, ExecutionTrace))
    while parent_trace is not None:
      parent_position = parent_trace.position
      if parent_position is not None:
        return ExecutionUnit.Position(
            parent_position, parent_trace.indexof(self, ExecutionUnit)
        )
      parent_trace = parent_trace.sym_ancestor(
          lambda x: isinstance(x, ExecutionTrace)
      )
    return ExecutionUnit.Position(None, 0)

  @property
  @abc.abstractmethod
  def execution_units(
      self,
  ) -> list['ExecutionUnit']:
    """Returns immediate child execution items."""

  @property
  @abc.abstractmethod
  def queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns queries issued by the execution item."""

  @property
  @abc.abstractmethod
  def actions(self) -> list['ActionInvocation']:
    """Returns immediate child action invocations."""

  @property
  @abc.abstractmethod
  def logs(self) -> list[lf.logging.LogEntry]:
    """Returns immediate logs under current execution item."""

  @property
  @abc.abstractmethod
  def all_queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns all queries from the subtree."""

  @property
  @abc.abstractmethod
  def all_actions(self) -> list['ActionInvocation']:
    """Returns all action invocations from the subtree."""

  @property
  @abc.abstractmethod
  def all_logs(self) -> list[lf.logging.LogEntry]:
    """Returns all logs from the subtree."""


# Type definition for traced item during execution.
TracedItem = Union[
    ExecutionUnit,
    lf_structured.QueryInvocation,
    'ExecutionTrace',
    # NOTE(daiyip): Consider remove log entry once we migrate existing agents.
    lf.logging.LogEntry,
]


class ExecutionTrace(pg.Object, pg.views.html.HtmlTreeView.Extension):
  """Trace of an execution, containing queries, logs, and sub-actions.

  `ExecutionTrace` records the sequence of operations performed during an
  action's execution or within a specific phase of execution (demarcated by
  `session.track_phase`). It captures `lf.query` calls, log entries, and
  nested `ActionInvocation` objects in the order they occurred. It also
  aggregates LLM usage summaries from its child items.
  """

  name: Annotated[
      str | None,
      (
          'The name of the execution trace. If None, the trace is unnamed, '
          'which is the case for the top-level trace of an action. An '
          'execution trace could have sub-traces, called phases, which are '
          'created and named by `session.track_phase()` context manager.'
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

  def _on_parent_change(self, *args, **kwargs):
    super()._on_parent_change(*args, **kwargs)
    self.__dict__.pop('id', None)
    self.__dict__.pop('parent_execution_unit', None)
    self.__dict__.pop('position', None)

  def indexof(
      self,
      item: TracedItem,
      count_item_cls: Type[Any] | tuple[Type[Any], ...]
  ) -> int:
    """Returns the index of the child item of given type."""
    pos = 0
    for x in self._iter_children(count_item_cls):
      if x is item:
        return pos
      pos += 1
    return -1

  @functools.cached_property
  def parent_execution_unit(self) -> ExecutionUnit:
    """Returns the parent execution unit of the current execution trace."""
    parent = self.sym_parent
    if isinstance(parent, ActionInvocation):
      # Current execution trace is the body of an action.
      return parent
    elif isinstance(parent, pg.List):
      container = parent.sym_parent
      if isinstance(container, ParallelExecutions):
        return container
      elif isinstance(container, ExecutionTrace):
        return container.parent_execution_unit
    assert False, (
        'Execution trace is not associated with any `ActionInvocation` or '
        f'`ParallelExecutions`: {self}'
    )

  @functools.cached_property
  def position(self) -> ExecutionUnit.Position | None:
    """Returns the execution position of the execution trace.

    Returns:
      The execution position of the execution trace, or None if the execution
      trace is either not associated with an execution unit or the execution
      trace is a phase under another execution trace.
    """
    parent = self.sym_parent
    if isinstance(parent, ActionInvocation):
      # Current execution trace is the body of an action.
      return parent.position
    elif isinstance(parent, pg.List):
      container = parent.sym_parent
      if isinstance(container, ParallelExecutions):
        return ExecutionUnit.Position(
            container.position, self.sym_path.key
        )
      elif isinstance(container, ExecutionTrace):
        # When execution trace is a phase under another execution trace,
        # we return None as the position.
        return None
    assert False, (
        'Execution trace is not associated with any `ActionInvocation` or '
        f'`ParallelExecutions`: {self}'
    )

  @functools.cached_property
  def id(self) -> str:
    parent = self.sym_parent
    if isinstance(parent, ActionInvocation):
      # Current execution trace is the body of an action.
      return parent.id
    elif isinstance(parent, pg.List):
      container = parent.sym_parent
      if isinstance(container, ExecutionTrace):
        # Current execution trace is a phase.
        group_id = (
            self.name or f'g{container.indexof(self, ExecutionTrace) + 1}'
        )
        return f'{container.id}/{group_id}'
      elif isinstance(container, ParallelExecutions):
        # Current execution trace is a parallel branch.
        return f'{container.id}/b{self.sym_path.key + 1}'
    return ''

  def reset(self) -> None:
    """Resets the execution trace."""
    self.rebind(items=[], skip_notification=True, raise_on_no_change=False)

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
    return list(self._iter_children(lf_structured.QueryInvocation))

  @property
  def actions(self) -> list['ActionInvocation']:
    """Returns action invocations from the sequence."""
    return list(self._iter_children(ActionInvocation))

  @property
  def execution_units(self) -> list[ExecutionUnit]:
    """Returns parallel executions from the sequence."""
    return list(self._iter_children(ExecutionUnit))

  @property
  def logs(self) -> list[lf.logging.LogEntry]:
    """Returns logs from the sequence."""
    return list(self._iter_children(lf.logging.LogEntry))

  @property
  def all_queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns all queries from current trace and its child execution items."""
    return list(self._iter_subtree(lf_structured.QueryInvocation))

  @property
  def all_actions(self) -> list['ActionInvocation']:
    """Returns all actions from current trace and its child execution items."""
    return list(self._iter_subtree(ActionInvocation))

  @property
  def all_logs(self) -> list[lf.logging.LogEntry]:
    """Returns all logs from current trace and its child execution items."""
    return list(self._iter_subtree(lf.logging.LogEntry))

  def _iter_children(
      self, item_cls: Type[Any] | tuple[Type[Any], ...]
  ) -> Iterator[TracedItem]:
    for item in self.items:
      if isinstance(item, item_cls):
        yield item
      elif isinstance(item, ExecutionTrace):
        for x in item._iter_children(item_cls):  # pylint: disable=protected-access
          yield x
      elif isinstance(item, ParallelExecutions):
        for branch in item.branches:
          for x in branch._iter_children(item_cls):  # pylint: disable=protected-access
            yield x

  def _iter_subtree(
      self,
      item_cls: Type[Any] | tuple[Type[Any], ...]
  ) -> Iterator[TracedItem]:
    for item in self.items:
      if isinstance(item, item_cls):
        yield item
      if isinstance(item, ActionInvocation):
        for x in item.execution._iter_subtree(item_cls):  # pylint: disable=protected-access
          yield x
      elif isinstance(item, ExecutionTrace):
        for x in item._iter_subtree(item_cls):  # pylint: disable=protected-access
          yield x
      elif isinstance(item, ParallelExecutions):
        for branch in item.branches:
          for x in branch._iter_subtree(item_cls):  # pylint: disable=protected-access
            yield x

  #
  # Shortcut methods to operate on the execution trace.
  #

  def __len__(self) -> int:
    return len(self.items)

  def __iter__(self) -> Iterator[TracedItem]:
    return iter(self.items)

  def __bool__(self) -> bool:
    return bool(self.items)

  def __getitem__(self, index: int) -> TracedItem:
    return self.items[index]

  def merge_usage_summary(self, usage_summary: lf.UsageSummary) -> None:
    if usage_summary.total.num_requests == 0:
      return
    current_invocation = self
    while current_invocation is not None:
      current_invocation.usage_summary.merge(usage_summary)
      current_invocation = typing.cast(
          ExecutionTrace,
          current_invocation.sym_ancestor(
              lambda x: isinstance(x, ExecutionTrace)
          )
      )

  def append(self, item: TracedItem) -> None:
    """Appends an item to the sequence."""
    with pg.notify_on_change(False):
      self.items.append(item)

    if isinstance(item, lf_structured.QueryInvocation):
      self.merge_usage_summary(item.usage_summary)

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

  def remove(self, item: TracedItem) -> None:
    """Removes an item from the sequence."""
    index = self.items.index(item)
    if index == -1:
      raise ValueError(f'Item not found in execution trace: {item!r}')

    with pg.notify_on_change(False):
      self.items.pop(index)

    if self._tab_control is not None:
      self._tab_control.remove(index)

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
        subtree=dict(
            num_actions=len(self.all_actions),
            num_action_failures=len([
                a for a in self.all_actions if a.has_error
            ]),
            num_queries=len(self.all_queries),
            num_oop_failures=len([
                q for q in self.all_queries if q.has_oop_error
            ]),
            num_non_oop_failures=len([
                q for q in self.all_queries
                if q.has_error and not q.has_oop_error
            ]),
            total_query_time=sum(q.elapse for q in self.all_queries),
        ),
        current_level=dict(
            num_actions=len(self.actions),
            num_action_failures=len([
                a for a in self.actions if a.has_error
            ]),
            num_queries=len(self.queries),
            num_oop_failures=len([
                q for q in self.queries if q.has_oop_error
            ]),
            num_non_oop_failures=len([
                q for q in self.queries
                if q.has_error and not q.has_oop_error
            ]),
            execution_breakdown=[
                dict(
                    action=action.action.__class__.__name__,
                    usage=dict(
                        total_tokens=action.usage_summary.total.total_tokens,
                        estimated_cost=action.usage_summary.total.estimated_cost,
                    ),
                    execution_time=action.execution.elapse,
                )
                for action in self.actions
            ]
        )
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
      css_class = f'log-{item.level}'
    elif isinstance(item, ExecutionTrace):
      css_class = 'phase'
    elif isinstance(item, ParallelExecutions):
      css_class = 'parallel'
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
          tooltip=f'[{item.id}] Action invocation',
      )
    elif isinstance(item, lf_structured.QueryInvocation):
      schema_title = 'str'
      if item.schema:
        schema_title = lf_structured.annotation(item.schema.spec)
      return pg.views.html.controls.Label(
          schema_title,
          tooltip=f'[{item.id}] lf.Query invocation'
      )
    elif isinstance(item, lf.logging.LogEntry):
      return pg.views.html.controls.Label(
          item.level.title(),
          tooltip=item.message,
      )
    elif isinstance(item, ExecutionTrace):
      return pg.views.html.controls.Label(
          item.name or 'Phase',
          tooltip=f'[{item.id}] Execution group {item.name!r}'
      )
    elif isinstance(item, ParallelExecutions):
      return pg.views.html.controls.Label(
          item.name or 'Parallel',
          tooltip=f'[{item.id}] Parallel executions'
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
          content: "G";
          font-weight: bold;
          color: purple;
          padding: 10px;
        }
        .tab-button.parallel > ::before {
          content: "P";
          font-weight: bold;
          color: blue;
          padding: 10px;
        }
        .tab-button.query > ::before {
          content: "Q";
          font-weight: bold;
          color: orange;
          padding: 10px;
        }
        .tab-button.log-debug > ::before {
          content: "🔍";
          padding: 7px;
        }
        .tab-button.log-info > ::before {
          content: "ⓘ";
          color: blue;
          padding: 7px;
        }
        .tab-button.log-warning > ::before {
          content: "❗";
          padding: 7px;
        }
        .tab-button.log-error > ::before {
          content: "‼️";
          padding: 7px;
        }
        .tab-button.log-fatal > ::before {
          content: "❌";
          padding: 7px;
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


class ParallelExecutions(ExecutionUnit, pg.views.html.HtmlTreeView.Extension):
  """A container for multiple parallel execution traces.

  When `session.concurrent_map` is used, it creates a `ParallelExecutions`
  object to hold an `ExecutionTrace` for each parallel branch of execution,
  allowing inspection of parallel workflows.
  """

  name: Annotated[
      str | None,
      'The name of the parallel execution.'
  ] = None

  branches: Annotated[
      list[ExecutionTrace],
      'The branches of the parallel execution.'
  ] = []

  def __getitem__(self, key: int) -> ExecutionTrace:
    return self.branches[key]

  def __len__(self) -> int:
    return len(self.branches)

  def _on_bound(self):
    super()._on_bound()
    self._tab_control = None
    self._lock = threading.Lock()

  def _on_parent_change(self, *args, **kwargs):
    super()._on_parent_change(*args, **kwargs)
    self.__dict__.pop('id', None)

  @functools.cached_property
  def id(self) -> str:
    parent = self.sym_parent
    if isinstance(parent, pg.List):
      container = parent.sym_parent
      if isinstance(container, ExecutionTrace):
        parallel_id = (
            self.name or f'p{container.indexof(self, ParallelExecutions) + 1}'
        )
        return f'{container.id}/{parallel_id}'
    return ''

  def add(self) -> ExecutionTrace:
    """Appends a branch to the parallel execution."""
    with self._lock, pg.notify_on_change(False):
      branch = ExecutionTrace(name=f'[{len(self)}]')
      self.branches.append(branch)
      if self._tab_control is not None:
        self._tab_control.append(self._branch_tab(branch))

      # Invalidate cached properties.
      self.__dict__.pop('all_queries', None)
      self.__dict__.pop('all_actions', None)
      self.__dict__.pop('all_logs', None)
      return branch

  #
  # ExecutionUnit interface.
  #

  @property
  def execution_units(self) -> list[ExecutionUnit]:
    """Returns immediate child execution items from execution sequence."""
    return []

  @property
  def queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns immediate queries made by the parallel execution."""
    return []

  @property
  def actions(self) -> list['ActionInvocation']:
    """Returns immediate child action invocations."""
    return []

  @property
  def logs(self) -> list[lf.logging.LogEntry]:
    """Returns immediate child logs from execution sequence."""
    return []

  @functools.cached_property
  def all_queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns all queries made by the action and its child execution items."""
    return list(
        itertools.chain.from_iterable(
            branch.all_queries for branch in self.branches
        )
    )

  @functools.cached_property
  def all_actions(self) -> list['ActionInvocation']:
    """Returns all actions made by the action and its child execution items."""
    return list(
        itertools.chain.from_iterable(
            branch.all_actions for branch in self.branches
        )
    )

  @property
  def all_logs(self) -> list[lf.logging.LogEntry]:
    """Returns all logs made by the action and its child execution items."""
    return list(
        itertools.chain.from_iterable(
            branch.all_logs for branch in self.branches
        )
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

  def _html_tree_view_content(
      self,
      *,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ):
    del kwargs
    extra_flags = extra_flags or {}
    interactive = extra_flags.get('interactive', True)
    if interactive or self.branches:
      self._tab_control = pg.views.html.controls.TabControl(
          [self._branch_tab(branch) for branch in self.branches],
          tab_position='left'
      )
      return self._tab_control.to_html()
    return '(no tracked parallel executions)'

  def _branch_tab(self, branch: ExecutionTrace) -> pg.views.html.controls.Tab:
    return pg.views.html.controls.Tab(
        label=pg.views.html.controls.Label(
            branch.name,
            tooltip=f'[{branch.id}] Branch {branch.name!r}'
        ),
        content=pg.view(branch),
    )


class ActionInvocation(ExecutionUnit, pg.views.html.HtmlTreeView.Extension):
  """An invocation of an action, capturing its execution and result.

  `ActionInvocation` represents a single call to an `Action`. It contains
  the `Action` object itself, its result or error, associated metadata,
  and an `ExecutionTrace` detailing the steps taken during its execution
  (queries, logs, sub-actions). Invocations form a tree structure within a
  `Session`, reflecting the hierarchy of agentic operations.
  """

  action: Annotated[
      Action,
      'The action being invoked.'
  ]

  result: Annotated[
      Any,
      'The result of the action.'
  ] = None

  metadata: Annotated[
      dict[str, Any],
      'The metadata returned by the action.'
  ] = {}

  error: Annotated[
      pg.ErrorInfo | None,
      'Error from the action if failed.'
  ] = None

  max_execution_time: Annotated[
      float | None,
      (
          'The maximum allowed execution time for the action. '
          'It is set when the action or any of its parent actions has been '
          'called with a `max_execution_time` argument. '
          'The value is the remaining time of the parent invocation or '
          'the `max_execution_time` for the current invocation, '
          'whichever is smaller. Since we are running action within the '
          'current thread, there is no guarantee that the action will be '
          'executed within this time limit, but we will try to stop the action '
          'as soon as the next operation on the session is called.'
      )
  ] = None

  execution: Annotated[
      ExecutionTrace,
      'The execution sequence of the action.'
  ] = ExecutionTrace()

  # Allow symbolic assignment without `rebind`.
  allow_symbolic_assignment = True

  def _on_bound(self):
    super()._on_bound()
    self._tab_control = None
    self.action._invocation = self    # pylint: disable=protected-access

  def _on_parent_change(self, *args, **kwargs):
    super()._on_parent_change(*args, **kwargs)
    self.__dict__.pop('id', None)
    self.__dict__.pop('position', None)

  @property
  def parent_action(self) -> Optional['ActionInvocation']:
    """Returns the parent action invocation."""
    return self.sym_ancestor(lambda x: isinstance(x, ActionInvocation))

  @property
  def max_remaining_execution_time(self) -> float | None:
    """Returns the remaining execution time for the action."""
    if self.max_execution_time is None:
      return None
    return max(0, self.max_execution_time - self.execution.elapse)

  @functools.cached_property
  def id(self) -> str:
    """Returns the id of the action invocation."""
    parent = self.sym_parent
    if isinstance(parent, Session):
      return f'{parent.id}:'
    elif isinstance(parent, pg.List):
      container = parent.sym_parent
      if isinstance(container, ExecutionTrace):
        action_id = f'a{container.indexof(self, ActionInvocation) + 1}'
        return f'{container.id}/{action_id}'
    return ''

  @property
  def has_error(self) -> bool:
    """Returns True if the action invocation has an error."""
    return self.error is not None

  @property
  def state(self) -> Any:
    """Returns the state of the action."""
    return self.action.state

  #
  # Implement `ExecutionUnit` interface.
  #

  @property
  def execution_units(self) -> list[ExecutionUnit]:
    """Returns immediate child execution items from execution sequence."""
    return self.execution.execution_units

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
  def all_actions(self) -> list['ActionInvocation']:
    """Returns all actions made by the action and its child execution items."""
    return self.execution.all_actions

  @property
  def all_logs(self) -> list[lf.logging.LogEntry]:
    """Returns all logs made by the action and its child execution items."""
    return self.execution.all_logs

  @property
  def usage_summary(self) -> lf.UsageSummary:
    """Returns the usage summary of the action."""
    return self.execution.usage_summary

  @property
  def elapse(self) -> float:
    """Returns the elapsed time of the action."""
    return self.execution.elapse

  def start(self) -> None:
    """Starts the execution of the action."""
    self.execution.start()

  def end(
      self,
      result: Any,
      error: pg.ErrorInfo | None = None,
      metadata: dict[str, Any] | None = None,
  ) -> None:
    """Ends the execution of the action with result and metadata."""
    with pg.notify_on_change(False):
      self.result = result
      self.error = error
      if metadata:
        self.metadata.update(metadata)

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
      if self.has_error:
        self._tab_control.insert(
            1,
            pg.views.html.controls.Tab(
                'error',
                pg.view(
                    self.error,
                    collapse_level=None,
                    enable_summary_tooltip=False
                ),
                name='error',
            )
        )
      else:
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
      self._tab_control.select(['error', 'metadata', 'result'])

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


class SessionEventHandler:
  """Interface for handling session events."""

  def get(
      self,
      session_cls: type['SessionEventHandler']
  ) -> Optional['SessionEventHandler']:
    """Returns this or a child event handler for the given session class."""
    if isinstance(self, session_cls):
      return self
    elif isinstance(self, SessionEventHandlerChain):
      for handler in self.handlers:
        if v := handler.get(session_cls):
          return v
    return None

  def on_session_start(
      self,
      session: 'Session'
  ) -> None:
    """Called when a session starts."""

  def on_session_end(
      self,
      session: 'Session'
  ) -> None:
    """Called when a session ends."""

  def on_action_start(
      self,
      session: 'Session',
      action: ActionInvocation
  ) -> None:
    """Called when an action starts."""

  def on_action_end(
      self,
      session: 'Session',
      action: ActionInvocation
  ) -> None:
    """Called when an action ends."""

  def on_action_progress(
      self,
      session: 'Session',
      action: ActionInvocation,
      title: str,
      **kwargs
  ) -> None:
    """Called when an action progress is updated."""

  def on_query_start(
      self,
      session: 'Session',
      action: ActionInvocation,
      query: lf_structured.QueryInvocation,
  ) -> None:
    """Called when a query starts."""

  def on_query_end(
      self,
      session: 'Session',
      action: ActionInvocation,
      query: lf_structured.QueryInvocation,
  ) -> None:
    """Called when a query ends."""


@dataclasses.dataclass
class SessionEventHandlerChain(SessionEventHandler):
  """A session event handler that chains multiple event handlers."""

  handlers: list[SessionEventHandler]

  def on_session_start(self, session: 'Session') -> None:
    """Called when a session starts."""
    for handler in self.handlers:
      handler.on_session_start(session)

  def on_session_end(self, session: 'Session') -> None:
    """Called when a session ends."""
    for handler in self.handlers:
      handler.on_session_end(session)

  def on_action_start(
      self,
      session: 'Session',
      action: ActionInvocation) -> None:
    """Called when an action starts."""
    for handler in self.handlers:
      handler.on_action_start(session, action)

  def on_action_end(
      self,
      session: 'Session',
      action: ActionInvocation) -> None:
    """Called when an action ends."""
    for handler in self.handlers:
      handler.on_action_end(session, action)

  def on_action_progress(
      self,
      session: 'Session',
      action: ActionInvocation,
      title: str,
      **kwargs
  ) -> None:
    """Called when an action progress is updated."""
    for handler in self.handlers:
      handler.on_action_progress(session, action, title, **kwargs)

  def on_query_start(
      self,
      session: 'Session',
      action: ActionInvocation,
      query: lf_structured.QueryInvocation,
  ) -> None:
    """Called when a query starts."""
    for handler in self.handlers:
      handler.on_query_start(session, action, query)

  def on_query_end(
      self,
      session: 'Session',
      action: ActionInvocation,
      query: lf_structured.QueryInvocation,
  ) -> None:
    """Called when a query ends."""
    for handler in self.handlers:
      handler.on_query_end(session, action, query)


@dataclasses.dataclass
class SessionLogging(SessionEventHandler):
  """An event handler that logs Session events."""

  verbose: bool = False

  def on_session_end(self, session: 'Session'):
    if session.has_error:
      session.error(
          f'Trajectory failed in {session.elapse:.2f} seconds.',
          error=session.final_error,
          metadata=session.root.metadata,
          keep=True,
      )
    elif self.verbose:
      session.info(
          f'Trajectory succeeded in {session.elapse:.2f} seconds.',
          result=session.final_result,
          metadata=session.root.metadata,
          keep=False,
      )

  def on_action_start(
      self,
      session: 'Session',
      action: ActionInvocation
  ) -> None:
    if self.verbose:
      session.info(
          'Action execution started.',
          action=action.action,
          keep=False,
      )

  def on_action_end(
      self,
      session: 'Session',
      action: ActionInvocation
  ) -> None:
    if action.has_error:
      session.warning(
          (
              f'Action execution failed in '
              f'{action.execution.elapse:.2f} seconds.'
          ),
          action=action.action,
          error=action.error,
          keep=True,
      )
    elif self.verbose:
      session.info(
          (
              f'Action execution succeeded in '
              f'{action.execution.elapse:.2f} seconds.'
          ),
          action=action.action,
          result=action.result,
          keep=False,
      )

  def on_query_start(
      self,
      session: 'Session',
      action: ActionInvocation,
      query: lf_structured.QueryInvocation,
  ) -> None:
    if self.verbose:
      session.info(
          'Querying LLM started.',
          lm=query.lm.model_id,
          output_type=(
              lf_structured.annotation(query.schema.spec)
              if query.schema is not None else None
          ),
          keep=False,
      )

  def on_query_end(
      self,
      session: 'Session',
      action: ActionInvocation,
      query: lf_structured.QueryInvocation,
  ) -> None:
    if query.has_error:
      session.warning(
          (
              f'Querying LLM failed in '
              f'{time.time() - query.start_time:.2f} seconds.'
          ),
          lm=query.lm.model_id,
          output_type=(
              lf_structured.annotation(query.schema.spec)
              if query.schema is not None else None
          ),
          error=query.error,
          keep=True,
      )
    elif self.verbose:
      session.info(
          (
              f'Querying LLM succeeded in '
              f'{time.time() - query.start_time:.2f} seconds.'
          ),
          lm=query.lm.model_id,
          output_type=(
              lf_structured.annotation(query.schema.spec)
              if query.schema is not None else None
          ),
          keep=False,
      )


class Session(pg.Object, pg.views.html.HtmlTreeView.Extension):
  """Manages the execution trajectory of agentic actions.

  A `Session` tracks the execution of a root `Action` and all its
  sub-actions, including LLM queries (`lf.query`), logging messages,
  and nested actions. It provides a complete, hierarchical trace of an
  agent's workflow, which is important for debugging, analysis, and
  visualization.

  Sessions can be created implicitly when an action is called without an
  active session, or explicitly for more control.

  **1. Implicit Session:**
  When an action is called without a session, Langfun creates one automatically.

  ```python
  action = MyAction()
  action()
  session = action.session  # Access the implicit session
  ```

  **2. Explicit Session:**
  Use a `with` statement to manage a session explicitly. This is useful for
  setting session IDs or capturing the trajectory of multiple top-level actions.

  ```python
  with lf.Session(id='my-session') as session:
    action1()
    action2()
  ```

  **3. Accessing Trajectory:**
  The `session.root` attribute provides access to the `ActionInvocation` tree.

  ```python
  with lf.Session() as session:
    my_action()

  # Get all queries in the session
  print(session.all_queries)

  # Get all top-level action calls in the session
  print(session.root.actions)
  ```
  """

  root: Annotated[
      ActionInvocation,
      'The root action invocation of the session.'
  ]

  id: Annotated[
      str | None,
      'An optional identifier for the session, which will be used for logging.'
  ]

  @pg.explicit_method_override
  def __init__(
      self,
      id: str | None = None,   # pylint: disable=redefined-builtin
      *,
      verbose: bool = False,
      event_handler: SessionEventHandler | None = None,
      root: ActionInvocation | None = None,
      **kwargs
  ):
    super().__init__(
        id=id,
        root=root or ActionInvocation(RootAction()),
        **kwargs
    )
    self._event_handler = event_handler or SessionLogging(verbose=verbose)

  @property
  def event_handler(self) -> SessionEventHandler:
    """Returns the event handler for the session."""
    return self._event_handler

  def _sym_clone(self, deep: bool, memo: Any = None) -> 'Session':
    other = super()._sym_clone(deep=deep, memo=memo)
    if deep:
      event_handler = pg.clone(self.event_handler, deep=deep, memo=memo)
    else:
      event_handler = self.event_handler
    other._event_handler = event_handler  # pylint: disable=protected-access
    return other

  #
  # Shortcut methods for accessing the root action invocation.
  #

  @property
  def metadata(self) -> dict[str, Any]:
    """Returns metadata associated with the root of the session."""
    return self.root.metadata

  @property
  def all_queries(self) -> list[lf_structured.QueryInvocation]:
    """Returns all queries made by the session."""
    return self.root.all_queries

  @property
  def all_actions(self) -> list[ActionInvocation]:
    """Returns all actions made by the session."""
    return self.root.all_actions

  @property
  def all_logs(self) -> list[lf.logging.LogEntry]:
    """Returns all logs made by the session."""
    return self.root.all_logs

  @property
  def usage_summary(self) -> lf.UsageSummary:
    """Returns the usage summary of the session."""
    return self.root.usage_summary

  @property
  def has_started(self) -> bool:
    """Returns True if the session has started."""
    return self.root.execution.has_started

  @property
  def has_stopped(self) -> bool:
    """Returns True if the session has stopped."""
    return self.root.execution.has_stopped

  @property
  def has_error(self) -> bool:
    """Returns True if the session has an error."""
    return self.root.has_error

  @property
  def final_result(self) -> Any:
    """Returns the final result of the session."""
    return self.root.result

  @property
  def final_error(self) -> pg.ErrorInfo | None:
    """Returns the error of the session."""
    return self.root.error

  @property
  def elapse(self) -> float:
    """Returns the elapsed time of the session."""
    return self.root.elapse

  # NOTE(daiyip): Action execution may involve multi-threading, hence current
  # action and execution are thread-local.

  @property
  def _current_action(self) -> ActionInvocation:
    """Returns the current invocation."""
    return getattr(self._tls, '__current_action__')

  @_current_action.setter
  def _current_action(self, value: ActionInvocation):
    setattr(self._tls, '__current_action__', value)

  @property
  def _current_execution(self) -> ExecutionTrace:
    """Returns the current execution."""
    return getattr(self._tls, '__current_execution__')

  @_current_execution.setter
  def _current_execution(self, value: ExecutionTrace):
    setattr(self._tls, '__current_execution__', value)

  def _on_bound(self):
    super()._on_bound()
    self._tls = threading.local()
    self._current_action = self.root
    self._current_execution = self.root.execution
    if self.id is None:
      self.rebind(
          id=f'agent@{uuid.uuid4().hex[-7:]}',
          skip_notification=True
      )

  def start(self) -> None:
    """Starts the session."""
    self.root.execution.start()
    self.event_handler.on_session_start(self)

  def end(
      self,
      result: Any,
      error: pg.ErrorInfo | None = None,
      metadata: dict[str, Any] | None = None,
  ) -> None:
    """Ends the session."""
    self.root.end(result, error, metadata)
    self.event_handler.on_session_end(self)

  def check_execution_time(self) -> None:
    """Checks the execution time of the current action."""
    action = self._current_action
    if action is not None and 0 == action.max_remaining_execution_time:
      # We raise error on the top-most action which has a time limit.
      current_action = action
      parent_action = current_action.parent_action
      while parent_action is not None and parent_action.max_execution_time:
        current_action = parent_action
        parent_action = current_action.parent_action
      raise ActionTimeoutError(
          f'Action {current_action.id} '
          f'({current_action.action.__class__.__name__}) has exceeded its '
          f'maximum execution time of {current_action.max_execution_time} '
          'seconds.'
      )

  def update_progress(self, title: str, **kwargs: Any) -> None:
    """Updates the progress of current action's execution.

    Args:
      title: The title of the progress update.
      **kwargs: Additional keyword arguments to pass to the event handler.
    """
    self.event_handler.on_action_progress(
        self,
        self._current_action,
        title,
        **kwargs
    )

  def __enter__(self):
    """Enters the session."""
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exits the session."""
    # We allow users to explicitly end the session with specified result
    # and metadata.
    if self.root.execution.has_stopped:
      return

    if exc_val is not None:
      result, metadata = None, None
      error = pg.ErrorInfo.from_exception(exc_val)
    else:
      actions = self.root.actions
      if actions:
        result = actions[-1].result
        error = actions[-1].error
        metadata = actions[-1].metadata
      else:
        result, error, metadata = None, None, None
    self.end(result, error, metadata)

  #
  # Context-manager for information tracking.
  #

  @contextlib.contextmanager
  def track_action(
      self,
      action: Action,
      max_execution_time: float | None = None
  ) -> Iterator[ActionInvocation]:
    """Track the execution of an action."""
    if not self.root.execution.has_started:
      raise ValueError(
          'Please call `Session.start() / Session.end()` explicitly, '
          'or use `with Session(...) as session: ...` context manager to '
          'signal the start and end of the session.'
      )

    invocation = ActionInvocation(
        pg.maybe_ref(action),
        max_execution_time=self._child_max_execution_time(max_execution_time)
    )
    parent_action = self._current_action
    parent_execution = self._current_execution
    parent_execution.append(invocation)

    try:
      self._current_action = invocation
      self._current_execution = invocation.execution
      # Start the execution of the current action.
      self._current_action.start()
      self.event_handler.on_action_start(self, self._current_action)
      yield invocation
    finally:
      self.event_handler.on_action_end(self, self._current_action)
      self._current_execution = parent_execution
      self._current_action = parent_action

  @contextlib.contextmanager
  def track_phase(self, name: str | None) -> Iterator[ExecutionTrace]:
    """Context manager for starting a new execution phase (group)."""
    parent_execution = self._current_execution
    if name is None:
      phase = parent_execution
    else:
      phase = ExecutionTrace(name=name)
      phase.start()
      parent_execution.append(phase)

    try:
      self._current_execution = phase
      yield phase
    finally:
      if phase is not parent_execution:
        phase.stop()
        self._current_execution = parent_execution

  @contextlib.contextmanager
  def track_queries(
      self,
      phase: str | None = None,
      track_if: Callable[
          [lf_structured.QueryInvocation],
          bool
      ] | None = None,
  ) -> Iterator[list[lf_structured.QueryInvocation]]:
    """Tracks `lf.query` made within the context.

    Args:
      phase: The name of a new phase to track the queries in. If not provided,
        the queries will be tracked in the parent phase.
      track_if: A function that takes a `lf_structured.QueryInvocation` and
        returns True if the query should be included in the result. If None,
        all queries (including failed queries) will be included.

    Yields:
      A list of `lf.QueryInvocation` objects, each for a single `lf.query`
      call.
    """
    def _query_start(invocation: lf_structured.QueryInvocation):
      # Early terminate the action if the execution time is exceeded.
      self.check_execution_time()

      execution = self._current_execution
      invocation.rebind(
          id=f'{execution.id}/q{len(execution.queries) + 1}',
          skip_notification=False, raise_on_no_change=False
      )
      execution.append(invocation)
      self.event_handler.on_query_start(self, self._current_action, invocation)

    def _query_end(invocation: lf_structured.QueryInvocation):
      if track_if is not None and not track_if(invocation):
        self._current_execution.remove(invocation)
      # Even if the query is not included in the execution trace, we still
      # count the usage summary to the current execution and trigger the
      # event handler to log the query.
      self._current_execution.merge_usage_summary(invocation.usage_summary)
      self.event_handler.on_query_end(self, self._current_action, invocation)

    with self.track_phase(phase), lf_structured.track_queries(
        include_child_scopes=False,
        start_callback=_query_start,
        end_callback=_query_end,
    ) as queries:
      try:
        yield queries
      finally:
        pass

  #
  # Operations with activity tracking.
  #

  def add_metadata(self, **kwargs: Any) -> None:
    """Adds metadata to the current invocation."""
    with pg.notify_on_change(False):
      self._current_action.metadata.update(kwargs)

  def query(
      self,
      prompt: Union[str, lf.Template, Any],
      schema: Union[
          lf_structured.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
      ] = None,
      default: Any = lf.RAISE_IF_HAS_ERROR,
      *,
      lm: lf.LanguageModel,
      examples: list[lf_structured.MappingExample] | None = None,
      track_if: Callable[[lf_structured.QueryInvocation], bool] | None = None,
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
      track_if: A function that takes a `lf_structured.QueryInvocation`
        and returns True if the query should be tracked.
        If None, all queries (including failed queries) will be tracked.
      **kwargs: Additional keyword arguments to pass to `lf.query`.

    Returns:
      The result of the query.
    """
    with self.track_queries(track_if=track_if):
      return lf_structured.query(
          prompt,
          schema=schema,
          default=default,
          lm=lm,
          examples=examples,
          **kwargs
      )

  def concurrent_map(
      self,
      func: Callable[[Any], Any],
      parallel_inputs: Iterable[Any],
      *,
      phase: str | None = None,
      max_workers: int = 32,
      timeout: int | None = None,
      silence_on_errors: Union[
          Type[BaseException], tuple[Type[BaseException], ...], None
      ] = Exception,
      ordered: bool = False,
      show_progress: bool | int = False,
  ) -> Iterator[Any]:
    """Starts and tracks parallel execution with `lf.concurrent_map`."""
    self.check_execution_time()
    parallel_inputs = list(parallel_inputs)
    parallel_execution = ParallelExecutions(name=phase)
    self._current_execution.append(parallel_execution)
    parent_action = self._current_action

    def _map_single(input_value):
      execution = parallel_execution.add()

      # This happens on a new thread. Therefore, we update the thread-local
      # states from the parent thread.
      self._current_execution = execution
      self._current_action = parent_action
      execution.start()
      try:
        with self.track_queries():
          return func(input_value)
      finally:
        execution.stop()

    for input_value, result, error in lf.concurrent_map(
        _map_single,
        parallel_inputs,
        max_workers=max_workers,
        timeout=self._child_max_execution_time(timeout),
        silence_on_errors=silence_on_errors,
        ordered=ordered,
        show_progress=show_progress,
    ):
      yield input_value, result, error

  def _child_max_execution_time(
      self, max_execution_time: float | None
  ) -> float | None:
    """Returns the max execution time for the child action."""
    max_remaining = None
    if self._current_action is not None:
      max_remaining = self._current_action.max_remaining_execution_time
    if max_remaining is None:
      return max_execution_time
    if max_execution_time is None:
      return max_remaining
    return min(max_remaining, max_execution_time)

  def _log(
      self,
      level: lf.logging.LogLevel,
      message: str,
      keep: bool,
      *,
      for_action: Action | ActionInvocation | None = None,
      **kwargs
  ) -> None:
    """Logs a message to the session.

    Args:
      level: The logging level.
      message: The message to log.
      keep: Whether to keep the log entry in the execution trace.
      for_action: The action to log the message for. If not provided, the
        current action will be used.
      **kwargs: Additional keyword arguments to pass to `lf.logging.log` as
        metadata to show.
    """
    # Early terminate the action if the execution time is exceeded.
    self.check_execution_time()

    execution = self._current_execution
    if for_action is None:
      for_action = self._current_action
    elif isinstance(for_action, Action):
      for_action = for_action.invocation
      assert for_action is not None, (
          f'Action must be called before it can be logged: {for_action}'
      )

    log_entry = lf.logging.log(
        level,
        (
            f'[{for_action.id} ({for_action.action.__class__.__name__})]: '
            f'{message}'
        ),
        **kwargs
    )
    if keep:
      execution.append(log_entry)

  def debug(
      self,
      message: str,
      keep: bool = True,
      *,
      for_action: Action | ActionInvocation | None = None,
      **kwargs
  ) -> None:
    """Logs a debug message to the session."""
    self._log(
        'debug', message, keep=keep, for_action=for_action, **kwargs
    )

  def info(
      self,
      message: str,
      keep: bool = True,
      *,
      for_action: Action | ActionInvocation | None = None,
      **kwargs
  ) -> None:
    """Logs an info message to the session."""
    self._log(
        'info', message, keep=keep, for_action=for_action, **kwargs
    )

  def warning(
      self,
      message: str,
      keep: bool = True,
      *,
      for_action: Action | ActionInvocation | None = None,
      **kwargs
  ) -> None:
    """Logs a warning message to the session."""
    self._log(
        'warning', message, keep=keep, for_action=for_action, **kwargs
    )

  def error(
      self,
      message: str,
      keep: bool = True,
      *,
      for_action: Action | ActionInvocation | None = None,
      **kwargs
  ):
    """Logs an error message to the session."""
    self._log(
        'error', message, keep=keep, for_action=for_action, **kwargs
    )

  def fatal(
      self,
      message: str,
      keep: bool = True,
      *,
      for_action: Action | ActionInvocation | None = None,
      **kwargs
  ):
    """Logs a fatal message to the session."""
    self._log(
        'fatal', message, keep=keep, for_action=for_action, **kwargs
    )

  def as_message(self) -> lf.AIMessage:
    """Returns the session as a message."""
    return lf.AIMessage(
        'Agentic task session.',
        result=self.root
    )

  @property
  def current_action(self) -> ActionInvocation:
    """Returns the current invocation."""
    return self._current_action

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


# Register the logging functions to skip the source of the logging functions.
pg.logging.register_frame_to_skip([
    Session._log,   # pylint: disable=protected-access
    Session.debug,
    Session.info,
    Session.warning,
    Session.error,
    Session.fatal
])
