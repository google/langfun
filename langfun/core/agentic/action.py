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
from typing import Annotated, Any, Iterable, Iterator, Optional, Type, Union
import langfun.core as lf
from langfun.core import structured as lf_structured
import pyglove as pg


class Action(pg.Object):
  """Base class for agent actions."""

  def _on_bound(self):
    super()._on_bound()
    self._result = None

  @property
  def result(self) -> Any:
    """Returns the result of the action."""
    return self._result

  def __call__(
      self, session: Optional['Session'] = None, **kwargs) -> Any:
    """Executes the action."""
    session = session or Session()
    with session.track(self):
      self._result = self.call(session=session, **kwargs)
      return self._result

  @abc.abstractmethod
  def call(self, session: 'Session', **kwargs) -> Any:
    """Subclasses to implement."""


class ActionInvocation(pg.Object, pg.views.html.HtmlTreeView.Extension):
  """A class for capturing the invocation of an action."""
  action: Action

  result: Annotated[
      Any,
      'The result of the action.'
  ] = None

  execution: Annotated[
      list[
          Union[
              lf_structured.QueryInvocation,
              'ActionInvocation',
              lf.logging.LogEntry
          ]
      ],
      'Execution execution.'
  ] = []

  # Allow symbolic assignment without `rebind`.
  allow_symbolic_assignment = True

  @property
  def logs(self) -> list[lf.logging.LogEntry]:
    """Returns logs from execution sequence."""
    return [v for v in self.execution if isinstance(v, lf.logging.LogEntry)]

  @property
  def child_invocations(self) -> list['ActionInvocation']:
    """Returns child action invocations."""
    return [v for v in self.execution if isinstance(v, ActionInvocation)]

  def queries(
      self,
      include_children: bool = False
  ) -> Iterable[lf_structured.QueryInvocation]:
    """Iterates over queries from the current invocation."""
    for v in self.execution:
      if isinstance(v, lf_structured.QueryInvocation):
        yield v
      elif isinstance(v, ActionInvocation):
        if include_children:
          yield from v.queries(include_children=True)

  def _html_tree_view_summary(
      self, *, view: pg.views.html.HtmlTreeView, **kwargs
  ):
    if isinstance(self.action, RootAction):
      return None
    kwargs.pop('title')
    return view.summary(
        self,
        title=view.render(
            self.action, name='action', collapse_level=0,
            css_classes='invocation-title',
        ),
        **kwargs
    )

  def _html_tree_view_content(
      self,
      *,
      root_path: pg.KeyPath | None = None,
      collapse_level: int | None = None,
      view: pg.views.html.HtmlTreeView,
      **kwargs
  ):
    prepare_phase = []
    current_phase = prepare_phase
    action_phases = []
    for item in self.execution:
      if isinstance(item, ActionInvocation):
        current_phase = []
        action_phases.append(current_phase)
      current_phase.append(item)

    def _render_phase(
        phase: list[ActionInvocation | lf.logging.LogEntry]
    ) -> pg.Html.WritableTypes:
      return pg.Html.element(
          'div',
          [
              view.render(item) for item in phase
          ]
      )

    def _render_action_phases(
        phases: list[list[ActionInvocation | lf.logging.LogEntry]]
    ) -> pg.Html.WritableTypes:
      if len(phases) == 1:
        return _render_phase(phases[0])
      return pg.views.html.controls.TabControl(
          [
              pg.views.html.controls.Tab(
                  label=f'Step {i + 1}',
                  content=_render_phase(phase),
              )
              for i, phase in enumerate(phases)
          ],
      )

    result_name = 'final_result' if isinstance(
        self.action, RootAction) else 'result'
    return pg.Html.element(
        'div',
        [
            view.render(
                self.result,
                name=result_name,
                css_classes=[
                    f'invocation-{result_name}'.replace('_', '-')
                ]
            ),
            _render_phase(prepare_phase) if prepare_phase else None,
            _render_action_phases(action_phases)
        ]
    )

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        details.invocation-title {
          display: inline-block;
          background-color: #b1f0ff;
          border: 1px solid white;
        }
        details.invocation-result {
          border: 1px solid #eee;
        }
        details.invocation-final-result {
          border: 1px solid #eee;
          background-color: #fef78f;
        }
        """
    ]


class RootAction(Action):
  """A placeholder action for the root of the action tree."""

  def call(self, session: 'Session', **kwargs) -> Any:
    raise NotImplementedError('Shall not be called.')


class Session(pg.Object):
  """Session for performing an agentic task."""

  root_invocation: ActionInvocation = ActionInvocation(RootAction())

  def _on_bound(self):
    super()._on_bound()
    self._invocation_stack = [self.root_invocation]

  @property
  def final_result(self) -> Any:
    """Returns the final result of the session."""
    return self.root_invocation.result

  @property
  def current_invocation(self) -> ActionInvocation:
    """Returns the current invocation."""
    assert self._invocation_stack
    return self._invocation_stack[-1]

  @contextlib.contextmanager
  def track(self, action: Action) -> Iterator[ActionInvocation]:
    """Track the execution of an action."""
    new_invocation = ActionInvocation(pg.maybe_ref(action))
    with pg.notify_on_change(False):
      self.current_invocation.execution.append(new_invocation)
    self._invocation_stack.append(new_invocation)

    try:
      yield new_invocation
    finally:
      assert self._invocation_stack
      invocation = self._invocation_stack.pop(-1)
      invocation.rebind(
          result=action.result, skip_notification=True, raise_on_no_change=False
      )
      assert invocation.action is action, (invocation.action, action)
      assert self._invocation_stack, self._invocation_stack

      if len(self._invocation_stack) == 1:
        self.root_invocation.rebind(
            result=invocation.result,
            skip_notification=True,
            raise_on_no_change=False
        )

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
    """Calls `lf.query` and associates it with the current invocation."""
    with lf_structured.track_queries() as queries:
      output = lf_structured.query(
          prompt,
          schema=schema,
          default=default,
          lm=lm,
          examples=examples,
          **kwargs
      )
    with pg.notify_on_change(False):
      self.current_invocation.execution.extend(queries)
    return output

  def _log(self, level: lf.logging.LogLevel, message: str, **kwargs):
    with pg.notify_on_change(False):
      self.current_invocation.execution.append(
          lf.logging.log(
              level, message, indent=len(self._invocation_stack) - 1, **kwargs
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
        result=self.root_invocation
    )
