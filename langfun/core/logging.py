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
"""Langfun event logging."""

import contextlib
import datetime
import functools
import typing
from typing import Any, Iterator, Literal

from langfun.core import component
from langfun.core import console
import pyglove as pg


LogLevel = Literal['debug', 'info', 'error', 'warning', 'fatal']
_LOG_LEVELS = list(typing.get_args(LogLevel))


@contextlib.contextmanager
def use_log_level(log_level: LogLevel = 'info') -> Iterator[None]:
  """Contextmanager to enable logging at a given level."""
  with component.context(__event_log_level__=log_level):
    try:
      yield
    finally:
      pass


def get_log_level() -> LogLevel:
  """Gets the current minimum log level."""
  return component.context_value('__event_log_level__', 'info')


class LogEntry(pg.Object):
  """Event log entry."""
  time: datetime.datetime
  level: LogLevel
  message: str
  metadata: dict[str, Any] = pg.Dict()
  indent: int = 0

  def should_output(self, min_log_level: LogLevel) -> bool:
    return _LOG_LEVELS.index(self.level) >= _LOG_LEVELS.index(min_log_level)

  def _html_tree_view_summary(
      self,
      view: pg.views.HtmlTreeView,
      title: str | pg.Html | None = None,
      max_summary_len_for_str: int = 80,
      **kwargs
      ) -> str:
    if len(self.message) > max_summary_len_for_str:
      message = self.message[:max_summary_len_for_str] + '...'
    else:
      message = self.message

    s = pg.Html(
        pg.Html.element(
            'span',
            [self.time.strftime('%H:%M:%S')],
            css_classes=['log-time']
        ),
        pg.Html.element(
            'span',
            [pg.Html.escape(message)],
            css_classes=['log-summary'],
        ),
    )
    return view.summary(
        self,
        title=title or s,
        max_summary_len_for_str=max_summary_len_for_str,
        **kwargs,
    )

  # pytype: disable=annotation-type-mismatch
  def _html_tree_view_content(
      self,
      view: pg.views.HtmlTreeView,
      root_path: pg.KeyPath,
      max_summary_len_for_str: int = 80,
      collapse_level: int | None = 1,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ) -> pg.Html:
    # pytype: enable=annotation-type-mismatch
    extra_flags = extra_flags if extra_flags is not None else {}
    collapse_log_metadata_level: int | None = extra_flags.get(
        'collapse_log_metadata_level', None
    )
    def render_message_text():
      if len(self.message) < max_summary_len_for_str:
        return None
      return pg.Html.element(
          'span',
          [pg.Html.escape(self.message)],
          css_classes=['log-text'],
      )

    def render_metadata():
      if not self.metadata:
        return None
      return pg.Html.element(
          'div',
          [
              view.render(
                  self.metadata,
                  name='metadata',
                  root_path=root_path + 'metadata',
                  parent=self,
                  collapse_level=view.get_collapse_level(
                      (collapse_level, -1), collapse_log_metadata_level,
                  ),
                  max_summary_len_for_str=max_summary_len_for_str,
                  extra_flags=extra_flags,
                  **view.get_passthrough_kwargs(**kwargs),
              )
          ],
          css_classes=['log-metadata'],
      )

    return pg.Html.element(
        'div',
        [
            render_message_text(),
            render_metadata(),
        ],
        css_classes=['complex_value'],
    )

  def _html_tree_view_config(self) -> dict[str, Any]:
    return pg.views.HtmlTreeView.get_kwargs(
        super()._html_tree_view_config(),
        dict(
            css_classes=[f'log-{self.level}'],
        )
    )

  @classmethod
  @functools.cache
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        /* Langfun LogEntry styles. */
        .log-time {
          color: #222;
          font-size: 12px;
          padding-right: 10px;
        }
        .log-summary {
          font-weight: normal;
          font-style: italic;
          padding: 4px;
        }
        .log-debug > summary > .summary_title::before {
          content: '🛠️ '
        }
        .log-info > summary > .summary_title::before {
          content: '💡 '
        }
        .log-warning > summary > .summary_title::before {
          content: '❗ '
        }
        .log-error > summary > .summary_title::before {
          content: '❌ '
        }
        .log-fatal > summary > .summary_title::before {
          content: '💀 '
        }
        .log-text {
          display: block;
          color: black;
          font-style: italic;
          padding: 20px;
          border-radius: 5px;
          background: rgba(255, 255, 255, 0.5);
          white-space: pre-wrap;
        }
        details.log-entry {
          margin: 0px 0px 10px;
          border: 0px;
        }
        div.log-metadata {
          margin: 10px 0px 0px 0px; 
        }
        .log-metadata > details {
          background-color: rgba(255, 255, 255, 0.5);
          border: 1px solid transparent;
        }
        .log-debug {
          background-color: #EEEEEE
        }
        .log-warning {
          background-color: #F8C471
        }
        .log-info {
          background-color: #A3E4D7
        }
        .log-error {
          background-color: #F5C6CB
        }
        .log-fatal {
          background-color: #F19CBB
        }
        """
    ]


def log(level: LogLevel,
        message: str,
        *,
        indent: int = 0,
        **kwargs) -> LogEntry:
  """Logs a message."""
  entry = LogEntry(
      indent=indent,
      level=level,
      time=datetime.datetime.now(),
      message=message,
      metadata=kwargs,
  )
  if entry.should_output(get_log_level()):
    if console.under_notebook():
      console.display(entry)
    else:
      # TODO(daiyip): Improve the console output formatting.
      console.write(entry)
  return entry


def debug(message: str, *, indent: int = 0, **kwargs) -> LogEntry:
  """Logs a debug message to the session."""
  return log('debug', message, indent=indent, **kwargs)


def info(message: str, *, indent: int = 0, **kwargs) -> LogEntry:
  """Logs an info message to the session."""
  return log('info', message, indent=indent, **kwargs)


def warning(message: str, *, indent: int = 0, **kwargs) -> LogEntry:
  """Logs an info message to the session."""
  return log('warning', message, indent=indent, **kwargs)


def error(message: str, *, indent: int = 0, **kwargs) -> LogEntry:
  """Logs an error message to the session."""
  return log('error', message, indent=indent, **kwargs)


def fatal(message: str, *, indent: int = 0, **kwargs) -> LogEntry:
  """Logs a fatal message to the session."""
  return log('fatal', message, indent=indent, **kwargs)
