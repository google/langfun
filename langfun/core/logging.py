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
import typing
from typing import Any, Iterator, Literal, Sequence

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
      max_str_len_for_summary: int = pg.View.PresetArgValue(80),  # pytype: disable=annotation-type-mismatch
      **kwargs
      ) -> str:
    if len(self.message) > max_str_len_for_summary:
      message = self.message[:max_str_len_for_summary] + '...'
    else:
      message = self.message

    s = pg.Html(
        pg.Html.element(
            'span',
            [self.time.strftime('%H:%M:%S')],
            css_class=['log-time']
        ),
        pg.Html.element(
            'span',
            [pg.Html.escape(message)],
            css_class=['log-summary'],
        ),
    )
    return view.summary(
        self,
        title=title or s,
        max_str_len_for_summary=max_str_len_for_summary,
        **kwargs,
    )

  # pytype: disable=annotation-type-mismatch
  def _html_tree_view_content(
      self,
      view: pg.views.HtmlTreeView,
      root_path: pg.KeyPath,
      collapse_log_metadata_level: int | None = pg.View.PresetArgValue(0),
      max_str_len_for_summary: int = pg.View.PresetArgValue(80),
      collapse_level: int | None = pg.View.PresetArgValue(1),
      **kwargs
  ) -> pg.Html:
    # pytype: enable=annotation-type-mismatch
    def render_message_text():
      if len(self.message) < max_str_len_for_summary:
        return None
      return pg.Html.element(
          'span',
          [pg.Html.escape(self.message)],
          css_class=['log-text'],
      )

    def render_metadata():
      if not self.metadata:
        return None
      child_path = root_path + 'metadata'
      return pg.Html.element(
          'div',
          [
              view.render(
                  self.metadata,
                  name='metadata',
                  root_path=child_path,
                  parent=self,
                  collapse_level=(
                      view.max_collapse_level(
                          collapse_level,
                          collapse_log_metadata_level,
                          child_path
                      )
                  )
              )
          ],
          css_class=['log-metadata'],
      )

    return pg.Html.element(
        'div',
        [
            render_message_text(),
            render_metadata(),
        ],
        css_class=['complex_value'],
    )

  def _html_style(self) -> list[str]:
    return super()._html_style() + [
        """
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

  def _html_element_class(self) -> Sequence[str] | None:
    return super()._html_element_class() + [f'log-{self.level}']


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
