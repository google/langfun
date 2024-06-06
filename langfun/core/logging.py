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

import datetime
import io
import typing
from typing import Any, Literal, ContextManager

from langfun.core import console
import pyglove as pg


LogLevel = Literal['debug', 'info', 'error', 'warning', 'fatal']
_LOG_LEVELS = list(typing.get_args(LogLevel))
_TLS_KEY_MIN_LOG_LEVEL = '_event_log_level'


def use_log_level(log_level: LogLevel | None = 'info') -> ContextManager[None]:
  """Contextmanager to enable logging at a given level."""
  return pg.object_utils.thread_local_value_scope(
      _TLS_KEY_MIN_LOG_LEVEL, log_level, 'info')


def get_log_level() -> LogLevel | None:
  """Gets the current minimum log level."""
  return pg.object_utils.thread_local_get(_TLS_KEY_MIN_LOG_LEVEL, 'info')


class LogEntry(pg.Object):
  """Event log entry."""
  time: datetime.datetime
  level: LogLevel
  message: str
  metadata: dict[str, Any] = pg.Dict()
  indent: int = 0

  def should_output(self, min_log_level: LogLevel) -> bool:
    return _LOG_LEVELS.index(self.level) >= _LOG_LEVELS.index(min_log_level)

  def _repr_html_(self) -> str:
    s = io.StringIO()
    padding_left = 50 * self.indent
    s.write(f'<div style="padding-left: {padding_left}px;">')
    s.write(self._message_display)
    if self.metadata:
      s.write('<div style="padding-left: 20px; margin-top: 10px">')
      s.write('<table style="border-top: 1px solid #EEEEEE;">')
      for k, v in self.metadata.items():
        if hasattr(v, '_repr_html_'):
          cs = v._repr_html_()  # pylint: disable=protected-access
        else:
          cs = f'<span style="white-space: pre-wrap">{str(v)}</span>'
        key_span = self._round_text(k, color='#F1C40F', margin_bottom='0px')
        s.write(
            '<tr>'
            '<td style="padding: 5px; vertical-align: top; '
            f'border-bottom: 1px solid #EEEEEE">{key_span}</td>'
            '<td style="padding: 5px; vertical-align: top; '
            f'border-bottom: 1px solid #EEEEEE">{cs}</td></tr>'
        )
      s.write('</table></div>')
    return s.getvalue()

  @property
  def _message_text_color(self) -> str:
    match self.level:
      case 'debug':
        return '#EEEEEE'
      case 'info':
        return '#A3E4D7'
      case 'error':
        return '#F5C6CB'
      case 'fatal':
        return '#F19CBB'
      case _:
        raise ValueError(f'Unknown log level: {self.level}')

  @property
  def _time_display(self) -> str:
    display_text = self.time.strftime('%H:%M:%S')
    alt_text = self.time.strftime('%Y-%m-%d %H:%M:%S.%f')
    return (
        '<span style="background-color: #BBBBBB; color: white; '
        'border-radius:5px; padding:0px 5px 0px 5px;" '
        f'title="{alt_text}">{display_text}</span>'
    )

  @property
  def _message_display(self) -> str:
    return self._round_text(
        self._time_display + '&nbsp;' + self.message,
        color=self._message_text_color,
    )

  def _round_text(
      self,
      text: str,
      *,
      color: str = '#EEEEEE',
      display: str = 'inline-block',
      margin_top: str = '5px',
      margin_bottom: str = '5px',
      whitespace: str = 'pre-wrap') -> str:
    return (
        f'<span style="background:{color}; display:{display};'
        f'border-radius:10px; padding:5px; '
        f'margin-top: {margin_top}; margin-bottom: {margin_bottom}; '
        f'white-space: {whitespace}">{text}</span>'
    )


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
