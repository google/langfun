# Copyright 2025 The Langfun Authors
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
"""Environment event logger."""

import re
import time
from typing import Annotated
from langfun.env import interface
import pyglove as pg


class EventLogger(pg.Object, interface.EventHandler):
  """Event handler for logging debugger."""

  colored: Annotated[
      bool,
      (
          'If True, log events with colors.'
      )
  ] = False

  regex: Annotated[
      str | list[str] | None,
      (
          'One or a list of regular expressions to filter event messages. '
          'If None, no filtering will be applied.'
      )
  ] = None

  error_only: Annotated[
      bool,
      (
          'If True, log events with errors only.'
      )
  ] = False

  sandbox_status: Annotated[
      bool,
      (
          'If True, log events for sandbox status changes.'
      )
  ] = True

  feature_status: Annotated[
      bool,
      (
          'If True, log events for feature setup/teardown updates.'
      )
  ] = True

  session_status: Annotated[
      bool,
      (
          'If True, log events for session start/end status update.'
      )
  ] = True

  housekeep_status: Annotated[
      bool,
      (
          'If True, log housekeeping events.'
      )
  ] = True

  stats_report_interval: Annotated[
      float | None,
      (
          'The minimum interval in seconds for reporting the environment '
          'stats. If None, stats will not be reported.'
      )
  ] = 300.0

  def _on_bound(self) -> None:
    super()._on_bound()

    regex_exps = self.regex
    if isinstance(regex_exps, str):
      regex_exps = [regex_exps]
    elif regex_exps is None:
      regex_exps = []
    self._regex_exps = [re.compile(x) for x in regex_exps]
    self._last_stats_report_time = None

  def _maybe_colored(
      self, message: str, color: str, styles: list[str] | None = None
  ) -> str:
    if self.colored:
      return pg.colored(message, color, styles=styles)
    return message

  def _format_message(
      self,
      message: str,
      error: BaseException | None,
  ) -> str:
    if error is not None:
      message = (
          f'{message} with error: {pg.utils.ErrorInfo.from_exception(error)}'
      )
    return message

  def _keep(
      self,
      message: str,
      error: BaseException | None,
  ) -> bool:
    if error is None and self.error_only:
      return False
    if self._regex_exps and all(
        not exp.match(message) for exp in self._regex_exps
    ):
      return False
    return True

  def on_environment_starting(
      self,
      environment: interface.Environment,
  ) -> None:
    """Called when the environment is starting."""
    self._print(
        f'[{environment.id}] environment starting',
        error=None,
        color='green',
        styles=['bold'],
    )

  def on_environment_shutting_down(
      self,
      environment: interface.Environment,
      offline_duration: float,
  ) -> None:
    """Called when the environment is shutting down."""
    self._print(
        f'[{environment.id}] environment shutting down '
        f'(offline_duration={offline_duration:.2f} seconds)',
        error=None,
        color='green',
        styles=['bold'],
    )

  def on_environment_start(
      self,
      environment: interface.Environment,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is started."""
    self._print(
        f'[{environment.id}] environment started '
        f'(duration={duration:.2f} seconds)',
        error=error,
        color='green',
        styles=['bold'],
    )

  def on_environment_housekeep(
      self,
      environment: interface.Environment,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when the environment is housekeeping."""
    if self.housekeep_status:
      self._print(
          f'[{environment.id}] environment housekeeping complete '
          f'(counter={counter}, duration={duration:.2f} seconds, '
          f'housekeep_info={kwargs})',
          error=error,
          color='green',
      )
    if (self.stats_report_interval is not None and
        (self._last_stats_report_time is None
         or time.time() - self._last_stats_report_time
         > self.stats_report_interval)):
      self._write_log(
          f'[{environment.id}] environment stats: {environment.stats()}',
          color='magenta',
          error=None,
      )
      self._last_stats_report_time = time.time()

  def on_environment_shutdown(
      self,
      environment: interface.Environment,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is shutdown."""
    self._print(
        f'[{environment.id}] environment shutdown '
        f'(duration={duration:.2f} seconds), lifetime={lifetime:.2f} seconds)',
        error=error,
        color='green',
        styles=['bold'],
    )

  def on_sandbox_start(
      self,
      sandbox: interface.Sandbox,
      duration: float,
      error: BaseException | None
  ) -> None:
    if self.sandbox_status:
      self._print(
          f'[{sandbox.id}] sandbox started '
          f'(duration={duration:.2f} seconds)',
          error=error,
          color='white',
          styles=['bold'],
      )

  def on_sandbox_status_change(
      self,
      sandbox: interface.Sandbox,
      old_status: interface.Sandbox.Status,
      new_status: interface.Sandbox.Status,
      span: float
  ) -> None:
    if self.sandbox_status:
      self._print(
          f'[{sandbox.id}] {old_status.value} '
          f'({span:.2f} seconds) -> {new_status.value}',
          error=None,
          color='white',
      )

  def on_sandbox_shutdown(
      self,
      sandbox: interface.Sandbox,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    if self.sandbox_status:
      self._print(
          f'[{sandbox.id}] sandbox shutdown '
          f'(duration={duration:.2f} seconds), '
          f'lifetime={lifetime:.2f} seconds)',
          error=error,
          color='white',
          styles=['bold'],
      )

  def on_sandbox_session_start(
      self,
      sandbox: interface.Sandbox,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session starts."""
    if self.session_status:
      self._print(
          f'[{sandbox.id}@{session_id}] sandbox session started '
          f'(duration={duration:.2f} seconds)',
          error=error,
          color='blue',
      )

  def on_sandbox_session_end(
      self,
      sandbox: interface.Sandbox,
      session_id: str,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session ends."""
    if self.session_status:
      self._print(
          f'[{sandbox.id}@{session_id}] sandbox session ended '
          f'(duration={duration:.2f} seconds), '
          f'lifetime={lifetime:.2f} seconds)',
          error=error,
          color='blue',
      )

  def on_sandbox_activity(
      self,
      name: str,
      sandbox: interface.Sandbox,
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    log_id = f'{sandbox.id}@{session_id or "<idle>"}'
    color = 'yellow' if session_id is None else 'cyan'
    self._print(
        f'[{log_id}] sandbox call {name!r} '
        f'(duration={duration:.2f} seconds, kwargs={kwargs}) ',
        error,
        color=color
    )

  def on_sandbox_housekeep(
      self,
      sandbox: interface.Sandbox,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox feature is housekeeping."""
    if self.sandbox_status and self.housekeep_status:
      self._print(
          f'[{sandbox.id}] sandbox housekeeping complete '
          f'(counter={counter}, duration={duration:.2f} seconds, '
          f'housekeep_info={kwargs})',
          error=error,
          color='white',
      )

  def on_feature_setup(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    if self.feature_status:
      self._print(
          f'[{feature.id}] feature setup complete '
          f'(duration={duration:.2f} seconds)',
          error=error,
          color='white',
      )

  def on_feature_teardown(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    if self.feature_status:
      self._print(
          f'[{feature.id}] feature teardown complete '
          f'(duration={duration:.2f} seconds)',
          error=error,
          color='white',
      )

  def on_feature_setup_session(
      self,
      feature: interface.Feature,
      session_id: str | None,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    if self.feature_status:
      self._print(
          f'[{feature.id}@{session_id or "<idle>"}] '
          f'feature setup complete (duration={duration:.2f} seconds)',
          error=error,
          color='yellow',
      )

  def on_feature_teardown_session(
      self,
      feature: interface.Feature,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    if self.feature_status:
      self._print(
          f'[{feature.id}@{session_id}] '
          f'feature teardown complete (duration={duration:.2f} seconds)',
          error=error,
          color='yellow',
      )

  def on_feature_activity(
      self,
      name: str,
      feature: interface.Feature,
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a feature activity is performed."""
    log_id = f'{feature.id}@{session_id or "<idle>"}'
    color = 'yellow' if session_id is None else 'cyan'
    self._print(
        f'[{log_id}] feature call {name!r} '
        f'(duration={duration:.2f} seconds, kwargs={kwargs}) ',
        error,
        color=color
    )

  def on_feature_housekeep(
      self,
      feature: interface.Feature,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox feature is housekeeping."""
    if self.feature_status and self.housekeep_status:
      self._print(
          f'[{feature.id}] feature housekeeping complete '
          f'(counter={counter}, (duration={duration:.2f} seconds, '
          f'housekeep_info={kwargs})',
          error=error,
          color='white',
      )

  def _print(
      self,
      message: str,
      error: BaseException | None,
      color: str | None = None,
      styles: list[str] | None = None,
  ):
    message = self._format_message(message, error)
    if not self._keep(message, error):
      return
    self._write_log(message, error, color, styles)

  def _write_log(
      self,
      message: str,
      error: BaseException | None,
      color: str | None = None,
      styles: list[str] | None = None,
  ):
    message = self._maybe_colored(
        message, color if error is None else 'red', styles=styles
    )
    if error is not None:
      pg.logging.error(message)
    else:
      pg.logging.info(message)


class ConsoleEventLogger(EventLogger):
  """Event handler for console debugger."""

  colored = True

  def _write_log(
      self,
      message: str,
      error: BaseException | None,
      color: str | None = None,
      styles: list[str] | None = None
  ):
    print(
        self._maybe_colored(
            message, color if error is None else 'red', styles=styles
        )
    )
