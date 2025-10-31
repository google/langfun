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
"""Environment event handler chain."""

from typing import Sequence
from langfun.env import interface


class EventHandlerChain(interface.EventHandler):
  """Environment event handler chain."""

  def __init__(self, handlers: Sequence[interface.EventHandler]):
    super().__init__()
    self._handlers = list(handlers)

  def add(self, handler: interface.EventHandler):
    self._handlers.append(handler)

  def remove(self, handler: interface.EventHandler):
    self._handlers.remove(handler)

  def on_environment_starting(
      self,
      environment: interface.Environment,
  ) -> None:
    """Called when the environment is starting."""
    for handler in self._handlers:
      handler.on_environment_starting(environment)

  def on_environment_shutting_down(
      self,
      environment: interface.Environment,
      offline_duration: float,
  ) -> None:
    """Called when the environment is shutting down."""
    for handler in self._handlers:
      handler.on_environment_shutting_down(environment, offline_duration)

  def on_environment_start(
      self,
      environment: interface.Environment,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is started."""
    for handler in self._handlers:
      handler.on_environment_start(environment, duration, error)

  def on_environment_housekeep(
      self,
      environment: interface.Environment,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when the environment is housekeeping."""
    for handler in self._handlers:
      handler.on_environment_housekeep(
          environment, counter, duration, error, **kwargs
      )

  def on_environment_shutdown(
      self,
      environment: interface.Environment,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is shutdown."""
    for handler in self._handlers:
      handler.on_environment_shutdown(environment, duration, lifetime, error)

  def on_sandbox_start(
      self,
      sandbox: interface.Sandbox,
      duration: float,
      error: BaseException | None
  ) -> None:
    for handler in self._handlers:
      handler.on_sandbox_start(sandbox, duration, error)

  def on_sandbox_status_change(
      self,
      sandbox: interface.Sandbox,
      old_status: interface.Sandbox.Status,
      new_status: interface.Sandbox.Status,
      span: float
  ) -> None:
    for handler in self._handlers:
      handler.on_sandbox_status_change(sandbox, old_status, new_status, span)

  def on_sandbox_shutdown(
      self,
      sandbox: interface.Sandbox,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    for handler in self._handlers:
      handler.on_sandbox_shutdown(sandbox, duration, lifetime, error)

  def on_sandbox_session_start(
      self,
      sandbox: interface.Sandbox,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session starts."""
    for handler in self._handlers:
      handler.on_sandbox_session_start(sandbox, session_id, duration, error)

  def on_sandbox_session_end(
      self,
      sandbox: interface.Sandbox,
      session_id: str,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session ends."""
    for handler in self._handlers:
      handler.on_sandbox_session_end(
          sandbox, session_id, duration, lifetime, error
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
    for handler in self._handlers:
      handler.on_sandbox_activity(
          name, sandbox, session_id, duration, error, **kwargs
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
    for handler in self._handlers:
      handler.on_sandbox_housekeep(sandbox, counter, duration, error, **kwargs)

  def on_feature_setup(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    for handler in self._handlers:
      handler.on_feature_setup(feature, duration, error)

  def on_feature_teardown(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    for handler in self._handlers:
      handler.on_feature_teardown(feature, duration, error)

  def on_feature_setup_session(
      self,
      feature: interface.Feature,
      session_id: str | None,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    for handler in self._handlers:
      handler.on_feature_setup_session(feature, session_id, duration, error)

  def on_feature_teardown_session(
      self,
      feature: interface.Feature,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    for handler in self._handlers:
      handler.on_feature_teardown_session(feature, session_id, duration, error)

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
    for handler in self._handlers:
      handler.on_feature_activity(
          name, feature, session_id, duration, error, **kwargs
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
    for handler in self._handlers:
      handler.on_feature_housekeep(feature, counter, duration, error, **kwargs)
