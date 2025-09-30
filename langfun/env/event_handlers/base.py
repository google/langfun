
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
"""Base classes for Langfun environment event handlers."""

from langfun.env import interface

Environment = interface.Environment
Sandbox = interface.Sandbox
Feature = interface.Feature


class _SessionEventHandler:
  """Base class for session event handlers."""

  def on_session_start(
      self,
      environment: Environment,
      sandbox: Sandbox,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session starts.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      session_id: The session ID.
      duration: The time spent on starting the session.
      error: The error that caused the session to start. If None, the session
        started normally.
    """

  def on_session_end(
      self,
      environment: Environment,
      sandbox: Sandbox,
      session_id: str,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session ends.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      session_id: The session ID.
      duration: The time spent on ending the session.
      lifetime: The session lifetime in seconds.
      error: The error that caused the session to end. If None, the session
        ended normally.
    """


class _FeatureEventHandler:
  """Base class for feature event handlers."""

  def on_feature_setup(
      self,
      environment: Environment,
      sandbox: Sandbox,
      feature: Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      feature: The feature.
      duration: The feature setup duration in seconds.
      error: The error happened during the feature setup. If None,
        the feature setup performed normally.
    """

  def on_feature_teardown(
      self,
      environment: Environment,
      sandbox: Sandbox,
      feature: Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      feature: The feature.
      duration: The feature teardown duration in seconds.
      error: The error happened during the feature teardown. If None,
        the feature teardown performed normally.
    """

  def on_feature_teardown_session(
      self,
      environment: Environment,
      sandbox: Sandbox,
      feature: Feature,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a feature is teardown with a session.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      feature: The feature.
      session_id: The session ID.
      duration: The feature teardown session duration in seconds.
      error: The error happened during the feature teardown session. If
        None, the feature teardown session performed normally.
    """

  def on_feature_setup_session(
      self,
      environment: Environment,
      sandbox: Sandbox,
      feature: Feature,
      session_id: str | None,
      duration: float,
      error: BaseException | None,
  ) -> None:
    """Called when a feature is setup with a session.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      feature: The feature.
      session_id: The session ID.
      duration: The feature setup session duration in seconds.
      error: The error happened during the feature setup session. If
        None, the feature setup session performed normally.
    """

  def on_feature_housekeep(
      self,
      environment: Environment,
      sandbox: Sandbox,
      feature: Feature,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs,
  ) -> None:
    """Called when a sandbox feature is housekeeping.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      feature: The feature.
      counter: Zero-based counter of the housekeeping round.
      duration: The feature housekeeping duration in seconds.
      error: The error happened during the feature housekeeping. If None, the
        feature housekeeping normally.
      **kwargs: Feature-specific properties computed during housekeeping.
    """


class _SandboxEventHandler(_FeatureEventHandler, _SessionEventHandler):
  """Base class for sandbox event handlers."""

  def on_sandbox_start(
      self,
      environment: Environment,
      sandbox: Sandbox,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox is started.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      duration: The time spent on starting the sandbox.
      error: The error that caused the sandbox to start. If None, the sandbox
        started normally.
    """

  def on_sandbox_status_change(
      self,
      environment: Environment,
      sandbox: Sandbox,
      old_status: 'Sandbox.Status',
      new_status: 'Sandbox.Status',
      span: float,
  ) -> None:
    """Called when a sandbox status changes.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      old_status: The old sandbox status.
      new_status: The new sandbox status.
      span: Time spent on the old status in seconds.
    """

  def on_sandbox_shutdown(
      self,
      environment: Environment,
      sandbox: Sandbox,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox is shutdown.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      duration: The time spent on shutting down the sandbox.
      lifetime: The sandbox lifetime in seconds.
      error: The error that caused the sandbox to shutdown. If None, the
        sandbox shutdown normally.
    """

  def on_sandbox_activity(
      self,
      name: str,
      environment: Environment,
      sandbox: Sandbox,
      feature: Feature | None,
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed.

    Args:
      name: The name of the sandbox activity.
      environment: The environment.
      sandbox: The sandbox.
      feature: The feature that is associated with the sandbox activity.
      session_id: The session ID.
      duration: The sandbox activity duration in seconds.
      error: The error that caused the sandbox activity to perform. If None,
        the sandbox activity performed normally.
      **kwargs: The keyword arguments of the sandbox activity.
    """

  def on_sandbox_housekeep(
      self,
      environment: Environment,
      sandbox: Sandbox,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox finishes a round of housekeeping.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      counter: Zero-based counter of the housekeeping round.
      duration: The sandbox housekeeping duration in seconds.
      error: The error that caused the sandbox to housekeeping. If None, the
        sandbox housekeeping normally.
      **kwargs: Sandbox-specific properties computed during housekeeping.
    """


class EventHandler(_SandboxEventHandler):
  """Base class for event handlers of an environment."""

  def on_environment_starting(self, environment: Environment) -> None:
    """Called when the environment is getting started.

    Args:
      environment: The environment.
    """

  def on_environment_start(
      self,
      environment: Environment,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is started.

    Args:
      environment: The environment.
      duration: The environment start duration in seconds.
      error: The error that failed the environment start. If None, the
        environment started normally.
    """

  def on_environment_housekeep(
      self,
      environment: Environment,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when the environment finishes a round of housekeeping.

    Args:
      environment: The environment.
      counter: Zero-based counter of the housekeeping round.
      duration: The environment start duration in seconds.
      error: The error that failed the housekeeping. If None, the
        housekeeping succeeded.
      **kwargs: Environment-specific properties computed during housekeeping.
    """

  def on_environment_shutting_down(
      self,
      environment: Environment,
      offline_duration: float,
  ) -> None:
    """Called when the environment is shutting down.

    Args:
      environment: The environment.
      offline_duration: The environment offline duration in seconds.
    """

  def on_environment_shutdown(
      self,
      environment: Environment,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is shutdown.

    Args:
      environment: The environment.
      duration: The environment shutdown duration in seconds.
      lifetime: The environment lifetime in seconds.
      error: The error that caused the environment to shutdown. If None, the
        environment shutdown normally.
    """
