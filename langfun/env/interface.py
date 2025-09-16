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
"""Interfaces for environments, sandboxes and features."""

import abc
import contextlib
import dataclasses
import os
from typing import Annotated, Any, Callable, ContextManager, ClassVar, Iterator, Optional
import uuid

import pyglove as pg

#
# Environemnt identifiers.
#


@dataclasses.dataclass(frozen=True)
class EnvironmentId:
  """Identifier for an environment."""
  environment_id: str

  def __str__(self) -> str:
    return self.environment_id

  def working_dir(self, root_dir: str | None) -> str | None:
    """Returns the download directory for the service."""
    if root_dir is None:
      return None
    return os.path.join(root_dir, _make_path_compatible(self.environment_id))

# Enable automatic conversion from str to EnvironmentId.
pg.typing.register_converter(str, EnvironmentId, EnvironmentId)


@dataclasses.dataclass(frozen=True)
class SandboxId:
  """Identifier for a sandbox."""
  environment_id: EnvironmentId
  sandbox_id: str

  def __str__(self) -> str:
    return f'{self.environment_id}/{self.sandbox_id}'

  def working_dir(self, root_dir: str | None) -> str | None:
    """Returns the download directory for the sandbox."""
    if root_dir is None:
      return None
    return os.path.join(
        self.environment_id.working_dir(root_dir),
        _make_path_compatible(self.sandbox_id)
    )


def _make_path_compatible(id_str: str) -> str:
  """Makes a path compatible with CNS."""
  return id_str.translate(
      str.maketrans({
          '@': '_',
          ':': '_',
          '#': '_',
          ' ': '',
      })
  )


#
# Environment errors.
#


class EnvironmentError(RuntimeError):  # pylint: disable=redefined-builtin
  """Base class for environment errors."""

  def __init__(
      self,
      message: str,
      *args,
      environment: 'Environment',
      **kwargs
  ) -> None:
    self.environment = environment
    super().__init__(f'[{environment.id}] {message}.', *args, **kwargs)


class EnvironmentOutageError(EnvironmentError):
  """Error that indicates environment is offline."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      offline_duration: float,
      **kwargs
  ):
    self.offline_duration = offline_duration
    super().__init__(
        message or f'Environment is offline for {offline_duration} seconds.',
        *args,
        **kwargs
    )


class EnvironmentOverloadError(EnvironmentError):
  """Error that indicates environment is overloaded."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      **kwargs
  ):
    super().__init__(
        message or 'All sandboxes in the pool are either busy or dead.',
        *args, **kwargs
    )


class SandboxError(RuntimeError):
  """Base class for sandbox errors."""

  def __init__(
      self,
      message: str,
      *args,
      sandbox: 'Sandbox',
      **kwargs
  ) -> None:
    self.sandbox = sandbox
    super().__init__(f'[{sandbox.id}] {message}.', *args, **kwargs)


class SandboxStateError(SandboxError):
  """Error that indicates sandbox is in an unexpected state.

  This error is raised when the sandbox is in an unexpected state and cannot
  be recovered. As a result, the sandbox will be shutdown and user session
  will be terminated.
  """

  def __init__(
      self,
      message: str | None = None,
      *args,
      code: str | None = None,
      **kwargs
  ):
    default_message = 'Sandbox is in an unexpected state'
    if code is not None:
      default_message = (
          f'Sandbox is in an unexpected state after executing code: {code!r}'
      )
    super().__init__(message or default_message, *args, **kwargs)


#
# Event handler.
#


class EnvironmentEventHandler:
  """Base class for environment event handlers."""

  def on_environment_start(
      self,
      environment: 'Environment',
      error: Exception | None
  ) -> None:
    """Called when the environment is started."""

  def on_environment_shutdown(
      self,
      environment: 'Environment',
      error: Exception | None
  ) -> None:
    """Called when the environment is shutdown."""

  def on_sandbox_start(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      error: Exception | None
  ) -> None:
    """Called when a sandbox is started."""

  def on_sandbox_shutdown(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      error: Exception | None
  ) -> None:
    """Called when a sandbox is shutdown."""

  def on_sandbox_feature_setup(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is setup."""

  def on_sandbox_feature_teardown(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is teardown."""

  def on_sandbox_feature_housekeep(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is housekeeping."""

  def on_sandbox_session_start(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      session_id: str,
      error: Exception | None
  ) -> None:
    """Called when a sandbox session starts."""

  def on_sandbox_session_activity(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      session_id: str,
      feature: Optional['Feature'],
      error: Exception | None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""

  def on_sandbox_session_end(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      session_id: str,
      error: Exception | None
  ) -> None:
    """Called when a sandbox session ends."""

  def on_sandbox_ping(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      error: Exception | None
  ) -> None:
    """Called when a sandbox is pinged."""


#
# Interface for sandbox-based environment.
#


class Environment(pg.Object):
  """Base class for an environment."""

  features: Annotated[
      dict[str, 'Feature'],
      'Features to be exposed by the environment.'
  ] = {}

  event_handler: Annotated[
      EnvironmentEventHandler | None,
      (
          'User handler for the environment events.'
      )
  ] = None

  _ENV_STACK: Annotated[
      ClassVar[list['Environment']],
      'Recording the environments stacked through context managers.'
  ] = []

  #
  # Subclasses must implement:
  #

  @property
  @abc.abstractmethod
  def id(self) -> EnvironmentId:
    """Returns the identifier for the environment."""

  @abc.abstractmethod
  def stats(self) -> dict[str, Any]:
    """Returns the stats of the environment."""

  @property
  @abc.abstractmethod
  def is_alive(self) -> bool:
    """Returns True if the environment is alive."""

  @abc.abstractmethod
  def start(self) -> None:
    """Starts the environment.

    Raises:
      EnvironmentError: If the environment is not available.
    """

  @abc.abstractmethod
  def shutdown(self) -> None:
    """Shuts down the environment.

    Raises:
      EnvironmentError: If the environment is not available.
    """

  @abc.abstractmethod
  def acquire(self) -> 'Sandbox':
    """Acquires a free sandbox from the environment.

    Returns:
      A free sandbox from the environment.

    Raises:
      EnvironmentOutageError: If the environment is out of service.
      EnvironmentOverloadError: If the environment is overloaded.
    """

  #
  # Environment lifecycle.
  #

  def __enter__(self) -> 'Environment':
    """Enters the environment and sets it as the current environment."""
    self.start()
    Environment._ENV_STACK.append(self)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the environment and reset the current environment."""
    assert Environment._ENV_STACK
    Environment._ENV_STACK.pop()
    self.shutdown()

  @classmethod
  def current(cls) -> Optional['Environment']:
    """Returns the current environment."""
    if not Environment._ENV_STACK:
      return None
    return Environment._ENV_STACK[-1]

  #
  # Environment operations.
  #

  def sandbox(
      self,
      session_id: str | None = None,
  ) -> ContextManager['Sandbox']:
    """Gets a sandbox from the environment and starts a new user session."""
    return self.acquire().new_session(session_id)

  def __getattr__(self, name: str) -> Any:
    """Gets a feature from a free sandbox from the environment.

    Example:
      ```
      with XboxEnvironment(
          features={'selenium': SeleniumFeature()}
      ) as env:
        driver = env.selenium.get_driver()
      ```

    Args:
      name: The name of the feature.

    Returns:
      A feature from a free sandbox from the environment.
    """
    if name in self.features:
      return self.acquire().features[name]
    raise AttributeError(name)

  #
  # Event handlers subclasses can override.
  #

  def on_start(self, error: Exception | None = None) -> None:
    """Called when the environment is started."""
    if self.event_handler:
      self.event_handler.on_environment_start(self, error)

  def on_shutdown(self, error: Exception | None = None) -> None:
    """Called when the environment is shutdown."""
    if self.event_handler:
      self.event_handler.on_environment_shutdown(self, error)

  def on_sandbox_start(
      self,
      sandbox: 'Sandbox',
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox is started."""
    if self.event_handler:
      self.event_handler.on_sandbox_start(self, sandbox, error)

  def on_sandbox_shutdown(
      self,
      sandbox: 'Sandbox',
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox is shutdown."""
    if self.event_handler:
      self.event_handler.on_sandbox_shutdown(self, sandbox, error)

  def on_sandbox_feature_setup(
      self,
      sandbox: 'Sandbox',
      feature: 'Feature',
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox feature is setup."""
    if self.event_handler:
      self.event_handler.on_sandbox_feature_setup(self, sandbox, feature, error)

  def on_sandbox_feature_teardown(
      self,
      sandbox: 'Sandbox',
      feature: 'Feature',
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    if self.event_handler:
      self.event_handler.on_sandbox_feature_teardown(
          self, sandbox, feature, error
      )

  def on_sandbox_feature_housekeep(
      self,
      sandbox: 'Sandbox',
      feature: 'Feature',
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox feature is housekeeping."""
    if self.event_handler:
      self.event_handler.on_sandbox_feature_housekeep(
          self, sandbox, feature, error
      )

  def on_sandbox_session_start(
      self,
      sandbox: 'Sandbox',
      session_id: str,
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox session starts."""
    if self.event_handler:
      self.event_handler.on_sandbox_session_start(
          self, sandbox, session_id, error
      )

  def on_sandbox_session_activity(
      self,
      sandbox: 'Sandbox',
      session_id: str,
      feature: Optional['Feature'] = None,
      error: Exception | None = None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    if self.event_handler:
      self.event_handler.on_sandbox_session_activity(
          self, sandbox, feature, session_id, error, **kwargs
      )

  def on_sandbox_session_end(
      self,
      sandbox: 'Sandbox',
      session_id: str,
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox session ends."""
    if self.event_handler:
      self.event_handler.on_sandbox_session_end(
          self, sandbox, session_id, error
      )

  def on_sandbox_ping(
      self,
      sandbox: 'Sandbox',
      error: Exception | None = None
  ) -> None:
    """Called when a sandbox is pinged."""
    if self.event_handler:
      self.event_handler.on_sandbox_ping(self, sandbox, error)


class Sandbox(pg.Object):
  """Interface for sandboxes."""

  @property
  @abc.abstractmethod
  def id(self) -> SandboxId:
    """Returns the identifier for the sandbox."""

  @property
  @abc.abstractmethod
  def environment(self) -> Environment:
    """Returns the environment for the sandbox."""

  @property
  @abc.abstractmethod
  def features(self) -> dict[str, 'Feature']:
    """Returns the features in the sandbox."""

  @property
  @abc.abstractmethod
  def is_alive(self) -> bool:
    """Returns True if the sandbox is alive."""

  @property
  @abc.abstractmethod
  def is_busy(self) -> bool:
    """Returns whether the sandbox is busy."""

  @abc.abstractmethod
  def set_pending(self) -> None:
    """Marks the sandbox pending after acquisition but before ready for use."""

  @property
  @abc.abstractmethod
  def is_pending(self) -> bool:
    """Returns whether the sandbox is pending."""

  @abc.abstractmethod
  def start(self) -> None:
    """Starts the sandbox.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
    """

  @abc.abstractmethod
  def shutdown(self) -> None:
    """Shuts down the sandbox.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
    """

  @abc.abstractmethod
  def ping(self) -> None:
    """Ping the sandbox to check if it is alive.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
    """

  @abc.abstractmethod
  def start_session(self, session_id: str) -> None:
    """Begins a user session with the sandbox.

    Args:
      session_id: The identifier for the user session.

    Raises:
      SandboxError: If the sandbox already has a user session
        or the session cannot be started.
    """

  @abc.abstractmethod
  def end_session(self) -> None:
    """Ends the user session with the sandbox."""

  @property
  @abc.abstractmethod
  def session_id(self) -> str | None:
    """Returns the current user session identifier."""

  #
  # API related to a user session.
  # A sandbox could be reused across different user sessions.
  # A user session is a sequence of stateful interactions with the sandbox,
  # Across different sessions the sandbox are considered stateless.
  #

  @contextlib.contextmanager
  def new_session(self, session_id: str | None = None) -> Iterator['Sandbox']:
    """Context manager for obtaining a sandbox for a user session."""
    if session_id is None:
      session_id = f'session-{uuid.uuid4().hex[:7]}'
    self.start_session(session_id)
    try:
      yield self
    finally:
      self.end_session()

  def __getattr__(self, name: str) -> Any:
    """Gets a feature from current sandbox.

    Example:
      ```
      with MyEnvironment(
          features={'feature1': Feature1()}
      ) as env:
        with env.sandbox('session1') as sb:
          driver = sb.feature1.feature_method()
      ```

    Args:
      name: The name of the feature.

    Returns:
      A feature from current sandbox.
    """
    if name in self.features:
      return self.features[name]
    raise AttributeError(name)

  #
  # Event handlers subclasses can override.
  #

  def on_start(self, error: Exception | None = None) -> None:
    """Called when the sandbox is started."""
    self.environment.on_sandbox_start(self, error)

  def on_shutdown(self, error: Exception | None = None) -> None:
    """Called when the sandbox is shutdown."""
    self.environment.on_sandbox_shutdown(self, error)

  def on_ping(self, error: Exception | None = None) -> None:
    """Called when the sandbox is pinged."""
    self.environment.on_sandbox_ping(self, error)

  def on_feature_setup(
      self,
      feature: 'Feature',
      error: Exception | None = None
  ) -> None:
    """Called when a feature is setup."""
    self.environment.on_sandbox_feature_setup(
        self, feature, error
    )

  def on_feature_teardown(
      self,
      feature: 'Feature',
      error: Exception | None = None
  ) -> None:
    """Called when a feature is teardown."""
    self.environment.on_sandbox_feature_teardown(
        self, feature, error
    )

  def on_feature_housekeep(
      self,
      feature: 'Feature',
      error: Exception | None = None
  ) -> None:
    """Called when a feature is housekeeping."""
    self.environment.on_sandbox_feature_housekeep(
        self, feature, error
    )

  def on_session_start(
      self,
      session_id: str,
      error: Exception | None = None
  ) -> None:
    """Called when the user session starts."""
    self.environment.on_sandbox_session_start(self, session_id, error)

  def on_session_activity(
      self,
      session_id: str,
      feature: Optional['Feature'] = None,
      error: Exception | None = None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    self.environment.on_sandbox_session_activity(
        sandbox=self,
        feature=feature,
        session_id=session_id,
        error=error,
        **kwargs
    )

  def on_session_end(
      self,
      session_id: str,
      error: Exception | None = None
  ) -> None:
    """Called when the user session ends."""
    self.environment.on_sandbox_session_end(self, session_id, error)


class Feature(pg.Object):
  """Interface for sandbox features."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Name of the feature, which will be used as key to access the feature."""

  @property
  @abc.abstractmethod
  def sandbox(self) -> Sandbox | None:
    """Returns the sandbox that the feature is running in."""

  @abc.abstractmethod
  def setup(self, sandbox: Sandbox) -> None:
    """Sets up the feature, which is called once when the sandbox is up.

    Args:
      sandbox: The sandbox that the feature is running in.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
    """

  @abc.abstractmethod
  def teardown(self) -> None:
    """Teardowns the feature, which is called once when the sandbox is down.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
    """

  @abc.abstractmethod
  def setup_session(self, session_id: str) -> None:
    """Sets up the feature for a user session."""

  @abc.abstractmethod
  def teardown_session(self, session_id: str) -> None:
    """Teardowns the feature for a user session."""

  @abc.abstractmethod
  def housekeep(self) -> None:
    """Performs housekeeping for the feature.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
    """

  @property
  @abc.abstractmethod
  def housekeep_interval(self) -> int | None:
    """Returns the interval in seconds for feature housekeeping.

    Returns:
      The interval in seconds for feature housekeeping. If None, the feature
      will not be housekeeping.
    """

  @property
  def session_id(self) -> str | None:
    """Returns the current user session identifier."""
    assert self.sandbox is not None
    return self.sandbox.session_id

  #
  # Event handlers subclasses can override.
  #

  def on_setup(
      self,
      error: BaseException | None = None
  ) -> None:
    """Called when the feature is setup."""
    self.sandbox.on_feature_setup(self, error)

  def on_teardown(
      self,
      error: BaseException | None = None
  ) -> None:
    """Called when the feature is teardown."""
    self.sandbox.on_feature_teardown(self, error)

  def on_housekeep(
      self,
      error: BaseException | None = None
  ) -> None:
    """Called when the feature has done housekeeping."""
    self.sandbox.on_feature_housekeep(self, error)

  def on_session_setup(
      self,
      session_id: str,
      error: BaseException | None = None,
  ) -> None:
    """Called when the feature is setup for a user session."""

  def on_session_activity(
      self,
      session_id: str,
      error: Exception | None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    self.sandbox.on_session_activity(
        feature=self, session_id=session_id, error=error, **kwargs
    )

  def on_session_teardown(
      self,
      session_id: str,
      error: BaseException | None = None,
  ) -> None:
    """Called when the feature is teardown for a user session."""


def call_with_event(
    action: Callable[[], None],
    event_handler: Callable[..., None],
    action_kwargs: dict[str, Any] | None = None,
    event_handler_kwargs: dict[str, Any] | None = None,
) -> None:
  """Triggers an event handler."""
  action_kwargs = action_kwargs or {}
  event_handler_kwargs = event_handler_kwargs or {}
  error = None
  try:
    action(**action_kwargs)
  except BaseException as e:  # pylint: disable=broad-except
    error = e
    raise
  finally:
    event_handler(error=error, **event_handler_kwargs)
