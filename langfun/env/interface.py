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
import enum
import os
from typing import Annotated, Any, ContextManager, ClassVar, Iterator, Optional

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


class FeatureTeardownError(SandboxError):
  """Base class for feature errors."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      errors: dict[str, BaseException],
      **kwargs
  ):
    self.errors = errors
    super().__init__(
        (message or
         f'Feature teardown failed with user-defined errors: {errors}.'),
        *args,
        **kwargs
    )

  @property
  def has_non_sandbox_state_error(self) -> bool:
    """Returns True if the feature teardown error has non-sandbox state error."""
    return any(
        not isinstance(e, SandboxStateError) for e in self.errors.values()
    )


class SessionTeardownError(SandboxError):
  """Base class for session errors."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      errors: dict[str, BaseException],
      **kwargs
  ):
    self.errors = errors
    super().__init__(
        (message or
         f'Session teardown failed with user-defined errors: {errors}.'),
        *args,
        **kwargs
    )

  @property
  def has_non_sandbox_state_error(self) -> bool:
    """Returns True if the feature teardown error has non-sandbox state error."""
    return any(
        not isinstance(e, SandboxStateError) for e in self.errors.values()
    )


#
# Event handler.
#


class SessionEventHandler:
  """Base class for session event handlers."""

  def on_session_start(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
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
      environment: 'Environment',
      sandbox: 'Sandbox',
      session_id: str,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session ends.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      session_id: The session ID.
      lifetime: The session lifetime in seconds.
      error: The error that caused the session to end. If None, the session
        ended normally.
    """


class FeatureEventHandler:
  """Base class for feature event handlers."""

  def on_feature_setup(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""

  def on_feature_teardown(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""

  def on_feature_teardown_session(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a feature is teardown with a session."""

  def on_feature_setup_session(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      session_id: str | None,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a feature is setup with a session."""

  def on_feature_housekeep(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      counter: int,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is housekeeping."""


class SandboxEventHandler(FeatureEventHandler, SessionEventHandler):
  """Base class for sandbox event handlers."""

  def on_sandbox_start(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
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
      environment: 'Environment',
      sandbox: 'Sandbox',
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
      environment: 'Environment',
      sandbox: 'Sandbox',
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox is shutdown.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      lifetime: The sandbox lifetime in seconds.
      error: The error that caused the sandbox to shutdown. If None, the
        sandbox shutdown normally.
    """

  def on_sandbox_activity(
      self,
      name: str,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: Optional['Feature'],
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
      environment: 'Environment',
      sandbox: 'Sandbox',
      counter: int,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox finishes a round of housekeeping.

    Args:
      environment: The environment.
      sandbox: The sandbox.
      counter: Zero-based counter of the housekeeping round.
      duration: The sandbox housekeeping duration in seconds.
      error: The error that caused the sandbox to housekeeping. If None, the
        sandbox housekeeping normally.
    """


class EnvironmentEventHandler(SandboxEventHandler):
  """Base class for environment event handlers."""

  def on_environment_start(
      self,
      environment: 'Environment',
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
      environment: 'Environment',
      counter: int,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment finishes a round of housekeeping.

    Args:
      environment: The environment.
      counter: Zero-based counter of the housekeeping round.
      duration: The environment start duration in seconds.
      error: The error that failed the housekeeping. If None, the
        housekeeping succeeded.
    """

  def on_environment_shutdown(
      self,
      environment: 'Environment',
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is shutdown.

    Args:
      environment: The environment.
      lifetime: The environment lifetime in seconds.
      error: The error that caused the environment to shutdown. If None, the
        environment shutdown normally.
    """


#
# Interface for sandbox-based environment.
#


class Environment(pg.Object):
  """Base class for an environment."""

  class Status(enum.Enum):
    """Environment state.

    State transitions:

    +---------------+               +-----------+
    |  <CREATED>    | --(start)---> |  ONLINE   | -(shutdown or outage detected)
    +---------------+      |        +-----------+              |
                           |        +-----------+              |
                           +------> | <OFFLINE> | <------------+
                                    +-----------+
    """
    CREATED = 'created'
    ONLINE = 'online'
    SHUTTING_DOWN = 'shutting_down'
    OFFLINE = 'offline'

  features: Annotated[
      dict[str, 'Feature'],
      'Features to be exposed by the environment.'
  ] = {}

  event_handlers: Annotated[
      list[EnvironmentEventHandler],
      (
          'User handler for the environment events.'
      )
  ] = []

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

  @property
  @abc.abstractmethod
  def status(self) -> Status:
    """Returns the status of the environment."""

  @abc.abstractmethod
  def stats(self) -> dict[str, Any]:
    """Returns the stats of the environment."""

  @abc.abstractmethod
  def start(self) -> None:
    """Starts the environment.

    Raises:
      EnvironmentError: If the environment is not available.
    """

  @abc.abstractmethod
  def shutdown(self) -> None:
    """Shuts down the environment.

    IMPORTANT: This method shall not raise any exceptions.
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

  @abc.abstractmethod
  def new_session_id(self) -> str:
    """Generates a new session ID."""

  #
  # Environment lifecycle.
  #

  @property
  def is_online(self) -> bool:
    """Returns True if the environment is alive."""
    return self.status == self.Status.ONLINE

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


class Sandbox(pg.Object):
  """Interface for sandboxes."""

  class Status(enum.Enum):
    """Sandbox state.

    State transitions:

            +---------------+          +---------------+
            |  <OFFLINE>    | <------  | SHUTTING_DOWN |
            +---------------+          +---------------+
                                           ^    ^
                                           |    |
                                 (shutdown)|    +------------------------+
                                           |                             |
    +-----------+    (call start) +------------+                         |
    | <CREATED> | ------------->  | SETTING_UP | <----------------+      |
    +-----------+                 +------------+                  |      |
                                       |                          |      |
                                       | (start succeeded)        |      |
                                       | OR (_setup_session)      |      |
                                       v                          |      |
                                   +---------+                    |      |
                                   |  READY  |                    |      |
                                   +---------+                    |      |
                                       |                          |      |
                                       | (set_acquired)           |      |
                                       v                          |      |
                                  +----------+                    |      |
                                  | ACQUIRED |                    |      |
                                  +----------+                    |      |
                                       |                          |      |
                                       | (call start_session)     |      |
                                 +------------+                   |      |
                                 | SETTING_UP |--(failed, call shutdown)-+
                                 +------------+                   |      |
                                       |                          |      |
                                       v  (succeeded)             |      |
                                 +--------------+                 |      |
                                 |  IN_SESSION  |                 |      |
                                 +--------------+                 |      |
                                         |                        |      |
                                  (call end_session)              |      |
                                         |                        |      |
                                         v                        |      |
                                +-----------------+               |      |
                                | EXITING_SESSION |--(succeeded)--+      |
                                +-----------------+                      |
                                         |                               |
                                         +------(failed, call shutdown)--+
    """

    # The sandbox is created, but not yet started.
    CREATED = 'created'

    # The sandbox is being setting up to serve user sessions.
    SETTING_UP = 'setting_up'

    # The sandbox is set up and free to be acquired by the user.
    READY = 'ready'

    # The sandbox is acquired by a thread, but not yet in a user session.
    ACQUIRED = 'acquired'

    # The sandbox is in a user session.
    IN_SESSION = 'in_session'

    # The sandbox is exiting a user session.
    EXITING_SESSION = 'exiting_session'

    # The sandbox is being shut down.
    SHUTTING_DOWN = 'shutting_down'

    # The sandbox is offline.
    OFFLINE = 'offline'

    @property
    def is_online(self) -> bool:
      """Returns True if the sandbox is online."""
      return self in (
          Sandbox.Status.SETTING_UP,
          Sandbox.Status.READY,
          Sandbox.Status.ACQUIRED,
          Sandbox.Status.IN_SESSION,
      )

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
  def status(self) -> Status:
    """Returns the status of the sandbox."""

  @property
  def is_online(self) -> bool:
    """Returns True if the sandbox is online."""
    return self.status.is_online

  @abc.abstractmethod
  def set_acquired(self) -> None:
    """Marks the sandbox as acquired."""

  @abc.abstractmethod
  def add_event_handler(
      self,
      event_handler: EnvironmentEventHandler
  ) -> None:
    """Sets the status of the sandbox."""

  @abc.abstractmethod
  def remove_event_handler(
      self,
      event_handler: EnvironmentEventHandler
  ) -> None:
    """Removes the status of the sandbox."""

  @property
  @abc.abstractmethod
  def state_errors(self) -> list[SandboxStateError]:
    """Returns state errors encountered during sandbox's lifecycle."""

  @abc.abstractmethod
  def start(self) -> None:
    """Starts the sandbox.

    State transitions:
      CREATED -> SETTING_UP -> READY: When all sandbox and feature setup
        succeeds.
      CREATED -> SETTING_UP -> SHUTTING_DOWN -> OFFLINE: When sandbox or feature
        setup fails.

    `start` and `shutdown` should be called in pairs, even when the sandbox
    fails to start. This ensures proper cleanup.

    Start may fail with two sources of errors:

    1. SandboxStateError: If sandbox or feature setup fail due to enviroment
       outage or sandbox state errors.
    2. BaseException: If feature setup failed with user-defined errors, this
       could happen when there is bug in the user code or non-environment code
       failure.

    In both cases, the sandbox will be shutdown automatically, and the error
    will be add to `errors`. The sandbox is considered dead and will not be
    further used.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
      BaseException: If feature setup failed with user-defined errors.
    """

  @abc.abstractmethod
  def shutdown(self) -> None:
    """Shuts down the sandbox.

    State transitions:
      SHUTTING_DOWN -> SHUTTING_DOWN: No operation.
      OFFLINE -> OFFLINE: No operation.
      SETTING_UP -> SHUTTING_DOWN -> OFFLINE: When sandbox and feature
        setup fails.
      IN_SESSION -> SHUTTING_DOWN -> OFFLINE: When user session exits while
        sandbox is set not to reuse, or session teardown fails.
      FREE -> SHUTTING_DOWN -> OFFLINE: When sandbox is shutdown when the
        environment is shutting down, or housekeeping loop shuts down the
        sandbox due to housekeeping failures.


    Please be aware that `shutdown` will be called whenever an operation on the
    sandbox encounters a critical error. This means, `shutdown` should not make
    the assumption that the sandbox is in a healthy state, even `start` could
    fail. As a result, `shutdown` must allow re-entry and be thread-safe with
    other sandbox operations.

    Shutdown may fail with two sources of errors:

    1. SandboxStateError: If the sandbox is in a bad state, and feature teardown
       logic depending on a healthy sandbox may fail. In such case, we do not
       raise error to the user as the user session is considered completed. The
       sandbox is abandoned and new user sessions will be served on other
       sandboxes.

    2. BaseException: The sandbox is in good state, but user code raises error
       due to bug or non-environment code failure. In such case, errors will be
       raised to the user so the error could be surfaced and handled properly.
       The sandbox is treated as shutdown and will not be further used.

    Raises:
      BaseException: If feature teardown failed with user-defined errors.
    """

  @abc.abstractmethod
  def start_session(
      self,
      session_id: str,
  ) -> None:
    """Begins a user session with the sandbox.

    State transitions:
      ACQUIRED -> SETTING_UP -> IN_SESSION: When session setup succeeds.
      ACQUIRED -> SETTING_UP -> SHUTTING_DOWN -> OFFLINE: When session setup
        fails.

    A session is a sequence of stateful interactions with the sandbox.
    Across different sessions the sandbox are considered stateless.
    `start_session` and `end_session` should always be called in pairs, even
    when the session fails to start. `Sandbox.new_session` context manager is
    the recommended way to use `start_session` and `end_session` in pairs.

    Starting a session may fail with two sources of errors:

    1. SandboxStateError: If the sandbox is in a bad state or session setup
       failed.

    2. BaseException: If session setup failed with user-defined errors.

    In both cases, the sandbox will be shutdown automatically and the
    session will be considered ended. The error will be added to `errors`.
    Future session will be served on other sandboxes.

    Args:
      session_id: The identifier for the user session.

    Raises:
      SandboxStateError: If the sandbox is already in a bad state or session
        setup failed.
      BaseException: If session setup failed with user-defined errors.
    """

  @abc.abstractmethod
  def end_session(self) -> None:
    """Ends the user session with the sandbox.

    State transitions:
      IN_SESSION -> EXITING_SESSION -> READY: When user session exits normally,
        and sandbox is set to reuse.
      IN_SESSION -> EXITING_SESSION -> SHUTTING_DOWN -> OFFLINE: When user
        session exits while
        sandbox is set not to reuse, or session teardown fails.
      IN_SESSION -> EXITING_SESSION -> SETTING_UP -> READY: When user session
        exits normally, and sandbox is set to reuse, and proactive session setup
        is enabled.
      IN_SESSION -> EXITING_SESSION -> SETTING_UP -> SHUTTING_DOWN -> OFFLINE:
        When user session exits normally, and proactive session setup is enabled
        but fails.
      EXITING_SESSION -> EXITING_SESSION: No operation.
      not IN_SESSION -> same state: No operation

    `end_session` should always be called for each `start_session` call, even
    when the session fails to start, to ensure proper cleanup.

    `end_session` may fail with two sources of errors:

    1. SandboxStateError: If the sandbox is in a bad state or session teardown
       failed.

    2. BaseException: If session teardown failed with user-defined errors.

    In both cases, the sandbox will be shutdown automatically and the
    session will be considered ended. The error will be added to `errors`.
    Future session will be served on other sandboxes.

    However, SandboxStateError encountered during `end_session` will NOT be
    raised to the user as the user session is considered completed.

    Raises:
      BaseException: If session teardown failed with user-defined errors.
    """

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
  def new_session(
      self,
      session_id: str | None = None,
  ) -> Iterator['Sandbox']:
    """Context manager for obtaining a sandbox for a user session.

    State transitions:
      ACQUIRED -> IN_SESSION -> READY: When session setup and teardown succeed.
      ACQUIRED -> IN_SESSINO -> OFFLINE: When session setup or teardown fails.

    Args:
      session_id: The identifier for the user session. If not provided, a random
        ID will be generated.

    Yields:
      The sandbox for the user session.

    Raises:
      SandboxStateError: If a session cannot be started on the sandbox.
      BaseException: If session setup or teardown failed with user-defined
        errors.
    """
    if session_id is None:
      session_id = self.environment.new_session_id()
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


class Feature(pg.Object):
  """Interface for sandbox features."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Name of the feature, which will be used as key to access the feature."""

  @property
  @abc.abstractmethod
  def sandbox(self) -> Sandbox:
    """Returns the sandbox that the feature is running in.

    Returns:
      The sandbox that the feature is running in.

    Raises:
      AssertError: If the feature is not set up with a sandbox yet.
    """

  @abc.abstractmethod
  def setup(self, sandbox: Sandbox) -> None:
    """Sets up the feature, which is called once when the sandbox is up.

    State transitions:
      SETTING_UP -> READY: When setup succeeds.
      SETTING_UP -> SHUTTING_DOWN -> OFFLINE: When setup fails.

    `setup` is called when a sandbox is started for the first time. When a
    feature's `setup` is called, its `teardown` is guaranteed to be called.

    Args:
      sandbox: The sandbox that the feature is running in.

    Raises:
      SandboxStateError: If setup failed due to sandbox state errors.
      BaseException: If setup failed with user-defined errors.
    """

  @abc.abstractmethod
  def teardown(self) -> None:
    """Teardowns the feature, which is called once when the sandbox is down.

    State transitions:
      SHUTTING_DOWN -> OFFLINE: When teardown succeeds or fails.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
    """

  @abc.abstractmethod
  def setup_session(self) -> None:
    """Sets up the feature for the upcoming user session.

    State transitions:
      SETTING_UP -> READY: When session setup succeeds.
      SETTING_UP -> SHUTTING_DOWN -> OFFLINE: When sessino setup fails.

    `setup_session` is called when a new user session starts for per-session
    setup. `setup_session` and `teardown_session` will be called in pairs, even
    `setup_session` fails.

    `setup_session` may fail with two sources of errors:

    1. SandboxStateError: If the sandbox is in a bad state or session setup
       failed.
    2. BaseException: If session setup failed with user-defined errors.

    In both cases, the error will be raised to the user and the session will be
    ended. The sandbox will shutdown automatically and will not be further used.

    Raises:
      SandboxStateError: If the sandbox is in a bad state or session setup
        failed.
      BaseException: If session setup failed with user-defined errors.
    """

  @abc.abstractmethod
  def teardown_session(self) -> None:
    """Teardowns the feature for an ending user session.

    State transitions:
      SHUTTING_DOWN -> OFFLINE: When session teardown succeeds or fails.

    `teardown_session` is called when a user session ends for per-
    session teardown. `teardown_session` will always be called upon a feature
    whose `setup_session` is called.

    `teardown_session` may fail with two sources of errors:

    1. SandboxStateError: If the sandbox is in a bad state or session teardown
       failed.
    2. BaseException: If session teardown failed with user-defined errors.

    In both cases, the session will be closed and the sandbox will be shutdown.
    However, SandboxStateError encountered during `teardown_session` will NOT be
    raised to the user as the user session is considered completed. Other errors
    will be raised to the user for proper error handling.

    Raises:
      BaseException: If session teardown failed with user-defined errors.
    """

  @abc.abstractmethod
  def housekeep(self) -> None:
    """Performs housekeeping for the feature.

    State transitions:
      (no state change): When housekeeping succeeds.
      original state -> SHUTTING_DOWN -> OFFLINE: When housekeeping fails.

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
