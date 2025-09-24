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
"""Common base class for sandboxes.

This module provides an base class `BaseSandbox` for sandboxes, which takes care
of the sandbox lifecycle and housekeeping. It also provides a decorator for
sandbox service methods to handle errors and trigger shutdowns. Please note that
this base class is intended to provide a convenient way to implement sandboxes,
and not all sandbox implementations need to subclass it. Also `BaseSandbox`
is not coupled with `BaseEnvironment` and `BaseFeature`, and is expected to
work with the `Environment` and `Feature` interfaces directly.
"""

import contextlib
import functools
import threading
import time
from typing import Annotated, Any, Callable, Iterator, Sequence, Type

from langfun.env import interface
import pyglove as pg


class BaseSandbox(interface.Sandbox):
  """Base class for a sandbox."""

  id: Annotated[
      interface.SandboxId,
      'The identifier for the sandbox.'
  ]

  environment: Annotated[
      pg.Ref[interface.Environment],
      'The parent environment.'
  ]

  reusable: Annotated[
      bool,
      (
          'If True, the sandbox can be reused for multiple user sessions at '
          'different times.'
      )
  ] = False

  keepalive_interval: Annotated[
      float | None,
      'Interval to ping the sandbox for keeping it alive..'
  ] = 60.0

  proactive_session_setup: Annotated[
      bool,
      (
          'If True, the sandbox will perform setup work before a user session '
          'is started. This is useful for sandboxes that need to perform '
          'heavy setup work, which could block the user thread for a long '
          'time. Applicable only when `reusable` is True.'
      )
  ] = True

  #
  # There is no required methods that subclasses must implement.
  # Subclasses can override the following methods:
  #

  def _start(self) -> None:
    """Implementation of start(). Subclasses can override.

    Raises:
      interface.SandboxStateError: If the sandbox is in a bad state.
    """

  def _shutdown(self) -> None:
    """Implementation of shutdown(). Subclasses can override.

    Raises:
      interface.SandboxStateError: If the sandbox is in a bad state.
    """

  def _set_status(self, status: interface.Sandbox.Status) -> None:
    """Sets the status of the sandbox."""
    assert self._status != status, (self._status, status)
    self.on_status_change(self._status, status)
    self._status = status
    self._status_start_time = time.time()

  def report_maybe_state_error(self, e: BaseException | None) -> None:
    """Reports sandbox state errors."""
    if (isinstance(e, interface.SandboxStateError)
        and e not in self._state_errors):
      self._state_errors.append(e)

  def _setup_features(self) -> None:
    """Starts the features in the sandbox."""
    # We keep track of the features that have setup called so we can teardown
    # the features when the sandbox is shutdown.
    self._features_with_setup_called.clear()

    for feature in self._features.values():
      self._features_with_setup_called.add(feature.name)
      feature.setup(self)

  def _setup_session(self) -> None:
    """Sets up a new session for the sandbox."""
    # We keep track of the features that have setup_session called so we can
    # teardown the session for them when the session ends.
    self._features_with_setup_session_called.clear()

    for feature in self._features.values():
      self._features_with_setup_session_called.add(feature.name)
      feature.setup_session()

  def _teardown_features(self) -> interface.FeatureTeardownError | None:
    """Tears down the features in the sandbox.

    IMPORTANT: This method shall not raise any exceptions.

    Returns:
      FeatureTeardownError: If feature teardown failed with errors.
        Otherwise None.
    """
    errors = {}
    for feature in self._features.values():
      if feature.name in self._features_with_setup_called:
        try:
          feature.teardown()
        except BaseException as e:  # pylint: disable=broad-except
          self.report_maybe_state_error(e)
          errors[feature.name] = e
    if errors:
      return interface.FeatureTeardownError(sandbox=self, errors=errors)
    return None

  def _start_session(self) -> None:
    """Starts a user session.

    Raises:
      BaseException: If feature setup failed with user-defined errors.
      SandboxStateError: If sandbox or feature setup fail due sandbox state
        errors.
    """
    # When pre-session setup is enabled, the session setup is performed
    # before the session is started. Otherwise we setup the session here.
    if not self._enable_pre_session_setup:
      self._setup_session()

  def _end_session(self) -> interface.SessionTeardownError | None:
    """Ends a user session.

    IMPORTANT: This method shall not raise any exceptions.

    Returns:
      SessionTeardownError: If session teardown failed with errors.
        Otherwise None.
    """
    feature_teardown_errors = {}
    for name, feature in self._features.items():
      if name in self._features_with_setup_session_called:
        try:
          feature.teardown_session()
        except BaseException as e:  # pylint: disable=broad-except
          self.report_maybe_state_error(e)
          feature_teardown_errors[name] = e

    return interface.SessionTeardownError(
        sandbox=self, errors=feature_teardown_errors
    ) if feature_teardown_errors else None

  def _ping(self) -> None:
    """Implementation of ping for health checking."""

  #
  # Sandbox basics.
  #

  def _on_bound(self) -> None:
    """Called when the sandbox is bound."""
    super()._on_bound()

    self._features = pg.Dict({
        name: pg.clone(feature)
        for name, feature in self.environment.features.items()
    })
    self._event_handlers = []
    self._enable_pre_session_setup = (
        self.reusable and self.proactive_session_setup
    )
    self._enables_housekeep = (
        self.keepalive_interval is not None
        or any(
            feature.housekeep_interval is not None
            for feature in self._features.values()
        )
    )
    self._housekeep_thread = None
    self._housekeep_counter = 0

    # Runtime state.
    self._status = self.Status.CREATED
    self._status_start_time = time.time()

    self._start_time = None
    self._state_errors = []

    self._features_with_setup_called = set()
    self._features_with_setup_session_called = set()

    self._session_id = None
    self._session_start_time = None

    # Thread local state for this sandbox.
    self._tls_state = threading.local()

  @functools.cached_property
  def working_dir(self) -> str | None:
    """Returns the working directory for the sandbox."""
    return self.id.working_dir(self.environment.root_dir)

  @property
  def status(self) -> interface.Sandbox.Status:
    """Returns the state of the sandbox."""
    return self._status

  def set_acquired(self) -> None:
    """Marks the sandbox as acquired."""
    self._set_status(self.Status.ACQUIRED)

  @property
  def housekeep_counter(self) -> int:
    """Returns the housekeeping counter."""
    return self._housekeep_counter

  def add_event_handler(
      self,
      event_handler: interface.EnvironmentEventHandler | None
  ) -> None:
    """Sets the event handler for the sandbox."""
    self._event_handlers.append(event_handler)

  def remove_event_handler(
      self,
      event_handler: interface.EnvironmentEventHandler | None
  ) -> None:
    """Removes the event handler for the sandbox."""
    self._event_handlers.remove(event_handler)

  @property
  def state_errors(self) -> list[interface.SandboxStateError]:
    """Returns all errors encountered during sandbox lifecycle."""
    return self._state_errors

  @property
  def is_shutting_down(self) -> bool:
    """Returns True if the sandbox is shutting down."""
    return self._status == self.Status.SHUTTING_DOWN or (
        self._state_errors and self._status == self.Status.EXITING_SESSION
    )

  @property
  def features(self) -> dict[str, interface.Feature]:
    """Returns the features in the sandbox."""
    return self._features

  def _enter_service_call(self) -> bool:
    """Enters a service call.

    Returns:
      True if the service call is at the top of the call stack.
    """
    v = getattr(self._tls_state, 'service_call_depth', None)
    if v is None:
      v = 0
    setattr(self._tls_state, 'service_call_depth', v + 1)
    return v == 0

  def _exit_service_call(self) -> bool:
    """Exits a service call.

    Returns:
      True if the service call is at the top of the call stack.
    """
    v = getattr(self._tls_state, 'service_call_depth')
    setattr(self._tls_state, 'service_call_depth', v - 1)
    return v == 1

  #
  # Sandbox start/shutdown.
  #

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
    will be added to `errors`. The sandbox is considered dead and will not be
    further used.

    Raises:
      SandboxStateError: If the sandbox is in a bad state.
      BaseException: If feature setup failed with user-defined errors.
    """
    assert self._status == self.Status.CREATED, (
        f'Sandbox {self.id} cannot be started because '
        f'it is in {self._status} status.'
    )

    starting_time = time.time()
    self._state = self.Status.SETTING_UP

    try:
      # Start the sandbox.
      self._start()

      # Setup the features.
      self._setup_features()

      # Setup the first session if pre-session setup is enabled.
      if self._enable_pre_session_setup:
        self._setup_session()

      if self._enables_housekeep:
        self._housekeep_thread = threading.Thread(
            target=self._housekeep_loop, daemon=True
        )
        self._housekeep_thread.start()

      self._start_time = time.time()

      # Mark the sandbox as ready when all setup succeeds.
      self._set_status(self.Status.READY)

      duration = time.time() - starting_time
      self.on_start(duration)
      pg.logging.info(
          '[%s]: Sandbox started in %.2f seconds.',
          self.id, duration
      )
    except BaseException as e:  # pylint: disable=broad-except
      duration = time.time() - starting_time
      pg.logging.error(
          '[%s]: Sandbox failed to start in %.2f seconds: %s',
          self.id, duration, e
      )
      self.report_maybe_state_error(e)
      self.on_start(duration, e)
      self.shutdown()
      raise e

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

    # Allow re-entry.
    if self._status in (
        interface.Sandbox.Status.SHUTTING_DOWN,
        interface.Sandbox.Status.OFFLINE
    ):
      return

    # End current session and shutdown the sandbox if the sandbox is in session.
    if self._status == self.Status.IN_SESSION:
      self.end_session(shutdown_sandbox=True)
      return

    self._set_status(interface.Sandbox.Status.SHUTTING_DOWN)
    shutdown_start_time = time.time()

    if (self._housekeep_thread is not None
        and threading.current_thread() is not self._housekeep_thread):
      self._housekeep_thread.join()
      self._housekeep_thread = None

    teardown_error = self._teardown_features()
    try:
      self._shutdown()
      self._set_status(interface.Sandbox.Status.OFFLINE)

      pg.logging.info(
          '[%s]: Sandbox shutdown in %.2f seconds. '
          '(lifetime: %.2f seconds, teardown errors: %s)',
          self.id,
          time.time() - shutdown_start_time,
          time.time() - self._start_time if self._start_time else 0,
          teardown_error
      )
      self.on_shutdown(teardown_error)
      shutdown_error = None
    except BaseException as e:  # pylint: disable=broad-except
      shutdown_error = e
      self.report_maybe_state_error(e)
      self._set_status(interface.Sandbox.Status.OFFLINE)
      pg.logging.error(
          '[%s]: Sandbox shutdown with error: %s',
          self.id, e
      )
      self.on_shutdown(teardown_error or shutdown_error)

    # We raise non-state errors to the user following timely order, so the user
    # code could be surfaced and handled properly.
    if (teardown_error is not None
        and teardown_error.has_non_sandbox_state_error):
      raise teardown_error

    if shutdown_error is not None and not isinstance(
        shutdown_error, interface.SandboxStateError
    ):
      raise shutdown_error

  def ping(self) -> None:
    """Pings the sandbox to check if it is alive."""
    self._ping()

  #
  # API related to a user session.
  # A sandbox could be reused across different user sessions.
  # A user session is a sequence of stateful interactions with the sandbox,
  # Across different sessions the sandbox are considered stateless.
  #

  @property
  def session_id(self) -> str | None:
    """Returns the current user session identifier.

    session_id is set to None when the sandbox is free, and set to 'pending'
    when the sandbox is acquired by a thread, before a user session is started.

    Returns:
      The current user session identifier or None if the sandbox is not busy.
    """
    return self._session_id

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
    assert self._status == self.Status.ACQUIRED, (
        f'Sandbox {self.id} is not in acquired state (status={self._status}).'
    )
    assert self._session_id is None, (
        f'A user session {self._session_id} is already active '
        f'for sandbox {self.id}.'
    )
    self._set_status(self.Status.SETTING_UP)

    self._session_id = session_id
    self._session_start_time = time.time()

    try:
      self._start_session()
      self._set_status(self.Status.IN_SESSION)
      self.on_session_start(session_id, time.time() - self._session_start_time)
    except BaseException as e:  # pylint: disable=broad-except
      self.report_maybe_state_error(e)
      self.on_session_start(
          session_id, time.time() - self._session_start_time, e
      )
      self.shutdown()
      raise e

  def end_session(self, shutdown_sandbox: bool = False) -> None:
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

    Args:
      shutdown_sandbox: If True, the sandbox will be shutdown after session
        teardown.

    Raises:
      BaseException: If session teardown failed with user-defined errors.
    """
    if self._status == self.Status.EXITING_SESSION:
      return

    if self._status not in (
        self.Status.IN_SESSION,
    ):
      return

    assert self._session_id is not None, (
        'No user session is active for this sandbox'
    )
    # Set sandbox status to EXITING_SESSION to avoid re-entry.
    self._set_status(self.Status.EXITING_SESSION)
    shutdown_sandbox = shutdown_sandbox or not self.reusable

    # Teardown features for the current session.
    end_session_error = self._end_session()
    previous_session_id = self._session_id
    self._session_id = None
    self._features_with_setup_session_called.clear()

    # If there is no state error, and proactive session setup is enabled,
    # set up the next session proactively.
    if not self.state_errors:
      if not shutdown_sandbox and self._enable_pre_session_setup:
        def _setup_next_session():
          try:
            self._setup_session()
            self._set_status(interface.Sandbox.Status.READY)
          except BaseException as e:  # pylint: disable=broad-except
            pg.logging.error(
                '[%s]: Shutting down sandbox due to practively setting up '
                'next session failed: %s',
                self.id,
                e
            )
            self.report_maybe_state_error(e)
            self.shutdown()

        # End session before setting up the next session.
        self.on_session_end(previous_session_id)

        # Mark the sandbox as setting up to prevent it from being acquired by
        # other threads.
        self._set_status(interface.Sandbox.Status.SETTING_UP)

        # TODO(daiyip): Consider using a thread pool to perform next session
        # setup.
        threading.Thread(target=_setup_next_session).start()
      else:
        # End session before reporting sandbox status change.
        self.on_session_end(previous_session_id)

        # If shutdown is requested, mark the sandbox as acquired to prevent it
        # from being acquired by other threads.
        self._set_status(
            interface.Sandbox.Status.ACQUIRED if shutdown_sandbox else
            interface.Sandbox.Status.READY
        )

    # Otherwise, shutdown the sandbox.
    else:
      self.on_session_end(previous_session_id, self.state_errors[0])
      self._set_status(interface.Sandbox.Status.ACQUIRED)
      shutdown_sandbox = True

    pg.logging.info(
        '[%s]: User session %s ended. '
        '(lifetime: %.2f seconds, teardown errors: %s).',
        self.id,
        self._session_id,
        time.time() - self._session_start_time,
        end_session_error
    )
    self._session_start_time = None
    self._session_event_handler = None

    if shutdown_sandbox:
      self.shutdown()

    # We only raise errors if teardown error contains non-sandbox-state error,
    # meaning that the user code may have bug or other non-environment
    # failures.
    if (end_session_error is not None
        and end_session_error.has_non_sandbox_state_error):
      raise end_session_error  # pylint: disable=raising-bad-type

  #
  # Housekeeping.
  #

  def _housekeep_loop(self) -> None:
    """Sandbox housekeeping loop."""
    now = time.time()
    last_ping = now
    last_housekeep_time = {name: now for name in self._features.keys()}

    while self._status not in (self.Status.SHUTTING_DOWN, self.Status.OFFLINE):
      housekeep_start = time.time()
      if self.keepalive_interval is not None:
        if time.time() - last_ping > self.keepalive_interval:
          try:
            self.ping()
          except interface.SandboxStateError as e:
            pg.logging.error(
                '[%s]: Shutting down sandbox because ping failed '
                'with error: %s.',
                self.id,
                str(e)
            )
            self._housekeep_counter += 1
            self.report_maybe_state_error(e)
            self.on_housekeep(time.time() - housekeep_start, e)
            self.shutdown()
            break
          last_ping = time.time()

      for name, feature in self._features.items():
        if feature.housekeep_interval is not None and (
            time.time() - last_housekeep_time[name]
            > feature.housekeep_interval
        ):
          try:
            feature.housekeep()
            last_housekeep_time[name] = time.time()
          except interface.SandboxStateError as e:
            pg.logging.error(
                '[%s/%s]: Feature housekeeping failed with error: %s. '
                'Shutting down sandbox...',
                self.id,
                feature.name,
                e,
            )
            self.report_maybe_state_error(e)
            self._housekeep_counter += 1
            self.on_housekeep(time.time() - housekeep_start, e)
            self.shutdown()
            break

      self._housekeep_counter += 1
      self.on_housekeep(time.time() - housekeep_start)
      time.sleep(1)

  #
  # Event handlers subclasses can override.
  #

  def on_start(
      self,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when the sandbox is started."""
    for handler in self._event_handlers:
      handler.on_sandbox_start(self.environment, self, duration, error)

  def on_status_change(
      self,
      old_status: interface.Sandbox.Status,
      new_status: interface.Sandbox.Status,
  ) -> None:
    """Called when the sandbox status changes."""
    for handler in self._event_handlers:
      handler.on_sandbox_status_change(
          self.environment,
          self,
          old_status,
          new_status,
          time.time() - self._status_start_time
      )

  def on_shutdown(self, error: BaseException | None = None) -> None:
    """Called when the sandbox is shutdown."""
    if self._start_time is None:
      lifetime = 0.0
    else:
      lifetime = time.time() - self._start_time
    for handler in self._event_handlers:
      handler.on_sandbox_shutdown(self.environment, self, lifetime, error)

  def on_housekeep(
      self,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when the sandbox finishes a round of housekeeping."""
    counter = self._housekeep_counter
    for handler in self._event_handlers:
      handler.on_sandbox_housekeep(
          self.environment, self, counter, duration, error
      )

  def on_feature_setup(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when a feature is setup."""
    for handler in self._event_handlers:
      handler.on_feature_setup(
          self.environment, self, feature, duration, error
      )

  def on_feature_teardown(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when a feature is teardown."""
    for handler in self._event_handlers:
      handler.on_feature_teardown(
          self.environment, self, feature, duration, error
      )

  def on_feature_setup_session(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when a feature is setup for a user session."""
    for handler in self._event_handlers:
      handler.on_feature_setup_session(
          self.environment, self, feature, self.session_id, duration, error
      )

  def on_feature_teardown_session(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when a feature is teardown for a user session."""
    for handler in self._event_handlers:
      handler.on_feature_teardown_session(
          self.environment, self, feature, self.session_id, duration, error
      )

  def on_feature_housekeep(
      self,
      feature: interface.Feature,
      counter: int,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when a feature is housekeeping."""
    for handler in self._event_handlers:
      handler.on_feature_housekeep(
          self.environment, self, feature, counter, duration, error
      )

  def on_session_start(
      self,
      session_id: str,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when the user session starts."""
    for handler in self._event_handlers:
      handler.on_session_start(
          self.environment, self, session_id, duration, error
      )

  def on_activity(
      self,
      name: str,
      duration: float,
      feature: interface.Feature | None = None,
      error: BaseException | None = None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    for handler in self._event_handlers:
      handler.on_sandbox_activity(
          name=name,
          environment=self.environment,
          sandbox=self,
          feature=feature,
          session_id=self.session_id,
          duration=duration,
          error=error,
          **kwargs
      )

  def on_session_end(
      self,
      session_id: str,
      error: BaseException | None = None
  ) -> None:
    """Called when the user session ends."""
    lifetime = time.time() - self._session_start_time
    for handler in self._event_handlers:
      handler.on_session_end(
          self.environment, self, session_id, lifetime, error
      )


#
# Sandbox service decorator.
#


def sandbox_service(
    critical_errors: Sequence[
        Type[BaseException] | tuple[Type[BaseException], str]
    ] | None = None
) -> Callable[..., Any]:
  """Decorator for Sandbox/Feature methods exposed as sandbox services.

  This decorator will catch errors and map to `SandboxStateError` if the
  error matches any of the critical errors. Consequently, the sandbox will be
  shutdown automatically when the error is raised.

  Example:

  ```
    with env:
      with env.sandbox() as sb:
        try:
          sb.test_feature.do_something_with_non_state_error()
        except ValueError:
          # sandbox will not be shutdown.
          pass

        try:
          sb.test_feature.do_something_with_state_error()
        except ValueError:
          assert sb.state == sb.Status.OFFLINE
  ```

  If the decorated method returns a context manager, a wrapper context manager
  will be returned, which will end the session when exiting the context.

  Example:

  ```
    with env:
      with env.test_feature.do_something_with_context_manager() as result:
        # sandbox will be alive during the whole context manager cycle.
  ```

  For sandbox service methods, an optional `session_id` argument can be passed
  to create a new session for the service call, even its signature does not
  contain a `session_id` argument.

  Args:
    critical_errors: A sequence of exception types or tuples of exception type
      and error messages (described in regular expression), when matched, treat
      the sandbox as in a bad state, which will trigger a shutdown.

  Returns:
    The decorator function.
  """
  critical_errors = critical_errors or []

  def decorator(func):
    signature = pg.typing.get_signature(func)
    if 'session_id' in signature.arg_names:
      raise ValueError(
          '`session_id` should not be used as argument for sandbox '
          'service method. Please use `self.session_id` instead.'
      )

    def to_kwargs(*args, **kwargs):
      num_non_self_args = len(signature.arg_names) - 1
      if len(args) > num_non_self_args:
        assert signature.varargs is not None, (signature, args)
        kwargs[signature.varargs.name] = tuple(args[num_non_self_args:])
        args = args[:num_non_self_args]
      for i in range(len(args)):
        # The first argument is `self`.
        kwargs[signature.arg_names[i + 1]] = args[i]
      return kwargs

    @functools.wraps(func)
    def method_wrapper(self, *args, **kwargs) -> Any:
      """Helper function to safely execute logics in the sandbox."""

      assert isinstance(self, (BaseSandbox, interface.Feature)), self
      sandbox = self.sandbox if isinstance(self, interface.Feature) else self

      # We count the service call stack depth so we could shutdown the sandbox
      # at the top upon sandbox state error.
      sandbox._enter_service_call()  # pylint: disable=protected-access

      # When a capability is directly accessed from the environment,
      # we create a new session for the capability call. This
      # prevents the sandbox from being reused for other feature calls.
      if sandbox.status == interface.Sandbox.Status.ACQUIRED:
        new_session = True
        new_session_id = kwargs.get('session_id')
        if new_session_id is None:
          new_session_id = sandbox.environment.new_session_id()

        # If it's a feature method called from the environment, start a new
        # session for the feature call.
        sandbox.start_session(new_session_id)
      else:
        new_session = False

      kwargs.pop('session_id', None)
      result = None
      error = None
      start_time = time.time()

      try:
        # Execute the service function.
        result = func(self, *args, **kwargs)

        # If the result is a context manager, wrap it with a context manager
        # to end the session when exiting.
        if isinstance(result, contextlib.AbstractContextManager):
          return _service_context_manager_wrapper(
              service=result,
              sandbox_or_feature=self,
              sandbox=sandbox,
              name=func.__name__,
              kwargs=to_kwargs(*args, **kwargs),
              start_time=start_time,
              new_session=new_session
          )

        # Otherwise, return the result and end the session in the finally block.
        return result
      except BaseException as e:
        error = e
        sandbox.report_maybe_state_error(e)
        if pg.match_error(e, critical_errors):
          state_error = interface.SandboxStateError(
              'Sandbox encountered an unexpected error executing '
              f'`{func.__name__}` (args={args!r}, kwargs={kwargs!r}): {e}',
              sandbox=self
          )
          sandbox.report_maybe_state_error(state_error)
          raise state_error from e
        raise
      finally:
        is_topmost_call = sandbox._exit_service_call()  # pylint: disable=protected-access
        if not isinstance(result, contextlib.AbstractContextManager):
          self.on_activity(
              name=func.__name__,
              duration=time.time() - start_time,
              error=error,
              **to_kwargs(*args, **kwargs),
          )
          if new_session:
            assert is_topmost_call

            # End the session if it's from a feature method and the result
            # is not a context manager.
            sandbox.end_session()

        # Shutdown the sandbox if it is at the top of the service call stack and
        # has state errors.
        if (is_topmost_call
            and sandbox.state_errors
            # Sandbox service method might be called during shutting down, in
            # that case we don't want to shutdown the sandbox again.
            and not sandbox.is_shutting_down):
          sandbox.shutdown()

    return method_wrapper
  return decorator


@contextlib.contextmanager
def _service_context_manager_wrapper(
    service: contextlib.AbstractContextManager[Any],
    sandbox_or_feature: BaseSandbox | interface.Feature,
    sandbox: interface.Sandbox,
    name: str,
    kwargs: dict[str, Any],
    new_session: bool,
    start_time: float,
) -> Iterator[Any]:
  """Context manager wrapper for ending a sandbox session when exiting."""
  error = None
  sandbox._enter_service_call()  # pylint: disable=protected-access

  try:
    with service as result:
      yield result
  except BaseException as e:
    error = e
    sandbox.report_maybe_state_error(error)
    raise
  finally:
    sandbox_or_feature.on_activity(
        name=name,
        error=error,
        duration=time.time() - start_time,
        **kwargs,
    )
    is_topmost_call = sandbox._exit_service_call()  # pylint: disable=protected-access

    if new_session:
      assert is_topmost_call
      sandbox.end_session()
    elif isinstance(error, interface.SandboxStateError):
      sandbox.shutdown()
