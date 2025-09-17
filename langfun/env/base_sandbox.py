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
import uuid

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

  def _setup_features(self) -> None:
    """Starts the features in the sandbox."""
    for feature in self._features.values():
      feature.setup(self)

  def _teardown_features(self) -> None:
    """Tears down the features in the sandbox."""
    for feature in self._features.values():
      feature.teardown()

  def _start_session(self, session_id: str) -> None:
    """Starts a user session."""
    self._session_id = session_id
    self._session_start_time = time.time()

    for feature in self._features.values():
      feature.setup_session(session_id)

  def _end_session(self) -> None:
    try:
      for feature in self._features.values():
        feature.teardown_session(self._session_id)
    finally:
      pg.logging.info(
          '[%s]: User session %s ended. (lifetime: %.2f seconds).',
          self.id,
          self._session_id,
          time.time() - self._session_start_time
      )
      self._session_id = None
      self._session_start_time = None

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
    self._session_id = None
    self._session_start_time = None
    self._alive = False
    self._start_time = None

    self._needs_housekeep = (
        self.keepalive_interval is not None
        or any(
            feature.housekeep_interval is not None
            for feature in self._features.values()
        )
    )
    self._housekeep_thread = None
    self._housekeep_count = 0

  @functools.cached_property
  def working_dir(self) -> str | None:
    """Returns the working directory for the sandbox."""
    return self.id.working_dir(self.environment.root_dir)

  @property
  def is_alive(self) -> bool:
    """Returns whether the sandbox is alive."""
    return self._alive

  @property
  def is_busy(self) -> bool:
    """Returns whether the sandbox is busy."""
    return self._session_id not in (None, 'pending')

  @property
  def features(self) -> dict[str, interface.Feature]:
    """Returns the features in the sandbox."""
    return self._features

  #
  # Sandbox start/shutdown.
  #

  def start(self) -> None:
    """Starts the sandbox.

    Raises:
      interface.SandboxStateError: If the sandbox fails to start.
    """
    assert not self._alive, 'Sandbox is already alive.'

    def start_impl():
      t = time.time()
      self._start()
      self._setup_features()

      # We mark the sandbox as alive after the setup before the maintenance
      # thread is started. This is to avoid the maintenance thread from
      # immediately shutting down the sandbox because it's not alive yet.
      self._alive = True
      self._start_time = time.time()

      if self._needs_housekeep:
        self._housekeep_thread = threading.Thread(
            target=self._housekeep_loop, daemon=True
        )
        self._housekeep_thread.start()

      pg.logging.info(
          '[%s]: Sandbox started in %.2f seconds.',
          self.id, time.time() - t
      )

    interface.call_with_event(
        action=start_impl,
        event_handler=self.on_start,
    )

  def shutdown(self) -> None:
    """Shuts down the sandbox.

    Raises:
      interface.SandboxStateError: If the sandbox is in a bad state.
    """
    if not self._alive:
      return

    self._alive = False
    shutdown_start_time = time.time()
    def shutdown_impl():
      self._teardown_features()
      self._shutdown()
      if (self._housekeep_thread is not None
          and threading.current_thread() is not self._housekeep_thread):
        self._housekeep_thread.join()
        self._housekeep_thread = None
      pg.logging.info(
          '[%s]: Sandbox shutdown in %.2f seconds. (lifetime: %.2f seconds)',
          self.id,
          time.time() - shutdown_start_time,
          time.time() - self._start_time if self._start_time else 0
      )

    interface.call_with_event(
        action=shutdown_impl,
        event_handler=self.on_shutdown,
    )

  def ping(self) -> None:
    """Pings the sandbox to check if it is alive."""
    interface.call_with_event(
        action=self._ping,
        event_handler=self.on_ping,
    )

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

  def set_pending(self) -> None:
    """Marks the sandbox as pending for new session."""
    self._session_id = 'pending'

  @property
  def is_pending(self) -> bool:
    """Returns whether the sandbox is pending for new session."""
    return self._session_id == 'pending'

  def start_session(self, session_id: str) -> None:
    """Begins a user session with the sandbox.

    Args:
      session_id: The identifier for the user session.

    Raises:
      interface.SandboxError: If the sandbox already has a user session
        or the session cannot be started.
    """
    assert self._session_id in (None, 'pending'), (
        'A user session is already active for this sandbox.'
    )
    interface.call_with_event(
        action=self._start_session,
        event_handler=self.on_session_start,
        action_kwargs={'session_id': session_id},
        event_handler_kwargs={'session_id': session_id},
    )

  def end_session(self) -> None:
    """Ends the user session with the sandbox."""
    assert self._session_id not in (None, 'pending'), (
        'No user session is active for this sandbox'
    )
    try:
      interface.call_with_event(
          action=self._end_session,
          event_handler=self.on_session_end,
          event_handler_kwargs={'session_id': self._session_id},
      )
    finally:
      if not self.reusable:
        self.shutdown()

  #
  # Housekeeping.
  #

  def _housekeep_loop(self) -> None:
    """Sandbox housekeeping loop."""
    now = time.time()
    last_ping = now
    last_housekeep_time = {name: now for name in self._features.keys()}

    while self._alive:
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
            self._housekeep_count += 1
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
                str(e)
            )
            self.shutdown()
            break
      self._housekeep_count += 1
      time.sleep(1)


def sandbox_service(
    critical_errors: Sequence[
        Type[BaseException] | tuple[Type[BaseException], str]
    ] | None = None
) -> Callable[..., Any]:
  """Decorator for Sandbox/Feature methods exposed as sandbox services.

  This decorator will catch errors and map to `SandboxStateError` if the
  error matches any of the critical errors. Consequently, the sandbox will be
  shutdown automatically when the error is raised.

  if the decorated method returns a context manager, a wrapper context manager
  will be returned, which will end the session when exiting the context.

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

    @functools.wraps(func)
    def method_wrapper(self, *args, **kwargs) -> Any:
      """Helper function to safely execute logics in the sandbox."""
      assert isinstance(self, (interface.Sandbox, interface.Feature)), self
      sandbox = self.sandbox if isinstance(self, interface.Feature) else self

      # When a capability is directly accessed from the environment,
      # we scope the function call within a short-lived sandbox session. This
      # prevents the sandbox from being reused for other feature calls.
      if sandbox.is_pending:
        new_session = True
        session_id = kwargs.get('session_id', f'session-{uuid.uuid4().hex[:7]}')
      else:
        new_session = False
        session_id = sandbox.session_id

      kwargs.pop('session_id', None)
      result = None
      error = None
      try:
        # If it's a feature method called from the environment, start a new
        # session for the feature call.
        if new_session:
          sandbox.start_session(session_id)

        # Execute the service function.
        result = func(self, *args, **kwargs)

        # If the result is a context manager, use it and end the session
        # afterwards.
        if new_session and isinstance(
            result, contextlib.AbstractContextManager
        ):
          return _end_session_when_exit(result, sandbox)

        # Otherwise, return the result and end the session in the finally block.
        return result
      except interface.SandboxStateError as e:
        error = e
        raise
      except BaseException as e:
        if pg.match_error(e, critical_errors):
          error = e
          raise interface.SandboxStateError(
              'Sandbox encountered an unexpected error executing '
              f'`{func.__name__}` (args={args!r}, kwargs={kwargs!r}): {e}',
              sandbox=self
          ) from e
        raise
      finally:
        if error is not None:
          sandbox.shutdown()

        # End the session if it's from a feature method and the result is not a
        # context manager.
        if (new_session
            and not isinstance(result, contextlib.AbstractContextManager)):
          sandbox.end_session()

        self.on_session_activity(
            session_id=session_id, error=error, args=args, **kwargs
        )
    return method_wrapper
  return decorator


@contextlib.contextmanager
def _end_session_when_exit(
    service: contextlib.AbstractContextManager[Any],
    sandbox: interface.Sandbox
) -> Iterator[Any]:
  """Context manager wrapper for ending a sandbox session when exiting."""
  try:
    with service as result:
      yield result
  finally:
    sandbox.end_session()
