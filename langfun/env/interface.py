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
import functools
import os
import random
import re
import time
from typing import Annotated, Any, Callable, ContextManager, ClassVar, Iterator, Optional, Sequence, Type, Union
import uuid

import langfun.core as lf
import pyglove as pg

#
# Environment errors.
#


class EnvironmentError(RuntimeError):  # pylint: disable=redefined-builtin
  """Base class for environment errors."""

  def __init__(
      self,
      message: str,
      *args,
      environment: 'AbstractEnvironment',
      **kwargs
  ) -> None:
    self.environment = environment
    super().__init__(f'[{environment.id}] {message}.', *args, **kwargs)


class SandboxServiceError(RuntimeError):
  """Base class for sandbox service errors."""

  def __init__(
      self,
      message: str,
      *args,
      sandbox_service: 'SandboxService',
      **kwargs
  ) -> None:
    self.sandbox_service = sandbox_service
    super().__init__(
        f'[{sandbox_service.id}] {message}.',
        *args,
        **kwargs
    )


class SandboxServiceOutageError(SandboxServiceError):
  """Error that indicates sandbox service is offline."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      offline_duration: float,
      **kwargs
  ):
    self.offline_duration = offline_duration
    super().__init__(
        message or (
            f'Sandbox service is offline for {offline_duration} seconds.'
        ),
        *args,
        **kwargs
    )


class SandboxServiceOverloadError(SandboxServiceError):
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


class SandboxFeaturesTeardownError(SandboxError):
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


class SandboxSessionTeardownError(SandboxError):
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
# Interface for feature errors.
#


class FeatureError(RuntimeError):
  """Base class for feature errors."""

  def __init__(
      self,
      message: str,
      *args,
      feature: 'Feature',
      **kwargs
  ) -> None:
    self.feature = feature
    super().__init__(f'[{feature.id}] {message}.', *args, **kwargs)


class FeatureSetupError(FeatureError):
  """Error that indicates feature setup failed."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      feature: 'Feature',
      **kwargs
  ) -> None:
    super().__init__(
        message or 'Feature setup failed.', *args, feature=feature, **kwargs
    )


class FeatureTeardownError(FeatureError):
  """Error that indicates feature teardown failed."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      feature: 'Feature',
      **kwargs
  ) -> None:
    super().__init__(
        message or 'Feature teardown failed.', *args, feature=feature, **kwargs
    )


class FeatureOutageError(FeatureError):
  """Error that indicates feature is offline."""

  def __init__(
      self,
      message: str | None = None,
      *args,
      offline_duration: float,
      **kwargs
  ):
    self.offline_duration = offline_duration
    super().__init__(
        message or (
            f'Feature is offline for {offline_duration} seconds.'
        ),
        *args,
        **kwargs
    )


#
# Interface for event handlers.
#


class _EnvironmentEventHandler:
  """Base class for event handlers of an environment."""

  def on_environment_starting(
      self,
      environment: 'AbstractEnvironment'
  ) -> None:
    """Called when the environment is getting started.

    Args:
      environment: The environment.
    """

  def on_environment_start(
      self,
      environment: 'AbstractEnvironment',
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

  def on_environment_shutting_down(
      self,
      environment: 'AbstractEnvironment',
  ) -> None:
    """Called when the environment is shutting down.

    Args:
      environment: The environment.
    """

  def on_environment_shutdown(
      self,
      environment: 'AbstractEnvironment',
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

  def on_environment_housekeep(
      self,
      environment: 'AbstractEnvironment',
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


class _SandboxServiceEventHandler:
  """Base class for event handlers of a sandbox service."""

  def on_sandbox_service_starting(
      self,
      sandbox_service: 'SandboxService',
  ) -> None:
    """Called when a sandbox service is getting started."""

  def on_sandbox_service_start(
      self,
      sandbox_service: 'SandboxService',
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox service is started."""

  def on_sandbox_service_shutting_down(
      self,
      sandbox_service: 'SandboxService',
      offline_duration: float,
  ) -> None:
    """Called when a sandbox service is shutting down."""

  def on_sandbox_service_shutdown(
      self,
      sandbox_service: 'SandboxService',
      duration: float,
      lifetime: float,
      error: BaseException | None,
  ) -> None:
    """Called when a sandbox service is shutting down."""

  def on_sandbox_service_housekeep(
      self,
      sandbox_service: 'SandboxService',
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox service finishes a round of housekeeping."""


class _SandboxEventHandler:
  """Base class for sandbox event handlers."""

  def on_sandbox_start(
      self,
      sandbox: 'Sandbox',
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox is started.

    Args:
      sandbox: The sandbox.
      duration: The time spent on starting the sandbox.
      error: The error that caused the sandbox to start. If None, the sandbox
        started normally.
    """

  def on_sandbox_status_change(
      self,
      sandbox: 'Sandbox',
      old_status: 'Sandbox.Status',
      new_status: 'Sandbox.Status',
      span: float,
  ) -> None:
    """Called when a sandbox status changes.

    Args:
      sandbox: The sandbox.
      old_status: The old sandbox status.
      new_status: The new sandbox status.
      span: Time spent on the old status in seconds.
    """

  def on_sandbox_shutdown(
      self,
      sandbox: 'Sandbox',
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox is shutdown.

    Args:
      sandbox: The sandbox.
      duration: The time spent on shutting down the sandbox.
      lifetime: The sandbox lifetime in seconds.
      error: The error that caused the sandbox to shutdown. If None, the
        sandbox shutdown normally.
    """

  def on_sandbox_session_start(
      self,
      sandbox: 'Sandbox',
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session starts.

    Args:
      sandbox: The sandbox.
      session_id: The session ID.
      duration: The time spent on starting the session.
      error: The error that caused the session to start. If None, the session
        started normally.
    """

  def on_sandbox_session_end(
      self,
      sandbox: 'Sandbox',
      session_id: str,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session ends.

    Args:
      sandbox: The sandbox.
      session_id: The session ID.
      duration: The time spent on ending the session.
      lifetime: The session lifetime in seconds.
      error: The error that caused the session to end. If None, the session
        ended normally.
    """

  def on_sandbox_activity(
      self,
      name: str,
      sandbox: 'Sandbox',
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed.

    Args:
      name: The name of the sandbox activity.
      sandbox: The sandbox.
      session_id: The session ID.
      duration: The sandbox activity duration in seconds.
      error: The error that caused the sandbox activity to perform. If None,
        the sandbox activity performed normally.
      **kwargs: The keyword arguments of the sandbox activity.
    """

  def on_sandbox_housekeep(
      self,
      sandbox: 'Sandbox',
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox finishes a round of housekeeping.

    Args:
      sandbox: The sandbox.
      counter: Zero-based counter of the housekeeping round.
      duration: The sandbox housekeeping duration in seconds.
      error: The error that caused the sandbox to housekeeping. If None, the
        sandbox housekeeping normally.
      **kwargs: Sandbox-specific properties computed during housekeeping.
    """


class _FeatureEventHandler:
  """Base class for feature event handlers."""

  def on_feature_setup(
      self,
      feature: 'Feature',
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup.

    Applicable to both sandbox-based and non-sandbox-based features.

    Args:
      feature: The feature.
      duration: The feature setup duration in seconds.
      error: The error happened during the feature setup. If None,
        the feature setup performed normally.
    """

  def on_feature_teardown(
      self,
      feature: 'Feature',
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown.

    Applicable to both sandbox-based and non-sandbox-based features.

    Args:
      feature: The feature.
      duration: The feature teardown duration in seconds.
      error: The error happened during the feature teardown. If None,
        the feature teardown performed normally.
    """

  def on_feature_teardown_session(
      self,
      feature: 'Feature',
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a feature is teardown with a session.

    Applicable to both sandbox-based and non-sandbox-based features.

    Args:
      feature: The feature.
      session_id: The session ID.
      duration: The feature teardown session duration in seconds.
      error: The error happened during the feature teardown session. If
        None, the feature teardown session performed normally.
    """

  def on_feature_setup_session(
      self,
      feature: 'Feature',
      session_id: str | None,
      duration: float,
      error: BaseException | None,
  ) -> None:
    """Called when a feature is setup with a session.

    Applicable to both sandbox-based and non-sandbox-based features.

    Args:
      feature: The feature.
      session_id: The session ID.
      duration: The feature setup session duration in seconds.
      error: The error happened during the feature setup session. If
        None, the feature setup session performed normally.
    """

  def on_feature_activity(
      self,
      name: str,
      feature: 'Feature',
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a feature activity is performed.

    Applicable to both sandbox-based and non-sandbox-based features.

    Args:
      name: The name of the feature activity.
      feature: The feature.
      session_id: The session ID. Session ID could be None if a feature
        activity is performed when setting up a session
        (e.g. BaseEnvironment.proactive_session_setup is on)
      duration: The feature activity duration in seconds.
      error: The error happened during the feature activity. If None,
        the feature activity performed normally.
      **kwargs: The keyword arguments of the feature activity.
    """

  def on_feature_housekeep(
      self,
      feature: 'Feature',
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs,
  ) -> None:
    """Called when a sandbox feature is housekeeping.

    Applicable to both sandbox-based and non-sandbox-based features.

    Args:
      feature: The feature.
      counter: Zero-based counter of the housekeeping round.
      duration: The feature housekeeping duration in seconds.
      error: The error happened during the feature housekeeping. If None, the
        feature housekeeping normally.
      **kwargs: Feature-specific properties computed during housekeeping.
    """


class EventHandler(
    _EnvironmentEventHandler,
    _SandboxServiceEventHandler,
    _SandboxEventHandler,
    _FeatureEventHandler,
):
  """Base class for langfun/env handlers."""


#
# Interfaces for features, sandboxes, sandbox services.
#


class Feature(lf.Component):
  """Interface for features that run in a Langfun environment.

  There are two type of features: sandbox-based and non-sandbox-based.
  Sandbox-based features run in a sandbox, which is emulated in a separate
  process. Non-sandbox-based features do not run in a sandbox.

  Features can be directly accessed through the environment, for example:

  ```python
  env = Environment(
      sandboxes={
          'docker': DockerSandboxService(
              features={
                  'feature1': SandboxBasedFeature(),
              }
          ),
      },
      feature={
          'feature2': NonSandboxBasedFeature(),
      }
  )
  # Start the environment.
  with env:
    # Access feature1, which involves acquiring a sandbox and return the feature
    # associated with the sandbox.
    with env.feature1() as f1:
      f1.feature_method()

    # Access feature2, which does not involve acquiring a sandbox.
    with env.feature2() as f2:
      f2.feature_method()
  ```

  Sandbox-based features can also be accessed through the sandbox, for example:

  ```python
  with env.sandbox('session1') as sb:
    # Access feature1 within the sandbox.
    sb.feature1.feature_method()

    # Attribute error.
    sb.feature2
  ```
  """

  @dataclasses.dataclass
  class Id:
    """Identifier of a feature."""
    container_id: Union['AbstractEnvironment.Id', 'Sandbox.Id', None]
    feature_name: str

    def __str__(self) -> str:
      if self.container_id is None:
        return self.feature_name
      return f'{self.container_id}/{self.feature_name}'

    def working_dir(self, root_dir: str | None) -> str | None:
      """Returns the working directory for the feature."""
      if root_dir is None:
        return None
      if self.container_id is None:
        return os.path.join(root_dir, _make_path_compatible(self.feature_name))
      return os.path.join(
          self.container_id.working_dir(root_dir),
          _make_path_compatible(self.feature_name)
      )

  is_sandbox_based: Annotated[
      bool,
      'Whether the feature is sandbox-based.'
  ] = True

  applicable_images: Annotated[
      list[str],
      (
          'A list of regular expressions for image IDs which enable '
          'this feature. Applicable only to sandbox-based features. '
          'By default, all images are enabled.'
      )
  ] = ['.*']

  housekeep_interval: Annotated[
      float | None,
      (
          'Interval in seconds for feature housekeeping. Applicable for both '
          'sandbox-based and non-sandbox-based features. If None, housekeeping '
          'is disabled.'
      )
  ] = None

  root_dir: Annotated[
      str | None,
      (
          'The root directory for the feature. If unspecified, the '
          'feature will use the environment-level root directory.'
      )
  ] = lf.contextual(default=None)

  event_handler: Annotated[
      'EventHandler',
      (
          'User handler for the feature events. '
          'If unspecified, the container\'s event handler (environment or '
          'sandbox) will be used. '
      )
  ] = lf.contextual(default=EventHandler())

  outage_grace_period: Annotated[
      float,
      (
          'The grace period in seconds before the feature is treated '
          'as out of service. Applicable when feature is not sandbox-based. '
          'When `env.<feature>()` is called, wait until the grace period has '
          'passed before raising an error. '
          'If unspecified (default), the feature will use the '
          'environment-level grace period instead.'
      )
  ] = lf.contextual(default=3600.0)

  # Disable symbolic comparison and hashing for sandbox objects.
  allow_symbolic_comparison = False

  @functools.cached_property
  def name(self) -> str:
    """Name of the feature, which will be used as key to access the feature."""
    assert isinstance(self.sym_parent, dict), 'Feature is not put into a dict.'
    return self.sym_path.key

  @functools.cached_property
  def id(self) -> Id:
    """Returns the identifier of the feature."""
    if self.is_sandbox_based:
      return Feature.Id(self.sandbox.id, self.name)
    if self.environment is not None:
      return Feature.Id(self.environment.id, self.name)
    return Feature.Id(None, pg.utils.camel_to_snake(self.__class__.__name__))

  @property
  def working_dir(self) -> str | None:
    """Returns the working directory for the feature."""
    return self.id.working_dir(self.root_dir) if self.root_dir else None

  def is_applicable(self, image_id: str) -> bool:
    """Returns True if the feature is applicable to the given image."""
    return any(
        re.fullmatch(regex, image_id) for regex in self.applicable_images
    )

  @property
  @abc.abstractmethod
  def environment(self) -> Optional['AbstractEnvironment']:
    """Returns the environment that the feature is running in."""

  @property
  @abc.abstractmethod
  def sandbox(self) -> Optional['Sandbox']:
    """Returns the sandbox that the feature is running in.

    Returns:
      The sandbox that the feature is running in. None if the feature is not
      sandbox-based or not yet bound with a sandbox.
    """

  @property
  @abc.abstractmethod
  def is_online(self) -> bool:
    """Returns True if the feature is online."""

  @property
  @abc.abstractmethod
  def offline_duration(self) -> float:
    """Returns the offline duration of non-sandbox-based feature."""

  @abc.abstractmethod
  def setup(self, sandbox: Optional['Sandbox'] = None) -> None:
    """Sets up the feature.

    For sandbox-based features, the setup will be called when a sandbox is
    started for the first time.

    For non-sandbox-based features, the setup will be called when the
    environment starts.

    When a feature's `setup` is called, its `teardown` is guaranteed to be
    called.

    Args:
      sandbox: The sandbox that the feature is running in.

    Raises:
      FeatureSetupError: If setup failed.
    """

  @abc.abstractmethod
  def teardown(self) -> None:
    """Teardowns the feature, which is called once when the sandbox is down.

    Raises:
      FeatureTeardownError: If teardown failed.
    """

  @abc.abstractmethod
  def setup_session(self) -> None:
    """Sets up the feature for the upcoming user session.

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
  def track_activity(
      self,
      name: str,
      **kwargs: Any
  ) -> ContextManager[None]:
    """Context manager that tracks a feature activity.

    Args:
      name: The name of the activity.
      **kwargs: Additional keyword arguments to pass to the activity handler.

    Returns:
      A context manager that tracks the activity, including duration and error.
    """

  @property
  def session_id(self) -> str | None:
    """Returns the current user session identifier."""
    if self.is_sandbox_based:
      return self.sandbox.session_id
    return self._non_sandbox_based_session_id

  @contextlib.contextmanager
  def new_session(self, session_id: str) -> Iterator['Feature']:
    """Context manager for obtaining a non-sandbox-based feature session."""
    assert not self.is_sandbox_based, (
        'Applicable only to non-sandbox-based features. '
        'For sandbox-based features, use `Sandbox.new_session` instead.'
    )
    if not self.is_online:
      raise FeatureOutageError(
          feature=self, offline_duration=self.offline_duration
      )

    try:
      self._non_sandbox_based_session_id = session_id
      self.setup_session()
      yield self
    finally:
      try:
        # Since the session is ended, we don't want to raise any errors during
        # session teardown to the user. So we catch all exceptions here.
        # However the event handler will still be notified and log the error.
        self.teardown_session()
      except BaseException:  # pylint: disable=broad-except
        pass
      self._non_sandbox_based_session_id = None

  def _on_bound(self) -> None:
    """Called when the feature is bound to a sandbox."""
    super()._on_bound()
    self._non_sandbox_based_session_id = None

  def __enter__(self) -> None:
    """Called when the non-sandbox-based feature is entered."""
    assert not self.is_sandbox_based, (
        'Applicable only to non-sandbox-based features.'
    )
    assert self.environment is None, (
        'Applicable only when the feature is not bound to an environment.'
    )
    self.setup()

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    """Called when the non-sandbox-based feature is exited."""
    del exc_type, exc_value, traceback
    self.teardown()


class Sandbox(lf.Component):
  """Interface for sandboxes.

  A sandbox is a container that runs a single image with a set of sandbox-based
  features. It will be brought up by the environment, setup the features,
  fullfill user requests, and then tear down features and finally the sandbox
  itself.
  """

  @dataclasses.dataclass(frozen=True, slots=True)
  class Id:
    """Identifier for a sandbox."""
    service_id: 'SandboxService.Id'
    image_id: str
    sandbox_id: str

    def __str__(self) -> str:
      return f'{self.service_id}/{self.image_id}:{self.sandbox_id}'

    def working_dir(self, root_dir: str | None) -> str | None:
      """Returns the download directory for the sandbox."""
      if root_dir is None:
        return None
      return os.path.join(
          self.service_id.working_dir(root_dir),
          _make_path_compatible(self.image_id),
          _make_path_compatible(self.sandbox_id)
      )

  class Status(enum.Enum):
    r"""Sandbox state.

    State transitions:

                    (sandbox / feature
    +------------+   teardown)            +---------------+
    | <OFFLINE>  | <--------------------- | SHUTTING_DOWN |
    +------------+                        +---------------+
                                            ^     ^
                                           /       \
                          (setup failed)  /         \
                                         /           \
    +-----------+   (start)  +------------+           \
    | <CREATED> | -------->  | SETTING_UP |            \
    +-----------+         ^  +------------+             \
                         /        |                      \
                        /         | (sandbox /            \
                       /          |  feature /session      \
                      /           v  setup succeeded)       \
                     /        +---------+                    \
                    /         |  READY  |                     \
                   /          +---------+                      \
                  /                |                            \
                 /                 |  (acquire)                  \
                /                  v                              \
               /              +----------+                         \
              |               | ACQUIRED |                          \
              |               +----------+                           |
              |                    |                                 |
              |                    |  (start_session)                |
              |               +------------+                         |
              |               | SETTING_UP |-- (setup failed) ------>+
              |               +------------+                         |
              |                    |                                 |
              |                    v  (succeeded)                    |
              |               +--------------+                       |
              |               |  IN_SESSION  |- (op failed) -------->+
              |               +--------------+                       |
              |                    |                                 |
              |                    |  (end_session)                  |
              |                    |                                 |
              |                    v             (session teardown   |
      (setup next            +-----------------+  failed OR          |
      session for  <---------| EXITING_SESSION |- non-reusable  -----+
      reusable sandbox)      +-----------------+  sandbox)

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

  # Disable symbolic comparison and hashing for sandbox objects.
  use_symbolic_comparison = False

  @property
  @abc.abstractmethod
  def id(self) -> Id:
    """Returns the identifier for the sandbox."""

  @property
  @abc.abstractmethod
  def image_id(self) -> str:
    """Returns the image ID used for bootstrapping the sandbox."""

  @property
  @abc.abstractmethod
  def sandbox_service(self) -> 'SandboxService':
    """Returns the sandbox service for the sandbox."""

  @property
  def environment(self) -> Optional['AbstractEnvironment']:
    """Returns the environment for the sandbox."""
    return self.sandbox_service.environment

  @functools.cached_property
  def working_dir(self) -> str | None:
    """Returns the working directory for the sandbox."""
    return self.id.working_dir(self.sandbox_service.root_dir)

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
  def report_state_error(self, error: SandboxStateError) -> None:
    """Reports state error the sandbox.

    If state errors are reported, the sandbox will be forcefully shutdown when
    `Sandbox.end_session()` is called, even if the sandbox is set to be
    reusable.

    Args:
      error: SandboxStateError to report.
    """

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

    When `end_session` is called with state errors reported, the sandbox will be
    forcefully shutdown even if the sandbox is set to be reusable.

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

  @abc.abstractmethod
  def track_activity(
      self,
      name: str,
      **kwargs: Any
  ) -> ContextManager[None]:
    """Context manager that tracks a sandbox activity.

    Args:
      name: The name of the activity.
      **kwargs: Additional keyword arguments to pass to the activity handler.

    Returns:
      A context manager that tracks the activity, including duration and error.
    """

  #
  # API related to a user session.
  # A sandbox could be reused across different user sessions.
  # A user session is a sequence of stateful interactions with the sandbox,
  # Across different sessions the sandbox are considered stateless.
  #

  @contextlib.contextmanager
  def new_session(self, session_id: str) -> Iterator['Sandbox']:
    """Context manager for obtaining a sandbox for a user session.

    State transitions:
      ACQUIRED -> IN_SESSION -> READY: When session setup and teardown succeed.
      ACQUIRED -> IN_SESSINO -> OFFLINE: When session setup or teardown fails.

    Args:
      session_id: The identifier for the user session.

    Yields:
      The sandbox for the user session.

    Raises:
      SandboxStateError: If a session cannot be started on the sandbox.
      BaseException: If session setup or teardown failed with user-defined
        errors.
    """
    self.start_session(session_id)
    try:
      yield self
    except SandboxStateError as e:
      self.report_state_error(e)
      raise
    finally:
      self.end_session()

  def __getattr__(self, name: str) -> Any:
    """Gets a feature from current sandbox.

    Example:
      ```
      with Environment(
          sandboxes={
              'docker': DockerSandboxService(
                  features={
                      'feature1': Feature1(),
                  }
              )
          }
      ) as env:
        with env.sandbox('session1') as sb:
          driver = sb.feature1.feature_method()
      ```

    Args:
      name: The name of the feature.

    Returns:
      A feature from current sandbox.
    """
    if name.startswith('_') or name == 'features':
      raise AttributeError(name)
    if name in self.features:
      return self.features[name]
    raise AttributeError(name)


class SandboxService(lf.Component):
  """Interface for sandbox services.

  A sandbox service is a provider of sandboxes running with images specialized
  for certain features. It can host multiple sandboxes with the same or
  different images running concurrently.
  """

  @dataclasses.dataclass(frozen=True)
  class Id:
    """Identifier for a sandboxing service."""
    environment_id: Optional['AbstractEnvironment.Id']
    name: str

    def __str__(self) -> str:
      if self.environment_id is None:
        return self.name
      return f'{self.environment_id}/{self.name}'

    def working_dir(self, root_dir: str | None) -> str | None:
      """Returns the working directory for the service."""
      if root_dir is None:
        return None
      if self.environment_id is None:
        return os.path.join(root_dir, self.name)
      return os.path.join(self.environment_id.working_dir(root_dir), self.name)

  image_ids: Annotated[
      list[str],
      (
          'A list of static image IDs served by the sandbox service. '
      )
  ]

  supports_dynamic_image_loading: Annotated[
      bool,
      (
          'Whether the sandbox service supports dynamic loading of images '
          'which is not included in the `image_ids`. `image_ids` could coexist '
          'with dynamic image loading, which allows users to specify an image '
          'id that is not included in the `image_ids`.'
      )
  ] = False

  features: Annotated[
      dict[str, 'Feature'],
      (
          'A dictionary of sandbox-based features that are served by the '
          'sandbox service. The key is the feature name, which will be used to '
          'access the feature.'
      )
  ] = {}

  #
  # Contextual fields which are defaulted by the environment.
  #

  root_dir: Annotated[
      str | None,
      (
          'The root directory for the sandbox service. If unspecified, the '
          'sandbox service will use the environment-level root directory.'
      )
  ] = lf.contextual(default=None)

  event_handler: Annotated[
      'EventHandler',
      (
          'User handler for the sandbox service events.'
          'If None, no-op handler will be used.'
      )
  ] = lf.contextual(default=EventHandler())

  outage_grace_period: Annotated[
      float,
      (
          'The grace period in seconds before the sandbox service is treated '
          'as out of service. When calling `environment.sandbox()`, '
          'wait until the grace period has passed before raising an error. '
          'If unspecified (default), the sandbox service will use the '
          'environment-level grace period instead.'
      )
  ] = lf.contextual(default=3600.0)

  outage_retry_interval: Annotated[
      float,
      (
          'The retry interval in seconds for sandbox service outage. '
          'When calling `environment.sandbox()`, retry after the interval '
          'if the sandbox service is out of service. If unspecified (default), '
          'the sandbox service will use the environment-level retry interval '
          'instead.'
      )
  ] = lf.contextual(default=10.0)

  housekeep_interval: Annotated[
      float,
      (
          'The interval in seconds for service housekeeping. It recycles '
          'the dead sandboxes in the pool. This interval is the minimal time '
          'to detect outage while there is no request to obtain new sandboxes.'
          'This is applicable only when the sandbox service enables pooling.'
      )
  ] = 10.0

  def _on_path_change(self, old_path: pg.KeyPath, new_path: pg.KeyPath) -> None:
    """Changed when the service is assigned to an environment."""
    super()._on_path_change(old_path, new_path)
    self.__dict__.pop('environment', None)
    self.__dict__.pop('name', None)
    self.__dict__.pop('working_dir', None)

  @functools.cached_property
  def environment(self) -> Optional['AbstractEnvironment']:
    """Returns the containing environment."""
    return self.sym_ancestor(lambda v: isinstance(v, AbstractEnvironment))

  @functools.cached_property
  def name(self) -> str:
    """Returns the root directory for the sandbox service."""
    assert isinstance(self.sym_parent, dict), (
        'Sandbox service must be assigned to an environment.'
    )
    return self.sym_path.key

  @functools.cached_property
  def working_dir(self) -> str | None:
    """Returns the working directory for the service."""
    return self.id.working_dir(self.root_dir)

  @property
  @abc.abstractmethod
  def id(self) -> Id:
    """Returns the identifier for the sandbox service."""

  @property
  @abc.abstractmethod
  def is_online(self) -> bool:
    """Returns True if the service is online."""

  @abc.abstractmethod
  def start(self) -> None:
    """Starts the service.

    Raises:
      SandboxServiceError: If the sandbox service is not available.
    """

  @abc.abstractmethod
  def shutdown(self) -> None:
    """Shuts down the sandbox service.

    IMPORTANT: This method shall not raise any exceptions.
    """

  @abc.abstractmethod
  def acquire(self, image_id: str | None = None) -> 'Sandbox':
    """Acquires a free sandbox from the environment.

    Args:
      image_id: The image ID to use for the sandbox. If None, it will be
        automatically determined by the environment.

    Returns:
      A free sandbox from the service.

    Raises:
      SandboxServiceOutageError: If the sandbox service is out of service.
      SandboxServiceOverloadError: If the sandbox service is overloaded.
    """

  def image_id_for(self, feature: Feature) -> str:
    """Returns the image ID for the feature."""
    for image_id in self.image_ids:
      if feature.is_applicable(image_id):
        return image_id
    if self.supports_dynamic_image_loading:
      raise ValueError(
          f'Feature {feature.name!r} is not applicable to any of the '
          f'pre-configured images ({self.image_ids!r}) served by {self.id}. '
          'Please specify an image ID explicitly and it will be dynamically '
          'loaded.'
      )
    else:
      raise ValueError(
          f'Feature {feature.name} is not applicable to any image served by '
          f'sandbox service {self.id}.'
      )

#
# Langfun Environment that orchastrates sandbox services and features.
#


class AbstractEnvironment(lf.Component):
  """Langfun Environment.

  An **Environment** is the central component for managing **Sandboxes** and
  **Features**. It acts as an abstraction layer, hiding the implementation
  details of the underlying sandboxing system
  (e.g., Docker, virtual machines).

  The core goal is to enable the development of features that are **agnostic**
  to how or where they are executed (sandboxed or not).

  -----------------------------------------------------------------------------

  ## Core Concepts

  1.  **Sandbox-Based Features:** Features that require isolated execution
      contexts (sandboxes). They become available as properties of the sandbox
      object (e.g., `sandbox.feature1`). Applicability is determined by an image
      ID regex.
  2.  **Non-Sandbox-Based Features:** Features that run directly within the host
      process or manage their own execution context outside of the Environment's
      managed sandboxes. They are accessed directly via the Environment (e.g.,
      `env.feature3()`).

  How to Use:

  The primary usage patterns are creating a **sandbox session** or directly
  accessing a specific **Feature**, which transparently handles sandbox creation
  if needed.

  ```python
  env = lf.env.Environment(
      sandboxes={
          'docker': lf.env.DockerSandboxService(
              image_ids=['image1', 'image2'],
              pool_size=(0, 256),
              features={
                  'feature1': Feature1(applicable_images=['image1.*']),
                  'feature2': Feature2(applicable_images=['image2.*']),
              }
          ),
      },
      features={
          'feature3': Feature3(is_sandbox_based=False)
      })

  # Context manager for the Environment lifetime.
  with env:

    # 1. Access a specific sandbox directly.
    # Upon exiting the context, the sandbox will be shutdown or returned to
    # the pool for reuse.
    with env.sandbox(image_id='image1') as sandbox:
      # Execute a shell command inside the sandbox.
      sandbox.shell('echo "hello world"')

      # Access a sandbox-based feature (feature1 is applicable to image1).
      sandbox.feature1.feature_method()

      # Attempts to access inapplicable features will raise AttributeError:
      # sandbox.feature2  # Not applicable to image1.
      # sandbox.feature3  # Not sandbox-based.

    # 2. Access a sandbox-based feature and let the Environment manage the
    # sandbox. A suitable sandbox (e.g., one built from an image matching
    # 'image1.*') will be provisioned, and the feature instance will be yielded.
    with env.feature1() as feature1:
      feature1.feature_method()

    # 3. Access a non-sandbox-based feature.
    with env.feature3() as feature3:
      feature3.feature_method()
  ```

  -----------------------------------------------------------------------------

  ## Multi-tenancy and Pooling

  The Environment supports multi-tenancy (working with multiple image types) and
  pooling (reusing sandboxes) to amortize setup costs across different user
  requests. Pooling is configured via the `pool_size` parameter on
  `SandboxService` implementations (e.g., `lf.env.DockerSandboxService`).

  | pool_size Value           | Behavior                                       |
  |---------------------------|------------------------------------------------|
  | 0 or (0, 0)	              | No pooling. Sandboxes are created and shut down|
  |                           | on demand (useful for local development).      |
  |---------------------------|------------------------------------------------|
  | (MIN, MAX) tuple          | Global Pool: Applies the same minimum and      |
  |                           | maximum pool size to sandboxes created from all|
  |                           | specified images.                              |
  |---------------------------|------------------------------------------------|
  | {image_regex: (MIN, MAX)} | Per-Image Pool: Allows customizing pool        |
  |                           | settings based on image ID regular expressions.|
  |                           | (e.g., 'image1.*': (64, 256), '.*': (0, 256)). |


  **Example 1: No Pooling (pool_size=0)**
  Sandboxes are created and shutdown immediately upon session end.

  ```python
  env = lf.env.Environment(
      sandboxes={
          'docker': lf.env.DockerSandboxService(
              image_ids=['image1', 'image2'], pool_size=0
          )
      }
  )

  # Sandbox created and shutdown on demand.
  with env.sandbox(image_id='image1') as sandbox1:
    ...
  ```

  **Exaxmple 2: Global Pooling (pool_size=(0, 256))**
  Up to 256 sandboxes will be created and pooled across both images as needed.
  None are created initially.

  ```python
  env = lf.env.Environment(
      sandboxes={
          'docker': lf.env.DockerSandboxService(
              image_ids=['image1', 'image2'], pool_size=(0, 256)
          )
      }
  )
  ```

  **Example 3: Per-Image Custom Pooling**:
  For images matching 'image1.*': 64 sandboxes are pre-created (MIN=64) and
  pooled, up to a MAX of 256.
  For all other images ('.*'): Sandboxes are created and pooled on demand
  (MIN=0), up to a MAX of 256.

  ```python
  env = lf.env.Environment(sandboxes={
      'docker': lf.env.DockerSandboxService(
          image_ids=['image1', 'image2'],
          pool_size={
              'image1.*': (64, 256),
              '.*': (0, 256),
          },
      )
  })
  ```

  ## Handling Sandbox Failures

  Sandboxes often run in distributed, ephemeral environments and must be treated
  as fault-tolerant. Langfun provides a protocol for handling unexpected sandbox
  state issues.

  ### Communicating Errors
  If a feature encounters an unexpected state in its sandbox (e.g., a process
  died), it should raise `lf.env.SandboxStateError`.

  The sandbox will be automatically shut down when its context manager exits.
  The Environment handles replacement and ensures future requests are routed to
  healthy sandboxes.

  For example:
  ```
  with env:
    with env.sandbox() as sb:
      # If SandboxStateError is raised within this block, the sandbox
      # will be forcefully shut down upon block exit.
      sb.shell('echo hi')
  ```

  ### Robust User Code
  A simple strategy for robust user code is to wrap critical operations in a
  retry loop:
  ```
  while True:
    try:
      result = do_something_that_involves_sandbox()
      break  # Success!
    except lf.env.SandboxStateError:
      # The sandbox failed; a new, healthy one will be provisioned on the next
      # iteration.
      # Wait briefly to avoid resource thrashing.
      time.sleep(1)
    except lf.env.SandboxServiceOutageError:
      # If the Environment is down for too long
      # (past BaseEnvironment.outage_grace_period)
      # and cannot provision a healthy replacement, this error is raised.
      # The retry loop should be broken or an outer failure reported.
      raise
  ```
  """

  @dataclasses.dataclass(frozen=True)
  class Id:
    """Identifier for an environment."""
    environment_id: str

    def __str__(self) -> str:
      return self.environment_id

    def working_dir(self, root_dir: str | None) -> str | None:
      """Returns the download directory for the service."""
      if root_dir is None:
        return None
      return os.path.join(root_dir, _make_path_compatible(self.environment_id))

  id: Annotated[
      Id,
      'Identifier for the environment.'
  ] = Id('env')

  root_dir: Annotated[
      str | None,
      (
          'The root directory for the environment for writting output files.'
          'If None, no output files will be allowed for the sandboxes.'
      )
  ] = None

  sandboxes: Annotated[
      dict[str, 'SandboxService'],
      'Sandboxes to be exposed by the environment.'
  ] = {}

  features: Annotated[
      dict[str, 'Feature'],
      'Non-sandbox-based features to be exposed by the environment.'
  ] = {}

  random_seed: Annotated[
      int | None,
      (
          'The random seed for generating session IDs with reproducibility. '
          'If None, no seed will be used.'
      )
  ] = None

  #
  # Contextual fields that will be used as the default for sandbox services and
  # non-sandbox-based features.
  #

  event_handler: Annotated[
      'EventHandler',
      (
          'User handler for the environment events.'
          'If unspecified, no-op handler will be used.'
      )
  ] = EventHandler()

  outage_grace_period: Annotated[
      float,
      (
          'The grace period for environment outage. '
          'If the environment is offline for longer than this period, '
          'an SandboxServiceOutageError will be raised.'
      )
  ] = 3600.0

  # Disable symbolic comparison and hashing for environment objects.
  use_symbolic_comparison = False

  _DEFAULT: ClassVar[Optional['AbstractEnvironment']] = None

  def as_default(self) -> 'AbstractEnvironment':
    """Sets the global default environment."""
    AbstractEnvironment.set_default(self)
    return self

  @classmethod
  def set_default(cls, env: 'AbstractEnvironment') -> None:
    """Sets the global default environment."""
    AbstractEnvironment._DEFAULT = env

  @classmethod
  def default(cls) -> Optional['AbstractEnvironment']:
    """Returns the global default environment if set, otherwise None."""
    return AbstractEnvironment._DEFAULT

  _ENV_STACK: Annotated[
      ClassVar[list['AbstractEnvironment']],
      'Recording the environments stacked through context managers.'
  ] = []

  def _on_bound(self) -> None:
    """Called when the environment is bound to a context manager."""
    super()._on_bound()

    self._start_time = None
    self._features_with_setup_called = set()
    self._sandbox_services_with_start_called = set()
    self._random = (
        random if self.random_seed is None else random.Random(self.random_seed)
    )

    # Index feature name -> (feature, sandbox service) mapping
    # for fast feature retrieval by name.
    all_features: dict[
        str,
        tuple[Feature, SandboxService | None]
    ] = {}

    for sandbox_service in self.sandboxes.values():
      for name, feature in sandbox_service.features.items():
        if name in all_features:
          service = all_features[name][1]
          assert service is not None
          raise ValueError(
              f'Feature {name!r} is already configured in {service.id.name!r}.'
          )
        else:
          all_features[name] = (feature, sandbox_service)

    # Check if sandbox-based features are configured as sandbox services.
    for name, feature in self.features.items():
      if feature.is_sandbox_based:
        raise ValueError(
            f'Feature {feature.name!r} is sandbox-based and should be '
            'configured as a feature under a sandbox service.'
        )
      if name in all_features:
        service = all_features[name][1]
        assert service is not None
        raise ValueError(
            f'Feature {name!r} is already configured in {service.id.name!r}.'
        )
      all_features[feature.name] = (feature, None)
    self._all_features = all_features

  #
  # Environment basics.
  #

  @property
  def has_started(self) -> bool:
    """Returns True if the environment has started."""
    return self._start_time is not None

  @property
  def all_online(self) -> bool:
    """Returns True if all sandbox services and features are online."""
    return (
        all(sandbox.is_online for sandbox in self.sandboxes.values())
        and all(feature.is_online for feature in self.features.values())
    )

  def has_feature(self, name: str) -> bool:
    """Returns True if the feature is available in the environment."""
    return name in self._all_features

  @property
  def start_time(self) -> float | None:
    """Returns the start time of the environment."""
    return self._start_time

  @functools.cached_property
  def working_dir(self) -> str | None:
    """Returns the working directory for the environment."""
    return self.id.working_dir(self.root_dir)

  #
  # Environment lifecycle.
  #

  def start(self) -> None:
    """Starts the environment.

    Raises:
      EnvironmentError: If the environment is not available.
    """
    assert self._start_time is None, (
        f'Environment {self.id} is already started.'
    )
    self.on_starting()
    starting_time = time.time()
    try:
      AbstractEnvironment._ENV_STACK.append(self)
      try:
        self._start()
      finally:
        AbstractEnvironment._ENV_STACK.pop()
      self._start_time = time.time()
      self.on_start(duration=time.time() - starting_time)
    except BaseException as e:
      self.on_start(duration=time.time() - starting_time, error=e)
      self.shutdown()
      raise e

  def _start(self) -> None:
    """Implementation of starting the environment."""
    self._features_with_setup_called.clear()
    self._sandbox_services_with_start_called.clear()

    for name, feature in self.features.items():
      self._features_with_setup_called.add(name)
      feature.setup(sandbox=None)

    for name, sandbox_service in self.sandboxes.items():
      self._sandbox_services_with_start_called.add(name)
      sandbox_service.start()

  def shutdown(self) -> None:
    """Shuts down the environment."""
    self.on_shutting_down()
    shutting_down_time = time.time()
    try:
      self._shutdown()
      self.on_shutdown(duration=time.time() - shutting_down_time)
    except BaseException as e:  # pylint: disable=broad-except
      self.on_shutdown(duration=time.time() - shutting_down_time, error=e)
      raise e
    finally:
      self._start_time = None

  def _shutdown(self) -> None:
    """Shuts down the environment.

    IMPORTANT: This method shall not raise any exceptions.
    """
    # Shutdown all sandbox services.
    for name, sandbox_service in self.sandboxes.items():
      if name in self._sandbox_services_with_start_called:
        self._sandbox_services_with_start_called.remove(name)
        sandbox_service.shutdown()

    # Teardown all non-sandbox-based features.
    for name, feature in self.features.items():
      if name in self._features_with_setup_called:
        self._features_with_setup_called.remove(name)
        feature.teardown()

  def __enter__(self) -> 'AbstractEnvironment':
    """Enters the environment and sets it as the current environment."""
    self.start()
    AbstractEnvironment._ENV_STACK.append(self)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the environment and reset the current environment."""
    assert AbstractEnvironment._ENV_STACK
    AbstractEnvironment._ENV_STACK.pop()
    self.shutdown()

  @classmethod
  def current(cls) -> Optional['AbstractEnvironment']:
    """Returns the current environment."""
    if not AbstractEnvironment._ENV_STACK:
      return None
    return AbstractEnvironment._ENV_STACK[-1]

  #
  # Environment operations.
  #

  def __getattr__(self, name: str) -> Any:
    """Gets a sandbox service or a feature session creator by name.

    Example:
      ```
      with lf.env.Environment(
          sandboxes={
              'docker': DockerSandboxService(
                  features={'selenium': SeleniumFeature()}
              ),
          }
          features={'github': GithubFeature()}
      ) as env:
        # Get a sandbox service.
        with env.docker.sandbox() as sb:
          sb.shell('echo "hello world"')

        # Get a non-sandbox-based feature session.
        with env.github() as github:
          github.list_repos()

        # Get a sandbox-based feature session.
        with env.selenium() as selenium:
          driver = selenium.get_driver()
      ```

    Args:
      name: The name of the feature or sandbox service.

    Returns:
      If a feature name, a callable `(image_id, *, session_id) ->
      ContextManager[Feature]` that creates a context manager for the requested
      feature under a new client session. If a sandbox service name, the
      sandbox service object.
    """
    if name.startswith('_') or name in ('sandboxes', 'features'):
      raise AttributeError(name)
    if name in self.sandboxes:
      return self.sandboxes[name]
    if self.has_feature(name):
      return self._feature_session_creator(name)
    raise AttributeError(name)

  def sandbox(
      self,
      session_id: str | None = None,
      image_id: str | None = None,
      sandbox_service: str | None = None,
  ) -> ContextManager['Sandbox']:
    """Gets a sandbox from the environment and starts a new user session."""
    return self._get_sandbox_service(
        image_id, sandbox_service
    ).acquire(image_id).new_session(session_id or self.new_session_id())

  def _get_sandbox_service(
      self,
      image_id: str | None = None,
      sandbox_service: str | None = None,
  ) -> 'SandboxService':
    """Returns the sandbox service for the given image ID."""
    if sandbox_service is not None:
      return self.sandboxes[sandbox_service]
    for sandbox_service in self.sandboxes.values():
      if image_id is None or image_id in sandbox_service.image_ids:
        return sandbox_service

    # Returns the first sandbox service that supports dynamic image loading
    # if image ID is not found in pre-configured image IDs.
    for sandbox_service in self.sandboxes.values():
      if sandbox_service.supports_dynamic_image_loading:
        return sandbox_service
    raise ValueError(
        f'Environment {self.id} does not serve image ID {image_id!r}.'
    )

  def stats(self) -> dict[str, Any]:
    """Returns the stats of the environment."""
    return {}

  def feature_session(
      self,
      name: str,
      session_id: str | None = None,
      image_id: str | None = None,
  ) -> ContextManager['Feature']:
    """Gets a feature from the environment and starts a new user session."""
    if not self.has_feature(name):
      raise ValueError(f'Feature {name!r} is not available on {self.id}.')

    feature, sandbox_service = self._all_features[name]
    if sandbox_service is not None:
      return self._sandbox_session_for_feature(
          sandbox_service, feature, image_id, session_id
      )
    assert image_id is None, (
        'Non-sandbox based feature does not support image ID.'
    )
    return feature.new_session(session_id or self.new_session_id(name))

  @contextlib.contextmanager
  def _sandbox_session_for_feature(
      self,
      sandbox_service: 'SandboxService',
      feature: 'Feature',
      image_id: str | None = None,
      session_id: str | None = None,
  ) -> Iterator['Feature']:
    """Returns a context manager for a session for a feature."""
    assert feature.is_sandbox_based
    if image_id is None:
      image_id = sandbox_service.image_id_for(feature)
    elif not feature.is_applicable(image_id):
      raise ValueError(
          f'Feature {feature.name!r} is not applicable to image {image_id!r}.'
      )
    sandbox = sandbox_service.acquire(image_id=image_id)
    with sandbox.new_session(
        session_id=session_id or self.new_session_id(feature.name)
    ):
      yield sandbox.features[feature.name]

  def new_session_id(self, feature_hint: str | None = None) -> str:
    """Generates a random session ID."""
    suffix = uuid.UUID(
        bytes=bytes(bytes(self._random.getrandbits(8) for _ in range(16))),
        version=4
    ).hex[:7]
    return f'{feature_hint or "unknown"}-session-{suffix}'

  def _feature_session_creator(self, name: str) -> Callable[
      [str | None, str | None], ContextManager['Feature']
  ]:
    """Returns a callable that returns a context manager for a feature session."""
    def _feature_session(
        session_id: str | None = None, image_id: str | None = None
    ) -> ContextManager['Feature']:
      return self.feature_session(
          name, session_id=session_id, image_id=image_id
      )
    return _feature_session

  #
  # Environment events.
  #

  def on_starting(self) -> None:
    """Called when the environment is getting started."""
    self.event_handler.on_environment_starting(self)

  def on_start(
      self,
      duration: float, error: BaseException | None = None
  ) -> None:
    """Called when the environment is started."""
    self.event_handler.on_environment_start(self, duration, error)

  def on_shutting_down(self) -> None:
    """Called when the environment is shutting down."""
    self.event_handler.on_environment_shutting_down(self)

  def on_shutdown(
      self,
      duration: float,
      error: BaseException | None = None) -> None:
    """Called when the environment is shutdown."""
    lifetime = (time.time() - self.start_time) if self.start_time else 0.0
    self.event_handler.on_environment_shutdown(self, duration, lifetime, error)


# Enable automatic conversion from str to Environment.Id.
pg.register_converter(str, AbstractEnvironment.Id, AbstractEnvironment.Id)


def _make_path_compatible(id_str: str) -> str:
  """Makes a path compatible with CNS."""
  return id_str.translate(
      str.maketrans({
          '@': '_',
          ':': '_',
          '#': '_',
          ' ': '',
          '<': '',
          '>': '',
      })
  )


def treat_as_sandbox_state_error(
    errors: Sequence[
        Type[BaseException] | tuple[Type[BaseException], str]
    ] | None = None
) -> Callable[..., Any]:
  """Decorator for Sandbox/Feature methods to convert errors to SandboxStateError.

  Args:
    errors: A sequence of exception types or tuples of (error_type, msg_regex).
      when matched, treat the error as SandboxStateError, which will lead to
      a sandbox shutdown when caught by `Sandbox.new_session()` context manager.

  Returns:
    The decorator function.
  """

  def decorator(func):
    @functools.wraps(func)
    def method_wrapper(self, *args, **kwargs) -> Any:
      """Helper function to safely execute logics in the sandbox."""

      assert isinstance(self, (Sandbox, Feature)), self
      sandbox = self.sandbox if isinstance(self, Feature) else self

      try:
        # Execute the service function.
        return func(self, *args, **kwargs)
      except BaseException as e:
        if pg.match_error(e, errors):
          state_error = SandboxStateError(
              'Sandbox encountered an unexpected error executing '
              f'`{func.__name__}` (args={args!r}, kwargs={kwargs!r}): {e}',
              sandbox=sandbox
          )
          raise state_error from e
        raise
    return method_wrapper
  return decorator


def log_activity(name: str | None = None):
  """Decorator for Sandbox/Feature methods to log sandbox/feature activity."""

  def decorator(func):
    signature = pg.typing.get_signature(func)
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

      assert isinstance(self, (Sandbox, Feature)), self
      with self.track_activity(
          name or func.__name__,
          **to_kwargs(*args, **kwargs)
      ):
        return func(self, *args, **kwargs)
    return method_wrapper
  return decorator
