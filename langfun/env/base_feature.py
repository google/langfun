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
"""Common base class for environment features.

This module provides an base class `BaseFeature` for environment features,
which provides event handlers for the feature lifecycle events, which can be
overridden by subclasses to provide custom behaviors. Please note that this base
class is intended to provide a convenient way to implement features, and not
all feature implementations need to subclass it. Also `BaseFeature` is not
coupled with `BaseEnvironment` and `BaseSandbox`, and is expected to work with
the `Environment` and `Sandbox` interfaces directly.
"""

import contextlib
import functools
import os
import threading
import time
from typing import Any, Callable, Iterator

from langfun.env import interface
import pyglove as pg


class BaseFeature(interface.Feature):
  """Common base class for environment features."""

  #
  # Subclasses can override:
  #

  def _setup(self) -> None:
    """Subclasses can override this for custom setup.

    NOTE: always call super()._setup() at the beginning of the implementation.
    """

  def _teardown(self) -> None:
    """Subclasses can override this for custom teardown.

    NOTE: always call super()._teardown() at the end of the implementation.
    """

  def _setup_session(self) -> None:
    """Subclasses can override this for custom setup session.

    NOTE: always call super()._setup_session() at the beginning of the
    implementation.
    """

  def _teardown_session(self) -> None:
    """Subclasses can override this for custom teardown session.

    NOTE: always call super()._teardown_session() at the end of the
    implementation.
    """

  def _housekeep(self) -> None:
    """Performs housekeeping for the feature.

    NOTE: always call super()._housekeep() at the beginning of the
    implementation.
    """

  #
  # Init and properties
  #

  def _on_bound(self) -> None:
    """Called when the feature is bound."""
    super()._on_bound()
    self._sandbox = None
    self._housekeep_counter = 0

    # Fields applicable only to non-sandbox-based features.
    self._is_online = False
    self._housekeep_thread = None
    self._housekeep_event = threading.Event()
    self._offline_start_time = None

  def _on_parent_change(
      self, old_parent: pg.Symbolic | None, new_parent: pg.Symbolic | None
  ) -> None:
    """Called when the feature is bound."""
    super()._on_parent_change(old_parent, new_parent)
    self.__dict__.pop('name', None)
    self.__dict__.pop('environment', None)

  @functools.cached_property
  def environment(self) -> interface.AbstractEnvironment | None:
    """Returns the environment that the feature is running in."""
    if self._sandbox is not None:
      return self._sandbox.environment
    env = self.sym_ancestor(
        lambda v: isinstance(v, interface.AbstractEnvironment)
    )
    return env

  @property
  def sandbox(self) -> interface.Sandbox | None:
    """Returns the sandbox that the feature is running in."""
    assert (
        self._sandbox is not None or not self.is_sandbox_based
    ), 'Feature has not been set up yet.'
    return self._sandbox

  @property
  def event_handler(self) -> interface.EventHandler:
    if hasattr(self, '_event_handler_ref'):
      return self._event_handler_ref
    return super().event_handler

  @property
  def is_online(self) -> bool:
    """Returns True if the feature is online."""
    if self.is_sandbox_based:
      return self.sandbox.is_online
    return self._is_online

  @property
  def offline_duration(self) -> float:
    """Returns the offline duration of the feature."""
    if self._offline_start_time is None:
      return 0.0
    return time.time() - self._offline_start_time

  @property
  def working_dir(self) -> str | None:
    """Returns the working directory of the feature."""
    if self.is_sandbox_based:
      sandbox_workdir = self.sandbox.working_dir
      if sandbox_workdir is None:
        return None
      return os.path.join(sandbox_workdir, self.name)
    if self.environment is None or self.environment.working_dir is None:
      return None
    return os.path.join(self.environment.working_dir, self.name)

  #
  # Setup and teardown of the feature.
  #

  def _do(
      self,
      action: Callable[[], None],
      event_handler: Callable[..., None],
  ) -> None:
    """Triggers an event handler."""
    error = None
    start_time = time.time()
    try:
      action()
    except BaseException as e:  # pylint: disable=broad-except
      error = e
      raise
    finally:
      event_handler(duration=time.time() - start_time, error=error)

  def setup(self, sandbox: interface.Sandbox | None = None) -> None:
    """Sets up the feature."""

    def _setup():
      try:
        self._setup()
        self._is_online = True
        if not self.is_sandbox_based and self.housekeep_interval is not None:
          self._housekeep_thread = threading.Thread(
              target=self._housekeep_loop, daemon=True
          )
          self._housekeep_thread.start()
      except BaseException as e:  # pylint: disable=broad-except
        if isinstance(
            e, (interface.EnvironmentError, interface.SandboxStateError)
        ):
          raise
        raise interface.FeatureSetupError(feature=self) from e

    self._sandbox = sandbox
    self._do(_setup, self.on_setup)

  def teardown(self) -> None:
    """Tears down the feature."""

    def _teardown():
      self._is_online = False
      if self._housekeep_event is not None:
        self._housekeep_event.set()
      if self._housekeep_thread is not None:
        self._housekeep_thread.join()
        self._housekeep_thread = None
      try:
        self._teardown()
      except BaseException as e:  # pylint: disable=broad-except
        raise interface.FeatureTeardownError(feature=self) from e

    self._do(_teardown, event_handler=self.on_teardown)

  def setup_session(self) -> None:
    """Sets up the feature for a user session."""
    self._do(self._setup_session, event_handler=self.on_setup_session)

  def teardown_session(self) -> None:
    """Teardowns the feature for a user session."""
    self._do(self._teardown_session, self.on_teardown_session)

  #
  # Housekeeping operation and loop.
  #

  def housekeep(self) -> None:
    """Performs housekeeping for the feature."""
    try:
      self._do(self._housekeep, self.on_housekeep)
    finally:
      self._housekeep_counter += 1

  def _housekeep_loop(self) -> None:
    """Housekeeping loop for the feature."""
    assert not self.is_sandbox_based and self.housekeep_interval
    while self._is_online:
      try:
        self.housekeep()
        self._offline_start_time = None
      except BaseException:  # pylint: disable=broad-except
        if self._offline_start_time is None:
          self._offline_start_time = time.time()
        if time.time() - self._offline_start_time > self.outage_grace_period:
          self._is_online = False
          break
      self._housekeep_event.wait(self.housekeep_interval)

  #
  # Event handlers subclasses can override.
  #

  def on_setup(
      self, duration: float, error: BaseException | None = None
  ) -> None:
    """Called when the feature is setup."""
    self.event_handler.on_feature_setup(
        feature=self, duration=duration, error=error
    )

  def on_teardown(
      self, duration: float, error: BaseException | None = None
  ) -> None:
    """Called when the feature is teardown."""
    self.event_handler.on_feature_teardown(
        feature=self, duration=duration, error=error
    )

  def on_housekeep(
      self, duration: float, error: BaseException | None = None, **kwargs
  ) -> None:
    """Called when the feature has done housekeeping."""
    self.event_handler.on_feature_housekeep(
        feature=self,
        counter=self._housekeep_counter,
        duration=duration,
        error=error,
        **kwargs,
    )

  def on_setup_session(
      self,
      duration: float,
      error: BaseException | None = None,
  ) -> None:
    """Called when the feature is setup for a user session."""
    self.event_handler.on_feature_setup_session(
        feature=self, session_id=self.session_id, duration=duration, error=error
    )

  def on_teardown_session(
      self,
      duration: float,
      error: BaseException | None = None,
  ) -> None:
    """Called when the feature is teardown for a user session."""
    self.event_handler.on_feature_teardown_session(
        feature=self, session_id=self.session_id, duration=duration, error=error
    )

  def on_activity(
      self,
      name: str,
      duration: float,
      error: BaseException | None = None,
      **kwargs,
  ) -> None:
    """Called when a sandbox activity is performed."""
    self.event_handler.on_feature_activity(
        name=f'{self.name}.{name}',
        feature=self,
        session_id=self.session_id,
        duration=duration,
        error=error,
        **kwargs,
    )

  @contextlib.contextmanager
  def track_activity(self, name: str, **kwargs: Any) -> Iterator[None]:
    """Context manager that tracks a feature activity."""
    start_time = time.time()
    error = None
    try:
      yield None
    except BaseException as e:  # pylint: disable=broad-except
      error = e
      raise
    finally:
      self.on_activity(
          name=name, duration=time.time() - start_time, error=error, **kwargs
      )
