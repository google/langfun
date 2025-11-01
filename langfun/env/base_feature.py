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
import re
import time
from typing import Annotated, Any, Callable, Iterator

from langfun.env import interface
import pyglove as pg


class BaseFeature(interface.Feature):
  """Common base class for environment features."""

  is_sandbox_based: Annotated[
      bool,
      'Whether the feature is sandbox-based.'
  ] = True

  applicable_images: Annotated[
      list[str],
      (
          'A list of regular expressions for image IDs which enable '
          'this feature. By default, all images are enabled.'
      )
  ] = ['.*']

  housekeep_interval: Annotated[
      float | None,
      'Interval in seconds for feature housekeeping.'
  ] = None

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

  @functools.cached_property
  def name(self) -> str:
    """Returns the name of the feature."""
    assert isinstance(self.sym_parent, dict), 'Feature is not put into a dict.'
    return self.sym_path.key

  def _on_parent_change(
      self,
      old_parent: pg.Symbolic | None,
      new_parent: pg.Symbolic | None
  ) -> None:
    """Called when the feature is bound."""
    super()._on_parent_change(old_parent, new_parent)
    self.__dict__.pop('name', None)

  @functools.cached_property
  def environment(self) -> interface.Environment:
    """Returns the environment that the feature is running in."""
    if self._sandbox is not None:
      return self._sandbox.environment
    env = self.sym_ancestor(lambda v: isinstance(v, interface.Environment))
    assert env is not None, 'Feature is not put into an environment.'
    return env

  @property
  def sandbox(self) -> interface.Sandbox | None:
    """Returns the sandbox that the feature is running in."""
    assert self._sandbox is not None or not self.is_sandbox_based, (
        'Feature has not been set up yet.'
    )
    return self._sandbox

  @property
  def working_dir(self) -> str | None:
    """Returns the working directory of the feature."""
    sandbox_workdir = self.sandbox.working_dir
    if sandbox_workdir is None:
      return None
    return os.path.join(sandbox_workdir, self.name)

  def is_applicable(self, image_id: str) -> bool:
    """Returns True if the feature is applicable to the given image."""
    return any(
        re.fullmatch(regex, image_id) for regex in self.applicable_images
    )

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
    self._sandbox = sandbox
    self._do(self._setup, self.on_setup)

  def teardown(self) -> None:
    """Tears down the feature."""
    self._do(self._teardown, event_handler=self.on_teardown)

  def setup_session(self) -> None:
    """Sets up the feature for a user session."""
    self._do(self._setup_session, event_handler=self.on_setup_session)

  def teardown_session(self) -> None:
    """Teardowns the feature for a user session."""
    self._do(self._teardown_session, self.on_teardown_session)

  #
  # Housekeeping.
  #

  def housekeep(self) -> None:
    """Performs housekeeping for the feature."""
    try:
      self._do(self._housekeep, self.on_housekeep)
    finally:
      self._housekeep_counter += 1

  #
  # Event handlers subclasses can override.
  #

  def on_setup(
      self,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when the feature is setup."""
    self.environment.event_handler.on_feature_setup(
        feature=self,
        duration=duration,
        error=error
    )

  def on_teardown(
      self,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when the feature is teardown."""
    self.environment.event_handler.on_feature_teardown(
        feature=self,
        duration=duration,
        error=error
    )

  def on_housekeep(
      self,
      duration: float,
      error: BaseException | None = None,
      **kwargs
  ) -> None:
    """Called when the feature has done housekeeping."""
    self.environment.event_handler.on_feature_housekeep(
        feature=self,
        counter=self._housekeep_counter,
        duration=duration,
        error=error,
        **kwargs
    )

  def on_setup_session(
      self,
      duration: float,
      error: BaseException | None = None,
  ) -> None:
    """Called when the feature is setup for a user session."""
    self.environment.event_handler.on_feature_setup_session(
        feature=self,
        session_id=self.session_id,
        duration=duration,
        error=error
    )

  def on_teardown_session(
      self,
      duration: float,
      error: BaseException | None = None,
  ) -> None:
    """Called when the feature is teardown for a user session."""
    self.environment.event_handler.on_feature_teardown_session(
        feature=self,
        session_id=self.session_id,
        duration=duration,
        error=error
    )

  def on_activity(
      self,
      name: str,
      duration: float,
      error: BaseException | None = None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    self.environment.event_handler.on_feature_activity(
        name=f'{self.name}.{name}',
        feature=self,
        session_id=self.session_id,
        duration=duration,
        error=error,
        **kwargs
    )

  @contextlib.contextmanager
  def track_activity(
      self,
      name: str,
      **kwargs: Any
  ) -> Iterator[None]:
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
          name=name,
          duration=time.time() - start_time,
          error=error,
          **kwargs
      )
