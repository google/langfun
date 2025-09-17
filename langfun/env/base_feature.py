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
"""Common base class for sandbox-based features.

This module provides an base class `BaseFeature` for sandbox-based features,
which provides event handlers for the feature lifecycle events, which can be
overridden by subclasses to provide custom behaviors. Please note that this base
class is intended to provide a convenient way to implement features, and not
all feature implementations need to subclass it. Also `BaseFeature` is not
coupled with `BaseEnvironment` and `BaseSandbox`, and is expected to work with
the `Environment` and `Sandbox` interfaces directly.
"""

import functools
from typing import Annotated

from langfun.env import interface
import pyglove as pg


class BaseFeature(interface.Feature):
  """Common base class for sandbox-based features."""

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

  def _setup_session(self, session_id: str) -> None:
    """Subclasses can override this for custom setup session.

    Args:
      session_id: The session ID.

    NOTE: always call super()._setup_session() at the beginning of the
    implementation.
    """

  def _teardown_session(self, session_id: str) -> None:
    """Subclasses can override this for custom teardown session.

    Args:
      session_id: The session ID.

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

  @property
  def sandbox(self) -> interface.Sandbox:
    """Returns the sandbox that the feature is running in."""
    assert self._sandbox is not None, 'Feature has not been set up yet.'
    return self._sandbox

  #
  # Setup and teardown of the feature.
  #

  def setup(self, sandbox: interface.Sandbox) -> None:
    """Sets up the feature."""
    self._sandbox = sandbox
    interface.call_with_event(
        action=self._setup,
        event_handler=self.on_setup,
    )

  def teardown(self) -> None:
    """Tears down the feature."""
    # If a sandbox is down during setting up, feature.shutdown might be called
    # before the feature is setup. In this case, we don't need to teardown the
    # feature.
    if self._sandbox is None:
      return

    interface.call_with_event(
        action=self._teardown,
        event_handler=self.on_teardown,
    )

  def setup_session(self, session_id: str) -> None:
    """Sets up the feature for a user session."""
    interface.call_with_event(
        action=self._setup_session,
        event_handler=self.on_session_setup,
        action_kwargs={'session_id': session_id},
        event_handler_kwargs={'session_id': session_id},
    )

  def teardown_session(self, session_id: str) -> None:
    """Teardowns the feature for a user session."""
    interface.call_with_event(
        action=self._teardown_session,
        event_handler=self.on_session_teardown,
        action_kwargs={'session_id': session_id},
        event_handler_kwargs={'session_id': session_id},
    )

  #
  # Housekeeping.
  #

  def housekeep(self) -> None:
    """Performs housekeeping for the feature."""
    interface.call_with_event(
        action=self._housekeep,
        event_handler=self.on_housekeep,
    )
