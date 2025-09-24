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
"""Common base class for sandbox-based environments.

This module provides `BaseEnvironment`, a common base class for sandbox-based
environments that handles pooling, load balancing, and maintenance.

Note that:
- Environments do not have to inherit from this class, especially if features
  like pooling or load balancing are not needed.
- `BaseEnvironment` is coupled with `BaseSandbox`.
"""

import abc
import functools
import random
import threading
import time
from typing import Annotated, Any
import uuid

import langfun.core as lf
from langfun.env import base_sandbox
from langfun.env import interface
from langfun.env import load_balancers
import pyglove as pg


class BaseEnvironment(interface.Environment):
  """Common base for environments.

  The base environment provides the common functionalities for sandbox-based
  environments, such as environment pooling, load balancing, and sandbox
  maintenance.
  """

  root_dir: Annotated[
      str | None,
      (
          'The root directory for the environment for writting output files.'
          'If None, no output files will be allowed for the sandboxes.'
      )
  ] = None

  pool_size: Annotated[
      int | tuple[int, int],
      (
          'The (min_size, max_size) of the sandbox pool. If an integer, it '
          'will be used as both min and max size. If 0, sandboxes will be '
          'created on demand and shutdown when user session ends.'
      )
  ] = (0, 256)

  load_balancer: Annotated[
      load_balancers.LoadBalancer,
      (
          'The load balancer for the environment.'
      )
  ] = load_balancers.RoundRobin()

  proactive_session_setup: Annotated[
      bool,
      (
          'If True, all sandboxes will perform setup work before a user '
          'session is started. This is useful for features that need to '
          'perform heavy setup work, which could block the user thread for a '
          'long time.'
      )
  ] = True

  outage_grace_period: Annotated[
      float,
      (
          'The grace period in seconds before the environment is treated '
          'as out of service. When calling `environment.sandbox()`, '
          'wait until the grace period has passed before raising an error.'
      )
  ] = 3600.0

  outage_retry_interval: Annotated[
      float,
      (
          'The retry interval in seconds for environment outage. '
          'When calling `environment.sandbox()`, retry after the interval '
          'if the environment is out of service.'
      )
  ] = 10.0

  stats_report_interval: Annotated[
      float,
      (
          'The interval in seconds for reporting the environment stats. '
          'If 0, stats will not be reported.'
      )
  ] = 60.0

  pool_operation_max_parallelism: Annotated[
      int,
      (
          'The maximum number of threads for bringing up or shutting down '
          'sandboxes in the pool.'
      )
  ] = 256

  random_seed: Annotated[
      int | None,
      (
          'The random seed for generating session IDs with reproducibility. '
          'If None, no seed will be used.'
      )
  ] = None

  def _on_bound(self) -> None:
    super()._on_bound()

    self._status = self.Status.CREATED
    self._start_time = None
    self._sandbox_pool = []
    self._next_pooled_sandbox_id = 0
    self._random = (
        random if self.random_seed is None else random.Random(self.random_seed)
    )

    self._housekeep_thread = None
    self._offline_start_time = None

  #
  # Subclasses must implement:
  #

  @abc.abstractmethod
  def _create_sandbox(
      self,
      sandbox_id: str,
      reusable: bool,
      proactive_session_setup: bool,
  ) -> base_sandbox.BaseSandbox:
    """Creates a sandbox with the given identifier.

    Args:
      sandbox_id: The identifier for the sandbox.
      reusable: Whether the sandbox is reusable across user sessions.
      proactive_session_setup: Whether the sandbox performs session setup work
        before a user session is started.

    Returns:
      The created sandbox.

    Raises:
      interface.EnvironmentError: If environment cannot create the sandbox.
      interface.SandboxStateError: If sandbox cannot be started.
    """

  def new_session_id(self) -> str:
    """Generates a random session ID."""
    suffix = uuid.UUID(
        bytes=bytes(bytes(self._random.getrandbits(8) for _ in range(16))),
        version=4
    ).hex[:7]
    return f'session-{suffix}'

  @property
  def housekeep_counter(self) -> int:
    """Returns the housekeeping counter."""
    return self._housekeep_counter

  #
  # Subclasses can override:
  #

  def stats(self) -> dict[str, Any]:
    """Returns the stats of the environment."""
    stats_dict = {
        status.value: 0
        for status in interface.Sandbox.Status
    }
    for sandbox in self._sandbox_pool:
      stats_dict[sandbox.status.value] += 1
    return {
        'sandbox': stats_dict,
    }

  def _start(self) -> None:
    """Implementation of starting the environment."""
    if self.min_pool_size > 0:
      self._sandbox_pool = [None] * self.min_pool_size
      for i, sandbox, _ in lf.concurrent_map(
          lambda i: self._bring_up_sandbox_with_retry(
              sandbox_id=str(i), shutdown_env_upon_outage=False
          ),
          range(self.min_pool_size),
          silence_on_errors=None,
          max_workers=min(
              self.pool_operation_max_parallelism,
              self.min_pool_size
          ),
      ):
        self._sandbox_pool[i] = sandbox
    self._next_sandbox_id = len(self._sandbox_pool)
    self._housekeep_thread = threading.Thread(
        target=self._housekeep_loop, daemon=True
    )
    self._housekeep_counter = 0
    self._housekeep_thread.start()

  def _shutdown(self) -> None:
    """Implementation of shutting down the environment."""
    if (self._housekeep_thread is not None
        and threading.current_thread() is not self._housekeep_thread):
      self._housekeep_thread.join()
      self._housekeep_thread = None

    def _shutdown_sandbox(sandbox: base_sandbox.BaseSandbox) -> None:
      if sandbox is not None:
        sandbox.shutdown()

    if self._sandbox_pool:
      _ = list(
          lf.concurrent_map(
              _shutdown_sandbox,
              self._sandbox_pool,
              silence_on_errors=None,
              max_workers=min(
                  self.pool_operation_max_parallelism,
                  len(self._sandbox_pool)
              ),
          )
      )
      self._sandbox_pool = []

  #
  # Environment basics.
  #

  @property
  def sandbox_pool(self) -> list[base_sandbox.BaseSandbox]:
    """Returns the sandbox pool."""
    return self._sandbox_pool

  @functools.cached_property
  def working_dir(self) -> str | None:
    """Returns the working directory for the environment."""
    return self.id.working_dir(self.root_dir)

  @property
  def enable_pooling(self) -> bool:
    """Returns whether the environment enables pooling."""
    return self.min_pool_size > 0

  @property
  def status(self) -> interface.Environment.Status:
    """Returns whether the environment is online."""
    return self._status

  def _set_status(self, status: interface.Environment.Status) -> None:
    """Sets the status of the environment."""
    self._status = status

  @property
  def min_pool_size(self) -> int:
    """Returns the minimum size of the sandbox pool."""
    if isinstance(self.pool_size, int):
      return self.pool_size
    return self.pool_size[0]

  @property
  def max_pool_size(self) -> int:
    """Returns the maximum size of the sandbox pool."""
    if isinstance(self.pool_size, int):
      return self.pool_size
    return self.pool_size[1]

  @property
  def start_time(self) -> float | None:
    """Returns the start time of the environment."""
    return self._start_time

  @property
  def offline_duration(self) -> float:
    """Returns the offline duration of the environment."""
    if self._offline_start_time is None:
      return 0.0
    return time.time() - self._offline_start_time

  #
  # Environment lifecycle.
  #

  def start(self) -> None:
    """Starts the environment.

    Raises:
      interface.EnvironmentOutageError: If the environment is out of service.
    """
    assert self._status == self.Status.CREATED, (
        f'Environment {self.id} cannot be started because '
        f'it is in {self._status.value!r} status.'
    )

    starting_time = time.time()
    try:
      self._start()
      self._start_time = time.time()
      self._set_status(self.Status.ONLINE)
      self.on_start(duration=time.time() - starting_time)

      pg.logging.info(
          '[%s]: %s started in %.2f seconds.',
          self.id, self.__class__.__name__, time.time() - starting_time
      )
    except BaseException as e:
      self.on_start(duration=time.time() - starting_time, error=e)
      self.shutdown()
      raise e

  def shutdown(self) -> None:
    """Shuts down the environment.

    This method should not raise any exceptions.
    """
    if self._status in (
        self.Status.SHUTTING_DOWN,
        self.Status.OFFLINE,
    ):
      return

    self._set_status(self.Status.SHUTTING_DOWN)

    try:
      with pg.timeit('env.shutdown') as t:
        self._shutdown()
      self.on_shutdown()
      pg.logging.info(
          '[%s]: %s shutdown in %.2f seconds.',
          self.id, self.__class__.__name__, t.elapse
      )
    except BaseException as e:  # pylint: disable=broad-except
      self.on_shutdown(error=e)
      raise e

  #
  # Environment operations.
  #

  def acquire(self) -> base_sandbox.BaseSandbox:
    """Acquires a sandbox from the environment.

    Returns:
      The acquired sandbox.

    Raises:
      interface.EnvironmentOutageError: If the environment is offline and the
        grace period has passed.
      interface.EnvironmentOverloadError: If the max pool size is reached and
        the grace period has passed.
    """

    if not self.is_online:
      raise interface.EnvironmentOutageError(
          f'Environment {self.id} is not alive.',
          environment=self,
          offline_duration=self.offline_duration,
      )

    if not self.enable_pooling:
      return self._bring_up_sandbox_with_retry(
          sandbox_id=str(self._increment_sandbox_id()),
          set_acquired=True,
      )

    allocation_start_time = time.time()
    while True:
      try:
        # We only append or replace items in the sandbox pool, therefore
        # there is no need to lock the pool.
        return self.load_balancer.acquire(self._sandbox_pool)
      except IndexError:
        if len(self._sandbox_pool) == self.max_pool_size:
          if time.time() - allocation_start_time > self.outage_grace_period:
            raise interface.EnvironmentOverloadError(  # pylint: disable=raise-missing-from
                environment=self
            )
          time.sleep(1)
        else:
          try:
            sandbox = self._bring_up_sandbox(
                sandbox_id=str(self._increment_sandbox_id()), set_acquired=True,
            )
            # Append is atomic and does not require locking.
            self._sandbox_pool.append(sandbox)
            self._offline_start_time = None
            return sandbox
          except (
              interface.EnvironmentError, interface.SandboxStateError
          ) as ex:
            self._report_outage_or_wait(ex)

  def _bring_up_sandbox(
      self,
      sandbox_id: str,
      set_acquired: bool = False,
  ) -> base_sandbox.BaseSandbox:
    """Brings up a new sandbox."""
    sandbox = self._create_sandbox(
        sandbox_id=sandbox_id,
        reusable=self.enable_pooling,
        proactive_session_setup=self.proactive_session_setup,
    )
    for handler in self.event_handlers:
      sandbox.add_event_handler(handler)
    sandbox.start()
    if set_acquired:
      sandbox.set_acquired()
    return sandbox

  def _bring_up_sandbox_with_retry(
      self,
      sandbox_id: str,
      set_acquired: bool = False,
      shutdown_env_upon_outage: bool = True,
  ) -> base_sandbox.BaseSandbox:
    """Brings up a new sandbox with retry until grace period is passed.

    Args:
      sandbox_id: The ID of the sandbox to bring up.
      set_acquired: If True, the sandbox will be marked as acquired.
      shutdown_env_upon_outage: Whether to shutdown the environment when the
        outage grace period is passed.

    Returns:
      A new sandbox ready to use.

    Raises:
      interface.EnvironmentOutageError: If the environment is offline and the
        grace period has passed.
    """
    while True:
      try:
        return self._bring_up_sandbox(
            sandbox_id=sandbox_id, set_acquired=set_acquired
        )
      except (interface.EnvironmentError, interface.SandboxStateError) as e:
        self._report_outage_or_wait(e, shutdown_env_upon_outage)

  def _increment_sandbox_id(self) -> int:
    """Returns the next pooled sandbox ID."""
    x = self._next_sandbox_id
    self._next_sandbox_id += 1
    return x

  def _report_outage_or_wait(
      self,
      error: interface.SandboxStateError,
      shutdown_env_upon_outage: bool = True
  ):
    """Raises error if the grace period has passed or wait for retry."""
    if self._offline_start_time is None:
      self._offline_start_time = time.time()
    if self.offline_duration > self.outage_grace_period:
      if shutdown_env_upon_outage:
        self.shutdown()
      raise interface.EnvironmentOutageError(
          environment=self,
          offline_duration=self.offline_duration,
      ) from error
    time.sleep(self.outage_retry_interval)

  #
  # Environment maintenance loop.
  #

  def _housekeep_loop(self) -> None:
    """Housekeeping loop for the environment."""
    pg.logging.info(
        '[%s]: %s maintenance thread started.', self.id, self.__class__.__name__
    )
    stats_report_time = time.time()
    while self._status not in (self.Status.SHUTTING_DOWN, self.Status.OFFLINE):
      housekeep_start_time = time.time()
      if time.time() - stats_report_time > self.stats_report_interval:
        pg.logging.info(
            '[%s] %s stats: %s.',
            self.id, self.__class__.__name__, self.stats()
        )
        stats_report_time = time.time()

      dead_pool_indices = [
          i for i, s in enumerate(self._sandbox_pool)
          if s.status == interface.Sandbox.Status.OFFLINE
      ]
      if dead_pool_indices and not self._replace_dead_sandboxes(
          dead_pool_indices
      ):
        self.shutdown()
        self._housekeep_counter += 1
        self.on_housekeep(
            time.time() - housekeep_start_time,
            interface.EnvironmentOutageError(
                environment=self,
                offline_duration=self.offline_duration
            )
        )
        break
      self._housekeep_counter += 1
      self.on_housekeep(time.time() - housekeep_start_time)
      time.sleep(1)

  def _replace_dead_sandboxes(self, dead_pool_indices: list[int]) -> bool:
    """Replaces a dead sandbox with a new one.

    Args:
      dead_pool_indices: The indices of the dead sandboxes to replace.

    Returns:
      Whether all of the dead sandboxes are replaced successfully.

    Raises:
      interface.EnvironmentOutageError: If the XBox sandboxes cannot be created
        within the wait time specified by `xbox_outage_grace_period`.
    """
    pg.logging.warning(
        '[%s]: %s maintenance: '
        'Replacing %d dead sandbox(es) with new ones...',
        self.id,
        self.__class__.__name__,
        len(dead_pool_indices),
    )
    def _replace(i: int):
      self._sandbox_pool[i] = self._bring_up_sandbox_with_retry(
          str(i), shutdown_env_upon_outage=False
      )

    # TODO(daiyip): Consider to loose the condition to allow some dead
    # sandboxes to be replaced successfully.
    return not any([
        error for _, _, error in lf.concurrent_map(
            _replace, dead_pool_indices,
            max_workers=min(
                self.pool_operation_max_parallelism,
                len(dead_pool_indices)
            ),
        )
    ])

  #
  # Event handlers subclasses can override.
  #

  def on_start(
      self,
      duration: float, error: BaseException | None = None
  ) -> None:
    """Called when the environment is started."""
    for handler in self.event_handlers:
      handler.on_environment_start(self, duration, error)

  def on_housekeep(
      self,
      duration: float,
      error: BaseException | None = None
  ) -> None:
    """Called when the environment finishes a round of housekeeping."""
    housekeep_counter = self.housekeep_counter
    for handler in self.event_handlers:
      handler.on_environment_housekeep(self, housekeep_counter, duration, error)

  def on_shutdown(self, error: BaseException | None = None) -> None:
    """Called when the environment is shutdown."""
    lifetime = (time.time() - self.start_time) if self.start_time else 0.0
    for handler in self.event_handlers:
      handler.on_environment_shutdown(self, lifetime, error)
