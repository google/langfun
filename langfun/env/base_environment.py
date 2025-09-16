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
- `BaseEnvironment` is not required to work with `BaseSandbox`.
"""

import abc
import functools
import threading
import time
from typing import Annotated, Any

import langfun.core as lf
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
  ] = 1

  load_balancer: Annotated[
      load_balancers.LoadBalancer,
      (
          'The load balancer for the environment.'
      )
  ] = load_balancers.RoundRobin()

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

  def _on_bound(self) -> None:
    super()._on_bound()

    self._alive = False
    self._start_time = None
    self._sandbox_pool = []
    self._next_pooled_sandbox_id = 0

    self._maintenance_thread = None
    self._offline_start_time = None

  #
  # Subclasses must implement:
  #

  @abc.abstractmethod
  def _create_sandbox(
      self,
      sandbox_id: str,
      reusable: bool
  ) -> interface.Sandbox:
    """Creates a sandbox with the given identifier.

    Args:
      sandbox_id: The identifier for the sandbox.
      reusable: Whether the sandbox is reusable across user sessions.

    Returns:
      The created sandbox.

    Raises:
      interface.EnvironmentError: If environment cannot create the sandbox.
      interface.SandboxStateError: If sandbox cannot be started.
    """

  #
  # Subclasses can override:
  #

  def stats(self) -> dict[str, Any]:
    """Returns the stats of the environment."""
    num_busy = 0
    num_free = 0
    num_dead = 0

    for sandbox in self._sandbox_pool:
      if sandbox.is_alive:
        if sandbox.is_busy:
          num_busy += 1
        else:
          num_free += 1
      else:
        num_dead += 1

    return {
        'sandbox': {
            'num_total': len(self._sandbox_pool),
            'num_busy': num_busy,
            'num_free': num_free,
            'num_dead': num_dead,
        },
    }

  def _start(self) -> None:
    """Implementation of starting the environment."""
    if self.min_pool_size > 0:
      self._sandbox_pool = [
          sandbox
          for _, sandbox, _ in lf.concurrent_map(
              lambda i: self._bring_up_sandbox_with_retry(sandbox_id=str(i)),
              range(self.min_pool_size),
              silence_on_errors=None,
              max_workers=min(
                  self.pool_operation_max_parallelism,
                  self.min_pool_size
              ),
          )
      ]
    self._next_sandbox_id = len(self._sandbox_pool)
    self._alive = True
    self._maintenance_thread = threading.Thread(
        target=self._maintenance_loop, daemon=True
    )
    self._maintenance_count = 0
    self._maintenance_thread.start()

  def _shutdown(self) -> None:
    """Implementation of shutting down the environment."""
    if (self._maintenance_thread is not None
        and threading.current_thread() is not self._maintenance_thread):
      self._maintenance_thread.join()
      self._maintenance_thread = None

    def _shutdown_sandbox(sandbox: interface.Sandbox) -> None:
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
  def sandbox_pool(self) -> list[interface.Sandbox]:
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
  def is_alive(self) -> bool:
    """Returns whether the environment is alive."""
    return self._alive

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
    assert not self._alive
    def _start_impl():
      with pg.timeit('env.start') as t:
        self._start()
      self._start_time = time.time()
      pg.logging.info(
          '[%s]: %s started in %.2f seconds.',
          self.id, self.__class__.__name__, t.elapse
      )
    interface.call_with_event(
        _start_impl, self.on_start,
    )

  def shutdown(self) -> None:
    """Shuts down the environment.

    This method should not raise any exceptions.
    """
    if not self._alive:
      return

    self._alive = False
    def _shutdown_impl():
      pg.logging.info(
          '[%s]: Shutting down %s...', self.id, self.__class__.__name__
      )
      with pg.timeit('env.shutdown') as t:
        self._shutdown()
      pg.logging.info(
          '[%s]: %s shutdown in %.2f seconds.',
          self.id, self.__class__.__name__, t.elapse
      )

    interface.call_with_event(
        _shutdown_impl, self.on_shutdown,
    )

  #
  # Environment operations.
  #

  def acquire(self) -> interface.Sandbox:
    """Acquires a sandbox from the environment.

    Returns:
      The acquired sandbox.

    Raises:
      interface.EnvironmentOutageError: If the environment is offline and the
        grace period has passed.
      interface.EnvironmentOverloadError: If the max pool size is reached and
        the grace period has passed.
    """

    if not self._alive:
      raise interface.EnvironmentOutageError(
          f'Environment {self.id} is not alive.',
          environment=self,
          offline_duration=self.offline_duration,
      )

    if not self.enable_pooling:
      return self._bring_up_sandbox_with_retry(
          sandbox_id=str(self._increment_sandbox_id()),
          set_pending=True,
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
            sandbox = self._create_sandbox(
                sandbox_id=str(self._increment_sandbox_id()),
                reusable=self.enable_pooling,
            )
            sandbox.start()
            sandbox.set_pending()

            # Append is atomic and does not require locking.
            self._sandbox_pool.append(sandbox)
            self._offline_start_time = None
            return sandbox
          except (
              interface.EnvironmentError, interface.SandboxStateError
          ) as ex:
            self._report_outage_or_wait(ex)

  def _bring_up_sandbox_with_retry(
      self,
      sandbox_id: str,
      set_pending: bool = False,
      shutdown_env_upon_outage: bool = True,
  ) -> interface.Sandbox:
    """Brings up a new sandbox with retry until grace period is passed.

    Args:
      sandbox_id: The ID of the sandbox to bring up.
      set_pending: Whether to mark the sandbox as pending.
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
        sandbox = self._create_sandbox(
            sandbox_id=sandbox_id,
            reusable=self.enable_pooling
        )
        sandbox.start()
        if set_pending:
          sandbox.set_pending()
        return sandbox
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

  def _maintenance_loop(self) -> None:
    """Maintains the server pool."""
    pg.logging.info(
        '[%s]: %s maintenance thread started.', self.id, self.__class__.__name__
    )
    stats_report_time = time.time()
    while self._alive:
      if time.time() - stats_report_time > self.stats_report_interval:
        pg.logging.info(
            '[%s] %s stats: %s.',
            self.id, self.__class__.__name__, self.stats()
        )
        stats_report_time = time.time()

      dead_pool_indices = [
          i for i, s in enumerate(self._sandbox_pool) if not s.is_alive
      ]
      if dead_pool_indices and not self._replace_dead_sandboxes(
          dead_pool_indices
      ):
        self.shutdown()
        self._maintenance_count += 1
        break
      self._maintenance_count += 1
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

    return not any([
        error for _, _, error in lf.concurrent_map(
            _replace, dead_pool_indices,
            max_workers=min(
                self.pool_operation_max_parallelism,
                len(dead_pool_indices)
            ),
        )
    ])
