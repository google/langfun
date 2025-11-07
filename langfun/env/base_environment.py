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
import collections
import functools
import random
import re
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

  image_ids: Annotated[
      list[str],
      (
          'A list of static image IDs served by the environment. '
      )
  ]

  supports_dynamic_image_loading: Annotated[
      bool,
      (
          'Whether the environment supports dynamic loading of images which is '
          'not included in the `image_ids`. `image_ids` could coexist with '
          'dynamic image loading, which allows users to specify an image id '
          'that is not included in the `image_ids`.'
      )
  ] = False

  root_dir: Annotated[
      str | None,
      (
          'The root directory for the environment for writting output files.'
          'If None, no output files will be allowed for the sandboxes.'
      )
  ] = None

  pool_size: Annotated[
      int | tuple[int, int] | dict[str, int | tuple[int, int]],
      (
          'The (min_size, max_size) of the sandbox pool. If an integer, it '
          'will be used as both min and max size. If 0, all sandboxes will be '
          'created on demand and shutdown when user session ends. If a dict, '
          'users could configure the pool size based on image IDs. The keys '
          'are regular expressions for image IDs, and the values are '
          '(min_size, max_size) tuples. For dynamic image IDs, min_size will '
          'ignored while max_size will be honored.'
      )
  ] = (0, 256)

  load_balancer: Annotated[
      load_balancers.LoadBalancer,
      (
          'The load balancer for the environment to acquire sandboxes.'
      )
  ] = load_balancers.RoundRobin()

  sandbox_keepalive_interval: Annotated[
      float | None,
      (
          'The interval in seconds to send keepalive pings to sandboxes. '
          'If None, sandbox keepalive is disabled. Please note that sandbox '
          'keepalive is different from feature housekeeping. Usually sandbox '
          'keepalive and feature housekeeping are different operations.'
      )
  ] = None

  proactive_session_setup: Annotated[
      bool,
      (
          'If True, all sandboxes will perform setup work before a user '
          'session is started. This is useful for features that need to '
          'perform heavy setup work, which could block the user thread for a '
          'long time.'
      )
  ] = True

  event_handler: Annotated[
      interface.EventHandler,
      (
          'User handler for the environment events.'
          'By default, the no-op event handler is used.'
      )
  ] = interface.EventHandler()

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

  housekeep_interval: Annotated[
      float,
      (
          'The interval in seconds for environment housekeeping. It recycles '
          'the dead sandboxes in the pool. This interval is the minimal time '
          'to detect outage while there is no request to obtain new sandboxes.'
          'This is applicable only when the environment enables pooling.'
      )
  ] = 10.0

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

    self._sandbox_pool: dict[str, list[base_sandbox.BaseSandbox]] = (
        collections.defaultdict(list)
    )
    self._next_sandbox_id: dict[str, int] = collections.defaultdict(int)
    self._random = (
        random if self.random_seed is None else random.Random(self.random_seed)
    )
    self._housekeep_thread = None
    self._offline_start_time = None
    self._non_sandbox_based_features_with_setup_called = set()

    # Check image IDs and feature requirements.
    self._check_image_ids()
    self._check_feature_requirements()

  def _check_image_ids(self) -> None:
    """Checks image ids. Subclass could override this method."""

  def _check_feature_requirements(self) -> None:
    """Checks if the image ID is supported by the feature."""
    if self.supports_dynamic_image_loading:
      return
    for name, feature in self.features.items():
      if not feature.is_sandbox_based or any(
          feature.is_applicable(image_id) for image_id in self.image_ids
      ):
        continue
      raise ValueError(
          f'Feature {name!r} is not applicable to all available images: '
          f'{self.image_ids!r}. '
          f'Applicable images: {feature.applicable_images}.'
      )

  #
  # Subclasses must implement:
  #

  @abc.abstractmethod
  def _create_sandbox(
      self,
      image_id: str,
      sandbox_id: str,
      reusable: bool,
      proactive_session_setup: bool,
      keepalive_interval: float | None,
  ) -> base_sandbox.BaseSandbox:
    """Creates a sandbox with the given identifier.

    Args:
      image_id: The image ID to use for the sandbox.
      sandbox_id: The identifier for the sandbox.
      reusable: Whether the sandbox is reusable across user sessions.
      proactive_session_setup: Whether the sandbox performs session setup work
        before a user session is started.
      keepalive_interval: Interval to ping the sandbox for keeping it alive.
        If None, the sandbox will not be pinged.

    Returns:
      The created sandbox.

    Raises:
      interface.EnvironmentError: If environment cannot create the sandbox.
      interface.SandboxStateError: If sandbox cannot be started.
    """

  def new_session_id(self, feature_hint: str | None = None) -> str:
    """Generates a random session ID."""
    suffix = uuid.UUID(
        bytes=bytes(bytes(self._random.getrandbits(8) for _ in range(16))),
        version=4
    ).hex[:7]
    return f'{feature_hint or "unknown"}-session-{suffix}'

  @property
  def housekeep_counter(self) -> int:
    """Returns the housekeeping counter."""
    return self._housekeep_counter

  #
  # Subclasses can override:
  #

  def stats(self) -> dict[str, Any]:
    """Returns the stats of the environment."""
    stats_by_image_id = {}
    for image_id, sandboxes in self._sandbox_pool.items():
      stats_dict = {
          status.value: 0
          for status in interface.Sandbox.Status
      }
      for sandbox in sandboxes:
        stats_dict[sandbox.status.value] += 1
      stats_by_image_id[image_id] = stats_dict
    return {
        'sandbox': stats_by_image_id,
    }

  def _start(self) -> None:
    """Implementation of starting the environment."""
    sandbox_startup_infos = []
    self._non_sandbox_based_features_with_setup_called.clear()
    # Setup all non-sandbox-based features.
    for feature in self.non_sandbox_based_features():
      self._non_sandbox_based_features_with_setup_called.add(feature.name)
      feature.setup(sandbox=None)

    # Setup sandbox pools.
    for image_id in self.image_ids:
      next_sandbox_id = 0
      if self.enable_pooling(image_id):
        min_pool_size = self.min_pool_size(image_id)
        for i in range(min_pool_size):
          sandbox_startup_infos.append((image_id, i))
        self._sandbox_pool[image_id] = [None] * min_pool_size
        next_sandbox_id = min_pool_size
      self._next_sandbox_id[image_id] = next_sandbox_id

    def _start_sandbox(sandbox_startup_info) -> None:
      image_id, index = sandbox_startup_info
      self._sandbox_pool[image_id][index] = self._bring_up_sandbox_with_retry(
          image_id=image_id,
          sandbox_id=f'{index}:0',
          shutdown_env_upon_outage=False
      )

    if sandbox_startup_infos:
      # Pre-allocate the sandbox pool before usage.
      _ = list(
          lf.concurrent_map(
              _start_sandbox,
              sandbox_startup_infos,
              silence_on_errors=None,
              max_workers=min(
                  self.pool_operation_max_parallelism,
                  len(sandbox_startup_infos)
              ),
          )
      )

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

    # Teardown all non-sandbox-based features.
    for feature in self.non_sandbox_based_features():
      if feature.name in self._non_sandbox_based_features_with_setup_called:
        try:
          feature.teardown()
        except BaseException:   # pylint: disable=broad-except
          pass

    # Shutdown sandbox pools.
    if self._sandbox_pool:
      sandboxes = []
      for sandbox in self._sandbox_pool.values():
        sandboxes.extend(sandbox)
      self._sandbox_pool = {}

      if sandboxes:
        def _shutdown_sandbox(sandbox: base_sandbox.BaseSandbox) -> None:
          if sandbox is not None:
            sandbox.shutdown()

        _ = list(
            lf.concurrent_map(
                _shutdown_sandbox,
                sandboxes,
                silence_on_errors=None,
                max_workers=min(
                    self.pool_operation_max_parallelism,
                    len(sandboxes)
                ),
            )
        )

  #
  # Environment basics.
  #

  @property
  def sandbox_pool(self) -> dict[str, list[base_sandbox.BaseSandbox]]:
    """Returns the sandbox pool."""
    return self._sandbox_pool

  @functools.cached_property
  def working_dir(self) -> str | None:
    """Returns the working directory for the environment."""
    return self.id.working_dir(self.root_dir)

  @property
  def status(self) -> interface.Environment.Status:
    """Returns whether the environment is online."""
    return self._status

  def _set_status(self, status: interface.Environment.Status) -> None:
    """Sets the status of the environment."""
    self._status = status

  def enable_pooling(self, image_id: str) -> bool:
    """Returns whether the environment enables pooling."""
    return self.max_pool_size(image_id) > 0

  def min_pool_size(self, image_id: str) -> int:
    """Returns the minimum size of the sandbox pool."""
    return self._pool_size(image_id)[0]

  def max_pool_size(self, image_id: str) -> int:
    """Returns the maximum size of the sandbox pool."""
    return self._pool_size(image_id)[1]

  def _pool_size(self, image_id: str) -> tuple[int, int]:
    """Returns the minimum and maximum size of the sandbox pool."""
    if isinstance(self.pool_size, dict):
      if image_id in self.pool_size:
        pool_size = self.pool_size[image_id]
      else:
        for k, v in self.pool_size.items():
          if re.match(k, image_id):
            pool_size = v
            break
        else:
          # Default pool size is 0 and 256.
          pool_size = (0, 256)
    else:
      pool_size = self.pool_size

    if isinstance(pool_size, int):
      return pool_size, pool_size
    else:
      assert isinstance(pool_size, tuple) and len(pool_size) == 2
      return pool_size

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

    self.on_starting()
    starting_time = time.time()
    try:
      self._start()
      self._start_time = time.time()
      self._set_status(self.Status.ONLINE)
      self.on_start(duration=time.time() - starting_time)
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
    self.on_shutting_down()

    shutting_down_time = time.time()
    try:
      self._shutdown()
      self.on_shutdown(duration=time.time() - shutting_down_time)
    except BaseException as e:  # pylint: disable=broad-except
      self.on_shutdown(duration=time.time() - shutting_down_time, error=e)
      raise e

  #
  # Environment operations.
  #

  def acquire(
      self,
      image_id: str | None = None
  ) -> base_sandbox.BaseSandbox:
    """Acquires a sandbox from the environment.

    Args:
      image_id: The image ID to use for the sandbox. If None, it will be
        automatically determined by the environment.

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
    if image_id is None:
      if not self.image_ids:
        raise ValueError(
            f'Environment {self.id} does not have a default image ID. '
            'Please specify the image ID explicitly.'
        )
      image_id = self.image_ids[0]
    elif (image_id not in self.image_ids
          and not self.supports_dynamic_image_loading):
      raise ValueError(
          f'Environment {self.id} does not serve image ID {image_id!r}. '
          f'Please use one of the following image IDs: {self.image_ids!r} or '
          f'set `{self.__class__.__name__}.supports_dynamic_image_loading` '
          'to True if dynamic image loading is supported.'
      )
    return self._acquire(image_id)

  def _acquire(
      self,
      image_id: str | None = None
  ) -> base_sandbox.BaseSandbox:
    """Acquires a sandbox from the environment."""
    if not self.enable_pooling(image_id):
      return self._bring_up_sandbox_with_retry(
          image_id=image_id,
          sandbox_id=str(self._increment_sandbox_id(image_id)),
          set_acquired=True,
      )

    allocation_start_time = time.time()
    sandbox_pool = self._sandbox_pool[image_id]
    while True:
      try:
        # We only append or replace items in the sandbox pool, therefore
        # there is no need to lock the pool.
        return self.load_balancer.acquire(sandbox_pool)
      except IndexError:
        if len(sandbox_pool) == self.max_pool_size(image_id):
          if time.time() - allocation_start_time > self.outage_grace_period:
            raise interface.EnvironmentOverloadError(  # pylint: disable=raise-missing-from
                environment=self
            )
          time.sleep(1)
        else:
          try:
            sandbox = self._bring_up_sandbox(
                image_id=image_id,
                sandbox_id=f'{self._increment_sandbox_id(image_id)}:0',
                set_acquired=True,
            )
            # Append is atomic and does not require locking.
            sandbox_pool.append(sandbox)
            return sandbox
          except (
              interface.EnvironmentError, interface.SandboxStateError
          ) as ex:
            self._report_outage_or_wait(ex)

  def _bring_up_sandbox(
      self,
      image_id: str,
      sandbox_id: str,
      set_acquired: bool = False,
  ) -> base_sandbox.BaseSandbox:
    """Brings up a new sandbox."""
    env_error = None
    try:
      sandbox = self._create_sandbox(
          image_id=image_id,
          sandbox_id=sandbox_id,
          reusable=self.enable_pooling(image_id),
          proactive_session_setup=self.proactive_session_setup,
          keepalive_interval=self.sandbox_keepalive_interval,
      )
      sandbox.start()
      if set_acquired:
        sandbox.set_acquired()
      return sandbox
    except (interface.EnvironmentError, interface.SandboxStateError) as e:
      env_error = e
      raise e
    finally:
      if env_error is None:
        self._offline_start_time = None
      elif self._offline_start_time is None:
        self._offline_start_time = time.time()

  def _bring_up_sandbox_with_retry(
      self,
      image_id: str,
      sandbox_id: str,
      set_acquired: bool = False,
      shutdown_env_upon_outage: bool = True,
  ) -> base_sandbox.BaseSandbox:
    """Brings up a new sandbox with retry until grace period is passed.

    Args:
      image_id: The image ID to use for the sandbox.
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
            image_id=image_id, sandbox_id=sandbox_id, set_acquired=set_acquired
        )
      except (interface.EnvironmentError, interface.SandboxStateError) as e:
        self._report_outage_or_wait(e, shutdown_env_upon_outage)

  def _increment_sandbox_id(self, image_id: str) -> int:
    """Returns the next pooled sandbox ID."""
    x = self._next_sandbox_id[image_id]
    self._next_sandbox_id[image_id] += 1
    return x

  def _report_outage_or_wait(
      self,
      error: interface.SandboxStateError,
      shutdown_env_upon_outage: bool = True
  ):
    """Raises error if the grace period has passed or wait for retry."""
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
    def _indices_by_image_id(
        entries: list[tuple[str, int, Any]]
    ) -> dict[str, list[int]]:
      indices_by_image_id = collections.defaultdict(list)
      for image_id, i, _ in entries:
        indices_by_image_id[image_id].append(i)
      return indices_by_image_id

    last_housekeep_time = {
        f.name: time.time() for f in self.non_sandbox_based_features()
    }

    while self._status not in (self.Status.SHUTTING_DOWN, self.Status.OFFLINE):
      housekeep_start_time = time.time()
      feature_housekeep_successes = []
      feature_housekeep_failures = []

      # Housekeeping non-sandbox-based features.
      for feature in self.non_sandbox_based_features():
        if feature.housekeep_interval is None:
          continue
        if (last_housekeep_time[feature.name]
            + feature.housekeep_interval < time.time()):
          try:
            feature.housekeep()
            last_housekeep_time[feature.name] = time.time()
            feature_housekeep_successes.append(feature.name)
          except BaseException as e:  # pylint: disable=broad-except
            pg.logging.error(
                '[%s/%s]: Feature housekeeping failed with error: %s.'
                'Shutting down environment...',
                self.id,
                feature.name,
                e,
            )
            feature_housekeep_failures.append(feature.name)
            self._housekeep_counter += 1
            self.on_housekeep(
                duration=time.time() - housekeep_start_time,
                error=e,
                feature_housekeep_successes=feature_housekeep_successes,
                feature_housekeep_failures=feature_housekeep_failures,
            )
            self.shutdown()
            return

      # Replace dead sandboxes.
      is_online = True
      dead_sandbox_entries = []
      for image_id, sandboxes in self._sandbox_pool.items():
        for i, sandbox in enumerate(sandboxes):
          if sandbox.status == interface.Sandbox.Status.OFFLINE:
            dead_sandbox_entries.append((image_id, i, sandbox))

      replaced_indices_by_image_id = {}

      if dead_sandbox_entries:
        replaced_indices_by_image_id = self._replace_dead_sandboxes(
            dead_sandbox_entries
        )
        if not replaced_indices_by_image_id:
          is_online = self.offline_duration < self.outage_grace_period

      self._housekeep_counter += 1
      duration = time.time() - housekeep_start_time

      kwargs = dict(
          feature_housekeep_successes=feature_housekeep_successes,
          feature_housekeep_failures=feature_housekeep_failures,
          dead_sandboxes=_indices_by_image_id(dead_sandbox_entries),
          replaced_sandboxes=replaced_indices_by_image_id,
          offline_duration=self.offline_duration,
      )
      if is_online:
        self.on_housekeep(duration, **kwargs)
        time.sleep(self.housekeep_interval)
      else:
        self.on_housekeep(
            duration,
            interface.EnvironmentOutageError(
                environment=self, offline_duration=self.offline_duration
            ),
            **kwargs
        )
        self.shutdown()

  def _replace_dead_sandboxes(
      self,
      dead_sandbox_entries: list[tuple[str, int, base_sandbox.BaseSandbox]]
  ) -> dict[str, list[int]]:
    """Replaces a dead sandbox with a new one.

    Args:
      dead_sandbox_entries: A list of tuples (image_id, index, sandbox) of
        dead sandboxes to replace.

    Returns:
      Successfully replaced sandboxes in a dict of image ID to a list of
        indices.
    """
    pg.logging.warning(
        '[%s]: %s maintenance: '
        'Replacing %d dead sandbox(es) with new ones...',
        self.id,
        self.__class__.__name__,
        len(dead_sandbox_entries),
    )
    def _replace(sandbox_entry: tuple[str, int, base_sandbox.BaseSandbox]):
      image_id, i, sandbox = sandbox_entry
      generation = int(sandbox.id.sandbox_id.split(':')[-1])
      replaced_sandbox = self._bring_up_sandbox(
          image_id=image_id,
          sandbox_id=f'{i}:{generation + 1}'
      )
      self._sandbox_pool[image_id][i] = replaced_sandbox

    # TODO(daiyip): Consider to loose the condition to allow some dead
    # sandboxes to be replaced successfully.
    replaced_indices_by_image_id = collections.defaultdict(list)
    num_replaced = 0
    for (image_id, index, _), _, error in lf.concurrent_map(
        _replace, dead_sandbox_entries,
        max_workers=min(
            self.pool_operation_max_parallelism,
            len(dead_sandbox_entries)
        ),
    ):
      if error is None:
        replaced_indices_by_image_id[image_id].append(index)
        num_replaced += 1

    pg.logging.warning(
        '[%s]: %s maintenance: '
        '%d/%d dead sandbox(es) have been replaced with new ones. (slots=%s)',
        self.id,
        self.__class__.__name__,
        num_replaced,
        len(dead_sandbox_entries),
        replaced_indices_by_image_id,
    )
    return replaced_indices_by_image_id

  #
  # Event handlers subclasses can override.
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

  def on_housekeep(
      self,
      duration: float,
      error: BaseException | None = None,
      **kwargs
  ) -> None:
    """Called when the environment finishes a round of housekeeping."""
    self.event_handler.on_environment_housekeep(
        self, self.housekeep_counter, duration, error, **kwargs
    )

  def on_shutting_down(self) -> None:
    """Called when the environment is shutting down."""
    self.event_handler.on_environment_shutting_down(self, self.offline_duration)

  def on_shutdown(
      self,
      duration: float,
      error: BaseException | None = None) -> None:
    """Called when the environment is shutdown."""
    lifetime = (time.time() - self.start_time) if self.start_time else 0.0
    self.event_handler.on_environment_shutdown(self, duration, lifetime, error)
