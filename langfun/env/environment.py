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
"""Langfun Environment."""

import contextlib
import functools
import random
import time
from typing import Annotated, Any, ContextManager, Iterator, Optional
import uuid

from langfun.env import interface
import pyglove as pg

AbstractEnvironment = interface.AbstractEnvironment
SandboxService = interface.SandboxService
Sandbox = interface.Sandbox
Feature = interface.Feature
EventHandler = interface.EventHandler


class Environment(AbstractEnvironment):
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

  # Disable symbolic comparison and hashing for environment objects.
  use_symbolic_comparison = False

  id: Annotated[
      AbstractEnvironment.Id,
      'Identifier for the environment.'
  ]

  root_dir: Annotated[
      str | None,
      (
          'The root directory for the environment for writting output files.'
          'If None, no output files will be allowed for the sandboxes.'
      )
  ] = None

  sandboxes: Annotated[
      dict[str, SandboxService],
      'Sandboxes to be exposed by the environment.'
  ]

  features: Annotated[
      dict[str, Feature],
      'Non-sandbox-based features to be exposed by the environment.'
  ]

  event_handler: Annotated[
      'EventHandler',
      (
          'User handler for the environment events.'
          'If None, no-op handler will be used.'
      )
  ]

  outage_grace_period: Annotated[
      float,
      (
          'The grace period for environment outage. '
          'If the environment is offline for longer than this period, '
          'an SandboxServiceOutageError will be raised.'
      )
  ] = 3600.0

  random_seed: Annotated[
      int | None,
      (
          'The random seed for generating session IDs with reproducibility. '
          'If None, no seed will be used.'
      )
  ] = None

  @pg.explicit_method_override
  def __init__(
      self,
      sandboxes: dict[str, SandboxService] | None = None,
      *,
      features: dict[str, Feature] | None = None,
      id: str | None = None,  # pylint: disable=redefined-builtin
      root_dir: str | None = None,
      event_handler: Optional['EventHandler'] = None,
      random_seed: int | None = None,
      **kwargs,
  ) -> None:
    super().__init__(
        sandboxes=sandboxes or {},
        features=features or {},
        id=self.Id(environment_id=id or 'env'),
        root_dir=root_dir,
        event_handler=event_handler or EventHandler(),
        random_seed=random_seed,
        **kwargs,
    )

  def _on_bound(self) -> None:
    """Called when the environment is bound to a context manager."""
    # pylint: disable=protected-access
    super()._on_bound()
    self._housekeep_counter = 0

    event_handler = self.event_handler
    outage_grace_period = self.outage_grace_period

    # Propagate to non-sandbox-based features.
    for feature in self.features.values():
      if not hasattr(feature, '_event_handler_ref'):
        feature._event_handler_ref = event_handler

      if feature.outage_grace_period == 3600.0:
        feature.rebind(
            outage_grace_period=outage_grace_period,
            skip_notification=True,
            raise_on_no_change=False,
        )

    # Propagate to sandbox services.
    for service in self.sandboxes.values():
      if not hasattr(service, '_event_handler_ref'):
        service._event_handler_ref = event_handler

      if service.outage_grace_period == 3600.0:
        service.rebind(
            outage_grace_period=outage_grace_period,
            skip_notification=True,
            raise_on_no_change=False,
        )

    self._start_time = None
    self._shutting_down = False

    self._features_with_setup_called = set()
    self._sandbox_services_with_start_called = set()
    self._random = (
        random if self.random_seed is None else random.Random(self.random_seed)
    )

  @property
  def is_online(self) -> bool:
    """Returns True if the environment is alive."""
    return self._start_time is not None and not self._shutting_down

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
      self._start()
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

  def stats(self) -> dict[str, Any]:
    """Returns the stats of the environment."""
    stats = {}
    for name, service in self.sandboxes.items():
      stats[name] = service.stats()
    return stats

  def shutdown(self) -> None:
    """Shuts down the environment.

    This method should not raise any exceptions.
    """
    self._shutting_down = True
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
      self._shutting_down = False

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
        try:
          feature.teardown()
        except BaseException:   # pylint: disable=broad-except
          pass

  def new_session_id(self, feature_hint: str | None = None) -> str:
    """Generates a random session ID."""
    suffix = uuid.UUID(
        bytes=bytes(bytes(self._random.getrandbits(8) for _ in range(16))),
        version=4
    ).hex[:7]
    return f'{feature_hint or "unknown"}-session-{suffix}'

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
        self, self._housekeep_counter, duration, error, **kwargs
    )

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

  #
  # Environment operations.
  #

  def sandbox(
      self,
      session_id: str | None = None,
      image_id: str | None = None,
      sandbox_service: str | None = None,
  ) -> ContextManager[Sandbox]:
    """Gets a sandbox from the environment and starts a new user session."""
    return self._get_sandbox_service(
        image_id, sandbox_service
    ).acquire(image_id).new_session(session_id or self.new_session_id())

  def _get_sandbox_service(
      self,
      image_id: str | None = None,
      sandbox_service: str | None = None,
  ) -> SandboxService:
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

  def feature_session(
      self,
      name: str,
      session_id: str | None = None,
      image_id: str | None = None,
  ) -> ContextManager[Feature]:
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
      sandbox_service: SandboxService,
      feature: Feature,
      image_id: str | None = None,
      session_id: str | None = None,
  ) -> Iterator[Feature]:
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
