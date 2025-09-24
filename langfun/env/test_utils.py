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
"""Test utils for base environment."""

import contextlib
import time
from typing import Iterator, Type

from langfun.env import base_environment
from langfun.env import base_feature
from langfun.env import base_sandbox
from langfun.env import interface
import pyglove as pg


class TestingEnvironment(base_environment.BaseEnvironment):
  """Testing environment for unit tests."""

  simulate_start_error: Type[BaseException] | None = None
  simulate_shutdown_error: Type[BaseException] | None = None
  simulate_ping_error: Type[BaseException] | None = None
  keepalive_interval: float | None = 60.0
  offline: bool = False

  @property
  def id(self) -> interface.EnvironmentId:
    return interface.EnvironmentId('testing-env')

  def wait_for_housekeeping(self):
    housekeep_counter = self.housekeep_counter
    while self.housekeep_counter == housekeep_counter:
      time.sleep(0.1)

  def _create_sandbox(
      self,
      sandbox_id: str,
      reusable: bool,
      proactive_session_setup: bool,
  ) -> base_sandbox.BaseSandbox:
    if self.offline:
      raise interface.EnvironmentError(
          'Unknown environment error.',
          environment=self,
      )
    return TestingSandbox(
        environment=self,
        id=interface.SandboxId(
            environment_id=self.id,
            sandbox_id=sandbox_id
        ),
        reusable=reusable,
        proactive_session_setup=proactive_session_setup,
        simulate_start_error=self.simulate_start_error,
        simulate_shutdown_error=self.simulate_shutdown_error,
        simulate_ping_error=self.simulate_ping_error,
        keepalive_interval=self.keepalive_interval,
    )


class TestingSandbox(base_sandbox.BaseSandbox):
  """Testing sandbox for unit tests."""

  simulate_start_error: Type[BaseException] | None = None
  simulate_shutdown_error: Type[BaseException] | None = None
  simulate_ping_error: Type[BaseException] | None = None

  def _on_bound(self) -> None:
    super()._on_bound()
    self._shell_history = []
    self._ping_history = []

  def _raise_error(self, message, error_type: Type[BaseException], **kwargs):
    if (error_type is interface.SandboxStateError or
        issubclass(error_type, interface.SandboxStateError)):
      kwargs['sandbox'] = self
      raise error_type(message, **kwargs)
    else:
      raise error_type(message)

  def wait_until(
      self,
      status: interface.Sandbox.Status | tuple[interface.Sandbox.Status, ...]
  ) -> None:
    if not isinstance(status, tuple):
      status = (status,)
    while self.status not in status:
      time.sleep(0.1)

  def wait_until_not(
      self,
      status: interface.Sandbox.Status | tuple[interface.Sandbox.Status, ...]
  ) -> None:
    if not isinstance(status, tuple):
      status = (status,)
    while self.status in status:
      time.sleep(0.1)

  def wait_until_next_housekeep(self) -> None:
    housekeep_counter = self.housekeep_counter
    while self.housekeep_counter == housekeep_counter:
      time.sleep(0.1)

  def _start(self) -> None:
    if self.simulate_start_error:
      self._raise_error('Sandbox start error', self.simulate_start_error)
    super()._start()

  def _shutdown(self) -> None:
    if self.simulate_shutdown_error:
      self._raise_error('Sandbox shutdown error', self.simulate_shutdown_error)
    super()._shutdown()

  @base_sandbox.sandbox_service(critical_errors=(RuntimeError,))
  def shell(
      self,
      code: str,
      raise_error: Type[BaseException] | None = None,
  ) -> str:
    self._shell_history.append(code)
    if raise_error is not None:
      self._raise_error(f'shell "{code}" failed', raise_error)
    return f'shell "{code}" succeeded'

  def _ping(self) -> None:
    self._ping_history.append(not self.simulate_ping_error)
    if self.simulate_ping_error:
      self._raise_error('Ping error', self.simulate_ping_error, code='ping')


class TestingFeature(base_feature.BaseFeature):
  """Testing feature for unit tests."""

  housekeep_interval = 0
  setup_session_delay: float = 0.0
  simulate_housekeep_error: Type[BaseException] | None = None
  simulate_setup_error: Type[BaseException] | None = None
  simulate_teardown_error: Type[BaseException] | None = None
  simulate_setup_session_error: Type[BaseException] | None = None
  simulate_teardown_session_error: Type[BaseException] | None = None
  call_end_session_on_teardown_session: bool = False

  class Service:
    """Sandbox."""

    def __init__(self, sandbox: interface.Sandbox):
      self._sandbox = sandbox

    def do(self, code: str, raise_error: Type[BaseException] | None = None):
      self._sandbox.shell(code, raise_error=raise_error)

  def _raise_error(self, message, error_type: Type[BaseException], **kwargs):
    self._sandbox._raise_error(message, error_type, **kwargs)  # pylint: disable=protected-access

  def _setup(self) -> None:
    if self.simulate_setup_error:
      self._raise_error(f'{self.name} setup error', self.simulate_setup_error)
    self.sandbox.shell(f'"{self.name}" setup')

  def _teardown(self) -> None:
    if self.simulate_teardown_error:
      self._raise_error(
          f'{self.name} teardown error', self.simulate_teardown_error
      )
    self.sandbox.shell(f'"{self.name}" teardown')

  def _setup_session(self) -> None:
    if self.setup_session_delay > 0:
      time.sleep(self.setup_session_delay)

    if self.simulate_setup_session_error:
      self._raise_error(
          'Feature session setup error', self.simulate_setup_session_error
      )
    self.sandbox.shell(f'"{self.name}" setup session')

  def _teardown_session(self) -> None:
    if self.simulate_teardown_session_error:
      self._raise_error(
          'Feature session teardown error', self.simulate_teardown_session_error
      )
    self.sandbox.shell(f'"{self.name}" teardown session')
    if self.call_end_session_on_teardown_session:
      self.sandbox.end_session()

  @base_sandbox.sandbox_service()
  def num_shell_calls(self) -> int:
    return len(self.sandbox._shell_history)  # pylint: disable=protected-access

  @base_sandbox.sandbox_service()
  def bad_shell_call(self) -> None:
    self.sandbox.shell('bad command', raise_error=RuntimeError)

  @base_sandbox.sandbox_service()
  def show_session_id(self):
    return self.session_id

  @base_sandbox.sandbox_service()
  def call_with_varargs(self, code: str, *args, **kwargs):
    del code, args, kwargs
    return 0

  def _on_bound(self) -> None:
    super()._on_bound()
    self._service = None

  @base_sandbox.sandbox_service()
  @contextlib.contextmanager
  def my_service(self) -> Iterator[Service]:
    try:
      self._service = TestingFeature.Service(sandbox=self.sandbox)
      yield self._service
    finally:
      self._service = None

  def _housekeep(self) -> None:
    if self.simulate_housekeep_error:
      raise interface.SandboxStateError(
          'House keeping error', sandbox=self.sandbox
      )


class TestingEnvironmentEventHandler(
    pg.Object, interface.EnvironmentEventHandler
):
  """Testing environment event handler for unit tests."""

  log_sandbox_status: bool = False
  log_feature_setup: bool = True
  log_session_setup: bool = False
  log_housekeep: bool = False

  def _on_bound(self) -> None:
    super()._on_bound()
    self._logs = []

  @property
  def logs(self) -> list[str]:
    return self._logs

  def _add_message(self, message: str, error: BaseException | None) -> None:
    """Adds a message to the history."""
    if error is None:
      self._logs.append(message)
    else:
      self._logs.append(f'{message} with {error.__class__.__name__}')

  def on_environment_start(
      self,
      environment: interface.Environment,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is started."""
    assert duration > 0
    self._add_message(f'[{environment.id}] environment started', error)

  def on_environment_housekeep(
      self,
      environment: interface.Environment,
      counter: int,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment finishes a round of housekeeping."""
    assert duration > 0
    if self.log_housekeep:
      self._add_message(
          f'[{environment.id}] environment housekeeping {counter}', error
      )

  def on_environment_shutdown(
      self,
      environment: interface.Environment,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when the environment is shutdown."""
    assert lifetime is not None
    self._add_message(f'[{environment.id}] environment shutdown', error)

  def on_sandbox_start(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      duration: float,
      error: BaseException | None
  ) -> None:
    assert isinstance(environment, TestingEnvironment)
    assert duration > 0
    self._add_message(f'[{sandbox.id}] sandbox started', error)

  def on_sandbox_status_change(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      old_status: interface.Sandbox.Status,
      new_status: interface.Sandbox.Status,
      span: float
  ) -> None:
    assert span > 0
    if self.log_sandbox_status:
      self._add_message(
          f'[{sandbox.id}] {old_status.value} -> {new_status.value}',
          None,
      )

  def on_sandbox_shutdown(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    assert lifetime is not None
    self._add_message(f'[{sandbox.id}] sandbox shutdown', error)

  def on_sandbox_housekeep(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      counter: int,
      duration: float,
      error: BaseException | None
  ) -> None:
    assert duration > 0
    if self.log_housekeep:
      self._add_message(
          f'[{sandbox.id}] sandbox housekeeping {counter}', error
      )

  def on_feature_setup(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    assert duration > 0
    if self.log_feature_setup:
      self._add_message(
          f'[{sandbox.id}/{feature.name}] feature setup', error
      )

  def on_feature_teardown(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    assert duration > 0
    if self.log_feature_setup:
      self._add_message(
          f'[{sandbox.id}/{feature.name}] feature teardown', error
      )

  def on_feature_setup_session(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      session_id: str | None,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    assert duration > 0
    if self.log_session_setup:
      self._add_message(
          f'[{sandbox.id}/{feature.name}] feature setup session', error
      )

  def on_feature_teardown_session(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    assert duration > 0
    if self.log_session_setup:
      self._add_message(
          f'[{sandbox.id}/{feature.name}] feature teardown session', error
      )

  def on_feature_housekeep(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      counter: int,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is housekeeping."""
    assert duration > 0
    if self.log_housekeep:
      self._add_message(
          f'[{sandbox.id}/{feature.name}] feature housekeeping {counter}', error
      )

  def on_session_start(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session starts."""
    assert duration > 0
    self._add_message(
        f'[{sandbox.id}] session {session_id!r} started', error
    )

  def on_session_end(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      session_id: str,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session ends."""
    assert lifetime > 0
    self._add_message(
        f'[{sandbox.id}] session {session_id!r} ended', error
    )

  def on_sandbox_activity(
      self,
      name: str,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature | None,
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      *,
      code: str | None = None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    assert duration > 0
    del environment, kwargs
    if session_id is None:
      log_id = sandbox.id
    else:
      log_id = f'{sandbox.id}/{session_id}'

    if feature is not None:
      log_id = f'{log_id}/{feature.name}'
    self._add_message(
        f'[{log_id}] {name}: {code}', error
    )
