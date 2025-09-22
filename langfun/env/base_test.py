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
import contextlib
import time
from typing import Any, Iterator, Type
import unittest

from langfun.env import base_environment
from langfun.env import base_feature
from langfun.env import base_sandbox
from langfun.env import interface
import pyglove as pg


class TestingEnvironment(base_environment.BaseEnvironment):

  simulate_start_error: Type[BaseException] | None = None
  simulate_shutdown_error: Type[BaseException] | None = None
  simulate_ping_error: Type[BaseException] | None = None
  keepalive_interval: float | None = 60.0
  offline: bool = False

  @property
  def id(self) -> interface.EnvironmentId:
    return interface.EnvironmentId('testing-env')

  def wait_for_next_maintenance(self):
    maintenance_count = self._maintenance_count
    while self._maintenance_count == maintenance_count:
      time.sleep(0.1)

  def _create_sandbox(
      self,
      sandbox_id: str,
      reusable: bool,
      proactive_session_setup: bool,
  ) -> interface.Sandbox:
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
    housekeep_count = self._housekeep_count
    while self._housekeep_count == housekeep_count:
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
  housekeep_interval = 0
  setup_session_delay: float = 0.0
  simulate_housekeep_error: Type[BaseException] | None = None
  simulate_setup_error: Type[BaseException] | None = None
  simulate_teardown_error: Type[BaseException] | None = None
  simulate_setup_session_error: Type[BaseException] | None = None
  simulate_teardown_session_error: Type[BaseException] | None = None

  class Service:
    """Sandbox."""

    def __init__(self, sandbox: interface.Sandbox):
      self._sandbox = sandbox

    def do(self, code: str, raise_error: Type[BaseException] | None = None):
      self._sandbox.shell(code, raise_error=raise_error)

  def _raise_error(self, message, error_type: Type[BaseException], **kwargs):
    self._sandbox._raise_error(message, error_type, **kwargs)

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

  @base_sandbox.sandbox_service()
  def num_shell_calls(self) -> None:
    return len(self.sandbox._shell_history)

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
  log_sandbox_status: bool = False
  log_feature_setup: bool = True
  log_session_setup: bool = False

  def _on_bound(self) -> None:
    super()._on_bound()
    self._logs = []

  @property
  def logs(self) -> list[str]:
    return self._logs

  def _add_message(self, message: str, error: Exception | None) -> None:
    """Adds a message to the history."""
    if error is None:
      self._logs.append(message)
    else:
      self._logs.append(f'{message} with {error.__class__.__name__}')

  def on_environment_start(
      self,
      environment: interface.Environment,
      error: Exception | None
  ) -> None:
    """Called when the environment is started."""
    self._add_message(f'[{environment.id}] environment started', error)

  def on_environment_shutdown(
      self,
      environment: interface.Environment,
      error: Exception | None
  ) -> None:
    """Called when the environment is shutdown."""
    self._add_message(f'[{environment.id}] environment shutdown', error)

  def on_sandbox_start(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      error: Exception | None
  ) -> None:
    del environment
    self._add_message(f'[{sandbox.id}] sandbox started', error)

  def on_sandbox_status_change(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      old_status: interface.Sandbox.Status,
      new_status: interface.Sandbox.Status,
  ) -> None:
    if self.log_sandbox_status:
      self._add_message(
          f'[{sandbox.id}] {old_status.value} -> {new_status.value}',
          None,
      )

  def on_sandbox_shutdown(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      error: Exception | None
  ) -> None:
    self._add_message(f'[{sandbox.id}] sandbox shutdown', error)

  def on_feature_setup(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    if self.log_feature_setup:
      self._add_message(
          f'[{sandbox.id}/{feature.name}] feature setup', error
      )

  def on_feature_teardown(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
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
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is setup."""
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
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    if self.log_session_setup:
      self._add_message(
          f'[{sandbox.id}/{feature.name}] feature teardown session', error
      )

  def on_session_start(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      session_id: str,
      error: Exception | None
  ) -> None:
    """Called when a sandbox session starts."""
    self._add_message(
        f'[{sandbox.id}] session {session_id!r} started', error
    )

  def on_session_end(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      session_id: str,
      error: Exception | None
  ) -> None:
    """Called when a sandbox session ends."""
    self._add_message(
        f'[{sandbox.id}] session {session_id!r} ended', error
    )

  def on_session_activity(
      self,
      session_id: str,
      name: str,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature | None,
      error: Exception | None,
      *,
      code: str | None = None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    del environment, kwargs
    self._add_message(
        f'[{sandbox.id}/{session_id}] {name}: {code}', error
    )

#
# Tests
#


class EnvironmentTests(unittest.TestCase):

  def test_basics(self):
    env = TestingEnvironment(
        root_dir='/tmp',
        pool_size=0,
        features={'test_feature': TestingFeature()},
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    self.assertIsNone(interface.Environment.current())
    self.assertEqual(env.status, interface.Environment.Status.CREATED)
    self.assertFalse(env.is_online)
    self.assertEqual(env.min_pool_size, 0)
    self.assertEqual(env.max_pool_size, 0)
    self.assertEqual(env.sandbox_pool, [])
    self.assertEqual(env.id, interface.EnvironmentId('testing-env'))
    self.assertEqual(env.outage_grace_period, 1)
    self.assertEqual(env.stats_report_interval, 60)
    self.assertEqual(env.features['test_feature'].name, 'test_feature')

    self.assertIsNone(env.start_time)

    with env:
      self.assertEqual(env.status, interface.Environment.Status.ONLINE)
      self.assertIs(interface.Environment.current(), env)
      self.assertTrue(env.is_online)
      self.assertIsNotNone(env.start_time)
      self.assertEqual(env.offline_duration, 0.0)
      self.assertEqual(env.sandbox_pool, [])
      self.assertEqual(env.working_dir, '/tmp/testing-env')

      with env.sandbox('session1') as sb:
        self.assertEqual(
            sb.id, interface.SandboxId(environment_id=env.id, sandbox_id='0')
        )
        self.assertEqual(sb.session_id, 'session1')
        self.assertEqual(sb.working_dir, '/tmp/testing-env/0')
        self.assertTrue(sb.is_online)
        self.assertIs(sb.test_feature, sb.features['test_feature'])
        with self.assertRaises(AttributeError):
          _ = sb.test_feature2
      self.assertFalse(sb.is_online)

      self.assertIsInstance(env.test_feature, TestingFeature)
      with self.assertRaises(AttributeError):
        _ = env.test_feature2

  def test_acquire_env_offline(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with self.assertRaises(interface.EnvironmentOutageError):
      env.acquire()

  def test_acquire_no_pooling(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with env:
      sb = env.acquire()
      self.assertEqual(sb.status, interface.Sandbox.Status.ACQUIRED)

  def test_acquire_no_pooling_with_error(self):
    env = TestingEnvironment(
        features={
            'test_feature': TestingFeature(
                simulate_setup_error=interface.SandboxStateError
            )
        },
        pool_size=0,
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with env:
      with self.assertRaises(interface.EnvironmentOutageError):
        env.acquire()

  def test_acquire_with_pooling(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=1,
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with env:
      sb = env.acquire()
      self.assertEqual(sb.status, interface.Sandbox.Status.ACQUIRED)

  def test_acquire_with_pooling_overload(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=1,
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with env:
      sb = env.acquire()
      self.assertEqual(sb.status, interface.Sandbox.Status.ACQUIRED)
      with self.assertRaises(interface.EnvironmentOverloadError):
        env.acquire()

  def test_acquire_with_growing_pool(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=(1, 3),
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with env:
      self.assertEqual(len(env.sandbox_pool), 1)
      sb = env.acquire()
      self.assertEqual(sb.status, interface.Sandbox.Status.ACQUIRED)
      self.assertEqual(len(env.sandbox_pool), 1)
      sb2 = env.acquire()
      self.assertEqual(sb2.status, interface.Sandbox.Status.ACQUIRED)
      self.assertEqual(len(env.sandbox_pool), 2)

  def test_acquire_with_growing_pool_failure(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=(1, 3),
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with env:
      self.assertEqual(len(env.sandbox_pool), 1)
      sb = env.acquire()
      self.assertEqual(sb.status, interface.Sandbox.Status.ACQUIRED)

      # Make future sandbox setup to fail.
      env.features.test_feature.rebind(
          simulate_setup_error=interface.SandboxStateError,
          skip_notification=True
      )
      with self.assertRaises(interface.EnvironmentOutageError):
        env.acquire()

  def test_maintenance_error(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=1,
        proactive_session_setup=True,
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    with env:
      self.assertEqual(len(env.sandbox_pool), 1)
      self.assertEqual(
          env.sandbox_pool[0].status, interface.Sandbox.Status.READY
      )
      # Make future sandbox setup to fail.
      env.features.test_feature.rebind(
          simulate_setup_error=interface.SandboxStateError,
          skip_notification=True
      )
      with env.sandbox() as sb:
        with self.assertRaises(interface.SandboxStateError):
          sb.shell('bad command', raise_error=interface.SandboxStateError)
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      env.wait_for_next_maintenance()
      self.assertFalse(env.is_online)


class SandboxStatusTests(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.event_handler = TestingEnvironmentEventHandler(
        log_sandbox_status=True,
        log_feature_setup=True,
        log_session_setup=True,
    )
    self.maxDiff = None

  def _create_env(
      self,
      features,
      *,
      pool_size=0,
      **kwargs
  ) -> TestingEnvironment:
    return TestingEnvironment(
        pool_size=pool_size,
        features=features,
        outage_grace_period=0,
        event_handlers=[self.event_handler],
        outage_retry_interval=0,
        **kwargs
    )

  def test_passive_session_setup(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(),
        },
    )
    with env:
      with env.sandbox('session1') as sb:
        sb.shell('echo "hello"')
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0/session1] shell: "feature1" setup session',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/session1] shell: "feature2" setup session',
              '[testing-env/0/feature2] feature setup session',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: echo "hello"',
              '[testing-env/0/session1] shell: "feature1" teardown session',
              '[testing-env/0/feature1] feature teardown session',
              '[testing-env/0/session1] shell: "feature2" teardown session',
              '[testing-env/0/feature2] feature teardown session',
              "[testing-env/0] session 'session1' ended",
              '[testing-env/0] in_session -> acquired',
              '[testing-env/0] acquired -> shutting_down',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0/feature2] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )

  def test_proactive_session_setup(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(setup_session_delay=0.1),
            'feature2': TestingFeature(),
        },
        pool_size=1,
        proactive_session_setup=True,
    )
    with env:
      with env.sandbox('session1') as sb:
        sb.shell('echo "hello"')
      sb.wait_until_not(
          (
              interface.Sandbox.Status.IN_SESSION,
              interface.Sandbox.Status.SETTING_UP
          )
      )
      self.assertEqual(sb.status, interface.Sandbox.Status.READY)
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/feature2] feature setup session',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env] environment started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: echo "hello"',
              '[testing-env/0/session1] shell: "feature1" teardown session',
              '[testing-env/0/feature1] feature teardown session',
              '[testing-env/0/session1] shell: "feature2" teardown session',
              '[testing-env/0/feature2] feature teardown session',
              "[testing-env/0] session 'session1' ended",
              '[testing-env/0] in_session -> setting_up',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/feature2] feature setup session',
              '[testing-env/0] setting_up -> ready'
          ]
      )

  def test_practive_session_setup_with_setup_session_error(self):
    env = self._create_env(
        features={'test_feature': TestingFeature(setup_session_delay=0.5)},
        pool_size=1,
    )
    event_handler = TestingEnvironmentEventHandler(
        log_sandbox_status=True,
        log_feature_setup=True,
        log_session_setup=True,
    )
    with env:
      with env.sandbox('session1') as sb:
        sb.add_event_handler(event_handler)
        sb.test_feature.rebind(
            simulate_setup_session_error=interface.SandboxStateError,
            skip_notification=True
        )
      sb.wait_until_not(
          (
              interface.Sandbox.Status.SETTING_UP,
              interface.Sandbox.Status.SHUTTING_DOWN
          )
      )
      self.assertEqual(len(sb.state_errors), 1)
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      self.assertEqual(
          event_handler.logs,
          [
              '[testing-env/0/session1] shell: "test_feature" teardown session',
              '[testing-env/0/test_feature] feature teardown session',
              "[testing-env/0] session 'session1' ended",
              '[testing-env/0] in_session -> setting_up',
              '[testing-env/0/test_feature] feature setup session with SandboxStateError',  # pylint: disable=line-too-long
              '[testing-env/0] setting_up -> shutting_down',
              '[testing-env/0/test_feature] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )

  def test_sandbox_start_non_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(),
        },
        simulate_start_error=ValueError,
    )
    with env:
      with self.assertRaises(ValueError):
        with env.sandbox('session1'):
          pass
      self.assertTrue(env.is_online)
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0] sandbox started with ValueError',
              '[testing-env/0] created -> shutting_down',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )

  def test_sandbox_start_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(),
        },
        pool_size=1,
        simulate_start_error=interface.SandboxStateError,
    )
    with self.assertRaises(interface.EnvironmentOutageError):
      with env:
        pass
    self.assertEqual(
        self.event_handler.logs,
        [
            '[testing-env/0] sandbox started with SandboxStateError',
            '[testing-env/0] created -> shutting_down',
            '[testing-env/0] shutting_down -> offline',
            '[testing-env/0] sandbox shutdown',
            '[testing-env] environment started with EnvironmentOutageError',
            '[testing-env] environment shutdown'
        ]
    )

  def test_sandbox_shutdown_non_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(),
        },
        simulate_shutdown_error=ValueError,
    )
    with env:
      with self.assertRaises(ValueError):
        with env.sandbox('session1') as sb:
          sb.shell('echo "hello"')
      self.assertEqual(len(sb.state_errors), 0)

    self.assertEqual(
        self.event_handler.logs,
        [
            '[testing-env] environment started',
            '[testing-env/0/feature1] feature setup',
            '[testing-env/0/feature2] feature setup',
            '[testing-env/0] created -> ready',
            '[testing-env/0] sandbox started',
            '[testing-env/0] ready -> acquired',
            '[testing-env/0] acquired -> setting_up',
            '[testing-env/0/session1] shell: "feature1" setup session',
            '[testing-env/0/feature1] feature setup session',
            '[testing-env/0/session1] shell: "feature2" setup session',
            '[testing-env/0/feature2] feature setup session',
            '[testing-env/0] setting_up -> in_session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] shell: echo "hello"',
            '[testing-env/0/session1] shell: "feature1" teardown session',
            '[testing-env/0/feature1] feature teardown session',
            '[testing-env/0/session1] shell: "feature2" teardown session',
            '[testing-env/0/feature2] feature teardown session',
            "[testing-env/0] session 'session1' ended",
            '[testing-env/0] in_session -> acquired',
            '[testing-env/0] acquired -> shutting_down',
            '[testing-env/0/feature1] feature teardown',
            '[testing-env/0/feature2] feature teardown',
            '[testing-env/0] shutting_down -> offline',
            '[testing-env/0] sandbox shutdown with ValueError',
            '[testing-env] environment shutdown'
        ]
    )

  def test_env_shutdown_non_state_error(self):
    env = self._create_env(
        pool_size=1,
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(),
        },
        simulate_shutdown_error=ValueError,
    )
    with self.assertRaises(ValueError):
      with env:
        pass

    self.assertEqual(
        self.event_handler.logs,
        [
            '[testing-env/0/feature1] feature setup',
            '[testing-env/0/feature2] feature setup',
            '[testing-env/0/feature1] feature setup session',
            '[testing-env/0/feature2] feature setup session',
            '[testing-env/0] created -> ready',
            '[testing-env/0] sandbox started',
            '[testing-env] environment started',
            '[testing-env/0] ready -> shutting_down',
            '[testing-env/0/feature1] feature teardown',
            '[testing-env/0/feature2] feature teardown',
            '[testing-env/0] shutting_down -> offline',
            '[testing-env/0] sandbox shutdown with ValueError',
            '[testing-env] environment shutdown with ValueError'
        ]
    )

  def test_sandbox_shutdown_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(),
        },
        simulate_shutdown_error=interface.SandboxStateError,
    )
    with env:
      with env.sandbox('session1') as sb:
        sb.shell('echo "hello"')
      self.assertEqual(len(sb.state_errors), 1)
    self.assertEqual(
        self.event_handler.logs,
        [
            '[testing-env] environment started',
            '[testing-env/0/feature1] feature setup',
            '[testing-env/0/feature2] feature setup',
            '[testing-env/0] created -> ready',
            '[testing-env/0] sandbox started',
            '[testing-env/0] ready -> acquired',
            '[testing-env/0] acquired -> setting_up',
            '[testing-env/0/session1] shell: "feature1" setup session',
            '[testing-env/0/feature1] feature setup session',
            '[testing-env/0/session1] shell: "feature2" setup session',
            '[testing-env/0/feature2] feature setup session',
            '[testing-env/0] setting_up -> in_session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] shell: echo "hello"',
            '[testing-env/0/session1] shell: "feature1" teardown session',
            '[testing-env/0/feature1] feature teardown session',
            '[testing-env/0/session1] shell: "feature2" teardown session',
            '[testing-env/0/feature2] feature teardown session',
            "[testing-env/0] session 'session1' ended",
            '[testing-env/0] in_session -> acquired',
            '[testing-env/0] acquired -> shutting_down',
            '[testing-env/0/feature1] feature teardown',
            '[testing-env/0/feature2] feature teardown',
            '[testing-env/0] shutting_down -> offline',
            '[testing-env/0] sandbox shutdown with SandboxStateError',
            '[testing-env] environment shutdown'
        ]
    )

  def test_feature_setup_non_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(
                simulate_setup_error=ValueError
            ),
        },
    )
    with env:
      with self.assertRaises(ValueError):
        with env.sandbox('session1'):
          pass
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup with ValueError',
              '[testing-env/0] sandbox started with ValueError',
              '[testing-env/0] created -> shutting_down',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0/feature2] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )

  def test_feature_setup_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(
                simulate_setup_error=interface.SandboxStateError
            ),
            'feature2': TestingFeature(),
        },
    )
    with env:
      with self.assertRaises(interface.EnvironmentOutageError):
        with env.sandbox('session1'):
          pass
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup with SandboxStateError',
              '[testing-env/0] sandbox started with SandboxStateError',
              '[testing-env/0] created -> shutting_down',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown',
              '[testing-env] environment shutdown'
          ]
      )

  def test_feature_teardown_non_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(
                simulate_teardown_error=ValueError
            ),
        },
    )
    with env:
      with self.assertRaises(interface.FeatureTeardownError):
        with env.sandbox('session1'):
          pass
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0/session1] shell: "feature1" setup session',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/session1] shell: "feature2" setup session',
              '[testing-env/0/feature2] feature setup session',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: "feature1" teardown session',
              '[testing-env/0/feature1] feature teardown session',
              '[testing-env/0/session1] shell: "feature2" teardown session',
              '[testing-env/0/feature2] feature teardown session',
              "[testing-env/0] session 'session1' ended",
              '[testing-env/0] in_session -> acquired',
              '[testing-env/0] acquired -> shutting_down',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0/feature2] feature teardown with ValueError',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown with FeatureTeardownError',
          ]
      )

  def test_feature_teardown_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(
                simulate_teardown_error=interface.SandboxStateError
            ),
            'feature2': TestingFeature(
            ),
        },
    )
    with env:
      with env.sandbox('session1') as sb:
        pass
      self.assertEqual(len(sb.state_errors), 1)
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0/session1] shell: "feature1" setup session',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/session1] shell: "feature2" setup session',
              '[testing-env/0/feature2] feature setup session',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: "feature1" teardown session',
              '[testing-env/0/feature1] feature teardown session',
              '[testing-env/0/session1] shell: "feature2" teardown session',
              '[testing-env/0/feature2] feature teardown session',
              "[testing-env/0] session 'session1' ended",
              '[testing-env/0] in_session -> acquired',
              '[testing-env/0] acquired -> shutting_down',
              '[testing-env/0/feature1] feature teardown with SandboxStateError',  # pylint: disable=line-too-long
              '[testing-env/0/feature2] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown with FeatureTeardownError'
          ]
      )

  def test_feature_setup_session_non_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(
                simulate_setup_session_error=ValueError
            ),
        },
    )
    with env:
      with self.assertRaises(ValueError):
        with env.sandbox('session1') as sb:
          sb.shell('echo "hello"')
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0/session1] shell: "feature1" setup session',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/feature2] feature setup session with ValueError',
              "[testing-env/0] session 'session1' started with ValueError",
              '[testing-env/0] setting_up -> shutting_down',
              '[testing-env/0/session1] shell: "feature1" teardown',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0/session1] shell: "feature2" teardown',
              '[testing-env/0/feature2] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )

  def test_feature_teardown_session_non_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(
                simulate_teardown_session_error=ValueError
            ),
            'feature2': TestingFeature(),
        },
    )
    with env:
      with self.assertRaises(interface.SessionTeardownError):
        with env.sandbox('session1') as sb:
          sb.shell('echo "hello"')
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0/session1] shell: "feature1" setup session',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/session1] shell: "feature2" setup session',
              '[testing-env/0/feature2] feature setup session',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: echo "hello"',
              '[testing-env/0/feature1] feature teardown session with ValueError',  # pylint: disable=line-too-long
              '[testing-env/0/session1] shell: "feature2" teardown session',
              '[testing-env/0/feature2] feature teardown session',
              "[testing-env/0] session 'session1' ended",
              '[testing-env/0] in_session -> acquired',
              '[testing-env/0] acquired -> shutting_down',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0/feature2] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )

  def test_feature_teardown_session_state_error(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(
                simulate_teardown_session_error=interface.SandboxStateError
            ),
            'feature2': TestingFeature(),
        },
    )
    with env:
      with env.sandbox('session1') as sb:
        sb.shell('echo "hello"')
      self.assertEqual(len(sb.state_errors), 1)
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env] environment started',
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature2] feature setup',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0/session1] shell: "feature1" setup session',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0/session1] shell: "feature2" setup session',
              '[testing-env/0/feature2] feature setup session',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: echo "hello"',
              '[testing-env/0/feature1] feature teardown session with SandboxStateError',  # pylint: disable=line-too-long
              '[testing-env/0/session1] shell: "feature2" teardown session',
              '[testing-env/0/feature2] feature teardown session',
              "[testing-env/0] session 'session1' ended with SandboxStateError",
              '[testing-env/0] in_session -> acquired',
              '[testing-env/0] acquired -> shutting_down',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0/feature2] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )

  def test_session_activity_non_state_error(self):
    env = self._create_env(
        pool_size=1,
        features={
            'feature1': TestingFeature(),
        },
    )
    with env:
      with env.sandbox('session1') as sb:
        with self.assertRaises(ValueError):
          sb.shell('echo foo', raise_error=ValueError)
        self.assertEqual(len(sb.state_errors), 0)
        sb.shell('echo bar')
        self.assertEqual(sb.status, interface.Sandbox.Status.IN_SESSION)
      self.assertEqual(sb.status, interface.Sandbox.Status.READY)
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env] environment started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: echo foo with ValueError',
              '[testing-env/0/session1] shell: echo bar',
              '[testing-env/0/session1] shell: "feature1" teardown session',
              '[testing-env/0/feature1] feature teardown session',
              "[testing-env/0] session 'session1' ended",
              '[testing-env/0] in_session -> setting_up',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0] setting_up -> ready',
          ]
      )

  def test_session_activity_state_error(self):
    env = self._create_env(
        pool_size=1,
        features={
            'feature1': TestingFeature(),
        },
    )
    with env:
      with env.sandbox('session1') as sb:
        with self.assertRaises(interface.SandboxStateError):
          sb.shell('echo foo', raise_error=RuntimeError)
        self.assertEqual(len(sb.state_errors), 1)
        self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
        sb.shell('echo bar')
      self.assertEqual(
          self.event_handler.logs,
          [
              '[testing-env/0/feature1] feature setup',
              '[testing-env/0/feature1] feature setup session',
              '[testing-env/0] created -> ready',
              '[testing-env/0] sandbox started',
              '[testing-env] environment started',
              '[testing-env/0] ready -> acquired',
              '[testing-env/0] acquired -> setting_up',
              '[testing-env/0] setting_up -> in_session',
              "[testing-env/0] session 'session1' started",
              '[testing-env/0/session1] shell: echo foo with RuntimeError',
              '[testing-env/0/session1] shell: "feature1" teardown session',
              '[testing-env/0/feature1] feature teardown session',
              "[testing-env/0] session 'session1' ended with SandboxStateError",
              '[testing-env/0] in_session -> acquired',
              '[testing-env/0] acquired -> shutting_down',
              '[testing-env/0/feature1] feature teardown',
              '[testing-env/0] shutting_down -> offline',
              '[testing-env/0] sandbox shutdown'
          ]
      )


class SandboxActivityTests(unittest.TestCase):

  def test_session_id(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0
    )
    with env:
      with env.sandbox() as sb:
        self.assertRegex(sb.session_id, r'session-[0-9a-f]{7}')

      self.assertEqual(
          env.test_feature.show_session_id(session_id='session1'),
          'session1'
      )
      self.assertRegex(
          env.test_feature.show_session_id(),
          r'session-[0-9a-f]{7}'
      )

    with self.assertRaisesRegex(ValueError, '`session_id` should not be used'):
      @base_sandbox.sandbox_service()
      def foo(session_id: str):
        del session_id

  def test_ping_error(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature(housekeep_interval=0)},
        pool_size=1,
        keepalive_interval=0,
    )
    with env:
      with env.sandbox('session1') as sb:
        sb.rebind(
            simulate_ping_error=interface.SandboxStateError,
            skip_notification=True
        )
        sb.wait_until_next_housekeep()
        self.assertEqual(sb.status, sb.Status.OFFLINE)

  def test_housekeep_error(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature(housekeep_interval=0)},
        pool_size=1,
        outage_grace_period=0,
        outage_retry_interval=0,
        keepalive_interval=0,
    )
    with env:
      with env.sandbox('session1') as sb:
        self.assertEqual(len(env.sandbox_pool), 1)
        self.assertEqual(sb.status, interface.Sandbox.Status.IN_SESSION)
        self.assertEqual(sb.session_id, 'session1')
        housekeep_count = sb._housekeep_count
        sb.test_feature.rebind(
            simulate_housekeep_error=interface.SandboxStateError,
            skip_notification=True
        )
        while sb._housekeep_count == housekeep_count or (
            sb.status == interface.Sandbox.Status.IN_SESSION
        ):
          time.sleep(0.1)
        self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)

  def test_remove_event_handler(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature(housekeep_interval=0)},
        pool_size=1,
        outage_grace_period=0,
        outage_retry_interval=0,
        keepalive_interval=0,
        stats_report_interval=1,
    )
    event_handler = TestingEnvironmentEventHandler()
    with env:
      with env.sandbox('session1') as sb:
        sb.add_event_handler(event_handler)
        sb.shell('test_feature')
        sb.remove_event_handler(event_handler)
      events = list(event_handler.logs)
      self.assertGreater(len(events), 0)
      with env.sandbox('session2') as sb:
        sb.shell('test_feature')
      self.assertEqual(len(events), len(event_handler.logs))


class SandboxServiceTests(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.maxDiff = None
    self.event_handler = TestingEnvironmentEventHandler()
    self.env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        outage_grace_period=0,
        outage_retry_interval=0,
        keepalive_interval=0,
        event_handlers=[self.event_handler],
        stats_report_interval=1,
        random_seed=1,
    )

  def test_service_call_activity_log(self):

    class CustomEventHandler(interface.EnvironmentEventHandler):

      def __init__(self):
        self.calls = []

      def on_session_activity(
          self,
          session_id: str,
          name: str,
          environment: interface.Environment,
          sandbox: interface.Sandbox,
          feature: interface.Feature | None,
          error: BaseException | None,
          **kwargs: Any):
        self.calls.append((session_id, name, kwargs))

    event_handler = CustomEventHandler()
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        event_handlers=[event_handler],
    )
    with env:
      env.test_feature.call_with_varargs(
          'sum', 1, 2, debug=True, session_id='session1'
      )
    self.assertEqual(
        event_handler.calls,
        [
            ('session1', 'shell', {'code': '"test_feature" setup session'}),
            ('session1', 'call_with_varargs', {'args': (1, 2), 'code': 'sum', 'debug': True}),   # pylint: disable=line-too-long
            ('session1', 'shell', {'code': '"test_feature" teardown session'}),
        ]
    )

  def test_service_call_from_feature(self):
    with self.env:
      with self.env.sandbox('session1') as sb:
        self.assertEqual(sb.test_feature.num_shell_calls(), 2)
        self.assertEqual(sb.test_feature.num_shell_calls(), 2)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session1] shell: "test_feature" setup session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] num_shell_calls: None',
            '[testing-env/0/session1] num_shell_calls: None',
            '[testing-env/0/session1] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session1' ended",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_call_from_feature_with_error(self):
    with self.env:
      with self.env.sandbox('session1') as sb:
        with self.assertRaises(interface.SandboxStateError):
          sb.test_feature.bad_shell_call()
        self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)

    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session1] shell: "test_feature" setup session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] shell: bad command with RuntimeError',
            '[testing-env/0/session1] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session1' ended with SandboxStateError",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env/0/session1] bad_shell_call: None with SandboxStateError',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_call_from_environment(self):
    with self.env:
      self.assertEqual(self.env.test_feature.num_shell_calls(), 2)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session-2291d8c] shell: "test_feature" setup session',
            "[testing-env/0] session 'session-2291d8c' started",
            '[testing-env/0/session-2291d8c] num_shell_calls: None',
            '[testing-env/0/session-2291d8c] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session-2291d8c' ended",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_call_from_environment_with_error(self):
    with self.env:
      with self.assertRaises(interface.SandboxStateError):
        self.env.test_feature.bad_shell_call(session_id='session1')
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session1] shell: "test_feature" setup session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] shell: bad command with RuntimeError',
            '[testing-env/0/session1] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session1' ended with SandboxStateError",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env/0/session1] bad_shell_call: None with SandboxStateError',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_context_manager_from_feature(self):
    with self.env:
      with self.env.sandbox('session1') as sb:
        with sb.test_feature.my_service() as service:
          service.do('hello')
        sb.shell('foo')
        self.assertEqual(sb.status, interface.Sandbox.Status.IN_SESSION)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session1] shell: "test_feature" setup session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] my_service: None',
            '[testing-env/0/session1] shell: hello',
            '[testing-env/0/session1] shell: foo',
            '[testing-env/0/session1] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session1' ended",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_context_manager_from_feature_with_error(self):
    with self.env:
      with self.env.sandbox('session1') as sb:
        with self.assertRaises(interface.SandboxStateError):
          with sb.test_feature.my_service() as service:
            service.do('hello', raise_error=interface.SandboxStateError)
        self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session1] shell: "test_feature" setup session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] my_service: None',
            '[testing-env/0/session1] shell: hello with SandboxStateError',
            '[testing-env/0/session1] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session1' ended with SandboxStateError",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_context_manager_from_environment(self):
    with self.env:
      with self.env.test_feature.my_service(session_id='session1') as service:
        service.do('foo')
      with self.env.test_feature.my_service() as service:
        service.do('bar')
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session1] shell: "test_feature" setup session',
            "[testing-env/0] session 'session1' started",
            '[testing-env/0/session1] my_service: None',
            '[testing-env/0/session1] shell: foo',
            '[testing-env/0/session1] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session1' ended",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env/1/test_feature] feature setup',
            '[testing-env/1] sandbox started',
            '[testing-env/1/session-2291d8c] shell: "test_feature" setup session',
            "[testing-env/1] session 'session-2291d8c' started",
            '[testing-env/1/session-2291d8c] my_service: None',
            '[testing-env/1/session-2291d8c] shell: bar',
            '[testing-env/1/session-2291d8c] shell: "test_feature" teardown session',
            "[testing-env/1] session 'session-2291d8c' ended",
            '[testing-env/1/test_feature] feature teardown',
            '[testing-env/1] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_context_manager_from_environment_with_error(self):
    with self.env:
      with self.assertRaises(interface.SandboxStateError):
        with self.env.test_feature.my_service() as service:
          service.do('hello', raise_error=interface.SandboxStateError)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0/session-2291d8c] shell: "test_feature" setup session',
            "[testing-env/0] session 'session-2291d8c' started",
            '[testing-env/0/session-2291d8c] my_service: None',
            '[testing-env/0/session-2291d8c] shell: hello with SandboxStateError',
            '[testing-env/0/session-2291d8c] shell: "test_feature" teardown session',
            "[testing-env/0] session 'session-2291d8c' ended with SandboxStateError",
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )


if __name__ == '__main__':
  unittest.main()
