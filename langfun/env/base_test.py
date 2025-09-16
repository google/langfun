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
import concurrent.futures
import contextlib
import time
from typing import Iterator
import unittest

from langfun.env import base_environment
from langfun.env import base_feature
from langfun.env import base_sandbox
from langfun.env import interface


class TestingEnvironment(base_environment.BaseEnvironment):

  ping_simulate_error: bool = False
  keepalive_interval: float | None = 60.0
  offline: bool = False

  @property
  def id(self) -> interface.EnvironmentId:
    return interface.EnvironmentId('testing-env')

  def set_offline(self, offline: bool) -> None:
    self.rebind(
        offline=offline, skip_notification=True, raise_on_no_change=False
    )
    for sandbox in self._sandbox_pool:
      sandbox.rebind(
          ping_simulate_error=offline,
          skip_notification=True,
          raise_on_no_change=False
      )

  def _create_sandbox(
      self,
      sandbox_id: str,
      reusable: bool
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
        ping_simulate_error=self.ping_simulate_error,
        keepalive_interval=self.keepalive_interval,
    )


class TestingSandbox(base_sandbox.BaseSandbox):

  ping_simulate_error: bool = False

  def _on_bound(self) -> None:
    super()._on_bound()
    self._shell_history = []
    self._ping_history = []

  @base_sandbox.sandbox_service(critical_errors=(RuntimeError,))
  def shell(
      self,
      code: str,
      must_succeed: bool = True,
  ) -> str:
    self._shell_history.append(code)
    if 'bad' not in code:
      return f'shell {len(self._shell_history)} succeeded'

    message = f'shell {len(self._shell_history)} failed'
    if must_succeed:
      raise RuntimeError(message)
    raise ValueError(message)

  def _ping(self) -> None:
    self._ping_history.append(not self.ping_simulate_error)
    if self.ping_simulate_error:
      raise interface.SandboxStateError(sandbox=self, code='ping')


class TestingFeature(base_feature.BaseFeature):
  housekeep_interval = 0
  simulate_housekeep_error: bool = False

  class Service:
    """Sandbox."""

    def __init__(self, sandbox: interface.Sandbox):
      self._sandbox = sandbox

    def do(self, code: str):
      self._sandbox.shell(code)

  def _setup(self) -> None:
    self.sandbox.shell(f'echo "setup {self.name}"')

  def _teardown(self) -> None:
    self.sandbox.shell(f'echo "teardown {self.name}"')

  @base_sandbox.sandbox_service()
  def num_shell_calls(self) -> None:
    return len(self.sandbox._shell_history)

  @base_sandbox.sandbox_service()
  def bad_shell_call(self) -> None:
    self.sandbox.shell('bad command')

  @base_sandbox.sandbox_service()
  def show_session_id(self):
    return self.session_id

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


class TestingEnvironmentEventHandler(interface.EnvironmentEventHandler):

  def __init__(self):
    self.history = []

  def _add_message(self, message: str, error: Exception | None) -> None:
    """Adds a message to the history."""
    if error is None:
      self.history.append(message)
    else:
      self.history.append(f'{message} with error: {error}')

  def on_environment_start(
      self,
      environment: 'Environment',
      error: Exception | None
  ) -> None:
    """Called when the environment is started."""
    self._add_message(f'[{environment.id}] environment started', error)

  def on_environment_shutdown(
      self,
      environment: 'Environment',
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

  def on_sandbox_shutdown(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      error: Exception | None
  ) -> None:
    self._add_message(f'[{sandbox.id}] sandbox shutdown', error)

  def on_sandbox_feature_setup(
      self,
      environment: interface.Environment,
      sandbox: interface.Sandbox,
      feature: interface.Feature,
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    self._add_message(
        f'[{sandbox.id}/{feature.name}] feature setup', error
    )

  def on_sandbox_feature_teardown(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      feature: 'Feature',
      error: Exception | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    self._add_message(
        f'[{sandbox.id}/{feature.name}] feature teardown', error
    )

  def on_sandbox_session_start(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      session_id: str,
      error: Exception | None
  ) -> None:
    """Called when a sandbox session starts."""
    self._add_message(
        f'[{sandbox.id}] session {session_id!r} started', error
    )

  def on_sandbox_session_end(
      self,
      environment: 'Environment',
      sandbox: 'Sandbox',
      session_id: str,
      error: Exception | None
  ) -> None:
    """Called when a sandbox session ends."""
    self._add_message(
        f'[{sandbox.id}] session {session_id!r} ended', error
    )


class EnvironmentTest(unittest.TestCase):

  def test_environment_no_pooling_normal(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        pool_size=0,
        features={'test_feature': TestingFeature()},
        outage_grace_period=1,
        event_handler=event_handler,
        outage_retry_interval=0,
    )
    self.assertFalse(env.is_alive)
    self.assertEqual(env.min_pool_size, 0)
    self.assertEqual(env.max_pool_size, 0)
    self.assertEqual(env.sandbox_pool, [])
    self.assertEqual(env.id, interface.EnvironmentId('testing-env'))
    self.assertEqual(env.outage_grace_period, 1)
    self.assertEqual(env.stats_report_interval, 60)
    self.assertEqual(env.features['test_feature'].name, 'test_feature')

    self.assertIsNone(env.start_time)

    with env:
      self.assertEqual(
          event_handler.history,
          [
              '[testing-env] environment started',
          ]
      )
      self.assertIs(interface.Environment.current(), env)
      self.assertTrue(env.is_alive)
      self.assertIsNotNone(env.start_time)
      self.assertEqual(env.offline_duration, 0.0)
      self.assertEqual(env.sandbox_pool, [])
      self.assertIsNone(env.working_dir)

      with env.sandbox('session1') as sb:
        self.assertEqual(env.sandbox_pool, [])
        self.assertTrue(sb.is_alive)
        self.assertTrue(sb.is_busy)
        self.assertEqual(sb.session_id, 'session1')
        self.assertIsNone(sb.working_dir)
        self.assertEqual(
            sb.id, interface.SandboxId(environment_id=env.id, sandbox_id='0')
        )
        sb.shell('echo "foo"')
        self.assertIsInstance(sb.test_feature, TestingFeature)
        self.assertEqual(sb.test_feature.session_id, 'session1')
        with self.assertRaises(AttributeError):
          _ = sb.test_feature2
        self.assertIsNone(sb.test_feature._service)
        with sb.test_feature.my_service() as service:
          self.assertIs(sb.test_feature._service, service)
        self.assertIsNone(sb.test_feature._service)
        self.assertEqual(sb.session_id, 'session1')

      with env.sandbox('session2') as sb:
        self.assertEqual(env.sandbox_pool, [])
        self.assertTrue(sb.is_alive)
        self.assertTrue(sb.is_busy)
        self.assertEqual(sb.session_id, 'session2')
        self.assertEqual(
            sb.id, interface.SandboxId(environment_id=env.id, sandbox_id='1')
        )
        sb.shell('echo "bar"')

      # Access the feature from the environment without obtaining a sandbox.
      self.assertEqual(
          env.test_feature.num_shell_calls(session_id='num_shell_session'), 1
      )
      with self.assertRaises(AttributeError):
        _ = env.test_feature2
      # Access the feature from the environment without obtaining a sandbox.
      with env.test_feature.my_service(
          session_id='my_service_session'
      ) as service:
        service.do('echo "baz"')

    self.assertFalse(env.is_alive)
    self.assertIsNone(interface.Environment.current())
    self.assertEqual(
        event_handler.history,
        [
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0] session \'session1\' started',
            '[testing-env/0] session \'session1\' ended',
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env/1/test_feature] feature setup',
            '[testing-env/1] sandbox started',
            '[testing-env/1] session \'session2\' started',
            '[testing-env/1] session \'session2\' ended',
            '[testing-env/1/test_feature] feature teardown',
            '[testing-env/1] sandbox shutdown',
            '[testing-env/2/test_feature] feature setup',
            '[testing-env/2] sandbox started',
            '[testing-env/2] session \'num_shell_session\' started',
            '[testing-env/2] session \'num_shell_session\' ended',
            '[testing-env/2/test_feature] feature teardown',
            '[testing-env/2] sandbox shutdown',
            '[testing-env/3/test_feature] feature setup',
            '[testing-env/3] sandbox started',
            '[testing-env/3] session \'my_service_session\' started',
            '[testing-env/3] session \'my_service_session\' ended',
            '[testing-env/3/test_feature] feature teardown',
            '[testing-env/3] sandbox shutdown',
            '[testing-env] environment shutdown',
        ]
    )

  def test_environment_no_pooling_state_error(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        root_dir='/tmp',
        pool_size=0,
        features={'test_feature': TestingFeature()},
        outage_grace_period=1,
        outage_retry_interval=0,
        event_handler=event_handler,
    )
    self.assertEqual(env.working_dir, '/tmp/testing-env')
    with env:
      with env.sandbox('session1') as sb:
        self.assertEqual(env.sandbox_pool, [])
        self.assertTrue(sb.is_alive)
        self.assertTrue(sb.is_busy)
        self.assertFalse(sb.is_pending)
        self.assertEqual(sb.session_id, 'session1')
        self.assertEqual(sb.working_dir, '/tmp/testing-env/0')
        self.assertEqual(
            sb.id, interface.SandboxId(environment_id=env.id, sandbox_id='0')
        )
        # Non-critical error.
        with self.assertRaises(ValueError):
          sb.shell('bad command', must_succeed=False)

        # Critical error.
        with sb.test_feature.my_service() as service:
          with self.assertRaises(interface.SandboxStateError):
            service.do('bad command')
        self.assertFalse(sb.is_alive)

      with self.assertRaises(interface.SandboxStateError):
        env.test_feature.bad_shell_call(session_id='bad_shell_session')

      # Access the feature from the environment without obtaining a sandbox.
      with env.test_feature.my_service(
          session_id='my_service_session'
      ) as service:
        with self.assertRaises(interface.SandboxStateError):
          service.do('bad command')

    self.assertFalse(env.is_alive)
    self.assertIsNone(interface.Environment.current())
    self.assertEqual(
        event_handler.history,
        [
            '[testing-env] environment started',
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            '[testing-env/0] session \'session1\' started',
            # Sandbox shutdown is triggered by the SandboxStateError before
            # the session end event is triggered.
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env/0] session \'session1\' ended',
            '[testing-env/1/test_feature] feature setup',
            '[testing-env/1] sandbox started',
            "[testing-env/1] session 'bad_shell_session' started",
            '[testing-env/1/test_feature] feature teardown',
            '[testing-env/1] sandbox shutdown',
            "[testing-env/1] session 'bad_shell_session' ended",
            '[testing-env/2/test_feature] feature setup',
            '[testing-env/2] sandbox started',
            "[testing-env/2] session 'my_service_session' started",
            '[testing-env/2/test_feature] feature teardown',
            '[testing-env/2] sandbox shutdown',
            "[testing-env/2] session 'my_service_session' ended",
            '[testing-env] environment shutdown',
        ]
    )

  def test_environment_with_pooling_normal(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        pool_size=(1, 2),
        features={'test_feature': TestingFeature()},
        event_handler=event_handler,
        # To make pool operation deterministic in the test.
        pool_operation_max_parallelism=1,
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    self.assertFalse(env.is_alive)
    self.assertEqual(env.min_pool_size, 1)
    self.assertEqual(env.max_pool_size, 2)
    self.assertEqual(env.sandbox_pool, [])
    self.assertEqual(env.id, interface.EnvironmentId('testing-env'))
    self.assertEqual(env.outage_grace_period, 1)
    self.assertEqual(env.stats_report_interval, 60)
    self.assertEqual(env.features['test_feature'].name, 'test_feature')

    self.assertIsNone(env.start_time)

    with env:
      self.assertEqual(
          event_handler.history,
          [
              '[testing-env/0/test_feature] feature setup',
              '[testing-env/0] sandbox started',
              '[testing-env] environment started',
          ]
      )
      self.assertIs(interface.Environment.current(), env)
      self.assertEqual(len(env.sandbox_pool), 1)
      self.assertTrue(env.sandbox_pool[0].is_alive)
      self.assertFalse(env.sandbox_pool[0].is_busy)
      self.assertFalse(env.sandbox_pool[0].is_pending)

      # Use the pooled server.
      with env.sandbox('session1') as sb:
        self.assertEqual(len(env.sandbox_pool), 1)
        self.assertTrue(sb.is_alive)
        self.assertFalse(sb.is_pending)
        self.assertTrue(sb.is_busy)
        self.assertEqual(sb.session_id, 'session1')
        self.assertEqual(
            sb.id, interface.SandboxId(environment_id=env.id, sandbox_id='0')
        )
        sb.shell('echo "foo"')

      self.assertEqual(len(env.sandbox_pool), 1)

      # Reuse the pooled server.
      with env.sandbox('session2') as sb2:
        self.assertEqual(len(env.sandbox_pool), 1)
        self.assertTrue(sb2.is_alive)
        self.assertTrue(sb2.is_busy)
        self.assertEqual(sb2.session_id, 'session2')
        self.assertEqual(
            sb2.id, interface.SandboxId(environment_id=env.id, sandbox_id='0')
        )
        sb2.shell('echo "bar"')

        # Dynamically bring up a new server in the pool.
        with env.sandbox('session3') as sb3:
          self.assertEqual(len(env.sandbox_pool), 2)
          self.assertTrue(sb3.is_alive)
          self.assertFalse(sb3.is_pending)
          self.assertTrue(sb3.is_busy)
          self.assertEqual(sb3.session_id, 'session3')
          self.assertEqual(
              sb3.id, interface.SandboxId(environment_id=env.id, sandbox_id='1')
          )
          sb3.shell('echo "baz"')

          self.assertEqual(
              env.stats(),
              dict(
                  sandbox=dict(
                      num_total=2,
                      num_busy=2,
                      num_free=0,
                      num_dead=0,
                  )
              )
          )

          # Environment overloaded as all pooled servers are busy, and there
          # is no more quota to bring up new servers.
          with self.assertRaises(interface.EnvironmentOverloadError):
            with env.sandbox('session4'):
              pass

        self.assertEqual(
            env.stats(),
            dict(
                sandbox=dict(
                    num_total=2,
                    num_busy=1,
                    num_free=1,
                    num_dead=0,
                )
            )
        )

        # Access the feature from the environment without obtaining a sandbox.
        self.assertEqual(
            env.test_feature.num_shell_calls(session_id='num_shell_session'), 2
        )

    self.assertFalse(env.is_alive)
    self.assertEqual(env.sandbox_pool, [])
    self.assertIsNone(interface.Environment.current())
    self.assertEqual(
        event_handler.history,
        [
            '[testing-env/0/test_feature] feature setup',
            '[testing-env/0] sandbox started',
            # Environment ready is after the first sandbox is started.
            '[testing-env] environment started',
            '[testing-env/0] session \'session1\' started',
            '[testing-env/0] session \'session1\' ended',
            '[testing-env/0] session \'session2\' started',
            '[testing-env/1/test_feature] feature setup',
            '[testing-env/1] sandbox started',
            '[testing-env/1] session \'session3\' started',
            '[testing-env/1] session \'session3\' ended',
            '[testing-env/1] session \'num_shell_session\' started',
            '[testing-env/1] session \'num_shell_session\' ended',
            '[testing-env/0] session \'session2\' ended',
            '[testing-env/0/test_feature] feature teardown',
            '[testing-env/0] sandbox shutdown',
            '[testing-env/1/test_feature] feature teardown',
            '[testing-env/1] sandbox shutdown',
            '[testing-env] environment shutdown',
        ]
    )

  def test_session_id(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        event_handler=event_handler,
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

  def test_environment_with_pooling_state_error(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        root_dir='/tmp',
        pool_size=(1, 2),
        outage_grace_period=1,
        outage_retry_interval=0,
        event_handler=event_handler,
    )
    self.assertEqual(env.working_dir, '/tmp/testing-env')
    with env:
      with env.sandbox('session1') as sb:
        self.assertEqual(len(env.sandbox_pool), 1)
        self.assertTrue(sb.is_alive)
        self.assertTrue(sb.is_busy)
        self.assertEqual(sb.session_id, 'session1')
        self.assertEqual(
            sb.id, interface.SandboxId(environment_id=env.id, sandbox_id='0')
        )

        # Non-critical error.
        with self.assertRaises(ValueError):
          sb.shell('bad command', must_succeed=False)

        # Critical error.
        with self.assertRaises(interface.SandboxStateError):
          sb.shell('bad command')
        maintenance_count = env._maintenance_count
        self.assertFalse(sb.is_alive)

      self.assertTrue(env.is_alive)

      # Wait dead sandbox to be replaced.
      while env._maintenance_count == maintenance_count:
        time.sleep(0.5)

      self.assertTrue(env.sandbox_pool[0].is_alive)
      self.assertEqual(
          env.stats(),
          dict(
              sandbox=dict(
                  num_total=1,
                  num_busy=0,
                  num_free=1,
                  num_dead=0,
              )
          )
      )

    self.assertFalse(env.is_alive)
    self.assertIsNone(interface.Environment.current())
    self.assertEqual(
        event_handler.history,
        [
            '[testing-env/0] sandbox started',
            '[testing-env] environment started',
            '[testing-env/0] session \'session1\' started',
            # Sandbox shutdown is triggered by the SandboxStateError before
            # the session end event is triggered.
            '[testing-env/0] sandbox shutdown',
            '[testing-env/0] session \'session1\' ended',
            # The maintenance loop will replace the dead sandbox and start a new
            # sandbox.
            '[testing-env/0] sandbox started',
            '[testing-env/0] sandbox shutdown',
            '[testing-env] environment shutdown',
        ]
    )
    self.assertEqual(
        env.stats(),
        dict(
            sandbox=dict(
                num_total=0,
                num_busy=0,
                num_free=0,
                num_dead=0,
            )
        )
    )

  def test_environment_outage_with_pooling(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=(1, 2),
        outage_grace_period=0,
        outage_retry_interval=0,
        keepalive_interval=0,
        event_handler=event_handler,
        stats_report_interval=1,
    )
    with env:
      with env.sandbox('session1') as sb:
        self.assertEqual(len(env.sandbox_pool), 1)
        self.assertTrue(sb.is_alive)
        self.assertTrue(sb.is_busy)
        self.assertEqual(sb.session_id, 'session1')
        env.set_offline(True)
        housekeep_count = sb._housekeep_count
        while sb._housekeep_count == housekeep_count:
          time.sleep(1.0)
        self.assertFalse(sb.is_alive)
        maintenance_count = env._maintenance_count
        while env._maintenance_count == maintenance_count:
          time.sleep(1.0)
        self.assertFalse(env.is_alive)

      with self.assertRaises(interface.EnvironmentOutageError):
        with env.sandbox('session2'):
          pass

      with self.assertRaises(interface.EnvironmentOutageError):
        with env.test_feature.my_service():
          pass
      self.assertGreater(env.offline_duration, 0)

  def test_environment_outage_during_acquire(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=(2, 3),
        outage_grace_period=0,
        outage_retry_interval=0,
        keepalive_interval=0,
        event_handler=event_handler,
        stats_report_interval=1,
    )

    def _thread_func(i) -> bool:
      time.sleep(0.8 * i)
      with env.sandbox(f'session{i}') as sb:
        return sb.shell('echo "foo"')

    with env:
      with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fs = [
            executor.submit(_thread_func, i) for i in range(5)
        ]
        time.sleep(1.0)
        env.set_offline(True)
        vs = []
        for f in fs:
          try:
            vs.append(f.result())
          except interface.EnvironmentError as e:
            vs.append(e)

  def test_environment_outage_during_acquire_pool_not_full(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        features={
            'test_feature': TestingFeature(simulate_housekeep_error=True)
        },
        pool_size=(1, 3),
        outage_grace_period=1,
        outage_retry_interval=0,
        keepalive_interval=0,
        event_handler=event_handler,
        stats_report_interval=1,
    )

    def _thread_func() -> bool:
      with env.sandbox('session1') as sb:
        return sb.shell('echo "foo"')

    with env:
      self.assertEqual(len(env.sandbox_pool), 1)
      self.assertFalse(env.sandbox_pool[0].is_alive)
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        env.set_offline(True)
        f = executor.submit(_thread_func)
        with self.assertRaises(interface.EnvironmentOutageError):
          f.result()

  def test_housekeep_error(self):
    event_handler = TestingEnvironmentEventHandler()
    env = TestingEnvironment(
        features={
            'test_feature': TestingFeature(
                housekeep_interval=0
            )
        },
        pool_size=1,
        outage_grace_period=0,
        outage_retry_interval=0,
        keepalive_interval=0,
        event_handler=event_handler,
        stats_report_interval=1,
    )
    with env:
      with env.sandbox('session1') as sb:
        self.assertEqual(len(env.sandbox_pool), 1)
        self.assertTrue(sb.is_alive)
        self.assertTrue(sb.is_busy)
        self.assertEqual(sb.session_id, 'session1')
        housekeep_count = sb._housekeep_count
        sb.test_feature.rebind(
            simulate_housekeep_error=True, skip_notification=True
        )
        while sb._housekeep_count == housekeep_count or sb.is_alive:
          time.sleep(1.0)
        self.assertFalse(sb.is_alive)


if __name__ == '__main__':
  unittest.main()
