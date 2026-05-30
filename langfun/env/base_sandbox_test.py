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
import time
from typing import Any
import unittest

from langfun.env import environment
from langfun.env import interface
from langfun.env import test_utils


TestingSandboxService = test_utils.TestingSandboxService
TestingSandbox = test_utils.TestingSandbox
TestingFeature = test_utils.TestingFeature
TestingEventHandler = test_utils.TestingEventHandler


class RestrictiveFeature(TestingFeature):
  applicable_images = ['other_image_regex']


class SandboxStateTests(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.event_handler = TestingEventHandler(
        log_sandbox_status=True,
        log_feature_setup=True,
        log_session_setup=True,
        strip_service_name=True,
        filter_service_lifecycle_logs=True,
    )
    self.maxDiff = None

  def _create_env(
      self,
      features,
      *,
      pool_size=0,
      **kwargs
  ) -> environment.Environment:
    return environment.Environment(
        id='testing-env',
        sandboxes={
            'ss': TestingSandboxService(
                pool_size=pool_size,
                features=features,
                outage_retry_interval=0,
                sandbox_keepalive_interval=0,
            )
        },
        outage_grace_period=0,
        event_handler=self.event_handler,
        **kwargs
    )

  def test_passive_session_setup(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(),
            'feature2': TestingFeature(),
        },
    )
    self.assertFalse(env.sandboxes['ss'].enable_pooling('test_image'))
    self.assertIs(env.sandboxes['ss'].event_handler, env.event_handler)
    with env:
      with env.sandbox('session1') as sb:
        sb.shell('echo "hello"')
      self.assertEqual(
          self.event_handler.logs,
          [
              # pylint: disable=line-too-long
              '[testing-env] environment started',
              '[testing-env/test_image:0@<idle>] shell: "feature1" setup',
              '[testing-env/test_image:0/feature1] feature setup',
              '[testing-env/test_image:0@<idle>] shell: "feature2" setup',
              '[testing-env/test_image:0/feature2] feature setup',
              '[testing-env/test_image:0] created -> ready',
              '[testing-env/test_image:0] sandbox started',
              '[testing-env/test_image:0] ready -> acquired',
              '[testing-env/test_image:0] acquired -> setting_up',
              '[testing-env/test_image:0@session1] shell: "feature1" setup session',
              '[testing-env/test_image:0/feature1@session1] feature setup session',
              '[testing-env/test_image:0@session1] shell: "feature2" setup session',
              '[testing-env/test_image:0/feature2@session1] feature setup session',
              '[testing-env/test_image:0] setting_up -> in_session',
              "[testing-env/test_image:0] session 'session1' started",
              '[testing-env/test_image:0@session1] shell: echo "hello"',
              '[testing-env/test_image:0] in_session -> exiting_session',
              '[testing-env/test_image:0@session1] shell: "feature1" teardown session',
              '[testing-env/test_image:0/feature1@session1] feature teardown session',
              '[testing-env/test_image:0@session1] shell: "feature2" teardown session',
              '[testing-env/test_image:0/feature2@session1] feature teardown session',
              "[testing-env/test_image:0] session 'session1' ended",
              '[testing-env/test_image:0] exiting_session -> acquired',
              '[testing-env/test_image:0] acquired -> shutting_down',
              '[testing-env/test_image:0@<idle>] shell: "feature1" teardown',
              '[testing-env/test_image:0/feature1] feature teardown',
              '[testing-env/test_image:0@<idle>] shell: "feature2" teardown',
              '[testing-env/test_image:0/feature2] feature teardown',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown'
              # pylint: enable=line-too-long
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
      sb.wait_until_not(interface.Sandbox.Status.SETTING_UP)
      self.assertEqual(sb.status, interface.Sandbox.Status.READY)
      self.assertEqual(
          self.event_handler.logs,
          [
              # pylint: disable=line-too-long
              '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup',
              '[testing-env/test_image:0:0/feature1] feature setup',
              '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup session',
              '[testing-env/test_image:0:0/feature1@<idle>] feature setup session',
              '[testing-env/test_image:0:0] created -> ready',
              '[testing-env/test_image:0:0] sandbox started',
              '[testing-env] environment started',
              '[testing-env/test_image:0:0] ready -> acquired',
              '[testing-env/test_image:0:0] acquired -> setting_up',
              '[testing-env/test_image:0:0] setting_up -> in_session',
              "[testing-env/test_image:0:0] session 'session1' started",
              '[testing-env/test_image:0:0@session1] shell: echo foo with ValueError',
              '[testing-env/test_image:0:0@session1] shell: echo bar',
              '[testing-env/test_image:0:0] in_session -> exiting_session',
              '[testing-env/test_image:0:0@session1] shell: "feature1" teardown session',
              '[testing-env/test_image:0:0/feature1@session1] feature teardown session',
              "[testing-env/test_image:0:0] session 'session1' ended",
              '[testing-env/test_image:0:0] exiting_session -> setting_up',
              '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup session',
              '[testing-env/test_image:0:0/feature1@<idle>] feature setup session',
              '[testing-env/test_image:0:0] setting_up -> ready',
              # pylint: enable=line-too-long
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
      with self.assertRaises(interface.SandboxStateError):
        with env.sandbox('session1') as sb:
          sb.shell('echo foo', raise_error=RuntimeError)
      self.assertEqual(len(sb.state_errors), 1)
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
    # The first 22 log entries are deterministic (sandbox lifecycle).
    # After that, the maintenance thread may or may not replace the dead
    # sandbox before shutdown, depending on timing.
    expected_logs = [
        # pylint: disable=line-too-long
        '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup',
        '[testing-env/test_image:0:0/feature1] feature setup',
        '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup session',
        '[testing-env/test_image:0:0/feature1@<idle>] feature setup session',
        '[testing-env/test_image:0:0] created -> ready',
        '[testing-env/test_image:0:0] sandbox started',
        '[testing-env] environment started',
        '[testing-env/test_image:0:0] ready -> acquired',
        '[testing-env/test_image:0:0] acquired -> setting_up',
        '[testing-env/test_image:0:0] setting_up -> in_session',
        "[testing-env/test_image:0:0] session 'session1' started",
        '[testing-env/test_image:0:0@session1] shell: echo foo with RuntimeError',
        '[testing-env/test_image:0:0] in_session -> exiting_session',
        '[testing-env/test_image:0:0@session1] shell: "feature1" teardown session',
        '[testing-env/test_image:0:0/feature1@session1] feature teardown session',
        "[testing-env/test_image:0:0] session 'session1' ended with SandboxStateError",
        '[testing-env/test_image:0:0] exiting_session -> acquired',
        '[testing-env/test_image:0:0] acquired -> shutting_down',
        '[testing-env/test_image:0:0@<idle>] shell: "feature1" teardown',
        '[testing-env/test_image:0:0/feature1] feature teardown',
        '[testing-env/test_image:0:0] shutting_down -> offline',
        '[testing-env/test_image:0:0] sandbox shutdown',
        # pylint: enable=line-too-long
    ]
    self.assertEqual(
        self.event_handler.logs[:len(expected_logs)], expected_logs
    )
    self.assertEqual(
        self.event_handler.logs[-1],
        '[testing-env] environment shutdown',
    )

  def test_feature_teardown_non_state_error(self):
    class FaultyTeardownFeature(TestingFeature):
      def _teardown(self) -> None:
        raise ValueError('Teardown error')

    env = self._create_env(
        features={
            'feature1': FaultyTeardownFeature(),
        },
    )
    with self.assertRaises(interface.SandboxFeaturesTeardownError) as context:
      with env:
        with env.sandbox('session1'):
          pass
    self.assertIn('feature1', context.exception.errors)
    self.assertIsInstance(
        context.exception.errors['feature1'].__cause__, ValueError
    )
    self.assertTrue(context.exception.has_non_sandbox_state_error)

  def test_mismatching_applicable_image(self):
    mock_service = TestingSandboxService(
        pool_size=0,
        features={'feature1': RestrictiveFeature()},
        supports_dynamic_image_loading=True,
    )

    with self.assertRaisesRegex(
        ValueError, "not applicable to image 'test_image'"
    ):
      TestingSandbox(
          sandbox_service=mock_service,
          id=interface.Sandbox.Id(mock_service.id, 'test_image', '0'),
          image_id='test_image',
          features={'feature1': RestrictiveFeature()},
      )

  def test_is_shutting_down(self):
    env = self._create_env(features={'feature1': TestingFeature()})
    with env:
      with env.sandbox('session1') as sb:
        self.assertFalse(sb.is_shutting_down)

        # Mock status to SHUTTING_DOWN
        sb._set_status(interface.Sandbox.Status.SHUTTING_DOWN)
        self.assertTrue(sb.is_shutting_down)

        # Mock status to EXITING_SESSION and inject state_errors
        sb._set_status(interface.Sandbox.Status.EXITING_SESSION)
        sb.report_state_error(
            interface.SandboxStateError('State error', sandbox=sb)
        )
        self.assertTrue(sb.is_shutting_down)


class SandboxActivityTests(unittest.TestCase):

  def test_session_id(self):
    env = environment.Environment(
        id='testing-env',
        sandboxes={
            'ss': TestingSandboxService(
                features={'test_feature': TestingFeature()},
                pool_size=0
            )
        }
    )
    with env:
      with env.sandbox() as sb:
        self.assertRegex(sb.session_id, r'session-[0-9a-f]{7}')

      with env.test_feature() as test_feature:
        self.assertIsInstance(test_feature, TestingFeature)
        self.assertRegex(
            test_feature.session_id,
            r'test_feature-session-[0-9a-f]{7}'
        )

  def test_ping_error(self):
    env = environment.Environment(
        id='testing-env',
        sandboxes={
            'ss': TestingSandboxService(
                features={'test_feature': TestingFeature(housekeep_interval=0)},
                pool_size=1,
                sandbox_keepalive_interval=0,
            )
        }
    )
    with env:
      with env.sandbox('session1') as sb:
        sb.rebind(
            simulate_ping_error=interface.SandboxStateError,
            skip_notification=True
        )
        sb.wait_until_next_housekeep()
        self.assertIn(sb.status, (sb.Status.SHUTTING_DOWN, sb.Status.OFFLINE))

  def test_housekeep_error(self):
    event_handler = TestingEventHandler(
        log_housekeep=False,
        strip_service_name=True,
        filter_service_lifecycle_logs=True,
    )
    env = environment.Environment(
        id='testing-env',
        event_handler=event_handler,
        sandboxes={
            'ss': TestingSandboxService(
                features={'test_feature': TestingFeature(housekeep_interval=0)},
                pool_size=1,
                housekeep_interval=1.0,
                outage_grace_period=0,
                outage_retry_interval=0.1,
                sandbox_keepalive_interval=0,
            )
        }
    )
    with env:
      with env.sandbox('session1') as sb:
        self.assertEqual(len(env.ss.sandbox_pool['test_image']), 1)
        self.assertEqual(sb.status, interface.Sandbox.Status.IN_SESSION)
        self.assertEqual(sb.session_id, 'session1')
        housekeep_count = sb.housekeep_counter
        sb.test_feature.rebind(
            simulate_housekeep_error=interface.SandboxStateError,
            skip_notification=True
        )
        while sb.housekeep_counter == housekeep_count or (
            sb.status == interface.Sandbox.Status.IN_SESSION
        ):
          time.sleep(0.01)
        time.sleep(1.0)
        self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      env.ss.wait_for_housekeeping()
    self.assertEqual(
        event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env/test_image:0:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0:0/test_feature] feature setup',
            '[testing-env/test_image:0:0@<idle>] shell: "test_feature" setup session',
            '[testing-env/test_image:0:0] sandbox started',
            '[testing-env] environment started',
            "[testing-env/test_image:0:0] session 'session1' started",
            '[testing-env/test_image:0:0@session1] shell: "test_feature" teardown session',
            "[testing-env/test_image:0:0] session 'session1' ended with SandboxStateError",
            '[testing-env/test_image:0:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0:0/test_feature] feature teardown',
            '[testing-env/test_image:0:0] sandbox shutdown',
            '[testing-env/test_image:0:1@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0:1/test_feature] feature setup',
            '[testing-env/test_image:0:1@<idle>] shell: "test_feature" setup session',
            '[testing-env/test_image:0:1] sandbox started',
            '[testing-env/test_image:0:1@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0:1/test_feature] feature teardown',
            '[testing-env/test_image:0:1] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )


class SandboxServiceTests(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.maxDiff = None
    self.event_handler = TestingEventHandler(
        strip_service_name=True,
        filter_service_lifecycle_logs=True,
    )
    self.env = environment.Environment(
        id='testing-env',
        event_handler=self.event_handler,
        random_seed=1,
        sandboxes={
            'ss': TestingSandboxService(
                features={'test_feature': TestingFeature()},
                pool_size=0,
                outage_grace_period=0,
                outage_retry_interval=0,
                sandbox_keepalive_interval=0,
            )
        }
    )

  def test_service_call_activity_log(self):

    class CustomEventHandler(interface.EventHandler):

      def __init__(self):
        self.calls = []

      def on_sandbox_activity(
          self,
          name: str,
          sandbox: interface.Sandbox,
          session_id: str | None,
          duration: float,
          error: BaseException | None,
          **kwargs: Any):
        self.calls.append((session_id, name, kwargs))

      def on_feature_activity(
          self,
          name: str,
          feature: interface.Feature,
          session_id: str | None,
          duration: float,
          error: BaseException | None,
          **kwargs: Any):
        self.calls.append((session_id, name, kwargs))

    event_handler = CustomEventHandler()
    env = environment.Environment(
        id='testing-env',
        event_handler=event_handler,
        sandboxes={
            'ss': TestingSandboxService(
                features={'test_feature': TestingFeature()},
                pool_size=0,
            )
        }
    )
    with env:
      with env.test_feature(session_id='session1') as test_feature:
        test_feature.call_with_varargs('sum', 1, 2, debug=True)
    self.assertEqual(
        event_handler.calls,
        [
            (None, 'shell', {'code': '"test_feature" setup'}),
            ('session1', 'shell', {'code': '"test_feature" setup session'}),
            ('session1', 'test_feature.call_with_varargs', {'args': (1, 2), 'code': 'sum', 'debug': True}),   # pylint: disable=line-too-long
            ('session1', 'shell', {'code': '"test_feature" teardown session'}),
            (None, 'shell', {'code': '"test_feature" teardown'}),
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
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@session1] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'session1' started",
            '[testing-env/test_image:0/test_feature@session1] test_feature.num_shell_calls: None',
            '[testing-env/test_image:0/test_feature@session1] test_feature.num_shell_calls: None',
            '[testing-env/test_image:0@session1] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'session1' ended",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
        ]
    )

  def test_service_call_from_feature_with_error(self):
    with self.env:
      with self.assertRaises(interface.SandboxStateError):
        with self.env.sandbox('session1') as sb:
          sb.test_feature.bad_shell_call()
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      self.assertEqual(len(sb.state_errors), 1)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@session1] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'session1' started",
            '[testing-env/test_image:0@session1] shell: bad command with RuntimeError',
            '[testing-env/test_image:0/test_feature@session1] test_feature.bad_shell_call: None with SandboxStateError',
            '[testing-env/test_image:0@session1] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'session1' ended with SandboxStateError",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_call_from_environment(self):
    with self.env:
      with self.env.test_feature() as test_feature:
        self.assertEqual(test_feature.num_shell_calls(), 2)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@test_feature-session-2291d8c] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'test_feature-session-2291d8c' started",
            '[testing-env/test_image:0/test_feature@test_feature-session-2291d8c] test_feature.num_shell_calls: None',
            '[testing-env/test_image:0@test_feature-session-2291d8c] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'test_feature-session-2291d8c' ended",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
        ]
    )

  def test_service_call_from_environment_with_error(self):
    with self.env:
      with self.assertRaises(interface.SandboxStateError):
        with self.env.test_feature(session_id='session1') as test_feature:
          test_feature.bad_shell_call()
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@session1] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'session1' started",
            '[testing-env/test_image:0@session1] shell: bad command with RuntimeError',
            '[testing-env/test_image:0/test_feature@session1] test_feature.bad_shell_call: None with SandboxStateError',
            '[testing-env/test_image:0@session1] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'session1' ended with SandboxStateError",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env] environment shutdown',
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
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@session1] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'session1' started",
            '[testing-env/test_image:0@session1] shell: hello',
            '[testing-env/test_image:0@session1] shell: foo',
            '[testing-env/test_image:0@session1] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'session1' ended",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
        ]
    )

  def test_service_context_manager_from_feature_with_error(self):
    with self.env:
      with self.assertRaises(interface.SandboxStateError):
        with self.env.sandbox('session1') as sb:
          with sb.test_feature.my_service() as service:
            service.do('hello', raise_error=interface.SandboxStateError)
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      self.assertEqual(len(sb.state_errors), 1)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@session1] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'session1' started",
            '[testing-env/test_image:0@session1] shell: hello with SandboxStateError',
            '[testing-env/test_image:0@session1] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'session1' ended with SandboxStateError",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
        ]
    )

  def test_service_context_manager_from_environment(self):
    with self.env:
      with self.env.test_feature(session_id='session1') as test_feature:
        with test_feature.my_service() as service:
          service.do('foo')

      with self.env.test_feature() as test_feature:
        with test_feature.my_service() as service:
          service.do('bar')
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@session1] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'session1' started",
            '[testing-env/test_image:0@session1] shell: foo',
            '[testing-env/test_image:0@session1] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'session1' ended",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env/test_image:1@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:1/test_feature] feature setup',
            '[testing-env/test_image:1] sandbox started',
            '[testing-env/test_image:1@test_feature-session-2291d8c] shell: "test_feature" setup session',
            "[testing-env/test_image:1] session 'test_feature-session-2291d8c' started",
            '[testing-env/test_image:1@test_feature-session-2291d8c] shell: bar',
            '[testing-env/test_image:1@test_feature-session-2291d8c] shell: "test_feature" teardown session',
            "[testing-env/test_image:1] session 'test_feature-session-2291d8c' ended",
            '[testing-env/test_image:1@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:1/test_feature] feature teardown',
            '[testing-env/test_image:1] sandbox shutdown',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
        ]
    )

  def test_service_context_manager_from_environment_with_error(self):
    with self.env:
      with self.assertRaises(interface.SandboxStateError):
        with self.env.test_feature() as test_feature:
          with test_feature.my_service() as service:
            service.do('hello', raise_error=interface.SandboxStateError)
    self.assertEqual(
        self.event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env] environment started',
            '[testing-env/test_image:0@<idle>] shell: "test_feature" setup',
            '[testing-env/test_image:0/test_feature] feature setup',
            '[testing-env/test_image:0] sandbox started',
            '[testing-env/test_image:0@test_feature-session-2291d8c] shell: "test_feature" setup session',
            "[testing-env/test_image:0] session 'test_feature-session-2291d8c' started",
            '[testing-env/test_image:0@test_feature-session-2291d8c] shell: hello with SandboxStateError',
            '[testing-env/test_image:0@test_feature-session-2291d8c] shell: "test_feature" teardown session',
            "[testing-env/test_image:0] session 'test_feature-session-2291d8c' ended with SandboxStateError",
            '[testing-env/test_image:0@<idle>] shell: "test_feature" teardown',
            '[testing-env/test_image:0/test_feature] feature teardown',
            '[testing-env/test_image:0] sandbox shutdown',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
        ]
    )


class EnvironmentTests(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.event_handler = TestingEventHandler()

  def test_service_root_dir_propagation(self):
    service = TestingSandboxService(pool_size=0)
    env = environment.Environment(
        id='testing-env',
        sandboxes={'ss': service},
        root_dir='/env/root',
    )
    with env:
      self.assertEqual(service.root_dir, '/env/root')

  def test_feature_name_collision_across_services(self):
    with self.assertRaisesRegex(
        ValueError, "Feature 'test_feature' is already configured in 'ss'"
    ):
      environment.Environment(
          id='testing-env',
          sandboxes={
              'ss': TestingSandboxService(
                  features={'test_feature': TestingFeature()}
              ),
              'ss2': TestingSandboxService(
                  features={'test_feature': TestingFeature()}
              ),
          },
      )

  def test_sandbox_based_feature_as_top_level(self):
    class MySandboxFeature(TestingFeature):
      is_sandbox_based = True

    with self.assertRaisesRegex(
        ValueError,
        'is sandbox-based and should be configured as a feature under a'
        ' sandbox service',
    ):
      environment.Environment(
          id='testing-env',
          features={'test_feature': MySandboxFeature()},
      )

  def test_top_level_feature_collides_with_sandbox_feature(self):
    class NonSandboxFeature(TestingFeature):
      is_sandbox_based = False

    with self.assertRaisesRegex(
        ValueError, "Feature 'test_feature' is already configured in 'ss'"
    ):
      environment.Environment(
          id='testing-env',
          features={'test_feature': NonSandboxFeature()},
          sandboxes={
              'ss': TestingSandboxService(
                  features={'test_feature': TestingFeature()}
              ),
          },
      )

  def test_environment_shutdown_error(self):
    class FaultyShutdownEnv(environment.Environment):

      def _shutdown(self) -> None:
        raise RuntimeError('Shutdown crash')

    env = FaultyShutdownEnv(id='testing-env')
    env.start()
    with self.assertRaisesRegex(RuntimeError, 'Shutdown crash'):
      env.shutdown()

  def test_environment_housekeep(self):
    event_handler = TestingEventHandler(log_housekeep=True)
    env = environment.Environment(id='testing-env', event_handler=event_handler)
    with env:
      env.on_housekeep(duration=0.1)
    self.assertTrue(
        any('environment housekeeping' in log for log in event_handler.logs)
    )

  def test_get_sandbox_service_by_name(self):
    service = TestingSandboxService(pool_size=0)
    env = environment.Environment(
        id='testing-env',
        sandboxes={'ss': service},
    )
    with env:
      with env.sandbox(sandbox_service='ss') as sb:
        self.assertIs(sb.sandbox_service, service)

  def test_top_level_non_sandbox_feature_propagation(self):
    class NonSandboxFeature(test_utils.TestingNonSandboxBasedFeature):
      pass

    feature = NonSandboxFeature()
    env = environment.Environment(
        id='testing-env',
        features={'top_feature': feature},
    )
    with env:
      self.assertIs(feature.event_handler, env.event_handler)
      self.assertEqual(feature.outage_grace_period, env.outage_grace_period)

  def test_missing_feature_session(self):
    env = environment.Environment(id='testing-env')
    with env:
      with self.assertRaisesRegex(
          ValueError, "Feature 'missing_feature' is not available"
      ):
        env.feature_session('missing_feature')


if __name__ == '__main__':
  unittest.main()
