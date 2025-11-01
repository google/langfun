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

from langfun.env import interface
from langfun.env import test_utils

TestingEnvironment = test_utils.TestingEnvironment
TestingSandbox = test_utils.TestingSandbox
TestingFeature = test_utils.TestingFeature
TestingEventHandler = test_utils.TestingEventHandler


class SandboxStateTests(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.event_handler = TestingEventHandler(
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
        event_handler=self.event_handler,
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
    self.assertFalse(env.enable_pooling('test_image'))
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

  def test_proactive_session_setup(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(setup_session_delay=0.1),
            'feature2': TestingFeature(),
        },
        pool_size=1,
        proactive_session_setup=True,
    )
    self.assertTrue(env.enable_pooling('test_image'))
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
              # pylint: disable=line-too-long
              '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup',
              '[testing-env/test_image:0:0/feature1] feature setup',
              '[testing-env/test_image:0:0@<idle>] shell: "feature2" setup',
              '[testing-env/test_image:0:0/feature2] feature setup',
              '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup session',
              '[testing-env/test_image:0:0/feature1@<idle>] feature setup session',
              '[testing-env/test_image:0:0@<idle>] shell: "feature2" setup session',
              '[testing-env/test_image:0:0/feature2@<idle>] feature setup session',
              '[testing-env/test_image:0:0] created -> ready',
              '[testing-env/test_image:0:0] sandbox started',
              '[testing-env] environment started',
              '[testing-env/test_image:0:0] ready -> acquired',
              '[testing-env/test_image:0:0] acquired -> setting_up',
              '[testing-env/test_image:0:0] setting_up -> in_session',
              "[testing-env/test_image:0:0] session 'session1' started",
              '[testing-env/test_image:0:0@session1] shell: echo "hello"',
              '[testing-env/test_image:0:0] in_session -> exiting_session',
              '[testing-env/test_image:0:0@session1] shell: "feature1" teardown session',
              '[testing-env/test_image:0:0/feature1@session1] feature teardown session',
              '[testing-env/test_image:0:0@session1] shell: "feature2" teardown session',
              '[testing-env/test_image:0:0/feature2@session1] feature teardown session',
              "[testing-env/test_image:0:0] session 'session1' ended",
              '[testing-env/test_image:0:0] exiting_session -> setting_up',
              '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup session',
              '[testing-env/test_image:0:0/feature1@<idle>] feature setup session',
              '[testing-env/test_image:0:0@<idle>] shell: "feature2" setup session',
              '[testing-env/test_image:0:0/feature2@<idle>] feature setup session',
              '[testing-env/test_image:0:0] setting_up -> ready'
              # pylint: enable=line-too-long
          ]
      )

  def test_proactive_session_setup_with_setup_session_error(self):
    env = self._create_env(
        features={'test_feature': TestingFeature(setup_session_delay=0.5)},
        pool_size=1,
        housekeep_interval=10.0,
    )
    with env:
      with env.sandbox('session1') as sb:
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
          self.event_handler.logs,
          [
              # pylint: disable=line-too-long
              '[testing-env/test_image:0:0@<idle>] shell: "test_feature" setup',
              '[testing-env/test_image:0:0/test_feature] feature setup',
              '[testing-env/test_image:0:0@<idle>] shell: "test_feature" setup session',
              '[testing-env/test_image:0:0/test_feature@<idle>] feature setup session',
              '[testing-env/test_image:0:0] created -> ready',
              '[testing-env/test_image:0:0] sandbox started',
              '[testing-env] environment started',
              '[testing-env/test_image:0:0] ready -> acquired',
              '[testing-env/test_image:0:0] acquired -> setting_up',
              '[testing-env/test_image:0:0] setting_up -> in_session',
              "[testing-env/test_image:0:0] session 'session1' started",
              '[testing-env/test_image:0:0] in_session -> exiting_session',
              '[testing-env/test_image:0:0@session1] shell: "test_feature" teardown session',
              '[testing-env/test_image:0:0/test_feature@session1] feature teardown session',
              "[testing-env/test_image:0:0] session 'session1' ended",
              '[testing-env/test_image:0:0] exiting_session -> setting_up',
              '[testing-env/test_image:0:0/test_feature@<idle>] feature setup session with SandboxStateError',
              '[testing-env/test_image:0:0] setting_up -> shutting_down',
              '[testing-env/test_image:0:0@<idle>] shell: "test_feature" teardown',
              '[testing-env/test_image:0:0/test_feature] feature teardown',
              '[testing-env/test_image:0:0] shutting_down -> offline',
              '[testing-env/test_image:0:0] sandbox shutdown'
              # pylint: enable=line-too-long
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
              '[testing-env/test_image:0] sandbox started with ValueError',
              '[testing-env/test_image:0] created -> shutting_down',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown'
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
            # pylint: disable=line-too-long
            '[testing-env/test_image:0:0] sandbox started with SandboxStateError',
            '[testing-env/test_image:0:0] created -> shutting_down',
            '[testing-env/test_image:0:0] shutting_down -> offline',
            '[testing-env/test_image:0:0] sandbox shutdown',
            '[testing-env] environment started with EnvironmentOutageError',
            '[testing-env] environment shutdown'
            # pylint: enable=line-too-long
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
            '[testing-env/test_image:0] sandbox shutdown with ValueError',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
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
            # pylint: disable=line-too-long
            '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup',
            '[testing-env/test_image:0:0/feature1] feature setup',
            '[testing-env/test_image:0:0@<idle>] shell: "feature2" setup',
            '[testing-env/test_image:0:0/feature2] feature setup',
            '[testing-env/test_image:0:0@<idle>] shell: "feature1" setup session',
            '[testing-env/test_image:0:0/feature1@<idle>] feature setup session',
            '[testing-env/test_image:0:0@<idle>] shell: "feature2" setup session',
            '[testing-env/test_image:0:0/feature2@<idle>] feature setup session',
            '[testing-env/test_image:0:0] created -> ready',
            '[testing-env/test_image:0:0] sandbox started',
            '[testing-env] environment started',
            '[testing-env/test_image:0:0] ready -> shutting_down',
            '[testing-env/test_image:0:0@<idle>] shell: "feature1" teardown',
            '[testing-env/test_image:0:0/feature1] feature teardown',
            '[testing-env/test_image:0:0@<idle>] shell: "feature2" teardown',
            '[testing-env/test_image:0:0/feature2] feature teardown',
            '[testing-env/test_image:0:0] shutting_down -> offline',
            '[testing-env/test_image:0:0] sandbox shutdown with ValueError',
            '[testing-env] environment shutdown with ValueError'
            # pylint: enable=line-too-long
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
            '[testing-env/test_image:0] sandbox shutdown with SandboxStateError',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
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
              # pylint: disable=line-too-long
              '[testing-env] environment started',
              '[testing-env/test_image:0@<idle>] shell: "feature1" setup',
              '[testing-env/test_image:0/feature1] feature setup',
              '[testing-env/test_image:0/feature2] feature setup with ValueError',
              '[testing-env/test_image:0] sandbox started with ValueError',
              '[testing-env/test_image:0] created -> shutting_down',
              '[testing-env/test_image:0@<idle>] shell: "feature1" teardown',
              '[testing-env/test_image:0/feature1] feature teardown',
              '[testing-env/test_image:0@<idle>] shell: "feature2" teardown',
              '[testing-env/test_image:0/feature2] feature teardown',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown'
              # pylint: enable=line-too-long
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
              # pylint: disable=line-too-long
              '[testing-env] environment started',
              '[testing-env/test_image:0/feature1] feature setup with SandboxStateError',
              '[testing-env/test_image:0] sandbox started with SandboxStateError',
              '[testing-env/test_image:0] created -> shutting_down',
              '[testing-env/test_image:0@<idle>] shell: "feature1" teardown',
              '[testing-env/test_image:0/feature1] feature teardown',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown',
              '[testing-env] environment shutdown',
              # pylint: enable=line-too-long
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
              '[testing-env/test_image:0/feature2] feature teardown with ValueError',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown with FeatureTeardownError',
              # pylint: enable=line-too-long
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
              '[testing-env/test_image:0] in_session -> exiting_session',
              '[testing-env/test_image:0@session1] shell: "feature1" teardown session',
              '[testing-env/test_image:0/feature1@session1] feature teardown session',
              '[testing-env/test_image:0@session1] shell: "feature2" teardown session',
              '[testing-env/test_image:0/feature2@session1] feature teardown session',
              "[testing-env/test_image:0] session 'session1' ended",
              '[testing-env/test_image:0] exiting_session -> acquired',
              '[testing-env/test_image:0] acquired -> shutting_down',
              '[testing-env/test_image:0/feature1] feature teardown with SandboxStateError',
              '[testing-env/test_image:0@<idle>] shell: "feature2" teardown',
              '[testing-env/test_image:0/feature2] feature teardown',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown with FeatureTeardownError',
              # pylint: enable=line-too-long
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
              '[testing-env/test_image:0/feature2@session1] feature setup session with ValueError',
              "[testing-env/test_image:0] session 'session1' started with ValueError",
              '[testing-env/test_image:0] setting_up -> shutting_down',
              '[testing-env/test_image:0@session1] shell: "feature1" teardown',
              '[testing-env/test_image:0/feature1] feature teardown',
              '[testing-env/test_image:0@session1] shell: "feature2" teardown',
              '[testing-env/test_image:0/feature2] feature teardown',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown'
              # pylint: enable=line-too-long
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
              '[testing-env/test_image:0/feature1@session1] feature teardown session with ValueError',
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
              '[testing-env/test_image:0] sandbox shutdown',
              # pylint: enable=line-too-long
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
              '[testing-env/test_image:0/feature1@session1] feature teardown session with SandboxStateError',
              '[testing-env/test_image:0@session1] shell: "feature2" teardown session',
              '[testing-env/test_image:0/feature2@session1] feature teardown session',
              "[testing-env/test_image:0] session 'session1' ended with SandboxStateError",
              '[testing-env/test_image:0] exiting_session -> acquired',
              '[testing-env/test_image:0] acquired -> shutting_down',
              '[testing-env/test_image:0@<idle>] shell: "feature1" teardown',
              '[testing-env/test_image:0/feature1] feature teardown',
              '[testing-env/test_image:0@<idle>] shell: "feature2" teardown',
              '[testing-env/test_image:0/feature2] feature teardown',
              '[testing-env/test_image:0] shutting_down -> offline',
              '[testing-env/test_image:0] sandbox shutdown',
              # pylint: enable=line-too-long
          ]
      )

  def test_feature_teardown_session_calling_end_session(self):
    env = self._create_env(
        features={
            'feature1': TestingFeature(
                call_end_session_on_teardown_session=True
            ),
            'feature2': TestingFeature(),
        },
    )
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

      with env.test_feature() as test_feature:
        self.assertIsInstance(test_feature, TestingFeature)
        self.assertRegex(
            test_feature.session_id,
            r'test_feature-session-[0-9a-f]{7}'
        )

  def test_ping_error(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature(housekeep_interval=0)},
        pool_size=1,
        sandbox_keepalive_interval=0,
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
    event_handler = TestingEventHandler(log_housekeep=False)
    env = TestingEnvironment(
        features={'test_feature': TestingFeature(housekeep_interval=0)},
        pool_size=1,
        housekeep_interval=1.0,
        outage_grace_period=0,
        outage_retry_interval=0.1,
        sandbox_keepalive_interval=0,
        event_handler=event_handler,
    )
    with env:
      with env.sandbox('session1') as sb:
        self.assertEqual(len(env.sandbox_pool), 1)
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
      env.wait_for_housekeeping()
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
    self.event_handler = TestingEventHandler()
    self.env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        outage_grace_period=0,
        outage_retry_interval=0,
        sandbox_keepalive_interval=0,
        event_handler=self.event_handler,
        random_seed=1,
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
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        event_handler=event_handler,
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


if __name__ == '__main__':
  unittest.main()
