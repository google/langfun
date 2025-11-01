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
import unittest

from langfun.env import interface
from langfun.env import test_utils

TestingEnvironment = test_utils.TestingEnvironment
TestingSandbox = test_utils.TestingSandbox
TestingFeature = test_utils.TestingFeature
TestingEventHandler = test_utils.TestingEventHandler


class BaseEnvironmentTests(unittest.TestCase):

  def test_basics(self):
    env = TestingEnvironment(
        image_ids=['test_image'],
        root_dir='/tmp',
        pool_size=0,
        features={'test_feature': TestingFeature()},
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    self.assertIsNone(interface.Environment.current())
    self.assertEqual(env.image_ids, ['test_image'])
    self.assertFalse(env.supports_dynamic_image_loading)
    self.assertEqual(env.status, interface.Environment.Status.CREATED)
    self.assertFalse(env.is_online)
    self.assertEqual(env.min_pool_size('test_image'), 0)
    self.assertEqual(env.max_pool_size('test_image'), 0)
    self.assertEqual(env.sandbox_pool, {})
    self.assertEqual(env.id, interface.Environment.Id('testing-env'))
    self.assertEqual(env.outage_grace_period, 1)
    self.assertEqual(env.features['test_feature'].name, 'test_feature')

    self.assertIsNone(env.start_time)

    with env:
      self.assertEqual(env.status, interface.Environment.Status.ONLINE)
      self.assertIs(interface.Environment.current(), env)
      self.assertTrue(env.is_online)
      self.assertIsNotNone(env.start_time)
      self.assertEqual(env.offline_duration, 0.0)
      self.assertEqual(env.sandbox_pool, {})
      self.assertEqual(env.working_dir, '/tmp/testing-env')

      with env.sandbox('session1') as sb:
        self.assertEqual(
            sb.id, interface.Sandbox.Id(
                environment_id=env.id,
                image_id=sb.image_id,
                sandbox_id='0')
        )
        self.assertEqual(sb.session_id, 'session1')
        self.assertEqual(sb.working_dir, '/tmp/testing-env/test_image/0')
        self.assertTrue(sb.is_online)
        self.assertIs(sb.test_feature, sb.features['test_feature'])
        self.assertEqual(
            sb.test_feature.working_dir,
            '/tmp/testing-env/test_image/0/test_feature'
        )
        with self.assertRaises(AttributeError):
          _ = sb.test_feature2
      self.assertFalse(sb.is_online)

      with self.assertRaisesRegex(
          ValueError, 'Environment .* does not serve image ID .*'
      ):
        env.sandbox(image_id='test_image2')

      with env.test_feature() as feature:
        self.assertIsInstance(feature, TestingFeature)
        self.assertEqual(
            feature.sandbox.status, interface.Sandbox.Status.IN_SESSION
        )
        self.assertTrue(
            feature.sandbox.session_id.startswith('test_feature-session')
        )

      with self.assertRaises(AttributeError):
        _ = env.test_feature2

  def test_dynamic_image_loading(self):
    env = TestingEnvironment(
        image_ids=[],
        supports_dynamic_image_loading=True,
        pool_size=0,
        features={'test_feature': TestingFeature()},
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    with env:
      with env.sandbox(image_id='test_image2') as sb:
        self.assertEqual(sb.image_id, 'test_image2')

      with self.assertRaisesRegex(
          ValueError, 'Environment .* does not have a default image ID.'
      ):
        env.sandbox()

  def test_dynamic_image_loading_with_pooling(self):
    env = TestingEnvironment(
        image_ids=[],
        supports_dynamic_image_loading=True,
        pool_size=2,
        features={'test_feature': TestingFeature()},
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    with env:
      with env.sandbox(image_id='test_image'):
        self.assertEqual(len(env.sandbox_pool['test_image']), 1)

        with env.sandbox(image_id='test_image'):
          self.assertEqual(len(env.sandbox_pool['test_image']), 2)

          with self.assertRaises(interface.EnvironmentOverloadError):
            with env.sandbox(image_id='test_image'):
              pass
        self.assertEqual(len(env.sandbox_pool['test_image']), 2)

        with env.sandbox(image_id='test_image'):
          self.assertEqual(len(env.sandbox_pool['test_image']), 2)

  def test_image_feature_mappings(self):
    env = TestingEnvironment(
        image_ids=[
            'test_image1',
            'test_image2',
        ],
        features={
            'test_feature': TestingFeature(
                applicable_images=['test_image1.*']
            ),
            'test_feature2': TestingFeature(
                applicable_images=['test_image2.*']
            ),
            'test_feature3': TestingFeature(
                applicable_images=['test_image.*']
            ),
        },
        pool_size=0,
        outage_grace_period=1,
        outage_retry_interval=0,
        sandbox_keepalive_interval=0,
    )
    with env:
      with env.sandbox(image_id='test_image1') as sb:
        self.assertIn('test_feature', sb.features)
        self.assertNotIn('test_feature2', sb.features)
        self.assertIn('test_feature3', sb.features)

      with env.sandbox(image_id='test_image2') as sb:
        self.assertNotIn('test_feature', sb.features)
        self.assertIn('test_feature2', sb.features)
        self.assertIn('test_feature3', sb.features)

      with env.test_feature() as feature:
        self.assertEqual(feature.sandbox.image_id, 'test_image1')

      with self.assertRaisesRegex(
          ValueError, 'Feature .* is not applicable to .*'
      ):
        with env.test_feature(image_id='test_image2'):
          pass

      with env.test_feature2() as feature:
        self.assertEqual(feature.sandbox.image_id, 'test_image2')

      with env.test_feature3() as feature:
        self.assertEqual(feature.sandbox.image_id, 'test_image1')

      with env.test_feature3(image_id='test_image2') as feature:
        self.assertEqual(feature.sandbox.image_id, 'test_image2')

  def test_feature_applicability_check(self):
    with self.assertRaisesRegex(
        ValueError, 'Feature .* is not applicable to .*'
    ):
      TestingEnvironment(
          image_ids=[
              'test_image1',
          ],
          features={
              'test_feature2': TestingFeature(
                  applicable_images=['test_image2.*']
              ),
          },
      )
    env = TestingEnvironment(
        image_ids=[],
        supports_dynamic_image_loading=True,
        features={
            'test_feature2': TestingFeature(
                applicable_images=['test_image2.*']
            ),
        },
        pool_size=0
    )
    with env:
      with self.assertRaisesRegex(
          ValueError, 'No image ID found for feature .*'
      ):
        with env.test_feature2():
          pass

      # Dynamically loaded IDs.
      with env.test_feature2(image_id='test_image2') as feature:
        self.assertEqual(feature.sandbox.image_id, 'test_image2')

  def test_pool_size(self):
    env = TestingEnvironment(
        image_ids=['test_image'],
        pool_size=1,
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    self.assertEqual(env.min_pool_size('test_image'), 1)
    self.assertEqual(env.max_pool_size('test_image'), 1)

    env = TestingEnvironment(
        image_ids=['test_image'],
        pool_size=(0, 256),
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    self.assertEqual(env.min_pool_size('test_image'), 0)
    self.assertEqual(env.max_pool_size('test_image'), 256)

    env = TestingEnvironment(
        image_ids=['test_image'],
        pool_size={
            'test_.*': (0, 128),
            'my.*': (5, 64),
            'exact_image_name': 10,
        },
        outage_grace_period=1,
        outage_retry_interval=0,
    )
    self.assertEqual(env.min_pool_size('test_image'), 0)
    self.assertEqual(env.max_pool_size('test_image'), 128)
    self.assertEqual(env.min_pool_size('my_image'), 5)
    self.assertEqual(env.max_pool_size('my_image'), 64)
    self.assertEqual(env.min_pool_size('exact_image_name'), 10)
    self.assertEqual(env.max_pool_size('exact_image_name'), 10)
    self.assertEqual(env.min_pool_size('some_image'), 0)  # default
    self.assertEqual(env.max_pool_size('some_image'), 256)  # default

  def test_del(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        outage_grace_period=1,
        outage_retry_interval=0,
        sandbox_keepalive_interval=0,
    )
    env.start()
    sb = env.acquire()
    del sb
    del env

  def test_acquire_env_offline(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        outage_grace_period=1,
        outage_retry_interval=0,
        sandbox_keepalive_interval=0,
    )
    with self.assertRaises(interface.EnvironmentOutageError):
      env.acquire()

  def test_acquire_no_pooling(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=0,
        outage_grace_period=1,
        outage_retry_interval=0,
        sandbox_keepalive_interval=0,
    )
    with env:
      sb = env.acquire()
      self.assertEqual(sb.status, interface.Sandbox.Status.ACQUIRED)
      self.assertIsNone(env.working_dir)
      self.assertIsNone(sb.working_dir)
      self.assertIsNone(sb.test_feature.working_dir)

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
        sandbox_keepalive_interval=0,
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
        sandbox_keepalive_interval=0,
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
        sandbox_keepalive_interval=0,
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
        sandbox_keepalive_interval=0,
    )
    with env:
      self.assertEqual(len(env.sandbox_pool['test_image']), 1)
      self.assertEqual(
          env.stats(),
          {
              'sandbox': {
                  'test_image': {
                      'created': 0,
                      'setting_up': 0,
                      'ready': 1,
                      'acquired': 0,
                      'in_session': 0,
                      'exiting_session': 0,
                      'shutting_down': 0,
                      'offline': 0,
                  }
              }
          }
      )
      sb = env.acquire()
      self.assertEqual(sb.status, interface.Sandbox.Status.ACQUIRED)
      self.assertEqual(
          env.stats(),
          {
              'sandbox': {
                  'test_image': {
                      'created': 0,
                      'setting_up': 0,
                      'ready': 0,
                      'acquired': 1,
                      'in_session': 0,
                      'exiting_session': 0,
                      'shutting_down': 0,
                      'offline': 0,
                  }
              }
          }
      )
      self.assertEqual(len(env.sandbox_pool['test_image']), 1)
      sb2 = env.acquire()
      self.assertEqual(sb2.status, interface.Sandbox.Status.ACQUIRED)
      self.assertEqual(len(env.sandbox_pool['test_image']), 2)
      self.assertEqual(
          env.stats(),
          {
              'sandbox': {
                  'test_image': {
                      'created': 0,
                      'setting_up': 0,
                      'ready': 0,
                      'acquired': 2,
                      'in_session': 0,
                      'exiting_session': 0,
                      'shutting_down': 0,
                      'offline': 0,
                  }
              }
          }
      )
    self.assertEqual(
        env.stats(),
        {
            'sandbox': {}
        }
    )

  def test_acquire_with_growing_pool_failure(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=(1, 3),
        outage_grace_period=1,
        outage_retry_interval=0,
        sandbox_keepalive_interval=0,
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

  def test_housekeep_error(self):
    env = TestingEnvironment(
        features={'test_feature': TestingFeature()},
        pool_size=1,
        proactive_session_setup=True,
        outage_grace_period=1,
        outage_retry_interval=0,
        sandbox_keepalive_interval=0,
    )
    with env:
      self.assertEqual(len(env.sandbox_pool), 1)
      self.assertIn('test_image', env.sandbox_pool)
      self.assertEqual(
          env.sandbox_pool['test_image'][0].status,
          interface.Sandbox.Status.READY
      )
      # Make future sandbox setup to fail.
      env.features.test_feature.rebind(
          simulate_setup_error=interface.SandboxStateError,
          skip_notification=True
      )
      with self.assertRaises(interface.SandboxStateError):
        with env.sandbox() as sb:
          sb.shell('bad command', raise_error=interface.SandboxStateError)
      self.assertEqual(sb.status, interface.Sandbox.Status.OFFLINE)
      self.assertEqual(len(sb.state_errors), 1)
      sb_offline_time = time.time()
      while time.time() - sb_offline_time < 10:
        if not env.is_online:
          break
        time.sleep(0.5)
      self.assertFalse(env.is_online)


if __name__ == '__main__':
  unittest.main()
