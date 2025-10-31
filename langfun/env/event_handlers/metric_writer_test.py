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

import unittest

from langfun.env import interface
from langfun.env import test_utils
from langfun.env.event_handlers import metric_writer as metric_writer_lib


class MetricWriterTest(unittest.TestCase):

  def test_write_metric(self):
    writer = metric_writer_lib.MetricWriter(app='test_app')
    env = test_utils.TestingEnvironment(
        features={
            'test_feature1': test_utils.TestingFeature(housekeep_interval=0),
            'test_feature2': test_utils.TestingFeature(housekeep_interval=None),
        },
        pool_size=2,
        outage_grace_period=0,
        outage_retry_interval=0,
        housekeep_interval=10.0,
        sandbox_keepalive_interval=1.0,
        event_handler=writer,
    )
    with env:
      with env.sandbox(session_id='session1') as sb:
        self.assertEqual(sb.test_feature1.num_shell_calls(), 4)

      with self.assertRaises(interface.SandboxStateError):
        with env.sandbox(session_id='session2') as sb:
          sb.shell('echo "bar"', raise_error=RuntimeError)

    self.assertIn(
        writer._sandbox_start.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            error='Success'
        ),
        (2, 3)
    )
    self.assertGreater(
        writer._sandbox_housekeep.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            error='Success'
        ),
        0,
    )
    self.assertEqual(
        writer._sandbox_shutdown.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            error='Success'
        ),
        2
    )
    self.assertEqual(
        writer._sandbox_count.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            status='ready',
        ),
        0
    )
    self.assertEqual(
        writer._sandbox_count.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            status='offline',
        ),
        0
    )
    self.assertEqual(
        writer._feature_setup.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature1',
            error='Success'
        ),
        2,
    )
    self.assertEqual(
        writer._feature_setup.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature2',
            error='Success'
        ),
        2,
    )
    self.assertEqual(
        writer._feature_setup_session.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature1',
            error='Success'
        ),
        3,
    )
    self.assertEqual(
        writer._feature_setup_session.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature2',
            error='Success'
        ),
        3,
    )
    self.assertEqual(
        writer._feature_teardown_session.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature1',
            error='Success'
        ),
        2,
    )
    self.assertEqual(
        writer._feature_teardown_session.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature2',
            error='Success'
        ),
        2,
    )
    self.assertEqual(
        writer._feature_teardown.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature1',
            error='Success'
        ),
        2,
    )
    self.assertEqual(
        writer._feature_teardown.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature2',
            error='Success'
        ),
        2,
    )
    self.assertGreater(
        writer._feature_housekeep.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature1',
            error='Success'
        ),
        0,
    )
    self.assertEqual(
        writer._feature_housekeep.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            feature_name='test_feature2',
            error='Success'
        ),
        0,
    )
    self.assertEqual(
        writer._sandbox_activity.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            activity='shell',
            error='Success'
        ),
        18
    )
    self.assertEqual(
        writer._sandbox_activity.value(
            app='test_app',
            environment_id='testing-env',
            image_id='test_image',
            activity='shell',
            error='RuntimeError'
        ),
        1
    )


if __name__ == '__main__':
  unittest.main()
