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
from typing import Iterator
import unittest
from unittest import mock

from langfun.env import base_feature
from langfun.env import base_sandbox_service
from langfun.env import interface
from langfun.env import test_utils


class IdTest(unittest.TestCase):

  def test_environment_id(self):
    env_id = interface.AbstractEnvironment.Id('env@1/a b:c#def')
    self.assertEqual(str(env_id), 'env@1/a b:c#def')
    self.assertEqual(
        env_id.working_dir(root_dir='/tmp'),
        '/tmp/env_1/ab_c_def'
    )
    self.assertIsNone(env_id.working_dir(root_dir=None))

  def test_sandbox_service_id(self):
    sandbox_service_id = interface.SandboxService.Id(
        environment_id=interface.AbstractEnvironment.Id('env'),
        name='service'
    )
    self.assertEqual(str(sandbox_service_id), 'env/service')
    self.assertEqual(
        sandbox_service_id.working_dir(root_dir='/tmp'),
        '/tmp/env/service'
    )
    self.assertIsNone(sandbox_service_id.working_dir(root_dir=None))

  def test_sandbox_id(self):
    sandbox_id = interface.Sandbox.Id(
        service_id=interface.SandboxService.Id(
            interface.AbstractEnvironment.Id('env'),
            name='docker'
        ),
        image_id='image:2025_01_01_00_00_00',
        sandbox_id='sandbox'
    )
    self.assertEqual(
        str(sandbox_id),
        'env/docker/image:2025_01_01_00_00_00:sandbox'
    )
    self.assertEqual(
        sandbox_id.working_dir(root_dir='/tmp'),
        '/tmp/env/docker/image_2025_01_01_00_00_00/sandbox'
    )
    self.assertIsNone(sandbox_id.working_dir(root_dir=None))

  def test_feature_id(self):
    # For non-sandboxed feature.
    feature_id = interface.Feature.Id(
        container_id=interface.AbstractEnvironment.Id('env'),
        feature_name='feature'
    )
    self.assertEqual(str(feature_id), 'env/feature')
    self.assertEqual(
        feature_id.working_dir(root_dir='/tmp'),
        '/tmp/env/feature'
    )
    self.assertIsNone(feature_id.working_dir(root_dir=None))

    # For sandboxed feature.
    feature_id = interface.Feature.Id(
        container_id=interface.Sandbox.Id(
            service_id=interface.SandboxService.Id(
                interface.AbstractEnvironment.Id('env'),
                name='docker'
            ),
            image_id='image1',
            sandbox_id='0'
        ),
        feature_name='feature'
    )
    self.assertEqual(str(feature_id), 'env/docker/image1:0/feature')
    self.assertEqual(
        feature_id.working_dir(root_dir='/tmp'),
        '/tmp/env/docker/image1/0/feature'
    )
    self.assertIsNone(feature_id.working_dir(root_dir=None))

    # For feature ID without container
    feature_id = interface.Feature.Id(
        container_id=None,
        feature_name='feature'
    )
    self.assertEqual(str(feature_id), 'feature')
    self.assertEqual(
        feature_id.working_dir(root_dir='/tmp'),
        '/tmp/feature'
    )


class TestingSandbox(interface.Sandbox):

  id = interface.Sandbox.Id(
      service_id=interface.SandboxService.Id(
          interface.AbstractEnvironment.Id('env'),
          name='test_sandbox_service'
      ),
      image_id='test_image',
      sandbox_id='0:0'
  )
  image_id = 'test_image'
  features: dict[str, interface.Feature] = {}
  status = interface.Sandbox.Status.READY
  session_id = None

  __test__ = False

  @property
  def sandbox_service(self) -> interface.SandboxService:
    raise NotImplementedError()

  def environment(self) -> interface.AbstractEnvironment:
    pass

  def _on_bound(self) -> None:
    self.activities = []

  def report_state_error(self, error: interface.SandboxStateError) -> None:
    pass

  def start(self) -> None:
    pass

  def shutdown(self) -> None:
    pass

  def start_session(self, session_id: str) -> None:
    pass

  def end_session(self, shutdown_sandbox: bool = False) -> None:
    pass

  @contextlib.contextmanager
  def track_activity(
      self,
      name: str,
      feature: interface.Feature | None = None,
      **kwargs
  ) -> Iterator[None]:
    error = None
    try:
      yield
    except BaseException as e:
      error = e
      raise
    finally:
      self.activities.append((name, error, kwargs))


class DecoratorTest(unittest.TestCase):

  def test_treat_as_sandbox_state_error(self):

    class SandboxA(TestingSandbox):

      @interface.treat_as_sandbox_state_error(errors=(ValueError,))
      def foo(self, bar: str) -> None:
        raise ValueError(bar)

    with self.assertRaises(interface.SandboxStateError):
      SandboxA().foo('foo')

  def test_log_sandbox_activity(self):

    class SandboxB(TestingSandbox):

      @interface.log_activity()
      def bar(self, x: str) -> None:
        pass

    sb = SandboxB()
    sb.bar('foo')
    self.assertEqual(sb.activities, [('bar', None, {'x': 'foo'})])


class InterfaceTests(unittest.TestCase):

  def test_environment_error(self):
    mock_env = mock.MagicMock()
    mock_env.id = interface.AbstractEnvironment.Id('env1')
    err = interface.EnvironmentError('custom error', environment=mock_env)
    self.assertEqual(err.environment, mock_env)
    self.assertEqual(str(err), '[env1] custom error.')

  def test_feature_outage_error(self):
    mock_feature = mock.MagicMock()
    err = interface.FeatureOutageError(
        offline_duration=15.0, feature=mock_feature
    )
    self.assertEqual(err.offline_duration, 15.0)
    self.assertIn('offline for 15.0 seconds', str(err))

  def test_feature_context_manager_errors(self):
    class NonSandboxFeature(base_feature.BaseFeature):
      is_sandbox_based = False

    class SandboxFeature(base_feature.BaseFeature):
      is_sandbox_based = True

    # 1. Sandbox-based feature enter throws AssertionError
    feature1 = SandboxFeature()
    with self.assertRaisesRegex(
        AssertionError, 'Applicable only to non-sandbox-based features'
    ):
      with feature1:
        pass

    # 2. Bound non-sandbox feature enter throws AssertionError
    feature2 = NonSandboxFeature()
    with mock.patch.object(
        NonSandboxFeature,
        'environment',
        new_callable=mock.PropertyMock,
    ) as mock_env_property:
      mock_env_property.return_value = mock.MagicMock()
      with self.assertRaisesRegex(
          AssertionError,
          'Applicable only when the feature is not bound to an environment',
      ):
        with feature2:
          pass

    # 3. Correct enter/exit calls setup/teardown
    feature3 = NonSandboxFeature()
    setup_called = False
    teardown_called = False

    def mock_setup(sandbox=None):
      del sandbox
      nonlocal setup_called
      setup_called = True

    def mock_teardown():
      nonlocal teardown_called
      teardown_called = True

    feature3.setup = mock_setup
    feature3.teardown = mock_teardown

    with feature3:
      self.assertTrue(setup_called)
    self.assertTrue(teardown_called)

  def test_sandbox_getattr_errors(self):
    sb = TestingSandbox()
    # starts with '_' raises AttributeError
    with self.assertRaises(AttributeError):
      sb.__getattr__('_private_attr')
    # is 'features' raises AttributeError
    with self.assertRaises(AttributeError):
      sb.__getattr__('features')

  def test_sandbox_service_id_str_no_env(self):
    sid = interface.SandboxService.Id(None, 'ss1')
    self.assertEqual(str(sid), 'ss1')
    self.assertEqual(sid.working_dir('/tmp'), '/tmp/ss1')

  def test_sandbox_service_image_id_for_error(self):
    class RestrictiveFeature(base_feature.BaseFeature):
      is_sandbox_based = True
      applicable_images = ['restricted_image']

    class DummyService(base_sandbox_service.BaseSandboxService):
      supports_dynamic_image_loading = False
      image_ids = ['test_image']

      def _create_sandbox(self, *args, **kwargs):
        pass

    service = DummyService(image_ids=['test_image'])
    feature = RestrictiveFeature()
    with mock.patch.object(
        RestrictiveFeature,
        'name',
        new_callable=mock.PropertyMock,
    ) as mock_name:
      mock_name.return_value = 'RestrictiveFeature'
      with self.assertRaisesRegex(
          ValueError,
          'is not applicable to any image served by sandbox service',
      ):
        service.image_id_for(feature)

  def test_abstract_environment_default(self):
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    env1 = BareEnvironment(id='env1')
    env2 = BareEnvironment(id='env2')

    # set_default / default
    interface.AbstractEnvironment.set_default(env1)
    self.assertIs(interface.AbstractEnvironment.default(), env1)

    # as_default
    env2.as_default()
    self.assertIs(interface.AbstractEnvironment.default(), env2)

  def test_abstract_environment_obsolete_implementations(self):
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    # Use TestingEventHandler to avoid strict MagicMock type check failure.
    mock_handler = test_utils.TestingEventHandler(
        log_environment_lifecycle=True
    )
    env = BareEnvironment(id='env1', event_handler=mock_handler)

    self.assertEqual(env.stats(), {})
    self.assertRegex(env.new_session_id('hint'), r'hint-session-[0-9a-f]{7}')

    env.on_starting()
    self.assertIn('[env1] environment starting', mock_handler.logs)

    env.on_start(1.0)
    self.assertIn('[env1] environment started', mock_handler.logs)

    env.on_shutting_down()
    self.assertIn('[env1] environment shutting down', mock_handler.logs)

    # Mock start_time for on_shutdown
    env._start_time = time.time() - 10.0
    env.on_shutdown(2.0)
    self.assertIn('[env1] environment shutdown', mock_handler.logs)

  def test_sandbox_session_for_feature_mismatching_image(self):
    class RestrictiveFeature(base_feature.BaseFeature):
      is_sandbox_based = True
      applicable_images = ['restricted_image']

    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    env = BareEnvironment(id='env1')
    mock_service = mock.MagicMock()
    feature = RestrictiveFeature()
    with mock.patch.object(
        RestrictiveFeature,
        'name',
        new_callable=mock.PropertyMock,
    ) as mock_name:
      mock_name.return_value = 'RestrictiveFeature'
      with self.assertRaisesRegex(
          ValueError, "is not applicable to image 'unsupported_image'"
      ):
        # Enter the generator context manager to trigger execution
        with env._sandbox_session_for_feature(
            mock_service, feature, image_id='unsupported_image'
        ):
          pass

  def test_sandbox_working_dir(self):
    sb = TestingSandbox()
    with mock.patch.object(
        TestingSandbox,
        'sandbox_service',
        new_callable=mock.PropertyMock,
    ) as mock_service:
      mock_service.return_value = mock.MagicMock(root_dir='/tmp')
      self.assertEqual(
          sb.working_dir,
          '/tmp/env/test_sandbox_service/test_image/0_0',
      )

  def test_unbound_feature_id_and_working_dir(self):
    # 1. Test base interface.Feature implementation of working_dir
    # (covers interface.py line 789)
    class BareFeature(interface.Feature):
      is_sandbox_based = False

      @property
      def environment(self):
        return None

      @property
      def sandbox(self):
        return None

      @property
      def is_online(self):
        return False

      @property
      def offline_duration(self):
        return 0.0

      def setup(self, sandbox=None):
        pass

      def teardown(self):
        pass

      def setup_session(self, session_id=None):
        pass

      def teardown_session(self, session_id=None):
        pass

      def track_activity(self, name, feature=None, **kwargs):
        pass

    feature1 = BareFeature(root_dir='/tmp/feat')
    self.assertEqual(str(feature1.id), 'bare_feature')
    self.assertEqual(feature1.working_dir, '/tmp/feat/bare_feature')

    # 2. Test base_feature.BaseFeature implementation
    # (covers base_feature.py line 154)
    class NonSandboxFeature(base_feature.BaseFeature):
      is_sandbox_based = False

    feature2 = NonSandboxFeature()
    with mock.patch.object(
        NonSandboxFeature,
        'environment',
        new_callable=mock.PropertyMock,
    ) as mock_env, mock.patch.object(
        NonSandboxFeature,
        'name',
        new_callable=mock.PropertyMock,
    ) as mock_name:
      mock_env.return_value = mock.MagicMock(working_dir='/tmp/env')
      mock_name.return_value = 'non_sandbox_feature'
      self.assertEqual(feature2.working_dir, '/tmp/env/non_sandbox_feature')

  def test_new_session_outage_error(self):
    class NonSandboxFeature(base_feature.BaseFeature):
      is_sandbox_based = False

    feature = NonSandboxFeature()
    with self.assertRaises(interface.FeatureOutageError):
      with feature.new_session('sess'):
        pass

  def test_abstract_environment_working_dir(self):
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    env = BareEnvironment(id='env1', root_dir='/tmp/env')
    self.assertEqual(env.working_dir, '/tmp/env/env1')

  def test_abstract_environment_double_start_error(self):
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    env = BareEnvironment(id='env1')
    env.start()
    with self.assertRaisesRegex(AssertionError, 'already started'):
      env.start()
    env.shutdown()

  def test_abstract_environment_starts_sandbox_services(self):
    # pylint: disable=useless-parent-delegation
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        super()._start()

      def _shutdown(self):
        super()._shutdown()
    # pylint: enable=useless-parent-delegation

    class BareFeature(interface.Feature):
      is_sandbox_based = False

      def _on_init(self):
        super()._on_init()
        self.setup_called = False
        self.teardown_called = False

      @property
      def environment(self):
        return None

      @property
      def sandbox(self):
        return None

      @property
      def is_online(self):
        return True

      @property
      def offline_duration(self):
        return 0.0

      def setup(self, sandbox=None):
        self.setup_called = True

      def teardown(self):
        self.teardown_called = True

      def setup_session(self, session_id=None):
        pass

      def teardown_session(self, session_id=None):
        pass

      def track_activity(self, name, feature=None, **kwargs):
        pass

    mock_service = test_utils.TestingSandboxService()
    feature = BareFeature()
    env = BareEnvironment(
        id='env1',
        sandboxes={'ss': mock_service},
        features={'feat': feature},
    )
    env.start()
    # Ensure starts_sandbox_services called mock_service.start()
    self.assertTrue(mock_service.is_online)
    self.assertTrue(feature.setup_called)
    env.shutdown()
    self.assertTrue(feature.teardown_called)

  def test_abstract_environment_start_errors(self):
    class FaultyEnvironment(interface.AbstractEnvironment):

      def _start(self):
        raise RuntimeError('Start crash')

      def _shutdown(self):
        pass

    mock_handler = test_utils.TestingEventHandler(
        log_environment_lifecycle=True
    )
    env = FaultyEnvironment(id='env1', event_handler=mock_handler)
    with self.assertRaisesRegex(RuntimeError, 'Start crash'):
      env.start()
    self.assertIn(
        '[env1] environment started with RuntimeError', mock_handler.logs
    )

  def test_abstract_environment_shutdown_errors(self):
    class FaultyEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        raise RuntimeError('Shutdown crash')

    mock_handler = test_utils.TestingEventHandler(
        log_environment_lifecycle=True
    )
    env = FaultyEnvironment(id='env1', event_handler=mock_handler)
    env.start()
    with self.assertRaisesRegex(RuntimeError, 'Shutdown crash'):
      env.shutdown()
    self.assertIn(
        '[env1] environment shutdown with RuntimeError', mock_handler.logs
    )

  def test_abstract_environment_getattr_errors(self):
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    env = BareEnvironment(id='env1')
    with self.assertRaises(AttributeError):
      env.__getattr__('_private_attr')
    with self.assertRaises(AttributeError):
      env.__getattr__('sandboxes')

  def test_abstract_environment_sandbox_session(self):
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    mock_service = test_utils.TestingSandboxService()
    env = BareEnvironment(id='env1', sandboxes={'ss': mock_service})
    mock_service._set_status(mock_service.Status.ONLINE)
    env.sandbox(sandbox_service='ss')

  def test_abstract_environment_get_sandbox_service(self):
    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    mock_service = test_utils.TestingSandboxService(
        supports_dynamic_image_loading=False,
        image_ids=['img1'],
    )
    dynamic_service = test_utils.TestingSandboxService(
        supports_dynamic_image_loading=True,
        image_ids=['img3'],
    )

    env = BareEnvironment(
        id='env1',
        sandboxes={'ss': mock_service, 'dynamic': dynamic_service},
    )

    # 1. Match by name
    self.assertIs(
        env._get_sandbox_service(sandbox_service='ss'), mock_service
    )

    # 2. Match by image ID
    self.assertIs(env._get_sandbox_service(image_id='img1'), mock_service)

    # 3. Fallback to dynamic image loading service
    self.assertIs(env._get_sandbox_service(image_id='img2'), dynamic_service)

  def test_abstract_environment_feature_session(self):
    class NonSandboxFeature(base_feature.BaseFeature):
      is_sandbox_based = False

    class SandboxFeature(base_feature.BaseFeature):
      is_sandbox_based = True
      applicable_images = ['img1']

    class BareEnvironment(interface.AbstractEnvironment):

      def _start(self):
        pass

      def _shutdown(self):
        pass

    feature_ns = NonSandboxFeature()
    feature_sb = SandboxFeature()

    mock_service = test_utils.TestingSandboxService(
        image_ids=['img1'],
        features={'feat_sb': feature_sb},
    )
    mock_service._set_status(mock_service.Status.ONLINE)

    env = BareEnvironment(id='env1')
    env._all_features = {
        'feat_ns': (feature_ns, None),
        'feat_sb': (feature_sb, mock_service),
    }

    # 1. Feature not available error
    with self.assertRaisesRegex(ValueError, 'is not available on env1'):
      with env.feature_session('missing'):
        pass

    # 2. Non-sandbox feature session creator returns context
    ctx_ns = env.feature_session('feat_ns')
    self.assertIsNotNone(ctx_ns)

    # 3. Sandbox feature session creator enters context
    # (image_id resolved contextually)
    with mock.patch.object(
        SandboxFeature,
        'name',
        new_callable=mock.PropertyMock,
    ) as mock_name:
      mock_name.return_value = 'feat_sb'
      with env.feature_session('feat_sb'):
        # Verify session acquired sandbox correctly
        self.assertTrue(mock_service.is_online)


if __name__ == '__main__':
  unittest.main()
