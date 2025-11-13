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
from typing import Iterator
import unittest
from langfun.env import interface


class IdTest(unittest.TestCase):

  def test_environment_id(self):
    env_id = interface.Environment.Id('env@1/a b:c#def')
    self.assertEqual(str(env_id), 'env@1/a b:c#def')
    self.assertEqual(
        env_id.working_dir(root_dir='/tmp'),
        '/tmp/env_1/ab_c_def'
    )
    self.assertIsNone(env_id.working_dir(root_dir=None))

  def test_sandbox_id(self):
    sandbox_id = interface.Sandbox.Id(
        environment_id=interface.Environment.Id('env'),
        image_id='image:2025_01_01_00_00_00',
        sandbox_id='sandbox'
    )
    self.assertEqual(str(sandbox_id), 'env/image:2025_01_01_00_00_00:sandbox')
    self.assertEqual(
        sandbox_id.working_dir(root_dir='/tmp'),
        '/tmp/env/image_2025_01_01_00_00_00/sandbox'
    )
    self.assertIsNone(sandbox_id.working_dir(root_dir=None))

  def test_feature_id(self):
    # For non-sandboxed feature.
    feature_id = interface.Feature.Id(
        container_id=interface.Environment.Id('env'),
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
            environment_id=interface.Environment.Id('env'),
            image_id='image1',
            sandbox_id='0'
        ),
        feature_name='feature'
    )
    self.assertEqual(str(feature_id), 'env/image1:0/feature')
    self.assertEqual(
        feature_id.working_dir(root_dir='/tmp'),
        '/tmp/env/image1/0/feature'
    )
    self.assertIsNone(feature_id.working_dir(root_dir=None))


class TestingSandbox(interface.Sandbox):

  id: interface.Sandbox.Id = interface.Sandbox.Id(
      environment_id=interface.Environment.Id('env'),
      image_id='test_image',
      sandbox_id='0:0'
  )
  image_id: str = 'test_image'
  features: dict[str, interface.Feature] = {}
  status: interface.Sandbox.Status = interface.Sandbox.Status.READY
  session_id: str | None = None

  __test__ = False

  def environment(self) -> interface.Environment:
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


if __name__ == '__main__':
  unittest.main()
