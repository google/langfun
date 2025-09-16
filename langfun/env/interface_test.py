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


class IdTest(unittest.TestCase):

  def test_environment_id(self):
    env_id = interface.EnvironmentId('env@1/a b:c#def')
    self.assertEqual(str(env_id), 'env@1/a b:c#def')
    self.assertEqual(
        env_id.working_dir(root_dir='/tmp'),
        '/tmp/env_1/ab_c_def'
    )
    self.assertIsNone(env_id.working_dir(root_dir=None))

  def test_sandbox_id(self):
    sandbox_id = interface.SandboxId(
        environment_id=interface.EnvironmentId('env'),
        sandbox_id='sandbox'
    )
    self.assertEqual(str(sandbox_id), 'env/sandbox')
    self.assertEqual(
        sandbox_id.working_dir(root_dir='/tmp'),
        '/tmp/env/sandbox'
    )
    self.assertIsNone(sandbox_id.working_dir(root_dir=None))


if __name__ == '__main__':
  unittest.main()
