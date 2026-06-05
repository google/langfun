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
import inspect
import os
import tempfile
import unittest

from langfun.core.coding.python import sandboxing
import pyglove as pg


class MultiProcessingSandboxTest(unittest.TestCase):
  """Tests for MultiProcessingSandbox."""

  def test_basics(self):
    with tempfile.TemporaryDirectory() as dir1:
      input_file = os.path.join(dir1, 'test.json')
      pg.save(pg.Dict(x=1, y=2), input_file)
      with sandboxing.MultiProcessingSandbox() as sandbox:
        self.assertIsNotNone(sandbox.working_dir)
        sandbox_input_file = sandbox.upload(input_file)
        code = (
            f"""
            import pyglove as pg
            print(pg.load({input_file!r}).x)
            """
        )
        self.assertEqual(
            sandbox.normalize_code(code),
            inspect.cleandoc(
                f"""
                import pyglove as pg
                print(pg.load({sandbox_input_file!r}).x)
                """
            ),
        )
        self.assertEqual(sandbox.run(code).stdout, '1\n')

  def test_bad_code(self):
    with sandboxing.MultiProcessingSandbox() as sandbox:
      with self.assertRaisesRegex(pg.coding.CodeError, '.* is not defined'):
        sandbox.run('print(x)')

  def test_bad_usages(self):
    sandbox = sandboxing.MultiProcessingSandbox()
    with self.assertRaisesRegex(ValueError, 'Sandbox is not set up.'):
      sandbox.upload('abc.txt')


if __name__ == '__main__':
  unittest.main()
