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

import asyncio
import contextlib
import time
import unittest

from langfun.core import async_support
import pyglove as pg


class AsyncSupportTest(unittest.TestCase):

  def test_invoke_async(self):

    def foo(x, *, y):
      time.sleep(2)
      return x + y + pg.contextual_value('z', 0)

    t = time.time()
    r = async_support.invoke_async(foo, 1, y=2)
    self.assertLess(time.time() - t, 1)
    with pg.contextual_override(z=3):
      self.assertEqual(asyncio.run(r), 6)

  def test_invoke_sync(self):
    @contextlib.asynccontextmanager
    async def bar(x):
      try:
        yield x
      finally:
        pass

    async def foo(x, *, y):
      time.sleep(2)
      return x + y + pg.contextual_value('z', 0)

    with pg.contextual_override(z=3):
      with async_support.sync_context_manager(bar(1)) as x:
        self.assertEqual(x, 1)
        with async_support.sync_context_manager(bar(2)) as y:
          self.assertEqual(y, 2)
          self.assertEqual(async_support.invoke_sync(foo, 1, y=2), 6)

    with pg.contextual_override(z=2):
      self.assertEqual(async_support.invoke_sync(foo, 1, y=2), 5)


if __name__ == '__main__':
  unittest.main()
