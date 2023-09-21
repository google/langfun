# Copyright 2023 The Langfun Authors
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
"""Tests for language model."""

import time
import unittest

from langfun.core.llms import fake
from langfun.core.llms.cache import in_memory


class InMemoryLMCacheTest(unittest.TestCase):

  def test_basics(self):
    in_memory.InMemory().reset()
    lm = fake.StaticSequence(['1', '2', '3'], cache=in_memory.InMemory())
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('b'), '2')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('c'), '3')

  def test_ttl(self):
    in_memory.InMemory().reset()
    lm = fake.StaticSequence(['1', '2', '3'], cache=in_memory.InMemory(ttl=1))
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    time.sleep(2)
    self.assertEqual(lm('a'), '2')

  def test_different_sampling_options(self):
    in_memory.InMemory().reset()
    lm = fake.StaticSequence(['1', '2', '3'], cache=in_memory.InMemory())
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a', temperature=1.0), '2')

  def test_different_model(self):
    lm1 = fake.StaticSequence(['1', '2', '3'], cache=in_memory.InMemory())
    lm2 = fake.Echo(cache=in_memory.InMemory())

    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm2('a'), 'a')
    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm1('b'), '2')
    self.assertEqual(lm2('b'), 'b')


if __name__ == '__main__':
  unittest.main()
