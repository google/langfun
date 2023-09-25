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

import copy
import os
import tempfile
import time
import unittest

import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.llms.cache import base
from langfun.core.llms.cache import in_memory

import pyglove as pg


class InMemoryLMCacheTest(unittest.TestCase):

  def test_basics(self):
    cache = in_memory.InMemory()
    lm = fake.StaticSequence(['1', '2', '3'], cache=cache)
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('b'), '2')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('c'), '3')
    self.assertEqual(cache.model_ids(), ['StaticSequence'])
    self.assertEqual(
        list(cache.keys()),
        [
            ('a', (0.0, 1024, 1, 40, None, None)),
            ('b', (0.0, 1024, 1, 40, None, None)),
            ('c', (0.0, 1024, 1, 40, None, None)),
        ])
    self.assertEqual(
        list(cache.keys('StaticSequence')),
        [
            ('a', (0.0, 1024, 1, 40, None, None)),
            ('b', (0.0, 1024, 1, 40, None, None)),
            ('c', (0.0, 1024, 1, 40, None, None)),
        ])

    def cache_entry(response_text):
      return base.LMCacheEntry(
          lf.LMSamplingResult([
              lf.LMSample(lf.AIMessage(response_text), score=1.0)
          ])
      )

    self.assertEqual(
        list(cache.values()),
        [
            cache_entry('1'),
            cache_entry('2'),
            cache_entry('3'),
        ])
    self.assertEqual(
        list(cache.values('StaticSequence')),
        [
            cache_entry('1'),
            cache_entry('2'),
            cache_entry('3'),
        ])
    self.assertEqual(
        list(cache.items()),
        [
            (
                ('a', (0.0, 1024, 1, 40, None, None)),
                cache_entry('1'),
            ),
            (
                ('b', (0.0, 1024, 1, 40, None, None)),
                cache_entry('2'),
            ),
            (
                ('c', (0.0, 1024, 1, 40, None, None)),
                cache_entry('3'),
            )
        ]
    )
    self.assertEqual(
        list(cache.items('StaticSequence')),
        [
            (
                ('a', (0.0, 1024, 1, 40, None, None)),
                cache_entry('1'),
            ),
            (
                ('b', (0.0, 1024, 1, 40, None, None)),
                cache_entry('2'),
            ),
            (
                ('c', (0.0, 1024, 1, 40, None, None)),
                cache_entry('3'),
            )
        ]
    )

    # Test clone/copy semantics.
    self.assertIs(cache.clone()._cache, cache._cache)
    self.assertIs(cache.clone(deep=True)._cache, cache._cache)
    self.assertIs(copy.copy(cache)._cache, cache._cache)
    self.assertIs(copy.deepcopy(cache)._cache, cache._cache)

  def test_ttl(self):
    lm = fake.StaticSequence(['1', '2', '3'], cache=in_memory.InMemory(ttl=1))
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    time.sleep(2)
    self.assertEqual(lm('a'), '2')

  def test_different_sampling_options(self):
    cache = in_memory.InMemory()
    lm = fake.StaticSequence(['1', '2', '3'], cache=cache)
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a', temperature=1.0), '2')
    self.assertEqual(
        list(cache.keys()),
        [
            ('a', (0.0, 1024, 1, 40, None, None)),
            ('a', (1.0, 1024, 1, 40, None, None))
        ])

  def test_different_model(self):
    cache = in_memory.InMemory()
    lm1 = fake.StaticSequence(['1', '2', '3'], cache=cache)
    lm2 = fake.Echo(cache=cache)

    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm2('a'), 'a')
    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm1('b'), '2')
    self.assertEqual(lm2('b'), 'b')

    self.assertEqual(
        list(cache.keys('StaticSequence')),
        [
            ('a', (0.0, 1024, 1, 40, None, None)),
            ('b', (0.0, 1024, 1, 40, None, None)),
        ])
    self.assertEqual(
        list(cache.keys('Echo')),
        [
            ('a', (0.0, 1024, 1, 40, None, None)),
            ('b', (0.0, 1024, 1, 40, None, None)),
        ])
    self.assertEqual(len(cache), 4)
    cache.reset('Echo')
    self.assertEqual(list(cache.keys('Echo')), [])
    cache.reset()
    self.assertEqual(list(cache.keys()), [])

  def test_save_load(self):
    pg.set_load_handler(pg.symbolic.default_load_handler)
    pg.set_save_handler(pg.symbolic.default_save_handler)

    cache = in_memory.InMemory()
    lm1 = fake.StaticSequence(['1', '2', '3'], cache=cache)
    lm2 = fake.Echo(cache=cache)

    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm2('a'), 'a')

    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, 'memory.json')
    cache.save(path)

    cache2 = in_memory.InMemory(path)
    self.assertEqual(cache2._cache, cache._cache)

    lm1 = fake.StaticSequence(['x', 'y'], cache=cache2)
    lm2 = fake.Echo(cache=cache2)

    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm2('a'), 'a')


class UseCacheTest(unittest.TestCase):

  def test_lm_cache(self):
    with in_memory.lm_cache() as c:
      lm = fake.Echo()
      self.assertIs(lm.cache, c)
      lm = fake.Echo(cache=in_memory.InMemory())
      self.assertIsNot(lm.cache, c)

  def test_lm_cache_load_save(self):
    pg.set_load_handler(pg.symbolic.default_load_handler)
    pg.set_save_handler(pg.symbolic.default_save_handler)

    cache = in_memory.InMemory()
    lm = fake.StaticSequence(['1', '2', '3'], cache=cache)
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('b'), '2')

    tmp_dir = tempfile.gettempdir()
    path1 = os.path.join(tmp_dir, 'memory1.json')
    cache.save(path1)

    path2 = os.path.join(tmp_dir, 'memory2.json')

    with in_memory.lm_cache(load=path1, save=path2) as c1:
      self.assertEqual(len(c1), 2)

      lm = fake.StaticSequence(['4', '5', '6'])
      self.assertEqual(lm('a'), '1')
      self.assertEqual(lm('b'), '2')

    with in_memory.lm_cache(load=path2, save=path2) as c2:
      self.assertEqual(len(c2), 2)


if __name__ == '__main__':
  unittest.main()
