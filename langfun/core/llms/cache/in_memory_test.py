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
    lm = fake.StaticSequence(['1', '2', '3', '4', '5', '6'], cache=cache)
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a', cache_seed=1), '2')
    self.assertEqual(lm('b'), '3')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('c'), '4')
    self.assertEqual(lm('a', cache_seed=None), '5')
    self.assertEqual(lm('a', cache_seed=None), '6')

    self.assertEqual(cache.model_ids(), ['StaticSequence'])
    self.assertEqual(
        list(cache.keys()),
        [
            ('a', (None, None, 1, 40, None, None), 0),
            ('a', (None, None, 1, 40, None, None), 1),
            ('b', (None, None, 1, 40, None, None), 0),
            ('c', (None, None, 1, 40, None, None), 0),
        ],
    )
    self.assertEqual(
        list(cache.keys('StaticSequence')),
        [
            ('a', (None, None, 1, 40, None, None), 0),
            ('a', (None, None, 1, 40, None, None), 1),
            ('b', (None, None, 1, 40, None, None), 0),
            ('c', (None, None, 1, 40, None, None), 0),
        ],
    )

    def cache_entry(response_text, cache_seed=0):
      return base.LMCacheEntry(
          lf.LMSamplingResult(
              [
                  lf.LMSample(
                      lf.AIMessage(response_text, cache_seed=cache_seed),
                      score=1.0
                  )
              ],
              usage=lf.LMSamplingUsage(
                  1,
                  len(response_text),
                  len(response_text) + 1,
              )
          )
      )

    self.assertEqual(
        list(cache.values()),
        [
            cache_entry('1'),
            cache_entry('2', 1),
            cache_entry('3'),
            cache_entry('4'),
        ],
    )
    self.assertEqual(
        list(cache.values('StaticSequence')),
        [
            cache_entry('1'),
            cache_entry('2', 1),
            cache_entry('3'),
            cache_entry('4'),
        ],
    )
    self.assertEqual(
        list(cache.items()),
        [
            (
                ('a', (None, None, 1, 40, None, None), 0),
                cache_entry('1'),
            ),
            (
                ('a', (None, None, 1, 40, None, None), 1),
                cache_entry('2', 1),
            ),
            (
                ('b', (None, None, 1, 40, None, None), 0),
                cache_entry('3'),
            ),
            (
                ('c', (None, None, 1, 40, None, None), 0),
                cache_entry('4'),
            ),
        ],
    )
    self.assertEqual(
        list(cache.items('StaticSequence')),
        [
            (
                ('a', (None, None, 1, 40, None, None), 0),
                cache_entry('1'),
            ),
            (
                ('a', (None, None, 1, 40, None, None), 1),
                cache_entry('2', 1),
            ),
            (
                ('b', (None, None, 1, 40, None, None), 0),
                cache_entry('3'),
            ),
            (
                ('c', (None, None, 1, 40, None, None), 0),
                cache_entry('4'),
            ),
        ],
    )

    # Test clone/copy semantics.
    self.assertIs(cache.clone()._stats, cache._stats)
    self.assertIs(cache.clone()._cache, cache._cache)
    self.assertIs(cache.clone(deep=True)._cache, cache._cache)
    self.assertIs(cache.clone(deep=True)._stats, cache._stats)
    self.assertIs(copy.copy(cache)._cache, cache._cache)
    self.assertIs(copy.copy(cache)._stats, cache._stats)
    self.assertIs(copy.deepcopy(cache)._cache, cache._cache)
    self.assertIs(copy.deepcopy(cache)._stats, cache._stats)

  def test_ttl(self):
    cache = in_memory.InMemory(ttl=1)
    lm = fake.StaticSequence(['1', '2', '3'], cache=cache)
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    time.sleep(2)
    self.assertEqual(lm('a'), '2')
    self.assertEqual(cache.stats.num_updates, 2)
    self.assertEqual(cache.stats.num_queries, 3)
    self.assertEqual(cache.stats.num_hits, 1)
    self.assertEqual(cache.stats.num_hit_expires, 1)
    self.assertEqual(cache.stats.num_misses, 1)

  def test_different_sampling_options(self):
    cache = in_memory.InMemory()
    lm = fake.StaticSequence(['1', '2', '3'], cache=cache)
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('a', temperature=1.0), '2')
    self.assertEqual(
        list(cache.keys()),
        [
            ('a', (None, None, 1, 40, None, None), 0),
            ('a', (1.0, None, 1, 40, None, None), 0),
        ],
    )

  def test_different_model(self):
    cache = in_memory.InMemory()
    lm1 = fake.StaticSequence(['1', '2', '3'], cache=cache, temperature=0.0)
    lm2 = fake.Echo(cache=cache, temperature=0.0)

    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm2('a'), 'a')
    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm1('b'), '2')
    self.assertEqual(lm2('b'), 'b')

    self.assertEqual(
        list(cache.keys('StaticSequence')),
        [
            ('a', (0.0, None, 1, 40, None, None), 0),
            ('b', (0.0, None, 1, 40, None, None), 0),
        ],
    )
    self.assertEqual(
        list(cache.keys('Echo')),
        [
            ('a', (0.0, None, 1, 40, None, None), 0),
            ('b', (0.0, None, 1, 40, None, None), 0),
        ],
    )
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
    self.assertEqual(cache.stats.num_updates, 2)

    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, 'memory.json')

    # Path does not exist at the moment.
    cache1 = in_memory.InMemory(path)
    self.assertEqual(len(cache1._cache), 0)

    # Now save the cache to path.
    with self.assertRaisesRegex(ValueError, '`path` must be specified'):
      cache.save()
    cache.save(path)

    cache2 = in_memory.InMemory(path)
    self.assertEqual(cache2._cache, cache._cache)

    # Do nothing since there is no updates.
    self.assertEqual(cache2.stats.num_updates, 0)
    cache2.save()

    lm1 = fake.StaticSequence(['x', 'y'], cache=cache2)
    lm2 = fake.Echo(cache=cache2)

    self.assertEqual(lm1('a'), '1')
    self.assertEqual(lm2('a'), 'a')

    # A new entry.
    self.assertEqual(lm2('b'), 'b')
    self.assertEqual(lm2('c'), 'c')
    self.assertEqual(cache2.stats.num_updates, 2)
    cache2.save()


class LmCacheTest(unittest.TestCase):

  def test_lm_cache(self):
    with in_memory.lm_cache() as c:
      lm = fake.Echo()
      self.assertIs(lm.cache, c)
      lm = fake.Echo(cache=in_memory.InMemory())
      self.assertIsNot(lm.cache, c)

  def test_lm_cache_with_file(self):
    pg.set_load_handler(pg.symbolic.default_load_handler)
    pg.set_save_handler(pg.symbolic.default_save_handler)

    cache = in_memory.InMemory()
    lm = fake.StaticSequence(['1', '2', '3'], cache=cache)
    self.assertEqual(lm('a'), '1')
    self.assertEqual(lm('b'), '2')

    tmp_dir = tempfile.gettempdir()
    path1 = os.path.join(tmp_dir, 'memory1.json')
    cache.save(path1)

    with in_memory.lm_cache(path1) as c1:
      self.assertEqual(len(c1), 2)

      lm = fake.StaticSequence(['4', '5', '6'])
      self.assertEqual(lm('a'), '1')
      self.assertEqual(lm('b'), '2')
      self.assertEqual(lm('c'), '4')

    with in_memory.lm_cache(path1) as c2:
      self.assertEqual(len(c2), 3)


if __name__ == '__main__':
  unittest.main()
