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

import contextlib
import io
import unittest
from langfun.core import concurrent
from langfun.core import language_model as lm_lib
from langfun.core import message as message_lib
import pyglove as pg


@pg.use_init_args(['failures_before_attempt'])
class MockModel(lm_lib.LanguageModel):
  """A mock model that echo back user prompts."""

  failures_before_attempt: int = 0

  def _sample(self,
              prompts: list[message_lib.Message]
              ) -> list[lm_lib.LMSamplingResult]:
    context = pg.Dict(attempt=0)

    def fake_sample(prompts):
      if context.attempt >= self.failures_before_attempt:
        return [
            lm_lib.LMSamplingResult([lm_lib.LMSample(  # pylint: disable=g-complex-comprehension
                response=prompt.text * self.sampling_options.top_k,
                score=self.sampling_options.temperature)])
            for prompt in prompts
        ]
      context.attempt += 1
      raise ValueError('Failed to sample prompts.')

    return concurrent.with_retry(
        fake_sample,
        retry_on_errors=ValueError,
        max_attempts=self.max_attempts,
        retry_interval=1,
    )(prompts)


class SimpleCache(lm_lib.LMCache):

  def _on_bound(self):
    super()._on_bound()
    self._cache = {}
    self.cache_hit = 0

  def get(self, lm, prompt):
    del lm
    r = self._cache.get(prompt.text)
    if r is not None:
      self.cache_hit += 1
    return r

  def put(self, lm, prompt, result):
    self._cache[prompt.text] = result

  @property
  def num_records(self):
    return len(self._cache)


class LMSamplingOptionsTest(unittest.TestCase):
  """Tests for LMSamplingOptions."""

  def test_cache_key(self):
    options = lm_lib.LMSamplingOptions()
    key1 = options.cache_key()
    self.assertEqual(key1, (0.0, 1024, 1, 40, None, None))
    with options.override(temperature=1.0, max_tokens=256):
      key2 = options.cache_key()
      self.assertEqual(key2, (1.0, 256, 1, 40, None, None))

      # Make sure key1 does not change upon override.
      self.assertEqual(key1, (0.0, 1024, 1, 40, None, None))


class LanguageModelTest(unittest.TestCase):
  """Tests for LanguageModel."""

  def test_init(self):
    lm = MockModel(1, temperature=0.5, top_k=2, max_attempts=2)
    self.assertEqual(lm.model_id, 'MockModel')
    self.assertEqual(lm.failures_before_attempt, 1)
    self.assertEqual(lm.sampling_options.temperature, 0.5)
    self.assertEqual(lm.sampling_options.top_k, 2)
    self.assertEqual(lm.max_attempts, 2)

  def test_sample(self):
    lm = MockModel(top_k=1)
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar']),
        [
            lm_lib.LMSamplingResult([lm_lib.LMSample('foo', score=0.0)]),
            lm_lib.LMSamplingResult([lm_lib.LMSample('bar', score=0.0)]),
        ],
    )
    # Test override sampling_options.
    self.assertEqual(
        lm.sample(
            prompts=['foo', 'bar'],
            sampling_options=lm_lib.LMSamplingOptions(top_k=2, temperature=0.5),
        ),
        [
            lm_lib.LMSamplingResult(
                [lm_lib.LMSample('foo' * 2, score=0.5)]
            ),
            lm_lib.LMSamplingResult(
                [lm_lib.LMSample('bar' * 2, score=0.5)]
            ),
        ],
    )
    # Test override individual flags within sampling_options.
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar'], temperature=1.0),
        [
            lm_lib.LMSamplingResult([lm_lib.LMSample('foo', score=1.0)]),
            lm_lib.LMSamplingResult([lm_lib.LMSample('bar', score=1.0)]),
        ],
    )
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar'], top_k=2, temperature=0.7),
        [
            lm_lib.LMSamplingResult(
                [lm_lib.LMSample('foo' * 2, score=0.7)]
            ),
            lm_lib.LMSamplingResult(
                [lm_lib.LMSample('bar' * 2, score=0.7)]
            ),
        ],
    )

  def test_call(self):
    lm = MockModel(sampling_options=lm_lib.LMSamplingOptions(top_k=1))
    self.assertEqual(lm(prompt='foo'), 'foo')

    # Test override sampling_options.
    self.assertEqual(
        lm('foo', sampling_options=lm_lib.LMSamplingOptions(top_k=2)), 'foo' * 2
    )
    # Test override individual flags within sampling_options.
    self.assertEqual(lm('foo', top_k=2), 'foo' * 2)

  def test_using_cache(self):
    cache = SimpleCache()
    lm = MockModel(cache=cache, top_k=1)
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar']),
        [
            lm_lib.LMSamplingResult([lm_lib.LMSample('foo', score=0.0)]),
            lm_lib.LMSamplingResult([lm_lib.LMSample('bar', score=0.0)]),
        ])

    self.assertEqual(cache.cache_hit, 0)
    self.assertEqual(cache.num_records, 2)
    self.assertEqual(
        lm.sample(prompts=['foo', 'baz'], temperature=1.0),
        [
            lm_lib.LMSamplingResult([lm_lib.LMSample('foo', score=0.0)]),
            lm_lib.LMSamplingResult([lm_lib.LMSample('baz', score=1.0)]),
        ])
    self.assertEqual(cache.cache_hit, 1)
    self.assertEqual(cache.num_records, 3)

    self.assertEqual(lm('baz', temperature=1.0), 'baz')
    self.assertEqual(cache.cache_hit, 2)
    self.assertEqual(cache.num_records, 3)

    lm = MockModel(cache=cache,
                   top_k=1,
                   failures_before_attempt=1,
                   max_attempts=1)
    try:
      lm.sample(['a']),
    except concurrent.RetryError:
      pass

    lm = MockModel(cache=cache, top_k=1)
    self.assertEqual(lm('a'), 'a')

  def test_retry(self):
    lm = MockModel(
        failures_before_attempt=1, top_k=1,
    )
    with self.assertRaisesRegex(
        concurrent.RetryError, 'Calling .* failed after 1 attempts'
    ):
      lm('foo', max_attempts=1)
    self.assertEqual(lm('foo', max_attempts=2), 'foo')

  def test_debug(self):
    string_io = io.StringIO()
    lm = MockModel(sampling_options=lm_lib.LMSamplingOptions(top_k=1))
    with contextlib.redirect_stdout(string_io):
      self.assertEqual(lm('hi', debug=True), 'hi')

    debug_info = string_io.getvalue()
    self.assertIn('[0] LM INFO', debug_info)
    self.assertIn('[0] PROMPT SENT TO LM', debug_info)
    self.assertIn('[0] LM RESPONSE', debug_info)

  def test_debug_modes(self):
    info_flag = lm_lib.LMDebugMode.INFO
    prompt_flag = lm_lib.LMDebugMode.PROMPT
    response_flag = lm_lib.LMDebugMode.RESPONSE
    debug_prints = {
        info_flag: 'LM INFO',
        prompt_flag: 'PROMPT SENT TO LM',
        response_flag: 'LM RESPONSE'
    }
    debug_modes = [
        info_flag,
        prompt_flag,
        response_flag,
        info_flag | prompt_flag,
        info_flag | response_flag,
        prompt_flag | response_flag,
        info_flag | prompt_flag | response_flag,
    ]

    for debug_mode in debug_modes:
      string_io = io.StringIO()
      lm = MockModel(sampling_options=lm_lib.LMSamplingOptions(top_k=1))

      with contextlib.redirect_stdout(string_io):
        self.assertEqual(lm('hi', debug=debug_mode), 'hi')

      debug_info = string_io.getvalue()
      expected_included = [
          debug_prints[f] for f in lm_lib.LMDebugMode if f in debug_mode
      ]
      expected_excluded = [
          debug_prints[f] for f in lm_lib.LMDebugMode if f not in debug_mode
      ]

      for expected_include in expected_included:
        self.assertIn('[0] ' + expected_include, debug_info)
      for expected_exclude in expected_excluded:
        self.assertNotIn('[0] ' + expected_exclude, debug_info)


if __name__ == '__main__':
  unittest.main()
