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
from langfun.core import language_model as lm_lib
import pyglove as pg


@pg.use_init_args(['failures_before_attempt'])
class MockModel(lm_lib.LanguageModel):
  """A mock model that echo back user prompts."""

  failures_before_attempt: int = 0

  def _sample(self, prompts: list[str]) -> list[lm_lib.LMSamplingResult]:
    context = pg.Dict(attempt=0)

    def fake_sample(prompts):
      if context.attempt >= self.failures_before_attempt:
        return [
            lm_lib.LMSamplingResult([lm_lib.LMSample(
                text=prompt * self.sampling_options.top_k,
                score=self.sampling_options.temperature)])
            for prompt in prompts
        ]
      context.attempt += 1
      raise ValueError('Failed to sample prompts.')

    sample_fn = self._with_max_attempts(
        fake_sample,
        errors_to_retry=ValueError,
        retry_wait_time_fn=lambda i, e: 0.1,
    )
    return sample_fn(prompts)


class LanguageModelTest(unittest.TestCase):
  """Tests for LanguageModel."""

  def test_init(self):
    lm = MockModel(1, temperature=0.5, top_k=2, max_attempts=2)
    self.assertEqual(lm.failures_before_attempt, 1)
    self.assertEqual(lm.sampling_options.temperature, 0.5)
    self.assertEqual(lm.sampling_options.top_k, 2)
    self.assertEqual(lm.max_attempts, 2)

  def test_sample(self):
    lm = MockModel(top_k=1)
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar']),
        [
            lm_lib.LMSamplingResult([lm_lib.LMSample(text='foo', score=0.0)]),
            lm_lib.LMSamplingResult([lm_lib.LMSample(text='bar', score=0.0)]),
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
                [lm_lib.LMSample(text='foo' * 2, score=0.5)]
            ),
            lm_lib.LMSamplingResult(
                [lm_lib.LMSample(text='bar' * 2, score=0.5)]
            ),
        ],
    )
    # Test override individual flags within sampling_options.
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar'], temperature=1.0),
        [
            lm_lib.LMSamplingResult([lm_lib.LMSample(text='foo', score=1.0)]),
            lm_lib.LMSamplingResult([lm_lib.LMSample(text='bar', score=1.0)]),
        ],
    )
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar'], top_k=2, temperature=0.7),
        [
            lm_lib.LMSamplingResult(
                [lm_lib.LMSample(text='foo' * 2, score=0.7)]
            ),
            lm_lib.LMSamplingResult(
                [lm_lib.LMSample(text='bar' * 2, score=0.7)]
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

  def test_retry(self):
    lm = MockModel(
        failures_before_attempt=1, top_k=1,
    )
    with self.assertRaisesRegex(
        ValueError, 'Calling MockModel failed after 1 attempt'
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


if __name__ == '__main__':
  unittest.main()
