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
from langfun.core import modality
from langfun.core.llms.cache import in_memory
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
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(  # pylint: disable=g-complex-comprehension
                        response=prompt.text * self.sampling_options.top_k,
                        score=self.sampling_options.temperature or -1.0,
                    )
                ],
                usage=lm_lib.LMSamplingUsage(
                    prompt_tokens=100,
                    completion_tokens=100,
                    total_tokens=200,
                ),
            )
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


class MockScoringModel(MockModel):

  def _score(
      self,
      prompt: message_lib.Message | list[message_lib.Message],
      completions: list[message_lib.Message],
      **kwargs
  ) -> list[lm_lib.LMScoringResult]:
    return [
        lm_lib.LMScoringResult(score=-i * 1.0) for i in range(len(completions))
    ]


class LMSamplingOptionsTest(unittest.TestCase):
  """Tests for LMSamplingOptions."""

  def test_cache_key(self):
    options = lm_lib.LMSamplingOptions()
    key1 = options.cache_key()
    self.assertEqual(key1, (None, None, 1, 40, None, None))
    with options.override(temperature=1.0, max_tokens=256):
      key2 = options.cache_key()
      self.assertEqual(key2, (1.0, 256, 1, 40, None, None))

      # Make sure key1 does not change upon override.
      self.assertEqual(key1, (None, None, 1, 40, None, None))


class LanguageModelTest(unittest.TestCase):
  """Tests for LanguageModel."""

  def test_init(self):
    lm = MockModel(1, temperature=0.5, top_k=2, max_attempts=2)
    self.assertEqual(lm.model_id, 'MockModel')
    self.assertEqual(lm.resource_id, 'MockModel')
    self.assertIsNone(lm.max_concurrency)
    self.assertEqual(lm.failures_before_attempt, 1)
    self.assertEqual(lm.sampling_options.temperature, 0.5)
    self.assertEqual(lm.sampling_options.top_k, 2)
    self.assertEqual(lm.max_attempts, 2)

  def test_sample(self):
    lm = MockModel(top_k=1)
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar']),
        [
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'foo',
                            score=-1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=-1.0,
                        logprobs=None,
                    )
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'bar',
                            score=-1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=-1.0,
                        logprobs=None,
                    )
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
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
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'foo' * 2,
                            score=0.5,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=0.5,
                        logprobs=None,
                    ),
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'bar' * 2,
                            score=0.5,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=0.5,
                        logprobs=None,
                    ),
                ],
                usage=lm_lib.LMSamplingUsage(
                    prompt_tokens=100, completion_tokens=100, total_tokens=200
                ),
            ),
        ]
    )
    # Test override individual flags within sampling_options.
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar'], temperature=1.0),
        [
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'foo',
                            score=1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=1.0,
                        logprobs=None,
                    ),
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'bar',
                            score=1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=1.0,
                        logprobs=None,
                    ),
                ],
                usage=lm_lib.LMSamplingUsage(
                    prompt_tokens=100, completion_tokens=100, total_tokens=200
                ),
            ),
        ]
    )
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar'], top_k=2, temperature=0.7),
        [
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'foo' * 2,
                            score=0.7,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=0.7,
                        logprobs=None,
                    ),
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'bar' * 2,
                            score=0.7,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=0.7,
                        logprobs=None,
                    ),
                ],
                usage=lm_lib.LMSamplingUsage(
                    prompt_tokens=100, completion_tokens=100, total_tokens=200
                ),
            ),
        ]
    )

  def test_call(self):
    lm = MockModel(sampling_options=lm_lib.LMSamplingOptions(top_k=1))
    response = lm(prompt='foo')
    self.assertEqual(response.text, 'foo')
    self.assertEqual(response.score, -1.0)
    self.assertIsNone(response.logprobs)
    self.assertEqual(response.usage, lm_lib.LMSamplingUsage(100, 100, 200))

    # Test override sampling_options.
    self.assertEqual(
        lm('foo', sampling_options=lm_lib.LMSamplingOptions(top_k=2)), 'foo' * 2
    )
    # Test override individual flags within sampling_options.
    self.assertEqual(lm('foo', top_k=2), 'foo' * 2)

  def test_using_cache(self):
    cache = in_memory.InMemory()
    lm = MockModel(cache=cache, top_k=1)
    self.assertEqual(
        lm.sample(prompts=['foo', 'bar']),
        [
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'foo',
                            cache_seed=0,
                            score=-1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=-1.0,
                        logprobs=None,
                    )
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'bar',
                            cache_seed=0,
                            score=-1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=-1.0,
                        logprobs=None,
                    )
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
        ],
    )
    self.assertEqual(cache.stats.num_queries, 2)
    self.assertEqual(cache.stats.num_hits, 0)
    self.assertEqual(cache.stats.num_updates, 2)

    self.assertEqual(lm('foo'), 'foo')
    self.assertEqual(lm('bar'), 'bar')
    self.assertEqual(cache.stats.num_queries, 4)
    self.assertEqual(cache.stats.num_hits, 2)
    self.assertEqual(cache.stats.num_updates, 2)

    # Test cache_seed=None will not trigger cache query.
    self.assertEqual(lm('foo', cache_seed=None), 'foo')
    self.assertEqual(cache.stats.num_queries, 4)
    self.assertEqual(cache.stats.num_updates, 2)

    self.assertEqual(
        lm.sample(prompts=['foo', 'baz'], temperature=1.0),
        [
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'foo',
                            cache_seed=0,
                            score=1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=1.0,
                        logprobs=None,
                    )
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
            lm_lib.LMSamplingResult(
                [
                    lm_lib.LMSample(
                        message_lib.AIMessage(
                            'baz',
                            cache_seed=0,
                            score=1.0,
                            logprobs=None,
                            usage=lm_lib.LMSamplingUsage(100, 100, 200),
                            tags=[message_lib.Message.TAG_LM_RESPONSE],
                        ),
                        score=1.0,
                        logprobs=None,
                    )
                ],
                usage=lm_lib.LMSamplingUsage(100, 100, 200),
            ),
        ],
    )
    self.assertEqual(cache.stats.num_queries, 6)
    self.assertEqual(cache.stats.num_hits, 2)
    self.assertEqual(cache.stats.num_updates, 4)

    self.assertEqual(lm('baz', temperature=1.0), 'baz')
    self.assertEqual(cache.stats.num_hits, 3)
    self.assertEqual(cache.stats.num_updates, 4)

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
    class Image(modality.Modality):
      def to_bytes(self):
        return b'fake_image'

    string_io = io.StringIO()
    lm = MockModel(sampling_options=lm_lib.LMSamplingOptions(top_k=1))
    with contextlib.redirect_stdout(string_io):
      self.assertEqual(
          lm(message_lib.UserMessage(
              'hi {{image}}', image=Image()), debug=True),
          'hi {{image}}')

    debug_info = string_io.getvalue()
    self.assertIn('[0] LM INFO', debug_info)
    self.assertIn('[0] PROMPT SENT TO LM', debug_info)
    self.assertIn('[0] MODALITY OBJECTS SENT TO LM', debug_info)
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
          debug_prints[f]
          for f in lm_lib.LMDebugMode
          if f != lm_lib.LMDebugMode.NONE and f in debug_mode
      ]
      expected_excluded = [
          debug_prints[f]
          for f in lm_lib.LMDebugMode
          if f != lm_lib.LMDebugMode.NONE and f not in debug_mode
      ]

      for expected_include in expected_included:
        self.assertIn('[0] ' + expected_include, debug_info)
      for expected_exclude in expected_excluded:
        self.assertNotIn('[0] ' + expected_exclude, debug_info)

  def test_score(self):
    info_flag = lm_lib.LMDebugMode.INFO
    prompt_flag = lm_lib.LMDebugMode.PROMPT
    response_flag = lm_lib.LMDebugMode.RESPONSE
    debug_prints = {
        info_flag: 'LM INFO',
        prompt_flag: 'SCORING LM WITH PROMPT',
        response_flag: 'SCORING COMPLETED',
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

    class Image(modality.Modality):
      def to_bytes(self):
        return b'fake_image'

    for debug_mode in debug_modes:
      string_io = io.StringIO()
      lm = MockScoringModel()

      with contextlib.redirect_stdout(string_io):
        self.assertEqual(
            lm.score(
                message_lib.UserMessage('hi {{image}}', image=Image()),
                ['1', '2'], debug=debug_mode),
            [
                lm_lib.LMScoringResult(score=-0.0),
                lm_lib.LMScoringResult(score=-1.0),
            ],
        )

        self.assertEqual(
            lm.score(
                [message_lib.UserMessage('hi {{image}}', image=Image()),
                 message_lib.UserMessage('hi {{image}}', image=Image())],
                ['1', '2'], debug=debug_mode),
            [
                lm_lib.LMScoringResult(score=-0.0),
                lm_lib.LMScoringResult(score=-1.0),
            ],
        )

      debug_info = string_io.getvalue()
      expected_included = [
          debug_prints[f]
          for f in lm_lib.LMDebugMode
          if f != lm_lib.LMDebugMode.NONE and f in debug_mode
      ]
      expected_excluded = [
          debug_prints[f]
          for f in lm_lib.LMDebugMode
          if f != lm_lib.LMDebugMode.NONE and f not in debug_mode
      ]

      for expected_include in expected_included:
        self.assertIn('[0] ' + expected_include, debug_info)
      for expected_exclude in expected_excluded:
        self.assertNotIn('[0] ' + expected_exclude, debug_info)

      if debug_mode & lm_lib.LMDebugMode.PROMPT:
        self.assertIn('[0] MODALITY OBJECTS SENT TO LM', debug_info)

  def test_score_with_unmatched_prompt_and_completions(self):
    with self.assertRaises(ValueError):
      MockScoringModel().score(['hi',], ['1', '2', '3'])

  def test_score_with_unsupported_model(self):
    with self.assertRaises(NotImplementedError):
      MockModel().score('hi', ['1', '2'])

  def test_rate_to_max_concurrency_no_rpm_no_tpm(self) -> None:
    lm = MockModel()
    self.assertEqual(
        lm_lib.DEFAULT_MAX_CONCURRENCY,
        lm.rate_to_max_concurrency(requests_per_min=0, tokens_per_min=0),
    )
    self.assertEqual(
        lm_lib.DEFAULT_MAX_CONCURRENCY,
        lm.rate_to_max_concurrency(requests_per_min=-1, tokens_per_min=-1),
    )

  def test_rate_to_max_concurrency_only_rpm_specified_uses_rpm(self) -> None:
    lm = MockModel()
    test_rpm = 1e4
    self.assertEqual(
        lm.rate_to_max_concurrency(requests_per_min=test_rpm),
        int(test_rpm / 60)
    )

  def test_rate_to_max_concurrency_tpm_specified_uses_tpm(self) -> None:
    lm = MockModel()
    test_tpm = 1e7
    self.assertEqual(
        lm.rate_to_max_concurrency(requests_per_min=1, tokens_per_min=test_tpm),
        int(test_tpm / lm_lib.TOKENS_PER_REQUEST / 60),
    )

  def test_rate_to_max_concurrency_small_rate_returns_one(self) -> None:
    lm = MockModel()
    self.assertEqual(lm.rate_to_max_concurrency(requests_per_min=1), 1)
    self.assertEqual(lm.rate_to_max_concurrency(tokens_per_min=1), 1)


if __name__ == '__main__':
  unittest.main()
