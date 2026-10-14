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
"""Tests for Gemini API."""

import base64
import datetime
from typing import Any
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core.llms import gemini
import pyglove as pg
import requests


example_image = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x18\x00\x00\x00\x18\x04'
    b'\x03\x00\x00\x00\x12Y \xcb\x00\x00\x00\x18PLTE\x00\x00'
    b'\x00fff_chaag_cg_ch^ci_ciC\xedb\x94\x00\x00\x00\x08tRNS'
    b'\x00\n\x9f*\xd4\xff_\xf4\xe4\x8b\xf3a\x00\x00\x00>IDATx'
    b'\x01c \x05\x08)"\xd8\xcc\xae!\x06pNz\x88k\x19\\Q\xa8"\x10'
    b'\xc1\x14\x95\x01%\xc1\n\xa143Ta\xa8"D-\x84\x03QM\x98\xc3'
    b'\x1a\x1a\x1a@5\x0e\x04\xa0q\x88\x05\x00\x07\xf8\x18\xf9'
    b'\xdao\xd0|\x00\x00\x00\x00IEND\xaeB`\x82'
)


def mock_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  c = pg.Dict(json['generationConfig'])
  parts = []
  if system_instruction := json.get('systemInstruction'):
    parts.extend([p['text'] for p in system_instruction.get('parts', [])])

  # Add text from the main contents.
  for c_item in json.get('contents', []):
    for p in c_item.get('parts', []):
      parts.append(p['text'])
  content = '\n'.join(parts)
  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'candidates': [
          {
              'content': {
                  'parts': [
                      {
                          'text': (
                              f'This is a response to {content} with '
                              f'temperature={c.temperature}, '
                              f'top_p={c.topP}, '
                              f'top_k={c.topK}, '
                              f'max_tokens={c.maxOutputTokens}, '
                              f'stop={"".join(c.stopSequences)}.'
                          ),
                      },
                      {
                          'text': 'This is the thought.',
                          'thought': True,
                      }
                  ],
              },
          },
      ],
      'usageMetadata': {
          'promptTokenCount': 3,
          'candidatesTokenCount': 4,
      }
  }).encode()
  return response


class GeminiTest(unittest.TestCase):
  """Tests for Vertex model with REST API."""

  def test_dir(self):
    self.assertIn('gemini-1.5-pro', gemini.Gemini.dir())

  def test_knowledge_cutoff(self):
    model = gemini.Gemini('gemini-3.1-pro-preview', api_endpoint='')
    self.assertEqual(
        model.model_info.knowledge_cutoff,
        datetime.date(2025, 1, 31),
    )

  def test_knowledge_cutoff_default(self):
    model = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
    self.assertIsNone(model.model_info.knowledge_cutoff)

  def test_estimate_cost(self):
    model = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
    self.assertEqual(
        model.estimate_cost(
            lf.LMSamplingUsage(
                prompt_tokens=100_000,
                completion_tokens=1000,
                total_tokens=101_000,
            )
        ),
        0.13
    )
    # Prompt length greater than 128K.
    self.assertEqual(
        model.estimate_cost(
            lf.LMSamplingUsage(
                prompt_tokens=200_000,
                completion_tokens=1000,
                total_tokens=201_000,
            )
        ),
        0.51
    )
    # Test with cached prompt tokens (regular pricing < 128K)
    self.assertEqual(
        model.estimate_cost(
            lf.LMSamplingUsage(
                prompt_tokens=100_000,
                completion_tokens=1000,
                total_tokens=101_000,
                cached_prompt_tokens=50_000,
            )
        ),
        # (50K non-cached * 1.25 + 50K cached * 0.3125 + 1K output * 5) / 1M
        # = (62500 + 15625 + 5000) / 1M = 0.083125
        0.083125,
    )
    # Test with cached prompt tokens (128K+ pricing)
    self.assertEqual(
        model.estimate_cost(
            lf.LMSamplingUsage(
                prompt_tokens=200_000,
                completion_tokens=1000,
                total_tokens=201_000,
                cached_prompt_tokens=100_000,
            )
        ),
        # (100K non-cached * 2.5 + 100K cached * 0.625 + 1K output * 10) / 1M
        # = (250000 + 62500 + 10000) / 1M = 0.3225
        0.3225,
    )

  def test_generation_config(self):
    model = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
    json_schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
        },
        'required': ['name'],
        'title': 'Person',
    }
    actual = model._generation_config(
        lf.UserMessage('hi', json_schema=json_schema),
        lf.LMSamplingOptions(
            temperature=2.0,
            top_p=1.0,
            top_k=20,
            max_tokens=1024,
            stop=['\n'],
        ),
    )
    self.assertEqual(
        actual,
        dict(
            candidateCount=1,
            temperature=2.0,
            topP=1.0,
            topK=20,
            maxOutputTokens=1024,
            stopSequences=['\n'],
            responseLogprobs=False,
            logprobs=None,
            seed=None,
            responseMimeType='application/json',
            responseSchema={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'}
                },
                'required': ['name'],
                'title': 'Person',
            }
        ),
    )

    # Add test for thinkingConfig.
    actual = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(
            max_thinking_tokens=100,
        ),
    )
    self.assertEqual(
        actual,
        dict(
            candidateCount=1,
            temperature=None,
            topP=None,
            topK=40,
            maxOutputTokens=None,
            stopSequences=None,
            responseLogprobs=False,
            logprobs=None,
            seed=None,
            thinkingConfig={'includeThoughts': True, 'thinkingBudget': 100},
        ),
    )

    # Add test for thinkingConfig with thinking_level.
    actual = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(
            thinking_level='high',
        ),
    )
    self.assertEqual(
        actual,
        dict(
            candidateCount=1,
            temperature=None,
            topP=None,
            topK=40,
            maxOutputTokens=None,
            stopSequences=None,
            responseLogprobs=False,
            logprobs=None,
            seed=None,
            thinkingConfig={'thinkingLevel': 'high'},
        ),
    )

    # Add test for thinkingConfig with both max_thinking_tokens and
    # thinking_level.
    actual = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(
            max_thinking_tokens=100,
            thinking_level='low',
        ),
    )
    self.assertEqual(
        actual,
        dict(
            candidateCount=1,
            temperature=None,
            topP=None,
            topK=40,
            maxOutputTokens=None,
            stopSequences=None,
            responseLogprobs=False,
            logprobs=None,
            seed=None,
            thinkingConfig={
                'includeThoughts': True,
                'thinkingBudget': 100,
                'thinkingLevel': 'low',
            },
        ),
    )

    with self.assertRaisesRegex(
        ValueError, '`json_schema` must be a dict, got'
    ):
      model._generation_config(
          lf.UserMessage('hi', json_schema='not a dict'),
          lf.LMSamplingOptions(),
      )

  def test_media_resolution_for_gemini3(self):
    model = gemini.Gemini('gemini-3-pro-preview', api_endpoint='')
    config = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(),
    )
    self.assertEqual(config.get('mediaResolution'), 'MEDIA_RESOLUTION_HIGH')

    model = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
    config = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(),
    )
    self.assertIsNone(config.get('mediaResolution'))

  def test_media_resolution_for_gemini4(self):
    model = gemini.Gemini('gemini-4-argon', api_endpoint='')
    config = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(),
    )
    self.assertEqual(config.get('mediaResolution'), 'MEDIA_RESOLUTION_HIGH')

  def test_no_media_resolution_for_unversioned_model(self):
    model = gemini.Gemini('gemini-exp-1206', api_endpoint='')
    config = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(),
    )
    self.assertIsNone(config.get('mediaResolution'))

  def test_thinking_level_medium(self):
    model = gemini.Gemini('gemini-4-argon', api_endpoint='')
    config = model._generation_config(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(thinking_level='medium'),
    )
    self.assertEqual(config['thinkingConfig'], {'thinkingLevel': 'medium'})

  def test_request_tool_config(self):
    model = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
    request = model.request(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(),
    )
    self.assertEqual(
        request.get('toolConfig'),
        {'functionCallingConfig': {'mode': 'NONE'}},
    )

  def test_call_model(self):
    with mock.patch('requests.Session.post') as mock_generate:
      mock_generate.side_effect = mock_requests_post

      lm = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
      r = lm(
          'hello',
          temperature=2.0,
          top_p=1.0,
          top_k=20,
          max_tokens=1024,
          stop='\n',
      )
      self.assertEqual(
          r.text,
          (
              'This is a response to hello with temperature=2.0, '
              'top_p=1.0, top_k=20, max_tokens=1024, stop=\n.'
          ),
      )
      self.assertEqual(r.metadata.thought, 'This is the thought.')
      self.assertEqual(r.metadata.usage.prompt_tokens, 3)
      self.assertEqual(r.metadata.usage.completion_tokens, 4)

  def test_call_model_with_context_limit_error(self):
    def mock_requests_post_error(*args, **kwargs):
      del args, kwargs
      response = requests.Response()
      response.status_code = 400
      response._content = b'exceeds the maximum number of tokens.'
      return response

    with mock.patch('requests.Session.post') as mock_generate:
      mock_generate.side_effect = mock_requests_post_error
      lm = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
      with self.assertRaisesRegex(
          lf.ContextLimitError, 'exceeds the maximum number of tokens.'
      ):
        lm('hello')

  def test_call_model_with_max_tokens_error(self):
    def mock_requests_post_error(*args, **kwargs):
      del args, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = pg.to_json_str({
          'candidates': [
              {
                  'finishReason': 'MAX_TOKENS',
                  'content': {
                      'parts': [
                          {
                              'text': 'This is'
                          }
                      ]
                  }
              },
          ],
          'usageMetadata': {
              'promptTokenCount': 3,
              'candidatesTokenCount': 4,
          }
      }).encode()
      return response

    with mock.patch('requests.Session.post') as mock_generate:
      mock_generate.side_effect = mock_requests_post_error
      lm = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
      m = lm('hello')
      self.assertEqual(m.metadata.finish_reason, 'MAX_TOKENS')
      self.assertEqual(m.text, 'This is')

  def test_result_with_missing_prompt_token_count(self):
    """Tests that missing promptTokenCount defaults to 0."""
    model = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
    json_response = {
        'candidates': [
            {
                'content': {
                    'parts': [{'text': 'response'}],
                },
            },
        ],
        'usageMetadata': {
            'candidatesTokenCount': 4,
            'totalTokenCount': 4,
        },
    }
    result = model.result(json_response)
    self.assertEqual(result.usage.prompt_tokens, 0)
    self.assertEqual(result.usage.completion_tokens, 4)
    self.assertEqual(result.usage.total_tokens, 4)

  def test_call_model_with_system_message(self):
    with mock.patch('requests.Session.post') as mock_generate:
      mock_generate.side_effect = mock_requests_post

      lm = gemini.Gemini('gemini-1.5-pro', api_endpoint='')
      r = lm(
          lf.UserMessage('hello', system_message=lf.SystemMessage('system')),
          temperature=2.0,
          top_p=1.0,
          top_k=20,
          max_tokens=1024,
          stop='\n',
      )
      self.assertEqual(
          r.text,
          (
              'This is a response to system\nhello with temperature=2.0, '
              'top_p=1.0, top_k=20, max_tokens=1024, stop=\n.'
          ),
      )


class GeminiEmptyGenerationTest(unittest.TestCase):
  """Tests `Gemini.result()` classification of answer-less generations."""

  def _model(self) -> gemini.Gemini:
    return gemini.Gemini('gemini-3.1-pro-preview', api_endpoint='')

  def _response(
      self,
      parts: list[dict[str, Any]] | None,
      finish_reason: str | None,
      *,
      candidates_token_count: int,
      thoughts_token_count: int = 0,
  ) -> dict[str, Any]:
    """Builds a single-candidate response payload.

    Args:
      parts: The candidate's content parts. `None` omits the enclosing
        `content` key entirely, the shape a service may return when it has no
        answer at all.
      finish_reason: The candidate's `finishReason`, omitted when None.
      candidates_token_count: Response-level `candidatesTokenCount`.
      thoughts_token_count: Response-level `thoughtsTokenCount`.

    Returns:
      A response payload accepted by `Gemini.result()`.
    """
    candidate: dict[str, Any] = {}
    if parts is not None:
      candidate['content'] = {'role': 'model', 'parts': parts}
    if finish_reason is not None:
      candidate['finishReason'] = finish_reason
    return {
        'candidates': [candidate],
        'usageMetadata': {
            'promptTokenCount': 12,
            'candidatesTokenCount': candidates_token_count,
            'thoughtsTokenCount': thoughts_token_count,
            'totalTokenCount': (
                12 + candidates_token_count + thoughts_token_count
            ),
        },
    }

  def test_error_is_not_retryable(self):
    """The new error must stay OUT of the retryable error hierarchy."""
    self.assertTrue(
        issubclass(gemini.GeminiEmptyGenerationError, lf.LMError)
    )
    self.assertFalse(
        issubclass(gemini.GeminiEmptyGenerationError, lf.RetryableLMError)
    )
    self.assertFalse(
        issubclass(gemini.GeminiEmptyGenerationError, lf.EmptyGenerationError)
    )

  def test_thought_only_max_tokens_raises(self):
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result(
          self._response(
              [{'text': 'thinking...', 'thought': True}],
              'MAX_TOKENS',
              candidates_token_count=0,
              thoughts_token_count=8192,
          )
      )
    message = str(ctx.exception)
    self.assertIn('MAX_TOKENS', message)
    self.assertIn('8192', message)

  def test_no_content_max_tokens_raises(self):
    """MAX_TOKENS with no parts at all is equally deterministic."""
    with self.assertRaises(gemini.GeminiEmptyGenerationError):
      self._model().result(
          self._response([], 'MAX_TOKENS', candidates_token_count=0)
      )

  def test_stop_with_zero_output_tokens_raises(self):
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result(
          self._response([], 'STOP', candidates_token_count=0)
      )
    self.assertIn('STOP', str(ctx.exception))

  def test_missing_content_raises(self):
    """A candidate with no `content` key must not surface as a raw KeyError."""
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result(
          self._response(
              None,
              'MAX_TOKENS',
              candidates_token_count=0,
              thoughts_token_count=61,
          )
      )
    message = str(ctx.exception)
    # A bare KeyError stringifies to just "'content'", carrying no diagnosis.
    self.assertNotEqual(message, "'content'")
    self.assertIn('content', message)
    self.assertIn('MAX_TOKENS', message)
    self.assertIn('61', message)
    self.assertIn(self._model().model_id, message)

  def test_null_content_raises(self):
    """A present-but-null `content` is just as unparseable as a missing one."""
    with self.assertRaises(gemini.GeminiEmptyGenerationError):
      self._model().result({
          'candidates': [{'content': None, 'finishReason': 'STOP'}],
          'usageMetadata': {'promptTokenCount': 12},
      })

  def test_missing_content_raises_under_unrelated_finish_reason(self):
    """The guard is unconditional: there is no answer to fall back to.

    Unlike an empty-but-present `content`, a missing one cannot be parsed into
    a message at all, so it can never reach the retryable path -- the pre-fix
    code crashed here. Reporting it stays non-retryable by construction.
    """
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result(
          self._response(None, 'SAFETY', candidates_token_count=0)
      )
    self.assertIn('SAFETY', str(ctx.exception))

  def test_answer_text_is_never_flagged(self):
    """Control: any non-thought answer text short-circuits the guard."""
    result = self._model().result(
        self._response(
            [{'text': 'thinking', 'thought': True}, {'text': 'answer'}],
            'MAX_TOKENS',
            candidates_token_count=2,
            thoughts_token_count=100,
        )
    )
    self.assertEqual(result.samples[0].response.text, 'answer')

  def test_modality_only_answer_is_never_flagged(self):
    """Control: an image-only answer is real output, not an empty generation."""
    result = self._model().result(
        self._response(
            [{
                'inlineData': {
                    'data': base64.b64encode(example_image).decode(),
                    'mimeType': 'image/png',
                }
            }],
            'STOP',
            candidates_token_count=0,
        )
    )
    self.assertNotEqual(result.samples[0].response.text, '')

  def test_stop_with_output_tokens_stays_on_legacy_path(self):
    """Empty text but tokens were emitted: not our deterministic signature.

    E.g. a function-call-only candidate. Left untouched so the pre-existing
    retryable `EmptyGenerationError` path keeps handling it.
    """
    result = self._model().result(
        self._response(
            [{'functionCall': {'name': 'f', 'args': {}}}],
            'STOP',
            candidates_token_count=7,
        )
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_other_finish_reason_stays_on_legacy_path(self):
    """Empty text under an unrelated finish reason is not reclassified."""
    result = self._model().result(
        self._response([], 'SAFETY', candidates_token_count=0)
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_missing_finish_reason_stays_on_legacy_path(self):
    result = self._model().result(
        self._response([], None, candidates_token_count=0)
    )
    self.assertEqual(result.samples[0].response.text, '')


def _hop(
    parts: list[dict[str, Any]],
    finish_reason: str,
    *,
    token: str | None = None,
    candidates_token_count: int = 0,
    thoughts_token_count: int = 0,
) -> dict[str, Any]:
  """Returns a single-candidate `generateContent` response payload."""
  candidate = {
      'content': {'role': 'model', 'parts': parts},
      'finishReason': finish_reason,
  }
  if token is not None:
    candidate['continuationToken'] = token
  return {
      'candidates': [candidate],
      'usageMetadata': {
          'promptTokenCount': 10,
          'candidatesTokenCount': candidates_token_count,
          'thoughtsTokenCount': thoughts_token_count,
          'totalTokenCount': 10 + candidates_token_count + thoughts_token_count,
      },
  }


class GeminiContinuationTest(unittest.TestCase):
  """Tests resuming generations stopped with finishReason=CONTINUATION."""

  def _serve(
      self, *responses: dict[str, Any] | int
  ) -> tuple[list[dict[str, Any]], Any]:
    """Returns (sent requests, POST side effect) serving `responses` in order.

    Args:
      *responses: Payloads to return with HTTP 200, or HTTP error codes. The
        last one is served again for any extra request.
    """
    sent = []

    def post(url, json, **kwargs):
      del url, kwargs
      sent.append(json)
      served = responses[min(len(sent), len(responses)) - 1]
      response = requests.Response()
      if isinstance(served, int):
        response.status_code = served
        response._content = b'Error.'
      else:
        response.status_code = 200
        response._content = pg.to_json_str(served).encode()
      return response

    return sent, post

  def _lm(self) -> gemini.Gemini:
    return gemini.Gemini(
        'gemini-4-argon', api_endpoint='', max_attempts=3, retry_interval=0
    )

  def test_follows_continuation_until_stop(self):
    sent, post = self._serve(
        _hop(
            [{'text': 'Plan.', 'thought': True}, {'text': 'Hello, wor'}],
            'CONTINUATION',
            token='token-1',
            candidates_token_count=4,
            thoughts_token_count=100,
        ),
        _hop([{'text': 'ld!'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello, world!')
    self.assertEqual(response.metadata.thought, 'Plan.')
    self.assertEqual(response.metadata.finish_reason, 'STOP')
    self.assertEqual(response.metadata.usage.prompt_tokens, 20)
    self.assertEqual(response.metadata.usage.completion_tokens, 6)
    self.assertEqual(response.metadata.usage.total_tokens, 126)
    self.assertEqual(
        response.metadata.usage.completion_tokens_details['thinking_tokens'],
        100,
    )
    self.assertEqual(len(sent), 2)
    self.assertNotIn('continuationToken', sent[0])
    self.assertEqual(sent[1], dict(sent[0], continuationToken='token-1'))

  def test_thought_only_hop_is_continued_not_resent(self):
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'CONTINUATION',
            token='token-1',
            thoughts_token_count=32_765,
        ),
        _hop(
            [{'text': 'The answer is 42.', 'thoughtSignature': 'sig'}],
            'STOP',
            candidates_token_count=6,
        ),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'The answer is 42.')
    self.assertEqual(len(sent), 2)

  def test_budget_exhausted_after_continuation_is_not_resent(self):
    sent, post = self._serve(
        _hop(
            [{'text': 'Thinking...', 'thought': True}],
            'CONTINUATION',
            token='token-1',
            thoughts_token_count=32_768,
        ),
        _hop(
            [{'text': ' still thinking', 'thought': True}],
            'MAX_TOKENS',
            thoughts_token_count=32_768,
        ),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaisesRegex(
          gemini.GeminiEmptyGenerationError, 'thoughtsTokenCount=65536'
      ):
        self._lm()('hi')
    self.assertEqual(len(sent), 2)

  def test_failed_continuation_hop_is_retried_in_place(self):
    sent, post = self._serve(
        _hop([{'text': 'Hello, wor'}], 'CONTINUATION', token='token-1'),
        503,
        _hop([{'text': 'ld!'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello, world!')
    self.assertEqual(len(sent), 3)
    self.assertEqual(sent[1], sent[2])
    self.assertEqual(sent[2]['continuationToken'], 'token-1')

  def test_continuation_hop_retries_are_bounded(self):
    sent, post = self._serve(
        _hop([{'text': 'Hello, wor'}], 'CONTINUATION', token='token-1'),
        503,
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(lf.concurrent.RetryError):
        self._lm()._sample_single(lf.UserMessage('hi'))
    # The first hop plus `max_attempts` tries of the second one.
    self.assertEqual(len(sent), 4)

  def test_continuation_hop_permanent_error_is_not_retried(self):
    sent, post = self._serve(
        _hop([{'text': 'Hello, wor'}], 'CONTINUATION', token='token-1'),
        400,
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaisesRegex(lf.LMError, '400'):
        self._lm()._sample_single(lf.UserMessage('hi'))
    self.assertEqual(len(sent), 2)

  def test_final_hop_without_candidates_raises_retryable_error(self):
    sent, post = self._serve(
        _hop([{'text': 'Hello, wor'}], 'CONTINUATION', token='token-1'),
        {'usageMetadata': {'promptTokenCount': 10}},
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(lf.TemporaryLMError):
        self._lm()._sample_single(lf.UserMessage('hi'))
    self.assertEqual(len(sent), 2)

  def test_stop_is_a_single_request(self):
    sent, post = self._serve(
        _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2)
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello!')
    self.assertEqual(len(sent), 1)

  def test_continuation_without_token_is_not_followed(self):
    sent, post = self._serve(
        _hop([{'text': 'Hello, wor'}], 'CONTINUATION', candidates_token_count=4)
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello, wor')
    self.assertEqual(len(sent), 1)

  def test_multiple_candidates_are_not_continued(self):
    payload = _hop([{'text': 'A'}], 'CONTINUATION', token='token-1')
    payload['candidates'].append({
        'content': {'role': 'model', 'parts': [{'text': 'B'}]},
        'finishReason': 'CONTINUATION',
        'continuationToken': 'token-2',
    })
    sent, post = self._serve(payload)
    with mock.patch('requests.Session.post', side_effect=post):
      result = self._lm()._sample_single(lf.UserMessage('hi'))
    self.assertEqual([s.response.text for s in result.samples], ['A', 'B'])
    self.assertEqual(len(sent), 1)

  def test_network_error_while_building_request_is_retryable(self):
    with mock.patch.object(
        gemini.Gemini,
        '_generation_config',
        side_effect=requests.exceptions.ConnectionError(
            'Connection reset by peer'
        ),
    ):
      with self.assertRaises(lf.TemporaryLMError):
        self._lm()._sample_single(lf.UserMessage('hi'))

  def test_stops_after_max_continuation_hops(self):
    sent, post = self._serve(
        _hop([{'text': 'x'}], 'CONTINUATION', token='token-1')
    )
    with mock.patch.object(gemini, '_MAX_CONTINUATION_HOPS', 3):
      with mock.patch('requests.Session.post', side_effect=post):
        with self.assertRaisesRegex(
            lf.LMError, 'did not finish within 3 requests'
        ):
          self._lm()._sample_single(lf.UserMessage('hi'))
    self.assertEqual(len(sent), 3)


if __name__ == '__main__':
  unittest.main()
