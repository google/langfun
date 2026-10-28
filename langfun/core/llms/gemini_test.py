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
      block_reason: str | None = None,
      drop_candidates: bool = False,
  ) -> dict[str, Any]:
    """Builds a single-candidate response payload.

    Args:
      parts: The candidate's content parts. `None` omits the enclosing
        `content` key entirely, the shape a service may return when it has no
        answer at all.
      finish_reason: The candidate's `finishReason`, omitted when None.
      candidates_token_count: Response-level `candidatesTokenCount`.
      thoughts_token_count: Response-level `thoughtsTokenCount`.
      block_reason: Response-level `promptFeedback.blockReason`, omitted when
        None.
      drop_candidates: Sends an EMPTY `candidates` list instead of the
        candidate built above -- a response that answered nothing, as opposed
        to one missing the key.

    Returns:
      A response payload accepted by `Gemini.result()`.
    """
    candidate: dict[str, Any] = {}
    if parts is not None:
      candidate['content'] = {'role': 'model', 'parts': parts}
    if finish_reason is not None:
      candidate['finishReason'] = finish_reason
    payload: dict[str, Any] = {
        'candidates': [] if drop_candidates else [candidate],
        'usageMetadata': {
            'promptTokenCount': 12,
            'candidatesTokenCount': candidates_token_count,
            'thoughtsTokenCount': thoughts_token_count,
            'totalTokenCount': (
                12 + candidates_token_count + thoughts_token_count
            ),
        },
    }
    if block_reason is not None:
      payload['promptFeedback'] = {'blockReason': block_reason}
    return payload

  def test_error_is_not_retryable(self):
    """Deterministic per prompt, so a re-send can only re-bill it."""
    self.assertTrue(
        issubclass(gemini.GeminiEmptyGenerationError, lf.LMError)
    )
    self.assertFalse(
        issubclass(gemini.GeminiEmptyGenerationError, lf.RetryableLMError)
    )
    # Staying off `lf.EmptyGenerationError` keeps the diagnosis -- finish
    # reason, budget, token counts -- instead of a generic empty-text message,
    # and keeps it off the retry path that class puts it on.
    self.assertFalse(
        issubclass(gemini.GeminiEmptyGenerationError, lf.EmptyGenerationError)
    )

  def test_error_is_not_retried_by_with_retry(self):
    """A caller's `lf.RetryableLMError` policy must not re-send the prompt.

    Everything that can rescue a starved generation -- following the
    `continuationToken`, extending an explicit ceiling -- happens inside the
    one request. Past that there is nothing a second full generation can do
    that the first did not, so it must not be paid for.
    """
    attempts = []

    def starve():
      attempts.append(None)
      self._model().result(
          self._response(
              [{'text': ''}],
              'MAX_TOKENS',
              candidates_token_count=3,
              thoughts_token_count=2045,
          )
      )

    retried = lf.with_retry(
        starve,
        retry_on_errors=lf.RetryableLMError,
        max_attempts=5,
        retry_interval=0,
    )
    with self.assertRaises(gemini.GeminiEmptyGenerationError):
      retried()
    self.assertEqual(len(attempts), 1)

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
    self.assertIn('before any answer token was emitted', message)
    # Thinking is what spent the budget, so spending less of it is a remedy.
    self.assertIn('lowering the thinking budget', message)

  def test_max_tokens_with_billed_output_is_not_called_starvation(self):
    """4,222 billed answer tokens is the opposite of a starved budget.

    The truncated hop returned none of them and no continuation recovered
    them, so they are lost rather than never produced -- and the diagnosis
    has to say which, because the two have different remedies. (More answer
    than thought tokens would be a misfire; see `GeminiMisfireTest`.)
    """
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result(
          self._response(
              [{'text': ''}],
              'MAX_TOKENS',
              candidates_token_count=4222,
              thoughts_token_count=61314,
          )
      )
    message = str(ctx.exception)
    self.assertIn('MAX_TOKENS', message)
    self.assertIn('4222 answer tokens were billed', message)
    self.assertIn('no continuation recovered them', message)
    self.assertNotIn('before any answer token was emitted', message)
    # The budget reached the answer, so the starvation remedy is withheld.
    self.assertIn('spending less of it on thinking is not the fix', message)
    self.assertNotIn('lowering the thinking budget', message)

  def test_the_diagnosis_names_the_retry_it_declines(self):
    """`max_attempts` times a full budget is the bill NOT being run up.

    Callers routinely set `max_attempts=80`; an operator reading one of these
    in a log needs to see that 80 full regenerations were declined and why,
    not infer it from the class hierarchy. The in-request rescue that was
    spent instead belongs in the same sentence.
    """
    model = gemini.Gemini(
        'gemini-3.1-pro-preview', api_endpoint='', max_attempts=80
    )
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      model.result(
          self._response(
              [{'text': ''}],
              'MAX_TOKENS',
              candidates_token_count=3,
              thoughts_token_count=65533,
          )
      )
    message = str(ctx.exception)
    self.assertIn('not retryable', message)
    self.assertIn('max_attempts=80', message)
    self.assertIn('billed in full', message)
    # The rescue that was already paid for, so nobody looks for it twice.
    self.assertIn('`continuationToken`', message)
    self.assertIn('answer tokens still owed', message)
    self.assertIn(f'{gemini._MAX_PINNED_BUDGET_HOPS} requests', message)  # pylint: disable=protected-access

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
    message = str(ctx.exception)
    self.assertIn('STOP', message)
    # The model stopped on its own, so no budget knob is the remedy.
    self.assertIn('decode budget was not the constraint', message)
    self.assertNotIn('lowering the thinking budget', message)

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
    # Nothing was generated, so the budget remedies do not apply either.
    self.assertIn('no budget knob applies', message)
    self.assertNotIn('lowering the thinking budget', message)

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

  def test_function_call_candidate_is_exempt(self):
    """The one carve-out: a tool call is an answer the converter drops.

    `GeminiMessageConverter.from_value` renders no text for a `functionCall`
    part, so the candidate looks empty while carrying a real answer. The
    exemption reads the parts, which is evidence; it does not read
    `candidatesTokenCount > 0`, which is not.
    """
    result = self._model().result(
        self._response(
            [{'functionCall': {'name': 'f', 'args': {}}}],
            'STOP',
            candidates_token_count=7,
        )
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_function_response_candidate_is_exempt(self):
    """A `functionResponse` part is dropped by the converter just the same."""
    result = self._model().result(
        self._response(
            [{'functionResponse': {'name': 'f', 'response': {}}}],
            'STOP',
            candidates_token_count=0,
        )
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_function_call_is_exempt_even_without_billed_tokens(self):
    """The exemption is the parts, not the count: zero tokens still exempt.

    This is what separates the evidence from the proxy it replaced. Under the
    old `output_tokens > 0` rule this candidate would have been reported as
    empty despite carrying a tool call.
    """
    result = self._model().result(
        self._response(
            [{'functionCall': {'name': 'f', 'args': {}}}],
            'MAX_TOKENS',
            candidates_token_count=0,
        )
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_content_policy_finish_reason_stays_retryable(self):
    """A policy flag on the sampled content is not a property of the request.

    The API raises every one of these on the *response candidate content*, so
    each describes the tokens that happened to be sampled -- which is exactly
    what the next sample changes. Admitting them would end a live episode
    over an outcome a re-send can clear, the same trade the closed rule lost
    28 times out of 28 on `MALFORMED_FUNCTION_CALL`. The prompt-level
    counterpart a re-send genuinely cannot change is
    `promptFeedback.blockReason`, which the API defines on the prompt and
    returns with no candidate at all; it is admitted, and tested, separately.
    """
    for finish_reason in (
        'SAFETY',
        'RECITATION',
        'BLOCKLIST',
        'PROHIBITED_CONTENT',
        'SPII',
        'LANGUAGE',
        'IMAGE_SAFETY',
    ):
      with self.subTest(finish_reason=finish_reason):
        result = self._model().result(
            self._response([], finish_reason, candidates_token_count=0)
        )
        self.assertEqual(result.samples[0].response.text, '')

  def test_tool_protocol_finish_reason_stays_retryable(self):
    """The live failure mode: MALFORMED_FUNCTION_CALL must NOT end the episode.

    Measured across 30 `gemini-4-argon` trajectories, 28 of 28 aborts raised
    by the closed rule carried this finish reason on a mid-episode turn whose
    request declared no tools at all -- so there was no tool declaration to
    correct, and the next sample succeeded. Returning hands it to
    `LanguageModel.sample()`, which raises the RETRYABLE
    `lf.EmptyGenerationError` and lets the episode recover.
    """
    result = self._model().result(
        self._response(
            [{'text': 'thinking...', 'thought': True}],
            'MALFORMED_FUNCTION_CALL',
            candidates_token_count=1304,
            thoughts_token_count=4821,
        )
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_missing_finish_reason_stays_retryable(self):
    """No reason reported is not evidence that a re-send cannot work.

    An unclassified signature is unclassified in both directions. Calling it
    deterministic costs a live episode on every false positive, while calling
    it transient costs at most `max_attempts` samples that already had to be
    budgeted for.
    """
    result = self._model().result(
        self._response([], None, candidates_token_count=0)
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_unrecognised_finish_reason_stays_retryable(self):
    """A reason this client has never seen is likewise not proven fatal."""
    result = self._model().result(
        self._response([], 'SOME_NEW_REASON', candidates_token_count=0)
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_stop_with_billed_tokens_stays_retryable(self):
    """A STOP over billed tokens is lost work, which is what a retry is for.

    The truncation this signature used to hide -- a `MAX_TOKENS` hop merged
    into a chain that takes the LAST hop's `STOP` over a SUMMED count -- is
    closed at its source now: `_merge_continuation_hops` keeps the chain
    labelled `MAX_TOKENS` (see
    `GeminiStarvedBudgetTest.test_an_exhausted_budget_is_not_laundered_into_stop`).
    What reaches here is a generation the service was billed for and lost, so
    re-sampling it is the recovery, not a wasted bill.
    """
    result = self._model().result(
        self._response(
            [{'text': ''}],
            'STOP',
            candidates_token_count=62848,
            thoughts_token_count=2688,
        )
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_stop_with_thinking_tokens_only_stays_retryable(self):
    """The boundary of the fatal STOP arm: thinking happened, so work exists.

    `STOP` is fatal only when the model was billed for NOTHING -- no answer
    and no thinking. One thought token is enough to make it a generation that
    ran and was lost.
    """
    result = self._model().result(
        self._response(
            [{'text': 'thinking...', 'thought': True}],
            'STOP',
            candidates_token_count=0,
            thoughts_token_count=1,
        )
    )
    self.assertEqual(result.samples[0].response.text, '')

  def test_blocked_prompt_raises(self):
    """`promptFeedback.blockReason` outranks every candidate-level reason.

    The prompt never reached decoding, so no budget knob is the remedy and the
    block reason is what an operator needs to read.
    """
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result(
          self._response(
              [],
              'STOP',
              candidates_token_count=0,
              block_reason='PROHIBITED_CONTENT',
          )
      )
    message = str(ctx.exception)
    self.assertIn('blockReason=PROHIBITED_CONTENT', message)
    self.assertIn('blocked before generation started', message)
    self.assertIn('never reached decoding', message)
    # The candidate-level STOP diagnosis must not be the one reported.
    self.assertNotIn('without emitting any output token', message)

  def test_empty_candidates_list_raises_instead_of_indexerror(self):
    """`candidates: []` must not reach `LanguageModel` as a bare IndexError.

    Left alone it yields `LMSamplingResult([])`, and the caller fails on
    `result.samples[0]` with an error naming neither the model nor the
    reason. With nothing explaining the empty list it is reported as the
    transient fault it looks like -- retryable, like the absent key -- but
    reported, and naming the model.
    """
    with self.assertRaises(lf.TemporaryLMError) as ctx:
      self._model().result(
          self._response(
              None, None, candidates_token_count=0, drop_candidates=True
          )
      )
    message = str(ctx.exception)
    self.assertIn(self._model().model_id, message)
    self.assertIn('empty candidate list', message)

  def test_empty_candidates_list_reports_the_block_reason(self):
    """The shape a blocked prompt actually returns: no candidate, a reason."""
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result(
          self._response(
              None,
              None,
              candidates_token_count=0,
              drop_candidates=True,
              block_reason='SAFETY',
          )
      )
    self.assertIn('blockReason=SAFETY', str(ctx.exception))

  def test_absent_candidates_key_stays_retryable(self):
    """The boundary of the change: a MISSING key is still the retryable path.

    An empty list and an absent key are different responses. Only the former
    is reclassified; the latter remains the transient Gemini API fault the
    pre-existing handling documents.
    """
    with self.assertRaises(lf.TemporaryLMError):
      self._model().result({'usageMetadata': {'promptTokenCount': 12}})

  def test_missing_usage_metadata_does_not_crash(self):
    """`usageMetadata` is read defensively: a response may omit it entirely.

    Before, `json['usageMetadata']` raised a bare `KeyError` that hid whatever
    the real failure was.
    """
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._model().result({
          'candidates': [{
              'content': {'role': 'model', 'parts': []},
              'finishReason': 'MAX_TOKENS',
          }]
      })
    self.assertIn('candidatesTokenCount=0', str(ctx.exception))

  def test_carries_unrendered_content_reads_the_parts(self):
    """Unit-level: the exemption is a pure function of the candidate's parts."""
    carries = gemini._carries_unrendered_content  # pylint: disable=protected-access
    self.assertTrue(
        carries({'content': {'parts': [{'functionCall': {'name': 'f'}}]}})
    )
    self.assertTrue(
        carries({'content': {'parts': [{'functionResponse': {'name': 'f'}}]}})
    )
    self.assertFalse(carries({'content': {'parts': [{'text': ''}]}}))
    self.assertFalse(carries({'content': {'parts': []}}))
    # The shapes `result()` can hand it: no parts, null content, no candidate.
    self.assertFalse(carries({'content': {}}))
    self.assertFalse(carries({'content': None}))
    self.assertFalse(carries({}))


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
    """A MAX_TOKENS hop with no token ends the generation, within one sample.

    Sampled through `_sample_single` so the assertion is about the hop loop;
    `GeminiEmptyGenerationError` is retryable, so the public entry point would
    re-send the whole prompt `max_attempts` times on top of this.
    """
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
        self._lm()._sample_single(lf.UserMessage('hi'))
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


class GeminiStarvedBudgetTest(unittest.TestCase):
  """Tests resuming a generation whose thinking spent the decode budget.

  The hop shapes replay what `gemini-4-argon` returns on Vertex AI when
  thinking consumes `maxOutputTokens` before an answer token is emitted:
  `finishReason=MAX_TOKENS`, `parts=[{'text': ''}]`, the budget accounted for
  exactly by `candidatesTokenCount + thoughtsTokenCount`, and a valid
  `continuationToken`.
  """

  def _serve(self, *responses: dict[str, Any] | int) -> tuple[list[Any], Any]:
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
        response._content = b'Error.'  # pylint: disable=protected-access
      else:
        response.status_code = 200
        response._content = pg.to_json_str(served).encode()  # pylint: disable=protected-access
      return response

    return sent, post

  def _resuming_service(self, answers_at: int) -> tuple[list[Any], Any]:
    """Returns (sent requests, POST side effect) replaying the real service.

    A fake rather than a canned script, because the behaviour under test is a
    rule the service applies hop by hop to the request we send it. It
    reproduces what was measured against `gemini-4-argon` on Vertex AI:

    * `maxOutputTokens` is a running total over the whole generation, so a
      resume that re-asserts a ceiling already reported spent is rejected
      with HTTP 400 and the generation cannot continue.
    * A resume that raises the ceiling is accepted and buys exactly the
      headroom it added -- which thinking spends too, until the ceiling
      finally leaves room for the answer.

    Args:
      answers_at: The ceiling at which the generation has room to answer.
        Every hop below it starves at `MAX_TOKENS`, offering a token.

    Returns:
      The list that records each request body, and the `requests.Session.post`
      side effect.
    """
    sent = []
    spent = [0]  # The running total the service has accumulated so far.

    def post(url, json, **kwargs):
      del url, kwargs
      sent.append(json)
      response = requests.Response()
      ceiling = (json.get('generationConfig') or {}).get('maxOutputTokens')
      if ceiling is not None and ceiling <= spent[0]:
        response.status_code = 400
        response._content = (  # pylint: disable=protected-access
            f'max_output_tokens ({ceiling}) has already been exhausted. '
            'Because max_output_tokens is accumulated across continuation '
            'requests, pass a higher max_output_tokens value to continue '
            'generation.'
        ).encode()
        return response
      if ceiling is not None and ceiling < answers_at:
        served = _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token=f'token-{len(sent)}',
            thoughts_token_count=ceiling - spent[0],
        )
        spent[0] = ceiling
      else:
        served = _hop([{'text': '5'}], 'STOP', candidates_token_count=1)
      response.status_code = 200
      response._content = pg.to_json_str(served).encode()  # pylint: disable=protected-access
      return response

    return sent, post

  def _lm(self, max_tokens: int | None = None) -> gemini.Gemini:
    return gemini.Gemini(
        'gemini-4-argon',
        api_endpoint='',
        max_attempts=3,
        retry_interval=0,
        sampling_options=lf.LMSamplingOptions(max_tokens=max_tokens),
    )

  def test_starved_hop_is_continued_until_the_answer_arrives(self):
    """Mode B: MAX_TOKENS carrying a token is resumed, not reported empty."""
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=3,
            thoughts_token_count=2045,
        ),
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-2',
            thoughts_token_count=2048,
        ),
        _hop(
            [{'text': '5'}],
            'STOP',
            candidates_token_count=1,
            thoughts_token_count=2500,
        ),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, '5')
    self.assertEqual(response.metadata.finish_reason, 'STOP')
    self.assertEqual(len(sent), 3)
    self.assertNotIn('continuationToken', sent[0])
    self.assertEqual(sent[1], dict(sent[0], continuationToken='token-1'))
    self.assertEqual(sent[2], dict(sent[0], continuationToken='token-2'))
    # Every hop is billed, so the usage of all three is summed.
    self.assertEqual(response.metadata.usage.completion_tokens, 4)
    self.assertEqual(
        response.metadata.usage.completion_tokens_details['thinking_tokens'],
        6593,
    )

  def test_empty_final_hop_keeps_text_from_earlier_hops(self):
    """A trailing `{'text': ''}` must not discard what earlier hops wrote."""
    sent, post = self._serve(
        _hop(
            [{'text': 'Hello, wor'}],
            'CONTINUATION',
            token='token-1',
            candidates_token_count=3,
        ),
        _hop(
            [{'text': 'ld!'}],
            'MAX_TOKENS',
            token='token-2',
            candidates_token_count=2,
        ),
        _hop([{'text': ''}], 'STOP', thoughts_token_count=16),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello, world!')
    self.assertEqual(len(sent), 3)

  def test_empty_part_is_kept_when_it_cannot_be_joined(self):
    """The merge keeps the placeholder; it is `from_chunks` that ignores it.

    A bare `{'text': ''}` is only joined into the preceding part when that
    part is text of the same thought-ness. Following a thought part it has
    nothing to join to, so it reaches the parser -- harmlessly, per
    `test_empty_text_chunks_do_not_change_the_parsed_text`.
    """
    merged = gemini._merge_continuation_hops([  # pylint: disable=protected-access
        _hop(
            [{'text': 'the answer'}, {'text': 'and more thinking',
                                      'thought': True}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=2,
        ),
        _hop([{'text': ''}], 'STOP'),
    ])
    self.assertEqual(
        merged['candidates'][0]['content']['parts'],
        [{'text': 'the answer'}, {'text': 'and more thinking',
                                  'thought': True}, {'text': ''}],
    )
    self.assertEqual(merged['candidates'][0]['finishReason'], 'STOP')
    self.assertEqual(
        lf.Message.from_value(
            merged['candidates'][0]['content'], format='gemini'
        ).text,
        'the answer',
    )

  def test_empty_text_chunks_do_not_change_the_parsed_text(self):
    """Why the merge needs no empty-part filter of its own.

    `Message.from_chunks` writes the separator before a chunk only when the
    previous one ended in a non-whitespace character, and an empty chunk
    resets that to `None`, so a run of empty chunks costs at most the single
    separator that a direct neighbour would have cost anyway -- and a trailing
    one is stripped. Dropping empty text parts is therefore unobservable in
    the parsed text, in every position.
    """
    for parts, expected in [
        ([{'text': ''}], ''),
        ([{'text': ''}, {'text': 'B'}], 'B'),
        ([{'text': 'A'}, {'text': ''}], 'A'),
        ([{'text': 'A'}, {'text': ''}, {'text': 'B'}], 'A B'),
        ([{'text': 'A'}, {'text': ''}, {'text': ''}, {'text': 'B'}], 'A B'),
        ([{'text': 'A'}, {'text': 'thinking', 'thought': True},
          {'text': ''}, {'text': 'B'}], 'A B'),
    ]:
      with self.subTest(parts=parts):
        self.assertEqual(
            lf.Message.from_value({'parts': parts}, format='gemini').text,
            expected,
        )
        self.assertEqual(
            lf.Message.from_value(
                {'parts': [p for p in parts if p != {'text': ''}]},
                format='gemini',
            ).text,
            expected,
        )

  def test_every_hop_empty_still_reports_the_empty_generation(self):
    """With nothing to preserve, the last hop's shape is reported as-is."""
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=3,
            thoughts_token_count=2045,
        ),
        _hop([{'text': ''}], 'MAX_TOKENS', thoughts_token_count=2048),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
        self._lm()._sample_single(lf.UserMessage('hi'))
    self.assertIn('MAX_TOKENS', str(ctx.exception))
    self.assertEqual(len(sent), 2)

  def test_failed_starvation_resume_falls_back_to_the_diagnosis(self):
    """A rejected resume must not replace what the generation itself did.

    A pinned `max_tokens` is rejected on continuation, because the service
    accumulates it across continuation requests. The caller is told about the
    exhausted budget rather than about the resume's HTTP error.
    """
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=3,
            thoughts_token_count=2045,
        ),
        400,
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
        self._lm()._sample_single(lf.UserMessage('hi'))
    self.assertIn('MAX_TOKENS', str(ctx.exception))
    self.assertEqual(len(sent), 2)

  def test_starved_hop_without_a_token_cannot_be_resumed(self):
    """Mode A: no token, so the error is raised after exactly one request."""
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            candidates_token_count=3,
            thoughts_token_count=2045,
        )
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(gemini.GeminiEmptyGenerationError):
        self._lm()._sample_single(lf.UserMessage('hi'))
    self.assertEqual(len(sent), 1)

  def test_a_starved_generation_that_cannot_resume_is_not_resent(self):
    """Mode A: non-retryable, so the prompt is paid for exactly once.

    `max_attempts=3` would otherwise buy three full regenerations of a budget
    that is spent the same way every time, and production launches set 80.
    """
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            candidates_token_count=3,
            thoughts_token_count=2045,
        )
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
        self._lm()('hi')
    self.assertEqual(len(sent), 1)  # `max_attempts=3`, and still one request.
    self.assertNotIsInstance(ctx.exception, lf.RetryableLMError)

  def test_explicitly_pinned_max_tokens_does_not_defeat_the_resume(self):
    """The caller's ceiling is honoured on hop 0 and doubled to resume.

    Against the real service rule -- a resume that re-sends a spent ceiling is
    rejected -- the only way to reach the answer is to raise it. This is the
    case an unset class default cannot cover, because the caller sets
    `max_tokens` itself, e.g. via a `?max_tokens=` model URI.
    """
    sent, post = self._resuming_service(answers_at=2048)
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=1024)('hi')
    self.assertEqual(response.text, '5')
    self.assertEqual(len(sent), 2)
    # Hop 0 spends exactly the budget the caller authorized.
    self.assertEqual(sent[0]['generationConfig']['maxOutputTokens'], 1024)
    # Hop 1 raises it, which is the only resume the service accepts.
    self.assertEqual(sent[1]['generationConfig']['maxOutputTokens'], 2048)
    self.assertEqual(sent[1]['continuationToken'], 'token-1')

  def test_the_budget_extension_climbs_to_four_times_the_cap(self):
    """One resume is not always enough, so the ladder runs 1x -> 2x -> 3x -> 4x.

    Each resume asks for the tokens already spent plus the answer tokens
    still owed, so the ceiling climbs by the pinned cap and no faster.
    Replays the measured wire behaviour for a caller pinning 2048: thinking
    spends every ceiling it is handed until 8192 finally leaves room for the
    answer.
    """
    sent, post = self._resuming_service(answers_at=8192)
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=2048)('hi')
    self.assertEqual(response.text, '5')
    self.assertEqual(
        [r['generationConfig']['maxOutputTokens'] for r in sent],
        [2048, 4096, 6144, 8192],
    )

  def test_the_budget_extension_stops_at_four_times_the_cap(self):
    """The ceiling the caller pinned still bounds the call.

    A generation that keeps starving is cut off rather than handed the
    service default: 512 authorized buys at most 2048, where dropping the
    ceiling instead was measured generating tens of thousands of tokens.
    """
    sent, post = self._resuming_service(answers_at=10**9)
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
        self._lm(max_tokens=512)('hi')
    self.assertIn('MAX_TOKENS', str(ctx.exception))
    self.assertEqual(
        [r['generationConfig']['maxOutputTokens'] for r in sent],
        [512, 1024, 1536, 2048],
    )
    # The bound and the guarantee it buys, pinned together: N requests for a
    # ceiling that climbs by the cap each time is exactly Nx the cap.
    self.assertEqual(len(sent), gemini._MAX_PINNED_BUDGET_HOPS)  # pylint: disable=protected-access
    self.assertEqual(
        sent[-1]['generationConfig']['maxOutputTokens'],
        512 * gemini._MAX_PINNED_BUDGET_HOPS,  # pylint: disable=protected-access
    )

  def test_a_cap_already_spent_on_answer_tokens_is_not_resumed(self):
    """`max_tokens` answer tokens were asked for and `max_tokens` arrived.

    The ceiling did exactly its job, so truncating here is the caller's own
    instruction rather than a failure to recover, and the `continuationToken`
    the service offered is declined.
    """
    sent, post = self._serve(
        _hop(
            [{'text': 'the answer, trunca'}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=512,
        ),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=512)('hi')
    self.assertEqual(response.text, 'the answer, trunca')
    self.assertEqual(response.metadata.finish_reason, 'MAX_TOKENS')
    self.assertEqual(len(sent), 1)

  def test_a_cap_spent_on_thinking_is_resumed_for_the_whole_answer(self):
    """Thinking is not what the caller bought, so none of the cap is lost.

    Hop 0 spends all 512 on thinking and emits no answer token. The resume
    asks for those 512 back plus the 512 answer tokens still owed, so the
    caller gets the full answer they paid for rather than nothing.
    """
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=0,
            thoughts_token_count=512,
        ),
        _hop([{'text': 'the answer'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=512)('hi')
    self.assertEqual(response.text, 'the answer')
    self.assertEqual(len(sent), 2)
    self.assertEqual(sent[0]['generationConfig']['maxOutputTokens'], 512)
    self.assertEqual(sent[1]['generationConfig']['maxOutputTokens'], 1024)

  def test_a_partly_spent_cap_is_resumed_for_only_the_rest_of_the_answer(self):
    """The resume buys the remainder of the cap, never a fresh one.

    509 of the 512 answer tokens arrived, so the resume authorizes the 512
    already burned plus the 3 still owed -- not another 512. Stopping here
    instead would hand back an answer cut off three tokens short because
    thinking took the difference.
    """
    sent, post = self._serve(
        _hop(
            [{'text': 'the answer, trunca'}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=509,
            thoughts_token_count=3,
        ),
        _hop([{'text': 'ted'}], 'STOP', candidates_token_count=3),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=512)('hi')
    self.assertEqual(response.text, 'the answer, truncated')
    self.assertEqual(len(sent), 2)
    self.assertEqual(sent[1]['generationConfig']['maxOutputTokens'], 515)

  def test_a_ceiling_above_the_model_maximum_is_still_asserted(self):
    """A ceiling the model cannot fill is sent anyway, not dropped.

    A caller pinning the model's own maximum makes every resume ask for more
    than the model can produce. Measured against the service, such values are
    accepted -- 65,537, 131,072 and 1,000,000 all return HTTP 200 -- so there
    is nothing to omit, and asserting them is what keeps the bound: dropping
    the key would hand the rest of the generation to the service default.
    The hop bound is what stops the generation.
    """
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=0,
            thoughts_token_count=65_536,
        ),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(gemini.GeminiEmptyGenerationError):
        self._lm(max_tokens=65_536)('hi')
    # Each resume asks for everything burned so far plus the full 65,536
    # answer tokens still owed, since no hop ever emits one.
    self.assertEqual(
        [r['generationConfig']['maxOutputTokens'] for r in sent],
        [65_536, 131_072, 196_608, 262_144],
    )
    self.assertEqual(len(sent), gemini._MAX_PINNED_BUDGET_HOPS)  # pylint: disable=protected-access

  def test_an_extended_ceiling_is_carried_by_every_later_hop(self):
    """The extension belongs to the generation, not to the one hop.

    A `CONTINUATION` hop following a `MAX_TOKENS` one is still part of a
    generation whose accumulated budget now stands at the extended ceiling.
    Reverting to the caller's original would be rejected exactly as
    re-asserting a spent ceiling was.
    """
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=3,
            thoughts_token_count=1021,
        ),
        _hop([{'text': 'the '}], 'CONTINUATION', token='token-2'),
        _hop([{'text': 'answer'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=1024)('hi')
    self.assertEqual(response.text, 'the answer')
    self.assertEqual(len(sent), 3)
    # 1024 spent (3 answer + 1021 thinking) plus the 1021 answer tokens still
    # owed, carried unchanged by the `CONTINUATION` hop that follows.
    self.assertEqual(
        [r['generationConfig']['maxOutputTokens'] for r in sent],
        [1024, 2045, 2045],
    )

  def test_an_unspent_ceiling_is_kept_while_the_generation_is_mid_flight(self):
    """`CONTINUATION` is the per-response cap, not budget exhaustion.

    Nothing is spent yet, so the caller's ceiling is still both legal and
    binding: dropping it here would hand the generation a budget nobody asked
    for.
    """
    sent, post = self._serve(
        _hop([{'text': 'the '}], 'CONTINUATION', token='token-1'),
        _hop([{'text': 'answer'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=1024)('hi')
    self.assertEqual(response.text, 'the answer')
    self.assertEqual(len(sent), 2)
    self.assertEqual(sent[1], dict(sent[0], continuationToken='token-1'))

  def test_a_request_without_a_ceiling_is_resumed_unchanged(self):
    """With nothing pinned there is nothing to extend, so nothing changes."""
    sent, post = self._serve(
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            candidates_token_count=3,
            thoughts_token_count=2045,
        ),
        _hop([{'text': '5'}], 'STOP', candidates_token_count=1),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, '5')
    self.assertEqual(sent[1], dict(sent[0], continuationToken='token-1'))

  def test_an_exhausted_budget_is_not_laundered_into_stop(self):
    """`ex 118`: 80 futile retries bought by the last hop's reason winning.

    Hop 0 spent the whole 65,536-token budget on thinking and returned no
    text; hop 1 added nothing and reported `STOP`. Taking `STOP` from the
    last hop while SUMMING the token counts produced `('STOP', >0)` -- a
    signature indistinguishable from a generation that ran and was lost, so
    it was re-sent `max_attempts` times, each re-send exhausting the same
    budget the same way.
    """
    merged = gemini._merge_continuation_hops([  # pylint: disable=protected-access
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            thoughts_token_count=65536,
        ),
        _hop([], 'STOP'),
    ])
    candidate = merged['candidates'][0]
    self.assertEqual(candidate['finishReason'], 'MAX_TOKENS')
    self.assertEqual(merged['usageMetadata']['thoughtsTokenCount'], 65536)
    # The resumption state of the last hop is still the chain's own.
    self.assertNotIn('continuationToken', candidate)

  def test_a_chain_that_answered_keeps_the_last_hops_reason(self):
    """The boundary: a truncation the continuation RECOVERED is not one.

    Only a chain that rendered nothing is relabelled. Once any hop returned
    answer text the generation did finish, and reporting `MAX_TOKENS` over it
    would turn a complete answer into a fatal budget exhaustion.
    """
    merged = gemini._merge_continuation_hops([  # pylint: disable=protected-access
        _hop(
            [{'text': ''}],
            'MAX_TOKENS',
            token='token-1',
            thoughts_token_count=65536,
        ),
        _hop([{'text': 'the answer'}], 'STOP', candidates_token_count=2),
    ])
    candidate = merged['candidates'][0]
    self.assertEqual(candidate['finishReason'], 'STOP')
    self.assertEqual(
        lf.Message.from_value(candidate['content'], format='gemini').text,
        'the answer',
    )


def _misfire(
    *,
    candidates_token_count: int = 2_836,
    thoughts_token_count: int = 717,
    finish_message: str | None = None,
) -> dict[str, Any]:
  """Returns a MALFORMED_FUNCTION_CALL response, shaped as Vertex sends it."""
  candidate = {
      'content': {'role': 'model'},
      'finishReason': 'MALFORMED_FUNCTION_CALL',
  }
  if finish_message is not None:
    candidate['finishMessage'] = finish_message
  return {
      'candidates': [candidate],
      'usageMetadata': {
          'promptTokenCount': 8_992,
          'candidatesTokenCount': candidates_token_count,
          'thoughtsTokenCount': thoughts_token_count,
          'totalTokenCount': (
              8_992 + candidates_token_count + thoughts_token_count
          ),
      },
  }


def _cap_misfire(
    answer_tokens: int = 65_536, thoughts: int = 2_000
) -> dict[str, Any]:
  """Returns an answer-less MAX_TOKENS hop that billed the cap as answer."""
  return _hop(
      [{'text': ''}],
      'MAX_TOKENS',
      token='token-1',
      candidates_token_count=answer_tokens,
      thoughts_token_count=thoughts,
  )


class GeminiMisfireTest(unittest.TestCase):
  """Tests the two misfire paths: short ones and ones that hit the cap."""

  def setUp(self):
    super().setUp()
    self.enterContext(
        mock.patch.object(gemini, '_CAP_MISFIRE_RETRY_INTERVAL', 0)
    )

  def _lm(
      self, max_tokens: int | None = None, max_attempts: int = 80
  ) -> gemini.Gemini:
    return gemini.Gemini(
        'gemini-4-argon',
        api_endpoint='',
        max_attempts=max_attempts,
        retry_interval=0,
        sampling_options=lf.LMSamplingOptions(max_tokens=max_tokens),
    )

  def _serve(self, *responses: dict[str, Any] | int) -> tuple[list[Any], Any]:
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
        response._content = b'Error.'  # pylint: disable=protected-access
      else:
        response.status_code = 200
        response._content = pg.to_json_str(served).encode()  # pylint: disable=protected-access
      return response

    return sent, post

  def test_short_misfire_is_logged_and_left_retryable(self):
    # The finish message can carry the model's whole answer: user content.
    finish_message = 'Malformed function call: ' + 'x' * 5_000
    with mock.patch.object(pg.logging, 'warning') as warning:
      result = self._lm().result(_misfire(finish_message=finish_message))
    self.assertEqual(result.samples[0].response.text, '')
    logged = warning.call_args.args[0]
    self.assertIn('finishReason=MALFORMED_FUNCTION_CALL', logged)
    self.assertIn('candidatesTokenCount=2836', logged)
    self.assertIn('thoughtsTokenCount=717', logged)
    self.assertIn(f'finishMessageChars={len(finish_message)}', logged)
    self.assertNotIn('Malformed function call:', logged)
    self.assertNotIn('xxxxxxxxxx', logged)

  def test_sticky_short_misfire_recovers_under_max_attempts(self):
    sent, post = self._serve(
        *[_misfire()] * 10,
        _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello!')
    self.assertEqual(len(sent), 11)

  def test_short_misfires_are_bounded_by_max_attempts(self):
    sent, post = self._serve(_misfire())
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(lf.concurrent.RetryError) as ctx:
        self._lm(max_attempts=12)('hi')
    self.assertEqual(len(sent), 12)
    self.assertTrue(
        all(
            isinstance(e, lf.EmptyGenerationError) for e in ctx.exception.errors
        )
    )

  def test_cap_misfire_is_a_retryable_misfire_error(self):
    with self.assertRaises(gemini.MalformedFunctionCallError) as ctx:
      self._lm().result(_cap_misfire(63_000, 2_536))
    self.assertIsInstance(ctx.exception, lf.EmptyGenerationError)
    message = str(ctx.exception)
    self.assertIn('finishReason=MAX_TOKENS', message)
    self.assertIn('candidatesTokenCount=63000', message)
    self.assertIn('thoughtsTokenCount=2536', message)
    self.assertIn('not thinking starvation', message)

  def test_starvation_keeps_its_diagnosis(self):
    with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
      self._lm().result(_cap_misfire(0, 65_536))
    message = str(ctx.exception)
    self.assertIn('before any answer token was emitted', message)
    self.assertNotIn('misfire', message)

  def test_sticky_cap_misfire_fails_after_three_samples(self):
    """The TBS signature: the whole pinned 65,536 cap billed, no text.

    It is re-sampled from scratch under the caller's ceiling and never
    continuation-resumed, also when the cap is not yet spent on answer tokens
    and a resume under a raised ceiling would otherwise be attempted. After
    three samples it fails, however large `max_attempts` is.
    """
    for answer_tokens, thoughts in (
        (65_536, 2_000),
        (65_536, 15_000),
        (60_000, 5_536),
    ):
      with self.subTest(answer_tokens=answer_tokens, thoughts=thoughts):
        sent, post = self._serve(_cap_misfire(answer_tokens, thoughts))
        with mock.patch('requests.Session.post', side_effect=post):
          with self.assertRaises(lf.concurrent.RetryError) as ctx:
            self._lm(max_tokens=65_536, max_attempts=100)('hi')
        self.assertEqual(len(sent), 3)
        for request in sent:
          self.assertNotIn('continuationToken', request)
          self.assertEqual(
              request['generationConfig']['maxOutputTokens'], 65_536
          )
        self.assertTrue(
            all(
                isinstance(e, gemini.MalformedFunctionCallError)
                for e in ctx.exception.errors
            )
        )

  def test_cap_misfire_budget_never_exceeds_max_attempts(self):
    """A caller running its own retries with `max_attempts=1` gets 1 sample."""
    sent, post = self._serve(_cap_misfire())
    with mock.patch('requests.Session.post', side_effect=post):
      with self.assertRaises(lf.concurrent.RetryError) as ctx:
        self._lm(max_tokens=65_536, max_attempts=1)('hi')
    self.assertEqual(len(sent), 1)
    (error,) = ctx.exception.errors
    self.assertIsInstance(error, gemini.MalformedFunctionCallError)
    self.assertIn('up to 1 per query', str(error))

  def test_cap_misfire_budget_is_per_query_not_per_retry(self):
    """Retrying a short misfire or a 429 does not reset the budget.

    Both are retried outside the cap-misfire loop, by sampling the prompt anew.
    A sticky prompt alternating them with cap misfires would otherwise draw
    about two cap misfires per attempt of `max_attempts`.
    """
    for name, between in (('short misfire', _misfire()), ('HTTP 429', 429)):
      with self.subTest(between=name):
        sent, post = self._serve(
            _cap_misfire(),
            between,
            _cap_misfire(),
            between,
            _cap_misfire(),
            _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2),
        )
        with mock.patch('requests.Session.post', side_effect=post):
          with self.assertRaises(lf.concurrent.RetryError) as ctx:
            self._lm(max_tokens=65_536)('hi')
        self.assertEqual(len(sent), 5)
        self.assertTrue(
            all(
                isinstance(e, gemini.MalformedFunctionCallError)
                for e in ctx.exception.errors
            )
        )

  def test_each_query_draws_on_its_own_budget(self):
    """Sending the same prompt object again starts a fresh budget."""
    prompt = lf.UserMessage('hi')
    answer = _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2)
    sent, post = self._serve(*[_cap_misfire(), _cap_misfire(), answer] * 2)
    lm = self._lm(max_tokens=65_536)
    with mock.patch('requests.Session.post', side_effect=post):
      self.assertEqual(lm(prompt).text, 'Hello!')
      self.assertEqual(lm(prompt).text, 'Hello!')
    self.assertEqual(len(sent), 6)

  def test_cap_misfire_then_answer_recovers(self):
    sent, post = self._serve(
        _cap_misfire(),
        _cap_misfire(),
        _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm(max_tokens=65_536)('hi')
    self.assertEqual(response.text, 'Hello!')
    self.assertEqual(len(sent), 3)
    self.assertFalse(any('continuationToken' in r for r in sent))

  def test_misfired_continuation_hop_is_resampled_not_resumed(self):
    sent, post = self._serve(
        _hop(
            [],
            'CONTINUATION',
            token='token-1',
            candidates_token_count=32_000,
            thoughts_token_count=1_500,
        ),
        _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello!')
    self.assertEqual(len(sent), 2)
    self.assertNotIn('continuationToken', sent[1])

  def test_misfire_after_a_thinking_hop_is_judged_on_its_own_counts(self):
    """A misfired later hop is raised where it is refused, within the budget.

    Summed with the 32,768 thought tokens of the hop before it, its 30,768
    answer tokens no longer outnumber the thoughts. Judged on the sums, a
    `MAX_TOKENS` hop would end in the non-retryable `GeminiEmptyGenerationError`
    and a `CONTINUATION` one would fall back to the plain empty-generation
    retry, outside the cap-misfire budget.
    """
    thinking = _hop(
        [{'text': ''}],
        'CONTINUATION',
        token='token-1',
        thoughts_token_count=32_768,
    )
    answer = _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2)
    for finish_reason in ('MAX_TOKENS', 'CONTINUATION'):
      with self.subTest(finish_reason=finish_reason):
        misfire = _hop(
            [{'text': ''}],
            finish_reason,
            token='token-2',
            candidates_token_count=30_768,
            thoughts_token_count=2_000,
        )
        sent, post = self._serve(*[thinking, misfire] * 3, answer)
        with mock.patch('requests.Session.post', side_effect=post):
          with self.assertRaises(lf.concurrent.RetryError) as ctx:
            self._lm(max_tokens=65_536)('hi')
        # Each thinking hop is resumed, and each misfire is sampled anew.
        self.assertEqual(
            ['continuationToken' in r for r in sent], [False, True] * 3
        )
        self.assertEqual(len(ctx.exception.errors), 3)
        for error in ctx.exception.errors:
          self.assertIsInstance(error, gemini.MalformedFunctionCallError)
          self.assertIn(f'finishReason={finish_reason}', str(error))
          self.assertIn('candidatesTokenCount=30768', str(error))
          self.assertIn('thoughtsTokenCount=2000', str(error))

  def test_misfire_after_partial_text_is_resampled_not_returned(self):
    """An answer cut off by a runaway call is not returned as the answer."""
    sent, post = self._serve(
        _hop(
            [{'text': 'The answer starts'}],
            'CONTINUATION',
            token='token-1',
            candidates_token_count=20_000,
        ),
        _hop(
            [{'text': ''}],
            'CONTINUATION',
            token='token-2',
            candidates_token_count=30_768,
            thoughts_token_count=2_000,
        ),
        _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2),
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello!')
    self.assertEqual(
        ['continuationToken' in r for r in sent], [False, True, False]
    )

  def test_answers_are_unaffected(self):
    sent, post = self._serve(
        _hop([{'text': 'Hello!'}], 'STOP', candidates_token_count=2)
    )
    with mock.patch('requests.Session.post', side_effect=post):
      with mock.patch.object(pg.logging, 'warning') as warning:
        response = self._lm()('hi')
    self.assertEqual(response.text, 'Hello!')
    self.assertEqual(len(sent), 1)
    warning.assert_not_called()


if __name__ == '__main__':
  unittest.main()
