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
"""Tests for Anthropic models."""

import base64
import copy
import datetime
import os
from typing import Any
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import anthropic
from langfun.core.llms import vertexai  # pylint: disable=unused-import
import pyglove as pg
import requests


def mock_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs

  response = requests.Response()
  response.status_code = 200

  # Construct base text from user/assistant messages payload
  messages_payload_text = '\n'.join(
      c['content'][0]['text']
      for c in json.get('messages', [])
      if c.get('content')
      and isinstance(c['content'], list)
      and c['content']
      and c['content'][0].get('type') == 'text'
      and 'text' in c['content'][0]
  )

  # Check for a system prompt in the request payload
  system_prompt_text = json.get('system')

  processed_text_parts = []
  if system_prompt_text:
    if isinstance(system_prompt_text, list):
      system_prompt_text = '\n'.join(
          [x['text'] for x in system_prompt_text if 'text' in x]
      )
    processed_text_parts.append(system_prompt_text)
  if messages_payload_text:
    processed_text_parts.append(messages_payload_text)

  processed_text = '\n'.join(processed_text_parts)

  response_content_text = (
      f'{processed_text} with temperature={json.get("temperature")}, '
      f'top_k={json.get("top_k")}, '
      f'top_p={json.get("top_p")}, '
      f'max_tokens={json.get("max_tokens")}, '
      f'stop={json.get("stop_sequences")}.'
  )

  response._content = pg.to_json_str({
      'content': [{'type': 'text', 'text': response_content_text}],
      'usage': {
          'input_tokens': (
              2
          ),  # Placeholder: adjust if tests need accurate token counts
          'output_tokens': (
              1
          ),  # Placeholder: adjust if tests need accurate token counts
      },
  }).encode()
  return response


image_content = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x18\x00\x00\x00\x18\x04'
    b'\x03\x00\x00\x00\x12Y \xcb\x00\x00\x00\x18PLTE\x00\x00'
    b'\x00fff_chaag_cg_ch^ci_ciC\xedb\x94\x00\x00\x00\x08tRNS'
    b'\x00\n\x9f*\xd4\xff_\xf4\xe4\x8b\xf3a\x00\x00\x00>IDATx'
    b'\x01c \x05\x08)"\xd8\xcc\xae!\x06pNz\x88k\x19\\Q\xa8"\x10'
    b'\xc1\x14\x95\x01%\xc1\n\xa143Ta\xa8"D-\x84\x03QM\x98\xc3'
    b'\x1a\x1a\x1a@5\x0e\x04\xa0q\x88\x05\x00\x07\xf8\x18\xf9'
    b'\xdao\xd0|\x00\x00\x00\x00IEND\xaeB`\x82'
)

pdf_content = (
    b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<<'
    b' /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page'
    b' /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0'
    b' obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 700 Td (Hello, PDF'
    b' content!) Tj ET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype'
    b' /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f'
    b' \n0000000010 00000 n \n0000000079 00000 n \n0000000178 00000 n'
    b' \n0000000278 00000 n \n0000000407 00000 n \ntrailer\n<< /Size 6 /Root 1'
    b' 0 R >>\nstartxref\n517\n%%EOF'
)


def mock_mm_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  v = json['messages'][0]['content'][0]
  content = lf_modalities.Mime.from_bytes(base64.b64decode(v['source']['data']))

  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'content': [{
          'type': 'text',
          'text': f'{v["type"]}: {content.mime_type}',
      }],
      'usage': {
          'input_tokens': 2,
          'output_tokens': 1,
      },
  }).encode()
  return response


def mock_requests_post_error(status_code, error_type, error_message):
  def _mock_requests(url: str, json: dict[str, Any], **kwargs):
    del url, json, kwargs
    response = requests.Response()
    response.status_code = status_code
    response._content = pg.to_json_str(
        {
            'error': {
                'type': error_type,
                'message': error_message,
            }
        }
    ).encode()
    return response

  return _mock_requests


class AnthropicTest(unittest.TestCase):

  def test_basics(self):
    self.assertEqual(
        anthropic.Claude3Haiku().model_id, 'claude-3-haiku-20240307'
    )
    self.assertGreater(anthropic.Claude3Haiku().max_concurrency, 0)

  def test_model_alias(self):
    # Alias will be normalized to the official version.
    self.assertEqual(
        anthropic.Anthropic('claude-3-5-sonnet-20241022').model_id,
        'claude-3-5-sonnet-20241022',
    )
    self.assertEqual(
        anthropic.Anthropic('claude-3-5-sonnet-v2@20241022').model_id,
        'claude-3-5-sonnet-20241022',
    )

  def test_api_key(self):
    lm = anthropic.Claude3Haiku()
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      lm('hi')

    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post

      lm = anthropic.Claude3Haiku(api_key='fake key')
      self.assertRegex(lm('hello').text, 'hello.*')

      os.environ['ANTHROPIC_API_KEY'] = 'abc'
      lm = anthropic.Claude3Haiku()
      self.assertRegex(lm('hello').text, 'hello.*')
      del os.environ['ANTHROPIC_API_KEY']

  def test_call(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = anthropic.Claude3Haiku(api_key='fake_key')
      response = lm('hello', temperature=0.0, top_k=0.1, top_p=0.2, stop=['\n'])
      self.assertEqual(
          response.text,
          (
              'hello with temperature=0.0, top_k=0.1, top_p=0.2, '
              "max_tokens=4096, stop=['\\n']."
          ),
      )
      self.assertIsNotNone(response.usage)
      self.assertIsNotNone(response.usage.prompt_tokens, 2)
      self.assertIsNotNone(response.usage.completion_tokens, 1)
      self.assertIsNotNone(response.usage.total_tokens, 3)
      self.assertGreater(response.usage.estimated_cost, 0)

  def test_call_with_system_message(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = anthropic.Claude3Haiku(api_key='fake_key')
      response = lm(
          lf.UserMessage('hello', system_message=lf.SystemMessage('system')),
          temperature=0.0,
          top_k=0.1,
          top_p=0.2,
          stop=['\n'],
      )
      self.assertEqual(
          response.text,
          (
              'system\nhello with temperature=0.0, top_k=0.1, top_p=0.2, '
              "max_tokens=4096, stop=['\\n']."
          ),
      )

  def test_mm_call(self):
    with mock.patch('requests.Session.post') as mock_mm_request:
      mock_mm_request.side_effect = mock_mm_requests_post
      lm = anthropic.Claude3Haiku(api_key='fake_key')
      response = lm(lf_modalities.Image.from_bytes(image_content), lm=lm)
      self.assertEqual(response.text, 'image: image/png')

  def test_pdf_call(self):
    with mock.patch('requests.Session.post') as mock_mm_request:
      mock_mm_request.side_effect = mock_mm_requests_post
      lm = anthropic.Claude35Sonnet(api_key='fake_key')
      response = lm(lf_modalities.PDF.from_bytes(pdf_content), lm=lm)
      self.assertEqual(response.text, 'document: application/pdf')

  def test_call_errors(self):
    for status_code, error_type, error_message in [
        (429, 'rate_limit', 'Rate limit exceeded.'),
        (529, 'service_unavailable', 'Service unavailable.'),
        (500, 'bad_request', 'Bad request.'),
    ]:
      with mock.patch('requests.Session.post') as mock_mm_request:
        mock_mm_request.side_effect = mock_requests_post_error(
            status_code, error_type, error_message
        )
        lm = anthropic.Claude3Haiku(api_key='fake_key')
        with self.assertRaisesRegex(
            Exception, f'.*{status_code}: .*{error_message}'
        ):
          lm('hello', max_attempts=1)

  def test_call_with_context_limit_error(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post_error(
          413, 'bad_request', 'Prompt is too long.'
      )
      lm = anthropic.Claude3Haiku(api_key='fake_key')
      with self.assertRaisesRegex(lf.ContextLimitError, 'Prompt is too long.'):
        lm('hello', max_attempts=1)

  def test_call_with_context_limit_error_400(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post_error(
          400,
          'invalid_request_error',
          'prompt is too long: 267093 tokens > 200000 maximum',
      )
      lm = anthropic.Claude3Haiku(api_key='fake_key')
      with self.assertRaisesRegex(lf.ContextLimitError, 'prompt is too long'):
        lm('hello', max_attempts=1)

  def test_text_mime_call(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = anthropic.Claude35Sonnet(api_key='fake_key')
      text_content = b'def hello():\n    print("hello")\n'
      response = lm(
          lf.Template(
              'Review this code: {{code}}',
              code=lf_modalities.Custom(
                  mime='text/plain', content=text_content
              ),
          ).render(),
          lm=lm,
      )
      self.assertIn('Review this code:', response.text)

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('claude-3-5-sonnet-latest'),
        anthropic.Anthropic,
    )

  def test_knowledge_cutoff(self):
    # Check that claude-opus-4-6 entries have the correct knowledge cutoff.
    opus_entries = [
        info
        for info in anthropic.SUPPORTED_MODELS
        if (
            info.model_id == 'claude-opus-4-6'
            or info.alias_for == 'claude-opus-4-6'
        )
    ]
    self.assertEqual(len(opus_entries), 2)
    for entry in opus_entries:
      self.assertEqual(
          entry.knowledge_cutoff,
          datetime.date(2025, 8, 31),
      )

  def test_knowledge_cutoff_default(self):
    model = anthropic.Anthropic('claude-3-5-sonnet-20241022')
    self.assertIsNone(model.model_info.knowledge_cutoff)

  def test_thinking_param_true_adaptive(self):
    """Claude 4.7 + thinking=True -> adaptive thinking, no budget needed."""
    lm = anthropic.Claude47Opus(api_key='fake', thinking=True)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )
    self.assertEqual(args['max_tokens'], 1000)
    self.assertNotIn('temperature', args)

  def test_thinking_param_true_manual(self):
    """Older model + thinking=True + max_thinking_tokens -> manual thinking."""
    lm = anthropic.Claude35Sonnet(api_key='fake', thinking=True)
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
        )
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )
    self.assertEqual(args['max_tokens'], 2024)
    self.assertNotIn('temperature', args)

  def test_thinking_param_true_manual_no_budget_defaults(self):
    """Older model + thinking=True WITHOUT budget -> defaults to 50% max."""
    lm = anthropic.Claude35Sonnet(api_key='fake', thinking=True)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=8192, temperature=0.5)
    )
    # 8192 // 2 = 4096
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 4096}
    )
    # max_tokens remains unchanged because 8192 > 4096
    self.assertEqual(args['max_tokens'], 8192)

  def test_thinking_param_false_no_thinking(self):
    """thinking=False -> no thinking config, temperature preserved."""
    lm = anthropic.Claude46Opus(api_key='fake', thinking=False)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertNotIn('thinking', args)
    self.assertEqual(args['temperature'], 0.5)

  def test_thinking_param_none_default_no_thinking(self):
    """Default thinking=None with no max_thinking_tokens -> no thinking."""
    lm = anthropic.Claude46Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertNotIn('thinking', args)
    self.assertEqual(args['temperature'], 0.5)

  def test_thinking_false_overrides_max_thinking_tokens(self):
    """thinking=False + max_thinking_tokens -> NO thinking (False wins)."""
    lm = anthropic.Claude46Opus(api_key='fake', thinking=False)
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
        )
    )
    self.assertNotIn('thinking', args)
    self.assertEqual(args['temperature'], 0.5)

  def test_thinking_false_overrides_max_thinking_tokens_older_model(self):
    """thinking=False + max_thinking_tokens on older model -> NO thinking."""
    lm = anthropic.Claude35Sonnet(api_key='fake', thinking=False)
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
        )
    )
    self.assertNotIn('thinking', args)
    self.assertEqual(args['temperature'], 0.5)

  def test_thinking_options_older_model(self):
    lm = anthropic.Claude35Sonnet(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
        )
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )
    self.assertEqual(args['max_tokens'], 2024)
    self.assertNotIn('temperature', args)

  def test_thinking_options_adaptive(self):
    lm = anthropic.Claude47Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
        )
    )
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )
    self.assertEqual(args['max_tokens'], 1000)
    self.assertNotIn('temperature', args)

  def test_thinking_options_opus_4_7(self):
    lm = anthropic.Claude47Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
        )
    )
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )
    self.assertEqual(args['output_config'], {'effort': 'high'})
    self.assertEqual(args['max_tokens'], 1000)
    self.assertNotIn('temperature', args)

  def test_thinking_options_opus_4_7_with_effort(self):
    lm = anthropic.Claude47Opus(api_key='fake', effort='xhigh')
    args = lm._request_args(
        lf.LMSamplingOptions(max_thinking_tokens=1024, max_tokens=1000)
    )
    self.assertEqual(args['output_config'], {'effort': 'xhigh'})

  def test_thinking_options_opus_4_7_with_reasoning_effort(self):
    lm = anthropic.Claude47Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, reasoning_effort='low'
        )
    )
    self.assertEqual(args['output_config'], {'effort': 'low'})

  def test_thinking_options_opus_4_7_no_effort(self):
    lm = anthropic.Claude47Opus(api_key='fake', effort=None)
    args = lm._request_args(
        lf.LMSamplingOptions(max_thinking_tokens=1024, max_tokens=1000)
    )
    self.assertNotIn('output_config', args)

  def test_opus47_no_thinking_removes_sampling_params(self):
    lm = anthropic.Claude47Opus(api_key='fake', thinking=False)
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_tokens=1000, temperature=0.5, top_k=40, top_p=0.9
        )
    )
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)
    self.assertNotIn('top_p', args)

  def test_thinking_param_true_adaptive_opus_4_7(self):
    """Claude 4.7 + thinking=True -> adaptive thinking with summarized display."""
    lm = anthropic.Claude47Opus(api_key='fake', thinking=True)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )

  def test_opus46_no_thinking_by_default(self):
    lm = anthropic.Claude46Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertNotIn('thinking', args)
    self.assertEqual(args['temperature'], 0.5)

  def test_older_model_no_thinking_by_default(self):
    lm = anthropic.Claude35Sonnet(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertNotIn('thinking', args)
    self.assertEqual(args['temperature'], 0.5)

  def test_thinking_adaptive_with_max_thinking_tokens_set(self):
    """Adaptive model with thinking=True AND max_thinking_tokens should use adaptive (not manual)."""
    model = anthropic.Claude47Opus(api_key='test_key', thinking=True)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1000, max_thinking_tokens=2048)
    )
    # Should be adaptive, not manual, even though max_thinking_tokens is set
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )
    # budget_tokens should NOT be present
    self.assertNotIn('budget_tokens', args['thinking'])

  def test_thinking_removes_temperature_top_k_top_p(self):
    """When thinking is enabled, temperature/top_k/top_p should be removed."""
    model = anthropic.Claude47Opus(api_key='test_key', thinking=True)
    args = model._request_args(
        lf.LMSamplingOptions(
            max_tokens=1000, temperature=0.7, top_k=40, top_p=0.9
        )
    )
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)
    self.assertNotIn('top_p', args)
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )

  def test_thinking_manual_max_tokens_adjustment(self):
    """When max_tokens < max_thinking_tokens, max_tokens should be increased."""
    model = anthropic.Claude35Sonnet(api_key='test_key', thinking=True)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024, max_thinking_tokens=4096)
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 4096}
    )
    # max_tokens should be adjusted: 1024 + 4096 = 5120
    self.assertEqual(args['max_tokens'], 5120)

  def test_thinking_manual_max_tokens_no_adjustment_when_sufficient(self):
    """When max_tokens >= max_thinking_tokens, max_tokens should NOT be increased."""
    model = anthropic.Claude35Sonnet(api_key='test_key', thinking=True)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=8192, max_thinking_tokens=4096)
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 4096}
    )
    # max_tokens should NOT be adjusted since 8192 >= 4096
    self.assertEqual(args['max_tokens'], 8192)

  def test_model_uri_instantiation_claude46_opus(self):
    """Test LLM instantiation from model URI string for Claude 4.6 Opus."""
    model = lf.LanguageModel.get('claude-opus-4-6?api_key=test_key')
    self.assertIsInstance(model, anthropic.Anthropic)
    self.assertFalse(model._use_adaptive_thinking)

  def test_model_uri_instantiation_with_thinking_true(self):
    """Test model URI with thinking=true parameter."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?api_key=test_key&thinking=true'
    )
    self.assertTrue(model.thinking)
    self.assertFalse(model._use_adaptive_thinking)
    # Now requires manual budget tokens.
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024, max_thinking_tokens=1024)
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )

  def test_model_uri_instantiation_with_thinking_true_uppercase(self):
    """Test model URI with thinking=True parameter (uppercase)."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?api_key=test_key&thinking=True'
    )
    self.assertTrue(model.thinking)
    self.assertFalse(model._use_adaptive_thinking)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024, max_thinking_tokens=1024)
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )

  def test_model_uri_instantiation_user_scenario(self):
    """Test model URI with user's specific scenario."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?project=lf-agent&location=us-east5&max_attempts=80&timeout=300&thinking=True'
    )
    self.assertTrue(model.thinking)
    self.assertFalse(model._use_adaptive_thinking)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024, max_thinking_tokens=1024)
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )

  def test_model_uri_instantiation_with_thinking_false(self):
    """Test model URI with thinking=false parameter."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?api_key=test_key&thinking=false'
    )
    self.assertFalse(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_model_uri_instantiation_without_thinking(self):
    """Test model URI without thinking parameter (default None)."""
    model = lf.LanguageModel.get('claude-opus-4-6?api_key=test_key')
    self.assertIsNone(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_model_uri_instantiation_older_model_with_thinking(self):
    """Test model URI for older model with thinking=true requires budget."""
    model = lf.LanguageModel.get(
        'claude-3-5-sonnet-20241022?api_key=test_key&thinking=true'
    )
    self.assertTrue(model.thinking)
    self.assertFalse(model._use_adaptive_thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )
    self.assertEqual(args['max_tokens'], 2048)

  def test_model_uri_instantiation_opus_4_7(self):
    """Test LLM instantiation from model URI for Claude Opus 4.7."""
    model = lf.LanguageModel.get('claude-opus-4-7?api_key=test_key')
    self.assertIsInstance(model, anthropic.Anthropic)
    self.assertTrue(model._use_adaptive_thinking)
    self.assertEqual(model.effort, 'high')

  def test_model_uri_instantiation_opus_4_7_with_thinking(self):
    """Test Opus 4.7 model URI with thinking=true."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?api_key=test_key&thinking=true'
    )
    self.assertTrue(model.thinking)
    self.assertTrue(model._use_adaptive_thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(
        args['thinking'],
        {'type': 'adaptive', 'display': 'summarized'},
    )
    self.assertEqual(args['output_config'], {'effort': 'high'})

  def test_model_uri_instantiation_opus_4_7_no_thinking(self):
    """Test Opus 4.7 model URI with thinking=false."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?api_key=test_key&thinking=false'
    )
    self.assertFalse(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)
    # Opus 4.7 still strips temperature/top_k/top_p
    self.assertNotIn('temperature', args)

  def test_model_uri_instantiation_opus_4_7_default(self):
    """Test Opus 4.7 model URI default (thinking=None)."""
    model = lf.LanguageModel.get('claude-opus-4-7?api_key=test_key')
    self.assertIsNone(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_opus47_default_no_thinking_strips_sampling_params(self):
    """Opus 4.7 strips temperature/top_k/top_p even without thinking."""
    lm = anthropic.Claude47Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_tokens=1000, temperature=0.7, top_k=40, top_p=0.9
        )
    )
    self.assertNotIn('thinking', args)
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)
    self.assertNotIn('top_p', args)

  def test_opus47_effort_max(self):
    """Test Opus 4.7 with effort='max'."""
    lm = anthropic.Claude47Opus(api_key='fake', effort='max', thinking=True)
    args = lm._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(args['output_config'], {'effort': 'max'})

  def test_opus47_effort_medium(self):
    """Test Opus 4.7 with effort='medium'."""
    lm = anthropic.Claude47Opus(api_key='fake', effort='medium', thinking=True)
    args = lm._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(args['output_config'], {'effort': 'medium'})

  def test_opus47_effort_low(self):
    """Test Opus 4.7 with effort='low'."""
    lm = anthropic.Claude47Opus(api_key='fake', effort='low', thinking=True)
    args = lm._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(args['output_config'], {'effort': 'low'})

  def test_opus47_reasoning_effort_overrides_model_effort(self):
    """reasoning_effort in sampling options overrides model-level effort."""
    lm = anthropic.Claude47Opus(api_key='fake', effort='high', thinking=True)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1024, reasoning_effort='low')
    )
    self.assertEqual(args['output_config'], {'effort': 'low'})

  def test_opus46_effort_config_no_effect(self):
    """Claude 4.6 has no adaptive effort configuration since it uses manual budget."""
    lm = anthropic.Claude46Opus(api_key='fake', thinking=True, effort='high')
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1024, max_thinking_tokens=1024)
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )
    self.assertNotIn('output_config', args)

  def test_thinking_overflow_guardrail_honors_model_limit(self):
    """Test that forcing overflow clamps max_tokens and contracts budget safely."""
    # Max output limit for Claude46 is 128,000.
    lm = anthropic.Claude46Opus(api_key='fake', thinking=True)

    # Case 1: Absolute overflow where user forces budget to equal max model cap.
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=128000, max_thinking_tokens=128000)
    )
    # Must clamp back to 128k
    self.assertEqual(args['max_tokens'], 128000)
    # Must contract budget to 128k - 1024 = 126976
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 126976}
    )

    # Case 2: Extreme total overage beyond physical limit.
    args2 = lm._request_args(
        lf.LMSamplingOptions(max_tokens=150000, max_thinking_tokens=150000)
    )
    self.assertEqual(args2['max_tokens'], 128000)
    self.assertEqual(args2['thinking']['budget_tokens'], 126976)

  def test_thinking_extreme_small_max_tokens(self):
    """Tests that absurdity small max_tokens correctly pads to safe harbor."""
    lm = anthropic.Claude46Opus(api_key='fake', thinking=True)
    args = lm._request_args(lf.LMSamplingOptions(max_tokens=10))
    # budget defaults to 1024. max_tokens becomes 10 + 1024 = 1034.
    self.assertEqual(args['max_tokens'], 1034)
    self.assertEqual(args['thinking']['budget_tokens'], 1024)

  def test_thinking_boundary_condition_exact_threshold(self):
    """Tests exact limit case for max_tokens vs budget calculation."""
    lm = anthropic.Claude46Opus(api_key='fake', thinking=True)
    args = lm._request_args(lf.LMSamplingOptions(max_tokens=1024))
    # budget defaults to 1024. 1024 <= 1024 triggers, doubling to 2048.
    self.assertEqual(args['max_tokens'], 2048)
    self.assertEqual(args['thinking']['budget_tokens'], 1024)

  def test_request_caching_default_on_no_extras_needed(self):
    # User-perspective: vanilla Anthropic instantiation, no extras, no flags.
    # Caching MUST be applied automatically (Gemini-implicit-style).
    # Use whatever Sonnet variant exists.
    model = anthropic.Claude35Sonnet_20241022()
    prompt = lf.UserMessage('hello', system_message=lf.SystemMessage('sys'))
    req = model.request(prompt, lf.LMSamplingOptions())  # NO extras at all
    self.assertIsInstance(req['system'], list)
    self.assertEqual(req['system'][0]['cache_control'], {'type': 'ephemeral'})
    self.assertEqual(
        req['messages'][-1]['content'][-1]['cache_control'],
        {'type': 'ephemeral'},
    )

  def test_request_caching_applied_to_anthropic_default(self):
    lm = anthropic.Anthropic('claude-3-5-sonnet-20241022')
    prompt = lf.UserMessage('hello', system_message=lf.SystemMessage('sys'))
    options = lf.LMSamplingOptions()
    req = lm.request(prompt, options)

    self.assertIsInstance(req['system'], list)
    self.assertEqual(req['system'][0]['cache_control'], {'type': 'ephemeral'})
    self.assertEqual(
        req['messages'][-1]['content'][-1]['cache_control'],
        {'type': 'ephemeral'},
    )

  def test_request_caching_works_on_opus_subclass(self):
    lm = anthropic.Claude3Opus(api_key='fake')
    prompt = lf.UserMessage('hello', system_message=lf.SystemMessage('sys'))
    options = lf.LMSamplingOptions(extras={'enable_prompt_caching': True})
    req = lm.request(prompt, options)

    self.assertIsInstance(req['system'], list)
    self.assertEqual(req['system'][0]['cache_control'], {'type': 'ephemeral'})
    self.assertEqual(
        req['messages'][-1]['content'][-1]['cache_control'],
        {'type': 'ephemeral'},
    )


class Claude48OpusTest(unittest.TestCase):
  """Tests for Claude Opus 4.8 model support."""

  def test_opus48_basics(self):
    """Test Claude Opus 4.8 basic instantiation."""
    lm = anthropic.Claude48Opus(api_key='fake')
    self.assertEqual(lm.model_id, 'claude-opus-4-8')
    self.assertTrue(lm._use_adaptive_thinking)

  def test_opus48_model_info(self):
    """Test Claude Opus 4.8 model info is registered."""
    opus_entries = [
        info
        for info in anthropic.SUPPORTED_MODELS
        if info.model_id == 'claude-opus-4-8'
    ]
    self.assertEqual(len(opus_entries), 1)
    self.assertEqual(opus_entries[0].provider, 'Anthropic')
    self.assertTrue(opus_entries[0].in_service)

  def test_opus48_vertexai_model_info(self):
    """Test Claude Opus 4.8 VertexAI model info is registered."""
    vertex_entries = [
        info
        for info in anthropic.SUPPORTED_MODELS
        if info.model_id == 'claude-opus-4-8@latest'
    ]
    self.assertEqual(len(vertex_entries), 1)
    self.assertEqual(vertex_entries[0].provider, 'VertexAI')
    self.assertEqual(vertex_entries[0].alias_for, 'claude-opus-4-8')

  def test_thinking_param_true_adaptive_opus_4_8(self):
    """Claude 4.8 + thinking=True -> adaptive thinking with summarized display."""
    lm = anthropic.Claude48Opus(api_key='fake', thinking=True)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )
    self.assertEqual(args['max_tokens'], 1000)
    self.assertNotIn('temperature', args)

  def test_opus48_no_thinking_removes_sampling_params(self):
    """Opus 4.8 strips temperature/top_k/top_p even with thinking=False."""
    lm = anthropic.Claude48Opus(api_key='fake', thinking=False)
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_tokens=1000, temperature=0.5, top_k=40, top_p=0.9
        )
    )
    self.assertNotIn('thinking', args)
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)
    self.assertNotIn('top_p', args)

  def test_opus48_default_strips_sampling_params(self):
    """Opus 4.8 strips temperature/top_k/top_p even without thinking."""
    lm = anthropic.Claude48Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_tokens=1000, temperature=0.7, top_k=40, top_p=0.9
        )
    )
    self.assertNotIn('thinking', args)
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)
    self.assertNotIn('top_p', args)

  def test_opus48_effort_max(self):
    """Test Opus 4.8 with effort='max'."""
    lm = anthropic.Claude48Opus(api_key='fake', effort='max', thinking=True)
    args = lm._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(args['output_config'], {'effort': 'max'})

  def test_opus48_effort_medium(self):
    """Test Opus 4.8 with effort='medium'."""
    lm = anthropic.Claude48Opus(api_key='fake', effort='medium', thinking=True)
    args = lm._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(args['output_config'], {'effort': 'medium'})

  def test_opus48_reasoning_effort_overrides_model_effort(self):
    """reasoning_effort in sampling options overrides model-level effort."""
    lm = anthropic.Claude48Opus(api_key='fake', effort='high', thinking=True)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1024, reasoning_effort='low')
    )
    self.assertEqual(args['output_config'], {'effort': 'low'})

  def test_opus48_no_effort(self):
    """Test Opus 4.8 with effort=None."""
    lm = anthropic.Claude48Opus(api_key='fake', effort=None)
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1024, max_thinking_tokens=1024)
    )
    self.assertNotIn('output_config', args)

  def test_opus48_thinking_options_adaptive(self):
    """Claude 4.8 with thinking options uses adaptive mode."""
    lm = anthropic.Claude48Opus(api_key='fake')
    args = lm._request_args(
        lf.LMSamplingOptions(
            max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
        )
    )
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )
    self.assertEqual(args['output_config'], {'effort': 'high'})
    self.assertEqual(args['max_tokens'], 1000)
    self.assertNotIn('temperature', args)

  def test_model_uri_instantiation_opus_4_8(self):
    """Test LLM instantiation from model URI for Claude Opus 4.8."""
    model = lf.LanguageModel.get('claude-opus-4-8?api_key=test_key')
    self.assertIsInstance(model, anthropic.Anthropic)
    self.assertTrue(model._use_adaptive_thinking)
    self.assertEqual(model.effort, 'high')

  def test_model_uri_instantiation_opus_4_8_with_thinking(self):
    """Test Opus 4.8 model URI with thinking=true."""
    model = lf.LanguageModel.get(
        'claude-opus-4-8?api_key=test_key&thinking=true'
    )
    self.assertTrue(model.thinking)
    self.assertTrue(model._use_adaptive_thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(
        args['thinking'],
        {'type': 'adaptive', 'display': 'summarized'},
    )
    self.assertEqual(args['output_config'], {'effort': 'high'})

  def test_model_uri_instantiation_opus_4_8_no_thinking(self):
    """Test Opus 4.8 model URI with thinking=false."""
    model = lf.LanguageModel.get(
        'claude-opus-4-8?api_key=test_key&thinking=false'
    )
    self.assertFalse(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)
    # Opus 4.8 still strips temperature/top_k/top_p
    self.assertNotIn('temperature', args)

  def test_opus48_call_e2e(self):
    """End-to-end call test for Claude Opus 4.8."""
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = anthropic.Claude48Opus(api_key='fake_key', thinking=False)
      response = lm('hello')
      self.assertIn('hello', response.text)
      # Verify sampling params are stripped (temperature/top_k/top_p = None)
      self.assertIn('temperature=None', response.text)
      self.assertIn('top_k=None', response.text)
      self.assertIn('top_p=None', response.text)


class AnthropicCachingTest(unittest.TestCase):

  def test_helper_stamps_system_string_to_block_list(self):
    request = {'system': 'sys text', 'messages': []}
    anthropic._apply_cache_breakpoints(request)
    self.assertIsInstance(request['system'], list)
    self.assertEqual(len(request['system']), 1)
    self.assertEqual(request['system'][0]['type'], 'text')
    self.assertEqual(request['system'][0]['text'], 'sys text')
    self.assertEqual(
        request['system'][0]['cache_control'], {'type': 'ephemeral'}
    )

  def test_helper_stamps_last_message_last_block(self):
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'a'},
            {'type': 'text', 'text': 'b'},
        ],
    }]
    request = {'messages': messages}
    anthropic._apply_cache_breakpoints(request)
    content = request['messages'][-1]['content']
    self.assertNotIn('cache_control', content[0])
    self.assertEqual(content[1]['cache_control'], {'type': 'ephemeral'})

  def test_helper_idempotent(self):
    request_in = {
        'system': 'sys text',
        'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}
        ],
    }
    once = anthropic._apply_cache_breakpoints(copy.deepcopy(request_in))
    twice = anthropic._apply_cache_breakpoints(copy.deepcopy(once))
    self.assertEqual(once, twice)
    # Also verify no double-stamp: system list still has exactly one element
    # with cache_control.
    self.assertEqual(len(twice['system']), 1)
    self.assertEqual(
        twice['system'][-1]['cache_control'], {'type': 'ephemeral'}
    )
    # And the last message's last content block has cache_control exactly once.
    last_block = twice['messages'][-1]['content'][-1]
    self.assertEqual(last_block['cache_control'], {'type': 'ephemeral'})

  def test_helper_handles_image_block_as_last(self):
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'foo'},
            {'type': 'image', 'source': {'data': 'img'}},
        ],
    }]
    request = {'messages': messages}
    anthropic._apply_cache_breakpoints(request)
    self.assertEqual(
        request['messages'][-1]['content'][-1]['cache_control'],
        {'type': 'ephemeral'},
    )

  def test_helper_skip_system_when_cache_system_false(self):
    request = {'system': 'sys text', 'messages': []}
    anthropic._apply_cache_breakpoints(request, cache_system=False)
    self.assertEqual(request['system'], 'sys text')

  def test_helper_skip_last_message_when_cache_last_message_false(self):
    request = {
        'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}]}
        ]
    }
    anthropic._apply_cache_breakpoints(request, cache_last_message=False)
    self.assertNotIn('cache_control', request['messages'][-1]['content'][0])

  def test_adversarial_empty_or_missing(self):
    # Empty messages - no op, no exception
    request = {'messages': []}
    anthropic._apply_cache_breakpoints(request)
    self.assertEqual(request['messages'], [])

    # Messages without content key
    request = {'messages': [{'role': 'user'}]}
    anthropic._apply_cache_breakpoints(request)
    self.assertNotIn('content', request['messages'][0])

    # System already list with cache control
    request = {
        'system': [{
            'type': 'text',
            'text': 'sys',
            'cache_control': {'type': 'existing'},
        }]
    }
    anthropic._apply_cache_breakpoints(request)
    # Should not replace existing cache control
    self.assertEqual(request['system'][0]['cache_control']['type'], 'existing')

  def test_adversarial_long_and_unicode(self):
    long_text = 'x' * 60000 + ' 日本語'
    request = {
        'system': long_text,
        'messages': [{
            'role': 'user',
            'content': [{'type': 'text', 'text': 'こんにちは'}],
        }],
    }
    anthropic._apply_cache_breakpoints(request)
    self.assertIn('cache_control', request['system'][0])
    self.assertIn('cache_control', request['messages'][-1]['content'][-1])


if __name__ == '__main__':
  unittest.main()
