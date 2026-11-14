# Copyright 2024 The Langfun Authors
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
"""Tests for VertexAI models."""

import datetime
import os
from typing import Any
import unittest
from unittest import mock

from google import auth
from google.auth import exceptions
import langfun.core as lf
from langfun.core.llms import anthropic
from langfun.core.llms import gemini
from langfun.core.llms import rest
from langfun.core.llms import vertexai
import pyglove as pg
import requests


class VertexAITest(unittest.TestCase):
  """Tests for Vertex model with REST API."""

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_and_location_check(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      _ = vertexai.VertexAIGemini15Pro()._api_initialized

    with self.assertRaisesRegex(ValueError, 'Please specify `location`'):
      _ = vertexai.VertexAIGemini15Pro(
          project='abc', location=None
      )._api_initialized

    self.assertTrue(
        vertexai.VertexAIGemini15Pro(
            project='abc', location='us-central1'
        )._api_initialized
    )

    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    model = vertexai.VertexAIGemini15Pro(location=pg.MISSING_VALUE)
    self.assertEqual(model.resource_id, 'vertexai://gemini-1.5-pro-002')
    self.assertIn('us-central1', model.api_endpoint)
    self.assertTrue(model._api_initialized)
    self.assertIsNotNone(model.session())
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_carbon(self):
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    # Unset via cleanup, so a failing assertion cannot leak into later tests.
    self.addCleanup(os.environ.pop, 'VERTEXAI_PROJECT', None)
    self.addCleanup(os.environ.pop, 'VERTEXAI_LOCATION', None)
    model = vertexai.VertexAIGemini4Carbon(location=pg.MISSING_VALUE)
    self.assertEqual(model.resource_id, 'vertexai://gemini-4-carbon')
    # Gemini 4 models default to 'global', overriding the environment.
    self.assertEqual(model.location, 'global')
    self.assertIn('global', model.api_endpoint)
    self.assertEqual(model.model_info.provider, 'VertexAI')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_carbon_string_id_resolves_to_global(self):
    # The bare string id must resolve to VertexAIGemini4Carbon, not the generic
    # VertexAIGemini, so it keeps the `global` location pin.
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    # Unset via cleanup, so a failing assertion cannot leak into later tests.
    self.addCleanup(os.environ.pop, 'VERTEXAI_PROJECT', None)
    self.addCleanup(os.environ.pop, 'VERTEXAI_LOCATION', None)
    model = lf.LanguageModel.get('gemini-4-carbon')
    self.assertIsInstance(model, vertexai.VertexAIGemini4Carbon)
    self.assertEqual(model.resource_id, 'vertexai://gemini-4-carbon')
    self.assertIn('global', model.api_endpoint)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_carbon_model_uri(self):
    model = lf.LanguageModel.get(
        'gemini-4-carbon?project=lf-agent&max_attempts=80&thinking_level=high'
    )
    self.assertIsInstance(model, vertexai.VertexAIGemini4Carbon)
    self.assertEqual(model.project, 'lf-agent')
    self.assertEqual(model.max_attempts, 80)
    self.assertEqual(model.timeout, 1200)
    self.assertEqual(model.sampling_options.thinking_level, 'high')
    self.assertEqual(model.sampling_options.max_tokens, 65_536)
    self.assertEqual(
        model.api_endpoint,
        'https://aiplatform.googleapis.com/v1beta1/projects/lf-agent/'
        'locations/global/publishers/google/models/gemini-4-carbon:'
        'generateContent',
    )
    config = model.request(lf.UserMessage('hi'), model.sampling_options)[
        'generationConfig'
    ]
    self.assertEqual(config['thinkingConfig'], {'thinkingLevel': 'high'})
    self.assertEqual(config['maxOutputTokens'], 65_536)
    self.assertEqual(config['mediaResolution'], 'MEDIA_RESOLUTION_HIGH')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_carbon_model_uri_overrides_defaults(self):
    model = lf.LanguageModel.get(
        'gemini-4-carbon?project=lf-agent&location=us&thinking_level=medium'
        '&max_tokens=131072&api_version=v1&timeout=60'
    )
    self.assertIsInstance(model, vertexai.VertexAIGemini4Carbon)
    self.assertEqual(model.sampling_options.thinking_level, 'medium')
    self.assertEqual(model.sampling_options.max_tokens, 131_072)
    self.assertEqual(model.timeout, 60)
    self.assertEqual(
        model.api_endpoint,
        'https://aiplatform.googleapis.com/v1/projects/lf-agent/'
        'locations/us/publishers/google/models/gemini-4-carbon:'
        'generateContent',
    )
    # The overrides apply to this instance only.
    self.assertEqual(
        vertexai.VertexAIGemini4Carbon(project='lf-agent').timeout, 1200
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_carbon_resumes_a_response_cut_at_32k_tokens(self):
    # The finish reasons and token counts are those of a live 56K-token
    # response on 2026-11-10, whose text the first request cut mid-line.
    hops = [
        {
            'candidates': [{
                'content': {
                    'role': 'model',
                    'parts': [{'text': '{"id": 1065}\n{"id": 10'}],
                },
                'finishReason': 'CONTINUATION',
                'continuationToken': 'token-1',
            }],
            'usageMetadata': {
                'promptTokenCount': 38_281,
                'candidatesTokenCount': 32_768,
                'totalTokenCount': 71_049,
            },
        },
        {
            'candidates': [{
                'content': {
                    'role': 'model',
                    'parts': [{'text': '66}\n{"id": 1067}'}],
                },
                'finishReason': 'STOP',
            }],
            'usageMetadata': {
                'promptTokenCount': 71_049,
                'candidatesTokenCount': 23_351,
                'totalTokenCount': 94_400,
                'cachedContentTokenCount': 38_282,
            },
        },
    ]
    calls = []

    def post(url, json, **kwargs):
      del kwargs
      calls.append((url, json))
      response = requests.Response()
      response.status_code = 200
      response._content = pg.to_json_str(hops[len(calls) - 1]).encode()  # pylint: disable=protected-access
      return response

    with mock.patch('requests.Session.post', side_effect=post):
      with _today(2026, 11, 30):
        response = vertexai.VertexAIGemini4Carbon(project='test-project')('hi')
    self.assertEqual(response.text, '{"id": 1065}\n{"id": 1066}\n{"id": 1067}')
    self.assertEqual(len(calls), 2)
    self.assertIn('/v1beta1/projects/test-project/', calls[0][0])
    self.assertEqual(calls[1][0], calls[0][0])
    self.assertNotIn('continuationToken', calls[0][1])
    self.assertEqual(calls[1][1]['continuationToken'], 'token-1')
    self.assertEqual(
        calls[1][1]['generationConfig']['maxOutputTokens'], 65_536
    )
    # Every request is billed, so the usage sums both.
    self.assertEqual(response.usage.prompt_tokens, 109_330)
    self.assertEqual(response.usage.cached_prompt_tokens, 38_282)
    self.assertEqual(response.usage.completion_tokens, 56_119)
    # (71,048 uncached * $2 + 38,282 cached * $0.10 + 56,119 output * $10) / 1M
    self.assertAlmostEqual(response.usage.estimated_cost, 0.7071142)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon(self):
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    # Unset via cleanup, so a failing assertion cannot leak into later tests.
    self.addCleanup(os.environ.pop, 'VERTEXAI_PROJECT', None)
    self.addCleanup(os.environ.pop, 'VERTEXAI_LOCATION', None)
    model = vertexai.VertexAIGemini4Argon(location=pg.MISSING_VALUE)
    self.assertEqual(model.resource_id, 'vertexai://gemini-4-argon')
    # Gemini 4 models default to 'global', overriding the environment.
    self.assertEqual(model.location, 'global')
    self.assertIn('global', model.api_endpoint)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon_string_id_resolves_to_global(self):
    # The bare string id must resolve to VertexAIGemini4Argon, not the generic
    # VertexAIGemini, so it keeps the `global` location pin.
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    # Unset via cleanup, so a failing assertion cannot leak into later tests.
    self.addCleanup(os.environ.pop, 'VERTEXAI_PROJECT', None)
    self.addCleanup(os.environ.pop, 'VERTEXAI_LOCATION', None)
    model = lf.LanguageModel.get('gemini-4-argon')
    self.assertIsInstance(model, vertexai.VertexAIGemini4Argon)
    self.assertEqual(model.resource_id, 'vertexai://gemini-4-argon')
    self.assertIn('global', model.api_endpoint)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon_model_uri(self):
    model = lf.LanguageModel.get(
        'gemini-4-argon?project=lf-agent&location=global&max_attempts=80'
        '&timeout=300&exponential_backoff=False&thinking_level=high'
    )
    self.assertIsInstance(model, vertexai.VertexAIGemini4Argon)
    self.assertEqual(model.project, 'lf-agent')
    self.assertEqual(model.max_attempts, 80)
    self.assertEqual(model.timeout, 300)
    self.assertFalse(model.exponential_backoff)
    self.assertEqual(model.sampling_options.thinking_level, 'high')
    self.assertEqual(model.sampling_options.max_tokens, 65_536)
    self.assertEqual(
        model.api_endpoint,
        'https://aiplatform.googleapis.com/v1beta1/projects/lf-agent/'
        'locations/global/publishers/google/models/gemini-4-argon:'
        'generateContent',
    )
    config = model.request(lf.UserMessage('hi'), model.sampling_options)[
        'generationConfig'
    ]
    self.assertEqual(config['thinkingConfig'], {'thinkingLevel': 'high'})
    self.assertEqual(config['maxOutputTokens'], 65_536)
    self.assertEqual(config['mediaResolution'], 'MEDIA_RESOLUTION_HIGH')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon_model_uri_overrides_defaults(self):
    model = lf.LanguageModel.get(
        'gemini-4-argon?project=lf-agent&thinking_level=medium'
        '&max_tokens=131072&api_version=v1'
    )
    self.assertEqual(model.sampling_options.thinking_level, 'medium')
    self.assertEqual(model.sampling_options.max_tokens, 131_072)
    self.assertEqual(model.timeout, 900)
    self.assertIn('aiplatform.googleapis.com/v1/projects/', model.api_endpoint)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon_caps_the_output_budget(self):
    """Unset, one generation can run to ~2^20 tokens across continuations."""
    model = vertexai.VertexAIGemini4Argon(project='test-project')
    self.assertEqual(model.sampling_options.max_tokens, 65_536)
    config = model.request(lf.UserMessage('hi'), model.sampling_options)[
        'generationConfig'
    ]
    self.assertEqual(config['maxOutputTokens'], 65_536)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon_resumes_a_budget_pinned_by_the_model_uri(self):
    """A generation that spends a pinned 65,536-token budget still resumes.

    Here the caller pinned the model's own 65,536-token output maximum and
    thinking spent all of it, so the resume has to ask for more than the model
    can itself produce. It asserts that number anyway rather than dropping the
    ceiling: measured against the service, `maxOutputTokens` above the model
    maximum is accepted, and dropping the key would hand the rest of the
    generation to the service default. The fake refuses only what the service
    refuses -- the ceiling already reported spent.
    """
    starved = {
        'candidates': [{
            'content': {'role': 'model', 'parts': [{'text': ''}]},
            'finishReason': 'MAX_TOKENS',
            'continuationToken': 'token-1',
        }],
        'usageMetadata': {
            'promptTokenCount': 12,
            'candidatesTokenCount': 3,
            'thoughtsTokenCount': 65_533,
            'totalTokenCount': 65_548,
        },
    }
    answered = {
        'candidates': [{
            'content': {'role': 'model', 'parts': [{'text': '42'}]},
            'finishReason': 'STOP',
        }],
        'usageMetadata': {
            'promptTokenCount': 12,
            'candidatesTokenCount': 1,
            'totalTokenCount': 13,
        },
    }
    sent = []

    def post(url, json, **kwargs):
      del url, kwargs
      sent.append(json)
      response = requests.Response()
      ceiling = json['generationConfig'].get('maxOutputTokens')
      if 'continuationToken' not in json:
        served = starved
      elif ceiling == 65_536:
        response.status_code = 400
        response._content = (  # pylint: disable=protected-access
            b'max_output_tokens (65536) has already been exhausted.'
        )
        return response
      else:
        served = answered
      response.status_code = 200
      response._content = pg.to_json_str(served).encode()  # pylint: disable=protected-access
      return response

    model = lf.LanguageModel.get(
        'gemini-4-argon?project=test-project&max_tokens=65536&max_attempts=2'
        '&retry_interval=0'
    )
    with mock.patch('requests.Session.post', side_effect=post):
      response = model('hi')
    self.assertEqual(response.text, '42')
    self.assertEqual(len(sent), 2)
    self.assertEqual(sent[0]['generationConfig']['maxOutputTokens'], 65_536)
    # 65,536 already burned, plus the 65,533 answer tokens still owed.
    self.assertEqual(
        sent[1]['generationConfig']['maxOutputTokens'], 131_069
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon_resumes_continuation_on_one_project(self):
    hops = [
        {
            'candidates': [{
                'content': {'role': 'model', 'parts': [{'text': ''}]},
                'finishReason': 'CONTINUATION',
                'continuationToken': 'token-1',
            }],
            'usageMetadata': {
                'promptTokenCount': 12,
                'thoughtsTokenCount': 32_765,
                'totalTokenCount': 32_777,
            },
        },
        {
            'candidates': [{
                'content': {'role': 'model', 'parts': [{'text': '42'}]},
                'finishReason': 'STOP',
            }],
            'usageMetadata': {
                'promptTokenCount': 12,
                'candidatesTokenCount': 1,
                'totalTokenCount': 13,
            },
        },
    ]
    calls = []

    def post(url, json, **kwargs):
      del kwargs
      calls.append((url, json))
      response = requests.Response()
      response.status_code = 200
      response._content = pg.to_json_str(hops[len(calls) - 1]).encode()  # pylint: disable=protected-access
      return response

    picks = []

    def alternate(projects):
      picks.append(projects)
      return projects[(len(picks) - 1) % len(projects)]

    with (
        mock.patch.object(vertexai.random, 'choice', side_effect=alternate),
        mock.patch('requests.Session.post', side_effect=post),
    ):
      response = vertexai.VertexAIGemini4Argon(project='p1,p2')('hi')
    self.assertEqual(response.text, '42')
    self.assertEqual(len(calls), 2)
    self.assertIn('/v1beta1/projects/p1/', calls[0][0])
    self.assertEqual(calls[1][0], calls[0][0])
    self.assertEqual(calls[1][1]['continuationToken'], 'token-1')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_37_flash(self):
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    model = vertexai.VertexAIGemini37Flash(location=pg.MISSING_VALUE)
    self.assertEqual(model.resource_id, 'vertexai://gemini-3.7-flash')
    # 3.x models default to 'global' location.
    self.assertIn('global', model.api_endpoint)
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_38_flash(self):
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    model = vertexai.VertexAIGemini38Flash(location=pg.MISSING_VALUE)
    self.assertEqual(model.resource_id, 'vertexai://gemini-3.8-flash')
    # 3.x models default to 'global' location.
    self.assertIn('global', model.api_endpoint)
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_31_flash_lite_preview(self):
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    model = vertexai.VertexAIGemini31FlashLitePreview(location=pg.MISSING_VALUE)
    self.assertEqual(
        model.resource_id, 'vertexai://gemini-3.1-flash-lite-preview'
    )
    # 3.x models default to 'global' location.
    self.assertIn('global', model.api_endpoint)
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_31_flash_lite(self):
    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    model = vertexai.VertexAIGemini31FlashLite(location=pg.MISSING_VALUE)
    self.assertEqual(model.resource_id, 'vertexai://gemini-3.1-flash-lite')
    # 3.x models default to 'global' location.
    self.assertIn('global', model.api_endpoint)
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_multi_project_support(self):
    # Test single project (backward compatibility)
    model = vertexai.VertexAIGemini15Pro(
        project='single-project', location='us-central1'
    )
    self.assertTrue(model._api_initialized)  # Trigger initialization
    self.assertEqual(model._projects, ['single-project'])
    # Single project always returns the same value
    self.assertEqual(model._project, 'single-project')
    self.assertEqual(model._project, 'single-project')  # Consistent
    self.assertIn('single-project', model.api_endpoint)

    # Test list of projects
    model = vertexai.VertexAIGemini15Pro(
        project=['proj-a', 'proj-b', 'proj-c'], location='us-central1'
    )
    self.assertTrue(model._api_initialized)
    self.assertEqual(model._projects, ['proj-a', 'proj-b', 'proj-c'])
    # _project property returns random project from list
    self.assertIn(model._project, ['proj-a', 'proj-b', 'proj-c'])

    # Test comma-separated string (for LanguageModel.get())
    model = vertexai.VertexAIGemini15Pro(
        project='proj-a,proj-b,proj-c', location='us-central1'
    )
    self.assertTrue(model._api_initialized)
    self.assertEqual(model._projects, ['proj-a', 'proj-b', 'proj-c'])

    # Test comma-separated with spaces
    model = vertexai.VertexAIGemini15Pro(
        project='proj-a, proj-b, proj-c', location='us-central1'
    )
    self.assertTrue(model._api_initialized)
    self.assertEqual(model._projects, ['proj-a', 'proj-b', 'proj-c'])

    # Test _project property returns random values across multiple accesses
    model = vertexai.VertexAIGemini15Pro(
        project=['proj-a', 'proj-b'], location='us-central1'
    )
    self.assertTrue(model._api_initialized)
    projects_seen = set()
    for _ in range(20):
      projects_seen.add(model._project)
    # With random selection, we should see both projects
    self.assertEqual(projects_seen, {'proj-a', 'proj-b'})

    # Test api_endpoint uses random project selection
    endpoints = set()
    for _ in range(20):
      endpoints.add(model.api_endpoint)
    # Should see both projects in endpoints
    self.assertEqual(len(endpoints), 2)
    self.assertTrue(any('proj-a' in ep for ep in endpoints))
    self.assertTrue(any('proj-b' in ep for ep in endpoints))

  # Placeholder for sidechannel metadata tests.

  def test_auth_refresh_error(self):
    def _auth_refresh_error(*args, **kwargs):
      del args, kwargs
      raise exceptions.RefreshError('Cannot refresh token')

    with self.assertRaisesRegex(
        lf.concurrent.RetryError,
        'Failed to refresh Google authentication credentials',
    ):
      with (
          mock.patch.object(auth, 'default') as mock_auth,
          mock.patch.object(rest.REST, '_post') as mock_post,
      ):
        mock_auth.return_value = mock.MagicMock(), None
        mock_post.side_effect = _auth_refresh_error
        model = vertexai.VertexAIGemini15Pro(
            project='abc', location='us-central1', max_attempts=1
        )
        model('hi')


class VertexAIAnthropicTest(unittest.TestCase):
  """Tests for VertexAI Anthropic models."""

  def test_basics(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      lm = vertexai.VertexAIClaude35Sonnet_20241022()
      lm('hi')

    model = vertexai.VertexAIClaude35Sonnet_20241022(project='langfun')
    self.assertEqual(model.resource_id, 'vertexai://claude-3-5-sonnet-20241022')
    # Map a Anthropic model back to VertexAI model.
    self.assertTrue(
        vertexai.VertexAIAnthropic(
            'claude-3-5-sonnet-20241022', project='langfun'
        ).model,
        'claude-3-5-sonnet-v2@20241022',
    )
    # NOTE(daiyip): For OSS users, default credentials are not available unless
    # users have already set up their GCP project. Therefore we ignore the
    # exception here.
    try:
      model._initialize()
    except exceptions.DefaultCredentialsError:
      pass

    self.assertEqual(
        model.api_endpoint,
        (
            'https://us-east5-aiplatform.googleapis.com/v1/projects/'
            'langfun/locations/us-east5/publishers/anthropic/'
            'models/claude-3-5-sonnet-v2@20241022:streamRawPredict'
        ),
    )
    self.assertEqual(
        model.headers,
        {
            'Content-Type': 'application/json; charset=utf-8',
        },
    )
    request = model.request(
        lf.UserMessage('hi'),
        lf.LMSamplingOptions(temperature=0.0),
    )
    self.assertEqual(
        request,
        {
            'anthropic_version': 'vertex-2023-10-16',
            'max_tokens': 8192,
            'messages': [{
                'content': [{
                    'cache_control': {'type': 'ephemeral'},
                    'text': 'hi',
                    'type': 'text',
                }],
                'role': 'user',
            }],
            'stream': True,
            'temperature': 0.0,
            'top_k': 40,
        },
    )

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('gemini-2.0-flash'),
        vertexai.VertexAIGemini,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('claude-3-5-sonnet-v2@20241022'),
        vertexai.VertexAIAnthropic,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('llama-3.1-405b-instruct-maas'),
        vertexai.VertexAILlama,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('mistral-large-2411'),
        vertexai.VertexAIMistral,
    )
    # Bare model IDs for Opus 4.6/4.7 resolve as VertexAI.
    self.assertIsInstance(
        lf.LanguageModel.get('claude-opus-4-6'),
        vertexai.VertexAIAnthropic,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('claude-opus-4-7'),
        vertexai.VertexAIAnthropic,
    )

  def test_lm_get_at_latest_resolves_anthropic(self):
    """@latest suffix for Opus 4.6/4.7 resolves as Anthropic direct API."""
    self.assertIsInstance(
        lf.LanguageModel.get('claude-opus-4-6@latest'),
        anthropic.Anthropic,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('claude-opus-4-7@latest'),
        anthropic.Anthropic,
    )

  def test_lm_get_opus_5_5_resolves_vertexai(self):
    """Bare `claude-opus-5-5` resolves to VertexAI; no `@latest` id exists.

    `_register_vertexai_models` only auto-registers rows whose provider is
    'VertexAI', and the `claude-opus-5-5` row declares provider='Anthropic',
    so without the explicit override the bare id would fall through to the
    direct `anthropic.Anthropic` class. That regression shipped once for Opus 5
    and had to be repaired afterwards; pin it so it cannot recur.

    Unlike Opus 4.6/4.7/4.8, Opus 5.5 registers no `@latest` alias: Vertex AI
    publishes only `claude-opus-5-5@default`, so the `@latest` id is
    unverifiable. Opus 5.5 mirrors `claude-opus-5` instead.
    """
    self.assertIsInstance(
        lf.LanguageModel.get('claude-opus-5-5'),
        vertexai.VertexAIClaude55Opus,
    )
    with self.assertRaises(ValueError):
      lf.LanguageModel.get('claude-opus-5-5@latest')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_vertexai_claude55_opus_global_default(self):
    """Verifies that Opus 5.5 defaults to 'global' and uses correct host."""
    model = vertexai.VertexAIClaude55Opus(project='test')
    self.assertEqual(model.location, 'global')
    self.assertTrue(model._api_initialized)
    self.assertIn('https://aiplatform.googleapis.com', model.api_endpoint)
    self.assertNotIn('global-aiplatform', model.api_endpoint)

  def test_thinking_param_true_adaptive_vertexai(self):
    """VertexAI Claude 4.7 + thinking=True -> adaptive thinking."""
    lm = vertexai.VertexAIClaude47Opus(
        project='test', location='global', thinking=True
    )
    args = lm._request_args(
        lf.LMSamplingOptions(max_tokens=1000, temperature=0.5)
    )
    self.assertEqual(
        args['thinking'], {'type': 'adaptive', 'display': 'summarized'}
    )
    self.assertEqual(args['max_tokens'], 1000)
    self.assertNotIn('temperature', args)

  def test_thinking_options_adaptive_vertexai(self):
    lm = vertexai.VertexAIClaude47Opus(project='test', location='global')
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

  def test_thinking_param_false_vertexai(self):
    """VertexAI Claude 4.6 with thinking=False should have no thinking."""
    model = vertexai.VertexAIClaude46Opus(
        project='test-project', location='us-east5', thinking=False
    )
    args = model._request_args(
        lf.LMSamplingOptions(max_thinking_tokens=1024, max_tokens=1000)
    )
    self.assertNotIn('thinking', args)

  def test_thinking_param_none_no_thinking_vertexai(self):
    """VertexAI Claude 4.6 with default thinking=None and no budget has no thinking."""
    model = vertexai.VertexAIClaude46Opus(
        project='test-project', location='us-east5'
    )
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1000))
    self.assertNotIn('thinking', args)

  def test_model_uri_instantiation_vertexai_claude46(self):
    """Test VertexAI model URI instantiation for Claude 4.6."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?project=lf-agent'
        '&location=us-east5&max_attempts=80&timeout=300'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertFalse(model._use_adaptive_thinking)

  def test_model_uri_vertexai_claude46_thinking_true(self):
    """Test VertexAI Claude 4.6 model URI with thinking=true."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?project=lf-agent&location=us-east5&thinking=true'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertTrue(model.thinking)
    self.assertFalse(model._use_adaptive_thinking)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024, max_thinking_tokens=1024)
    )
    self.assertEqual(
        args['thinking'], {'type': 'enabled', 'budget_tokens': 1024}
    )
    self.assertNotIn('temperature', args)

  def test_model_uri_vertexai_claude46_thinking_false(self):
    """Test VertexAI Claude 4.6 model URI with thinking=false."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?project=lf-agent&location=us-east5&thinking=false'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertFalse(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_model_uri_vertexai_claude46_default(self):
    """Test VertexAI Claude 4.6 model URI default (thinking=None)."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?project=lf-agent&location=us-east5'
    )
    self.assertIsNone(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_model_uri_instantiation_vertexai_claude47(self):
    """Test VertexAI model URI instantiation for Claude Opus 4.7."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?project=lf-agent&location=us-east5'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertTrue(model._use_adaptive_thinking)
    self.assertEqual(model.effort, 'high')

  def test_model_uri_instantiation_vertexai_claude47_thinking(self):
    """Test VertexAI Claude 4.7 model URI with thinking=true."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?project=lf-agent&location=us-east5&thinking=true'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertTrue(model.thinking)
    self.assertTrue(model._use_adaptive_thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(
        args['thinking'],
        {'type': 'adaptive', 'display': 'summarized'},
    )
    self.assertEqual(args['output_config'], {'effort': 'high'})

  def test_model_uri_vertexai_claude47_thinking_false(self):
    """Test VertexAI Claude 4.7 with thinking=false."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?project=lf-agent&location=us-east5&thinking=false'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertFalse(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)
    self.assertNotIn('output_config', args)

  def test_model_uri_vertexai_claude47_default(self):
    """Test VertexAI Claude 4.7 model URI default (thinking=None)."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?project=lf-agent&location=us-east5'
    )
    self.assertIsNone(model.thinking)
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)
    self.assertNotIn('output_config', args)

  # --- Claude Opus 4.7: VertexAI direct instantiation tests ---

  def test_vertexai_claude47_opus_direct(self):
    """Test direct VertexAIClaude47Opus instantiation."""
    model = vertexai.VertexAIClaude47Opus(project='test', location='us-east5')
    self.assertTrue(model._use_adaptive_thinking)
    self.assertEqual(model.effort, 'high')

  def test_vertexai_claude47_opus_thinking_true(self):
    """Test VertexAIClaude47Opus with thinking=True."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5', thinking=True
    )
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024, temperature=0.5)
    )
    self.assertEqual(
        args['thinking'],
        {'type': 'adaptive', 'display': 'summarized'},
    )
    self.assertEqual(args['output_config'], {'effort': 'high'})
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)

  def test_vertexai_claude47_opus_thinking_false(self):
    """Test VertexAIClaude47Opus with thinking=False."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5', thinking=False
    )
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_vertexai_claude47_opus_default(self):
    """Test VertexAIClaude47Opus default (no thinking, no budget)."""
    model = vertexai.VertexAIClaude47Opus(project='test', location='us-east5')
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_vertexai_claude47_opus_default_strips_sampling_params(self):
    """Opus 4.7 strips temperature/top_k/top_p even without thinking."""
    model = vertexai.VertexAIClaude47Opus(project='test', location='us-east5')
    args = model._request_args(
        lf.LMSamplingOptions(
            max_tokens=1024, temperature=0.7, top_k=40, top_p=0.9
        )
    )
    self.assertNotIn('thinking', args)
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)
    self.assertNotIn('top_p', args)

  def test_vertexai_claude47_opus_effort_xhigh(self):
    """Test VertexAIClaude47Opus with effort='xhigh'."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5', thinking=True, effort='xhigh'
    )
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(args['output_config'], {'effort': 'xhigh'})

  def test_vertexai_claude47_opus_effort_max(self):
    """Test VertexAIClaude47Opus with effort='max'."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5', thinking=True, effort='max'
    )
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertEqual(args['output_config'], {'effort': 'max'})

  def test_vertexai_claude47_opus_effort_none(self):
    """Test VertexAIClaude47Opus with effort=None (no output_config)."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5', thinking=True, effort=None
    )
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('output_config', args)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_vertexai_claude47_opus_global_default(self):
    """Verifies that Opus 4.7 defaults to 'global' and uses correct host."""
    model = vertexai.VertexAIClaude47Opus(project='test')
    self.assertEqual(model.location, 'global')
    self.assertTrue(model._api_initialized)
    self.assertIn('https://aiplatform.googleapis.com', model.api_endpoint)
    self.assertNotIn('global-aiplatform', model.api_endpoint)

    model2 = lf.LanguageModel.get('claude-opus-4-7', project='test')
    self.assertIsInstance(model2, vertexai.VertexAIClaude47Opus)
    self.assertEqual(model2.location, 'global')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_vertexai_anthropic_explicit_global(self):
    """Verifies explicit 'global' location works and uses correct host."""
    model = vertexai.VertexAIAnthropic(
        'claude-opus-4-7', project='test', location='global'
    )
    self.assertEqual(model.location, 'global')
    self.assertTrue(model._api_initialized)
    self.assertIn('https://aiplatform.googleapis.com', model.api_endpoint)

  def test_vertexai_claude47_opus_reasoning_effort_override(self):
    """reasoning_effort in sampling options overrides model-level effort."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5', thinking=True, effort='high'
    )
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024, reasoning_effort='low')
    )
    self.assertEqual(args['output_config'], {'effort': 'low'})

  def test_vertexai_anthropic_request_passes_through_caching(self):
    model = vertexai.VertexAIClaude3Opus_20240229(project='langfun')
    prompt = lf.UserMessage('hello', system_message=lf.SystemMessage('sys'))
    req = model.request(prompt, lf.LMSamplingOptions())  # NO extras
    # Verify cache lands by default
    self.assertIsInstance(req['system'], list)
    self.assertEqual(req['system'][0]['cache_control'], {'type': 'ephemeral'})
    self.assertEqual(
        req['messages'][-1]['content'][-1]['cache_control'],
        {'type': 'ephemeral'},
    )
    # Verify vertexai fields
    self.assertEqual(req['anthropic_version'], 'vertex-2023-10-16')
    self.assertNotIn('model', req)


def _vertex_response(
    parts: list[dict[str, Any]],
    finish_reason: str,
    *,
    candidates_token_count: int,
    thoughts_token_count: int = 0,
) -> dict[str, Any]:
  """Builds a Vertex `generateContent` response body with one candidate."""
  return {
      'candidates': [{
          'content': {'role': 'model', 'parts': parts},
          'finishReason': finish_reason,
      }],
      'usageMetadata': {
          'promptTokenCount': 12,
          'candidatesTokenCount': candidates_token_count,
          'thoughtsTokenCount': thoughts_token_count,
          'totalTokenCount': (
              12 + candidates_token_count + thoughts_token_count
          ),
      },
  }


class VertexAIEmptyGenerationTest(unittest.TestCase):
  """Tests the guard against deterministically answer-less generations.

  These exercise the real Vertex route (`VertexAIGemini31ProPreview` ->
  `VertexAIGemini` -> `gemini.Gemini.result()`) end to end, with only the HTTP
  POST faked, so they cover both the classification and the retry behavior.
  """

  def _fake_post(self, response_json: dict[str, Any]) -> tuple[list[Any], Any]:
    """Returns (call log, POST side effect) serving `response_json` forever."""
    calls = []

    def _post(*args, **kwargs):
      del args, kwargs
      calls.append(1)
      response = requests.Response()
      response.status_code = 200
      response._content = pg.to_json_str(response_json).encode()  # pylint: disable=protected-access
      return response

    return calls, _post

  def _lm(self) -> vertexai.VertexAIGemini31ProPreview:
    # `retry_interval=0` keeps a (regressed) retry loop fast instead of
    # sleeping through exponential backoff.
    return vertexai.VertexAIGemini31ProPreview(
        project='abc', max_attempts=5, retry_interval=0
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_thought_only_max_tokens_is_not_resent(self):
    """Thinking ate the decode budget: reported once, not re-bought.

    No `continuationToken` is offered here, so there is no in-request rescue
    left to try and a re-send would only reproduce the same spend.
    `max_attempts=5` must buy exactly one request.
    """
    calls, post = self._fake_post(
        _vertex_response(
            [{'text': 'Let me think about this...', 'thought': True}],
            'MAX_TOKENS',
            candidates_token_count=0,
            thoughts_token_count=8192,
        )
    )
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = post
      with self.assertRaises(gemini.GeminiEmptyGenerationError):
        self._lm()('hello')
    self.assertEqual(len(calls), 1)  # `max_attempts=5`, and still one call.

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_thought_only_max_tokens_raises_non_retryable_error(self):
    _, post = self._fake_post(
        _vertex_response(
            [{'text': 'Let me think about this...', 'thought': True}],
            'MAX_TOKENS',
            candidates_token_count=0,
            thoughts_token_count=8192,
        )
    )
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = post
      with self.assertRaises(gemini.GeminiEmptyGenerationError) as ctx:
        self._lm()('hello')
    self.assertNotIsInstance(ctx.exception, lf.RetryableLMError)
    message = str(ctx.exception)
    self.assertIn('MAX_TOKENS', message)
    self.assertIn('gemini-3.1-pro-preview', message)
    self.assertIn('max_tokens', message)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_stop_with_zero_output_tokens_is_not_resent(self):
    """finish_reason=STOP with 0 candidate tokens is deterministic too."""
    calls, post = self._fake_post(
        _vertex_response([], 'STOP', candidates_token_count=0)
    )
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = post
      with self.assertRaises(gemini.GeminiEmptyGenerationError):
        self._lm()('hello')
    self.assertEqual(len(calls), 1)  # `max_attempts=5`, and still one call.

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_normal_generation_unaffected(self):
    """Control: a generation with answer text still returns normally."""
    calls, post = self._fake_post(
        _vertex_response(
            [
                {'text': 'thinking', 'thought': True},
                {'text': 'the answer'},
            ],
            'STOP',
            candidates_token_count=3,
            thoughts_token_count=100,
        )
    )
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = post
      response = self._lm()('hello')
    self.assertEqual(response.text, 'the answer')
    self.assertEqual(len(calls), 1)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  @mock.patch.object(gemini, '_CAP_MISFIRE_RETRY_INTERVAL', 0)
  def test_cap_misfire_budget_does_not_grow_with_max_attempts(self):
    """A model that keeps misfiring into the cap stops after 3 samples."""
    calls, post = self._fake_post(
        _vertex_response(
            [{'text': ''}],
            'MAX_TOKENS',
            candidates_token_count=65_536,
            thoughts_token_count=2_000,
        )
    )
    lm = vertexai.VertexAIGemini4Argon(
        project='abc', max_attempts=100, retry_interval=0
    )
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = post
      with self.assertRaises(lf.concurrent.RetryError):
        lm('hello')
    self.assertEqual(len(calls), 3)


def _serve_json(*bodies: dict[str, Any]) -> Any:
  """Returns a POST side effect serving `bodies` with HTTP 200, in order."""
  served = []

  def post(url, json, **kwargs):
    del url, json, kwargs
    body = bodies[len(served)]
    served.append(body)
    response = requests.Response()
    response.status_code = 200
    response._content = (  # pylint: disable=protected-access
        pg.to_json_str(body).encode()
    )
    return response

  return post


def _today(year: int, month: int, day: int) -> Any:
  """Pins the date that dated pricing resolves on."""
  return mock.patch(
      'langfun.core.language_model._utc_today',
      return_value=datetime.date(year, month, day),
  )


class VertexAIBillingTest(unittest.TestCase):
  """Tests the usage and cost reported for the latest models on Vertex AI.

  The token counts in the fake responses are those Vertex AI reported for live
  calls on 2026-11-04 (on 2026-11-10 for Gemini 4 Carbon). Each prompt had been
  cached by an earlier call, except in the first Claude response, which wrote
  the prompt to the cache, and the first Gemini 4 Carbon response.
  """

  _ARGON_RESPONSE = {
      'candidates': [{
          'content': {'role': 'model', 'parts': [{'text': '25'}]},
          'finishReason': 'STOP',
      }],
      'usageMetadata': {
          'promptTokenCount': 17_387,
          'candidatesTokenCount': 6,
          'totalTokenCount': 18_099,
          'cachedContentTokenCount': 15_333,
          'thoughtsTokenCount': 706,
      },
  }

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_argon_bills_cache_hits_and_thinking(self):
    lm = vertexai.VertexAIGemini4Argon(project='test-project')
    with mock.patch(
        'requests.Session.post',
        side_effect=_serve_json(self._ARGON_RESPONSE, self._ARGON_RESPONSE),
    ):
      with _today(2026, 11, 30):
        promotional = lm('promotional').usage
      with _today(2026, 12, 1):
        standard = lm('standard').usage

    self.assertEqual(promotional.prompt_tokens, 17_387)
    self.assertEqual(promotional.cached_prompt_tokens, 15_333)
    self.assertEqual(promotional.completion_tokens, 6)
    self.assertEqual(
        promotional.completion_tokens_details, {'thinking_tokens': 706}
    )
    # (2,054 uncached * $2 + 15,333 cached * $0.10
    #  + (6 answer + 706 thinking) output * $10) / 1M
    self.assertAlmostEqual(promotional.estimated_cost, 0.0127613)
    # The standard rates from 2026-12-01 are twice the promotional ones.
    self.assertAlmostEqual(standard.estimated_cost, 0.0255226)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_4_carbon_bills_cache_hits_and_thinking(self):
    # Token counts of two live calls on 2026-11-10 with the same 17K-token
    # prompt: the first missed the implicit cache, the second hit it.
    miss = {
        'candidates': [{
            'content': {'role': 'model', 'parts': [{'text': '9920'}]},
            'finishReason': 'STOP',
        }],
        'usageMetadata': {
            'promptTokenCount': 17_343,
            'candidatesTokenCount': 8,
            'totalTokenCount': 17_381,
            'thoughtsTokenCount': 30,
        },
    }
    hit = {
        'candidates': [{
            'content': {'role': 'model', 'parts': [{'text': '9920'}]},
            'finishReason': 'STOP',
        }],
        'usageMetadata': {
            'promptTokenCount': 17_343,
            'candidatesTokenCount': 8,
            'totalTokenCount': 17_351,
            'cachedContentTokenCount': 15_289,
        },
    }
    lm = vertexai.VertexAIGemini4Carbon(project='test-project')
    with mock.patch(
        'requests.Session.post', side_effect=_serve_json(miss, hit, hit)
    ):
      with _today(2026, 11, 30):
        miss_usage = lm('miss').usage
        hit_usage = lm('hit').usage
      with _today(2026, 12, 1):
        standard = lm('standard').usage

    self.assertEqual(miss_usage.prompt_tokens, 17_343)
    self.assertEqual(miss_usage.cached_prompt_tokens, 0)
    self.assertEqual(miss_usage.completion_tokens, 8)
    self.assertEqual(
        miss_usage.completion_tokens_details, {'thinking_tokens': 30}
    )
    # (17,343 input * $2 + (8 answer + 30 thinking) output * $10) / 1M
    self.assertAlmostEqual(miss_usage.estimated_cost, 0.035066)
    self.assertEqual(hit_usage.prompt_tokens, 17_343)
    self.assertEqual(hit_usage.cached_prompt_tokens, 15_289)
    # (2,054 uncached * $2 + 15,289 cached * $0.10 + 8 output * $10) / 1M
    self.assertAlmostEqual(hit_usage.estimated_cost, 0.0057169)
    # The standard rates from 2026-12-01 are twice the promotional ones.
    self.assertAlmostEqual(standard.estimated_cost, 0.0114338)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_gemini_38_flash_bills_cache_hits_and_thinking(self):
    response = {
        'candidates': [{
            'content': {'role': 'model', 'parts': [{'text': '1369'}]},
            'finishReason': 'STOP',
        }],
        'usageMetadata': {
            'promptTokenCount': 17_340,
            'candidatesTokenCount': 4,
            'totalTokenCount': 17_396,
            'cachedContentTokenCount': 12_264,
            'thoughtsTokenCount': 52,
        },
    }
    lm = vertexai.VertexAIGemini38Flash(project='test-project')
    with mock.patch(
        'requests.Session.post', side_effect=_serve_json(response, response)
    ):
      with _today(2026, 12, 31):
        introductory = lm('introductory').usage
      with _today(2027, 1, 1):
        standard = lm('standard').usage

    self.assertEqual(introductory.prompt_tokens, 17_340)
    self.assertEqual(introductory.cached_prompt_tokens, 12_264)
    # (5,076 uncached * $0.75 + 12,264 cached * $0.075
    #  + (4 answer + 52 thinking) output * $3.75) / 1M
    self.assertAlmostEqual(introductory.estimated_cost, 0.0049368)
    # The standard rates from 2027-01-01 are twice the introductory ones.
    self.assertAlmostEqual(standard.estimated_cost, 0.0098736)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_claude_opus_55_bills_cache_writes_and_reads(self):

    def response(cache_creation_tokens, cache_read_tokens, output_tokens):
      return {
          'id': 'msg_1',
          'type': 'message',
          'role': 'assistant',
          'model': 'claude-opus-5-5',
          'content': [{'type': 'text', 'text': '25'}],
          'stop_reason': 'end_turn',
          'usage': {
              'input_tokens': 4,
              'cache_creation_input_tokens': cache_creation_tokens,
              'cache_read_input_tokens': cache_read_tokens,
              'cache_creation': {
                  'ephemeral_5m_input_tokens': cache_creation_tokens,
                  'ephemeral_1h_input_tokens': 0,
              },
              'output_tokens': output_tokens,
          },
      }

    lm = vertexai.VertexAIClaude55Opus(project='test-project')
    with mock.patch(
        'requests.Session.post',
        side_effect=_serve_json(
            response(14_422, 0, 168), response(0, 14_422, 183)
        ),
    ):
      write = lm('first').usage
      read = lm('second').usage

    self.assertEqual(write.prompt_tokens, 14_426)
    self.assertEqual(write.cache_creation_prompt_tokens, 14_422)
    self.assertEqual(write.cached_prompt_tokens, 0)
    # (4 uncached * $4 + 14,422 written * $5 + 168 output * $20) / 1M
    self.assertAlmostEqual(write.estimated_cost, 0.075486)
    self.assertEqual(read.prompt_tokens, 14_426)
    self.assertEqual(read.cache_creation_prompt_tokens, 0)
    self.assertEqual(read.cached_prompt_tokens, 14_422)
    # (4 uncached * $4 + 14,422 read * $0.20 + 183 output * $20) / 1M
    self.assertAlmostEqual(read.estimated_cost, 0.0065604)

  def test_non_global_endpoints_bill_a_premium(self):
    # Gemini 4 Argon on the US multi-region endpoint.
    pricing = vertexai.VertexAIGemini4Argon(
        project='test-project', location='us'
    ).pricing
    promotional = pricing.effective_on(datetime.date(2026, 11, 30))
    self.assertEqual(promotional.cost_per_1m_input_tokens, 2.2)
    self.assertEqual(promotional.cost_per_1m_cached_input_tokens, 0.11)
    self.assertEqual(promotional.cost_per_1m_output_tokens, 11.0)
    standard = pricing.effective_on(datetime.date(2026, 12, 1))
    self.assertEqual(standard.cost_per_1m_input_tokens, 4.4)
    self.assertEqual(standard.cost_per_1m_cached_input_tokens, 0.22)
    self.assertEqual(standard.cost_per_1m_output_tokens, 22.0)

    # Claude Opus 4.6 on its default endpoint, the us-east5 region.
    lm = vertexai.VertexAIClaude46Opus(project='test-project')
    self.assertEqual(lm.location, 'us-east5')
    self.assertEqual(lm.pricing.cost_per_1m_input_tokens, 5.5)
    self.assertEqual(lm.pricing.cost_per_1m_cached_input_tokens, 0.55)
    self.assertEqual(lm.pricing.cost_per_1m_output_tokens, 27.5)
    # A 5-minute cache write costs 1.25x the regional input rate.
    self.assertAlmostEqual(
        lm.estimate_cost(
            lf.LMSamplingUsage(
                prompt_tokens=1_000_000,
                completion_tokens=0,
                total_tokens=1_000_000,
                cache_creation_prompt_tokens=1_000_000,
            )
        ),
        6.875,
    )

  def test_gemini_4_carbon_bills_a_premium_off_the_global_endpoint(self):
    for location in ('us', 'eu'):
      with self.subTest(location=location):
        pricing = vertexai.VertexAIGemini4Carbon(
            project='test-project', location=location
        ).pricing
        promotional = pricing.effective_on(datetime.date(2026, 11, 30))
        self.assertEqual(promotional.cost_per_1m_input_tokens, 2.2)
        self.assertEqual(promotional.cost_per_1m_cached_input_tokens, 0.11)
        self.assertEqual(promotional.cost_per_1m_output_tokens, 11.0)
        standard = pricing.effective_on(datetime.date(2026, 12, 1))
        self.assertEqual(standard.cost_per_1m_input_tokens, 4.4)
        self.assertEqual(standard.cost_per_1m_cached_input_tokens, 0.22)
        self.assertEqual(standard.cost_per_1m_output_tokens, 22.0)
        # The premium keeps the promotion flag, so the list price is scaled.
        self.assertEqual(
            pricing.list_pricing_on(datetime.date(2026, 11, 30)), standard
        )
    # The default endpoint is the global one, which bills the list rates.
    pricing = vertexai.VertexAIGemini4Carbon(project='test-project').pricing
    self.assertEqual(
        pricing.effective_on(
            datetime.date(2026, 11, 30)
        ).cost_per_1m_input_tokens,
        2.0,
    )

  def test_global_endpoints_and_older_models_bill_list_rates(self):
    self.assertEqual(
        vertexai.VertexAIClaude55Opus(
            project='test-project'
        ).pricing.cost_per_1m_input_tokens,
        4.0,
    )
    self.assertEqual(
        vertexai.VertexAIGemini38Flash(project='test-project')
        .pricing.effective_on(datetime.date(2026, 12, 31))
        .cost_per_1m_input_tokens,
        0.75,
    )
    # Vertex AI bills Gemini 2.5 and Claude Opus 4 alike on every endpoint.
    self.assertEqual(
        vertexai.VertexAIGemini25Flash(
            project='test-project', location='us-central1'
        ).pricing.cost_per_1m_input_tokens,
        0.3,
    )
    self.assertEqual(
        vertexai.VertexAIClaude4Opus_20250514(
            project='test-project'
        ).pricing.cost_per_1m_input_tokens,
        15.0,
    )

  def test_gemini_25_pro_keeps_its_rates_on_vertex_ai(self):
    # Unlike the Gemini API, Vertex AI keeps serving Gemini 2.5 Pro.
    pricing = vertexai.VertexAIGemini25Pro(project='test-project').pricing
    in_2027 = pricing.effective_on(datetime.date(2027, 1, 1))
    self.assertEqual(in_2027.cost_per_1m_input_tokens, 1.25)

  def test_non_global_premium_follows_the_resolved_location(self):
    pricing = vertexai.VertexAIGemini38Flash(
        project='test-project', location='us-central1'
    ).pricing.effective_on(datetime.date(2026, 12, 31))
    self.assertEqual(pricing.cost_per_1m_input_tokens, 0.825)
    self.assertEqual(pricing.cost_per_1m_cached_input_tokens, 0.0825)
    self.assertEqual(pricing.cost_per_1m_output_tokens, 4.125)

    # Without a location, the endpoint location comes from the environment.
    with mock.patch.dict(os.environ, {'VERTEXAI_LOCATION': 'eu'}):
      pricing = vertexai.VertexAIGemini4Argon(
          project='test-project', location=None
      ).pricing.effective_on(datetime.date(2026, 12, 1))
    self.assertEqual(pricing.cost_per_1m_input_tokens, 4.4)

  def test_every_priced_model_is_classified_for_the_non_global_premium(self):
    """Fails when a model is added without deciding its non-global rates.

    Vertex AI bills generally available Gemini 3 and later models, and Claude
    Sonnet 4.5, Haiku 4.5, Opus 4.5 and later models, 10% more off the global
    endpoint. A new model belongs either in
    `vertexai._MODELS_WITH_NON_GLOBAL_PRICE_PREMIUM` or in `same_everywhere`.
    """
    same_everywhere = {
        # Gemini previews and models before Gemini 3.
        'gemini-3.1-pro-preview',
        'gemini-3.1-flash-lite-preview',
        'gemini-3-pro-preview',
        'gemini-3-flash-preview',
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-pro-preview-06-05',
        'gemini-2.5-flash-preview-05-20',
        'gemini-2.5-pro-preview-05-06',
        'gemini-2.5-flash-preview-04-17',
        'gemini-2.5-pro-preview-03-25',
        'gemini-2.5-pro-exp-03-25',
        'gemini-2.0-flash',
        'gemini-2.0-flash-001',
        'gemini-2.0-flash-lite-preview-02-05',
        'gemini-1.5-pro',
        'gemini-1.5-pro-001',
        'gemini-1.5-pro-002',
        'gemini-1.5-flash',
        'gemini-1.5-flash-001',
        'gemini-1.5-flash-002',
        # Claude models before Sonnet 4.5.
        'claude-4-opus-20250514',
        'claude-opus-4-20250514',
        'claude-4-sonnet-20250514',
        'claude-sonnet-4@20250514',
        'claude-3-7-sonnet-20250219',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
    }
    priced = {
        m.alias_for or m.model_id
        for m in gemini.SUPPORTED_MODELS
        if m.pricing is not None
        and (
            m.provider == 'VertexAI'
            or (
                isinstance(m.provider, pg.hyper.OneOf)
                and 'VertexAI' in m.provider.candidates
            )
        )
    } | {
        m.alias_for or m.model_id
        for m in anthropic.SUPPORTED_MODELS
        if m.pricing is not None
    }
    premium = vertexai._MODELS_WITH_NON_GLOBAL_PRICE_PREMIUM
    self.assertEqual(priced - premium - same_everywhere, set())
    self.assertEqual(premium - priced, set())
    self.assertEqual(premium & same_everywhere, set())


if __name__ == '__main__':
  unittest.main()
