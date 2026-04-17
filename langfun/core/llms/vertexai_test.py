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

import os
import unittest
from unittest import mock

from google import auth
from google.auth import exceptions
import langfun.core as lf
from langfun.core.llms import rest
from langfun.core.llms import vertexai
import pyglove as pg


class VertexAITest(unittest.TestCase):
  """Tests for Vertex model with REST API."""

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_and_location_check(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      _ = vertexai.VertexAIGemini15Pro()._api_initialized

    with self.assertRaisesRegex(ValueError, 'Please specify `location`'):
      _ = vertexai.VertexAIGemini15Pro(
          project='abc', location=None)._api_initialized

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
        'Failed to refresh Google authentication credentials'
    ):
      with (
          mock.patch.object(auth, 'default') as mock_auth,
          mock.patch.object(rest.REST, '_sample_single') as mock_sample_single
      ):
        mock_auth.return_value = mock.MagicMock(), None
        mock_sample_single.side_effect = _auth_refresh_error
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
        )
    )
    self.assertEqual(
        model.headers,
        {
            'Content-Type': 'application/json; charset=utf-8',
        },
    )
    request = model.request(
        lf.UserMessage('hi'), lf.LMSamplingOptions(temperature=0.0),
    )
    self.assertEqual(
        request,
        {
            'anthropic_version': 'vertex-2023-10-16',
            'max_tokens': 8192,
            'messages': [
                {'content': [{'text': 'hi', 'type': 'text'}], 'role': 'user'}
            ],
            'stream': False,
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
    from langfun.core.llms import anthropic  # pylint: disable=g-import-not-at-top
    self.assertIsInstance(
        lf.LanguageModel.get('claude-opus-4-6@latest'),
        anthropic.Anthropic,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('claude-opus-4-7@latest'),
        anthropic.Anthropic,
    )

  def test_thinking_param_true_adaptive_vertexai(self):
    """VertexAI Claude 4.6 + thinking=True -> adaptive thinking."""
    lm = vertexai.VertexAIClaude46Opus(
        project='test', location='us-east5', thinking=True
    )
    args = lm._request_args(lf.LMSamplingOptions(
        max_tokens=1000, temperature=0.5
    ))
    self.assertEqual(args['thinking'], {'type': 'adaptive'})
    self.assertEqual(args['max_tokens'], 1000)
    self.assertNotIn('temperature', args)

  def test_thinking_options_adaptive_vertexai(self):
    lm = vertexai.VertexAIClaude46Opus(project='test', location='us-east5')
    args = lm._request_args(lf.LMSamplingOptions(
        max_thinking_tokens=1024, max_tokens=1000, temperature=0.5
    ))
    self.assertEqual(args['thinking'], {'type': 'adaptive'})
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
    self.assertTrue(model._use_adaptive_thinking)

  def test_model_uri_vertexai_claude46_thinking_true(self):
    """Test VertexAI Claude 4.6 model URI with thinking=true."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?project=lf-agent'
        '&location=us-east5&thinking=true'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertTrue(model.thinking)
    self.assertTrue(model._use_adaptive_thinking)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
    self.assertEqual(args['thinking'], {'type': 'adaptive'})
    self.assertNotIn('temperature', args)

  def test_model_uri_vertexai_claude46_thinking_false(self):
    """Test VertexAI Claude 4.6 model URI with thinking=false."""
    model = lf.LanguageModel.get(
        'claude-opus-4-6?project=lf-agent'
        '&location=us-east5&thinking=false'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertFalse(model.thinking)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
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
        'claude-opus-4-7?project=lf-agent'
        '&location=us-east5'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertTrue(model._use_adaptive_thinking)
    self.assertEqual(model.effort, 'high')

  def test_model_uri_instantiation_vertexai_claude47_thinking(self):
    """Test VertexAI Claude 4.7 model URI with thinking=true."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?project=lf-agent'
        '&location=us-east5&thinking=true'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertTrue(model.thinking)
    self.assertTrue(model._use_adaptive_thinking)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
    self.assertEqual(
        args['thinking'],
        {'type': 'adaptive', 'display': 'summarized'},
    )
    self.assertEqual(args['output_config'], {'effort': 'high'})

  def test_model_uri_vertexai_claude47_thinking_false(self):
    """Test VertexAI Claude 4.7 with thinking=false."""
    model = lf.LanguageModel.get(
        'claude-opus-4-7?project=lf-agent'
        '&location=us-east5&thinking=false'
    )
    self.assertIsInstance(model, vertexai.VertexAIAnthropic)
    self.assertFalse(model.thinking)
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
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
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5'
    )
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
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
    self.assertNotIn('thinking', args)

  def test_vertexai_claude47_opus_default(self):
    """Test VertexAIClaude47Opus default (no thinking, no budget)."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5'
    )
    args = model._request_args(lf.LMSamplingOptions(max_tokens=1024))
    self.assertNotIn('thinking', args)

  def test_vertexai_claude47_opus_default_strips_sampling_params(self):
    """Opus 4.7 strips temperature/top_k/top_p even without thinking."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5'
    )
    args = model._request_args(lf.LMSamplingOptions(
        max_tokens=1024, temperature=0.7, top_k=40, top_p=0.9
    ))
    self.assertNotIn('thinking', args)
    self.assertNotIn('temperature', args)
    self.assertNotIn('top_k', args)
    self.assertNotIn('top_p', args)

  def test_vertexai_claude47_opus_effort_xhigh(self):
    """Test VertexAIClaude47Opus with effort='xhigh'."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5',
        thinking=True, effort='xhigh'
    )
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
    self.assertEqual(args['output_config'], {'effort': 'xhigh'})

  def test_vertexai_claude47_opus_effort_max(self):
    """Test VertexAIClaude47Opus with effort='max'."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5',
        thinking=True, effort='max'
    )
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
    self.assertEqual(args['output_config'], {'effort': 'max'})

  def test_vertexai_claude47_opus_effort_none(self):
    """Test VertexAIClaude47Opus with effort=None (no output_config)."""
    model = vertexai.VertexAIClaude47Opus(
        project='test', location='us-east5',
        thinking=True, effort=None
    )
    args = model._request_args(
        lf.LMSamplingOptions(max_tokens=1024)
    )
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
        project='test', location='us-east5',
        thinking=True, effort='high'
    )
    args = model._request_args(lf.LMSamplingOptions(
        max_tokens=1024, reasoning_effort='low'
    ))
    self.assertEqual(args['output_config'], {'effort': 'low'})


if __name__ == '__main__':
  unittest.main()
