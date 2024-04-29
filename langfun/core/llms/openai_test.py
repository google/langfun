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
"""Tests for OpenAI models."""

import unittest
from unittest import mock

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import openai
import pyglove as pg


def mock_completion_query(prompt, *, n=1, **kwargs):
  del kwargs
  choices = []
  for i, _ in enumerate(prompt):
    for k in range(n):
      choices.append(pg.Dict(
          index=i,
          text=f'Sample {k} for prompt {i}.',
          logprobs=k / 10,
      ))
  return pg.Dict(
      choices=choices,
      usage=lf.LMSamplingUsage(
          prompt_tokens=100,
          completion_tokens=100,
          total_tokens=200,
      ),
  )


def mock_chat_completion_query(messages, *, n=1, **kwargs):
  del messages, kwargs
  choices = []
  for k in range(n):
    choices.append(pg.Dict(
        message=pg.Dict(
            content=f'Sample {k} for message.'
        ),
        logprobs=None,
    ))
  return pg.Dict(
      choices=choices,
      usage=lf.LMSamplingUsage(
          prompt_tokens=100,
          completion_tokens=100,
          total_tokens=200,
      ),
  )


def mock_chat_completion_query_vision(messages, *, n=1, **kwargs):
  del kwargs
  choices = []
  urls = [
      c['image_url']['url']
      for c in messages[0]['content'] if c['type'] == 'image_url'
  ]
  for k in range(n):
    choices.append(pg.Dict(
        message=pg.Dict(
            content=f'Sample {k} for message: {"".join(urls)}'
        ),
        logprobs=None,
    ))
  return pg.Dict(
      choices=choices,
      usage=lf.LMSamplingUsage(
          prompt_tokens=100,
          completion_tokens=100,
          total_tokens=200,
      ),
  )


class OpenAITest(unittest.TestCase):
  """Tests for OpenAI language model."""

  def test_model_id(self):
    self.assertEqual(
        openai.Gpt35(api_key='test_key').model_id, 'OpenAI(text-davinci-003)')

  def test_resource_id(self):
    self.assertEqual(
        openai.Gpt35(api_key='test_key').resource_id, 'OpenAI(text-davinci-003)'
    )

  def test_max_concurrency(self):
    self.assertGreater(openai.Gpt35(api_key='test_key').max_concurrency, 0)

  def test_get_request_args(self):
    self.assertEqual(
        openai.Gpt35(api_key='test_key', timeout=90.0)._get_request_args(
            lf.LMSamplingOptions(
                temperature=2.0,
                n=2,
                max_tokens=4096,
                top_p=1.0)),
        dict(
            engine='text-davinci-003',
            logprobs=False,
            top_logprobs=None,
            n=2,
            temperature=2.0,
            max_tokens=4096,
            stream=False,
            timeout=90.0,
            top_p=1.0,
        )
    )
    self.assertEqual(
        openai.Gpt4(api_key='test_key')._get_request_args(
            lf.LMSamplingOptions(temperature=1.0, stop=['\n'], n=1)
        ),
        dict(
            model='gpt-4',
            logprobs=False,
            top_logprobs=None,
            n=1,
            temperature=1.0,
            stream=False,
            timeout=120.0,
            stop=['\n'],
        ),
    )

  def test_call_completion(self):
    with mock.patch('openai.Completion.create') as mock_completion:
      mock_completion.side_effect = mock_completion_query
      lm = openai.OpenAI(api_key='test_key', model='text-davinci-003')
      self.assertEqual(
          lm('hello', sampling_options=lf.LMSamplingOptions(n=2)),
          'Sample 0 for prompt 0.',
      )

  def test_call_chat_completion(self):
    with mock.patch('openai.ChatCompletion.create') as mock_chat_completion:
      mock_chat_completion.side_effect = mock_chat_completion_query
      lm = openai.OpenAI(api_key='test_key', model='gpt-4')
      self.assertEqual(
          lm('hello', sampling_options=lf.LMSamplingOptions(n=2)),
          'Sample 0 for message.',
      )

  def test_call_chat_completion_vision(self):
    with mock.patch('openai.ChatCompletion.create') as mock_chat_completion:
      mock_chat_completion.side_effect = mock_chat_completion_query_vision
      lm_1 = openai.Gpt4Turbo(api_key='test_key')
      lm_2 = openai.Gpt4VisionPreview(api_key='test_key')
      for lm in (lm_1, lm_2):
        self.assertEqual(
            lm(
                lf.UserMessage(
                    'hello {{image}}',
                    image=lf_modalities.Image.from_uri('https://fake/image')
                ),
                sampling_options=lf.LMSamplingOptions(n=2)
            ),
            'Sample 0 for message: https://fake/image',
        )

  def test_sample_completion(self):
    with mock.patch('openai.Completion.create') as mock_completion:
      mock_completion.side_effect = mock_completion_query
      lm = openai.OpenAI(api_key='test_key', model='text-davinci-003')
      results = lm.sample(
          ['hello', 'bye'], sampling_options=lf.LMSamplingOptions(n=3)
      )

    self.assertEqual(len(results), 2)
    self.assertEqual(
        results[0],
        lf.LMSamplingResult(
            [
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 0 for prompt 0.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 1 for prompt 0.',
                        score=0.1,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.1,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 2 for prompt 0.',
                        score=0.2,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.2,
                    logprobs=None,
                ),
            ],
            usage=lf.LMSamplingUsage(
                prompt_tokens=100, completion_tokens=100, total_tokens=200
            ),
        ),
    )
    self.assertEqual(
        results[1],
        lf.LMSamplingResult(
            [
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 0 for prompt 1.',
                        score=0.0,
                        logprobs=None,
                        usage=None,
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 1 for prompt 1.',
                        score=0.1,
                        logprobs=None,
                        usage=None,
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.1,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 2 for prompt 1.',
                        score=0.2,
                        logprobs=None,
                        usage=None,
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.2,
                    logprobs=None,
                ),
            ],
        ),
    )

  def test_sample_chat_completion(self):
    with mock.patch('openai.ChatCompletion.create') as mock_chat_completion:
      mock_chat_completion.side_effect = mock_chat_completion_query
      lm = openai.OpenAI(api_key='test_key', model='gpt-4')
      results = lm.sample(
          ['hello', 'bye'], sampling_options=lf.LMSamplingOptions(n=3)
      )

    self.assertEqual(len(results), 2)
    self.assertEqual(
        results[0],
        lf.LMSamplingResult(
            [
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 0 for message.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 1 for message.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 2 for message.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
            ],
            usage=lf.LMSamplingUsage(
                prompt_tokens=100, completion_tokens=100, total_tokens=200
            ),
        ),
    )
    self.assertEqual(
        results[1],
        lf.LMSamplingResult(
            [
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 0 for message.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 1 for message.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 2 for message.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
            ],
            usage=lf.LMSamplingUsage(
                prompt_tokens=100, completion_tokens=100, total_tokens=200
            ),
        ),
    )

  def test_sample_with_contextual_options(self):
    with mock.patch('openai.Completion.create') as mock_completion:
      mock_completion.side_effect = mock_completion_query
      lm = openai.OpenAI(api_key='test_key', model='text-davinci-003')
      with lf.use_settings(sampling_options=lf.LMSamplingOptions(n=2)):
        results = lm.sample(['hello'])

    self.assertEqual(len(results), 1)
    self.assertEqual(
        results[0],
        lf.LMSamplingResult(
            [
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 0 for prompt 0.',
                        score=0.0,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=50,
                            completion_tokens=50,
                            total_tokens=100,
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 1 for prompt 0.',
                        score=0.1,
                        logprobs=None,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=50,
                            completion_tokens=50,
                            total_tokens=100,
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.1,
                    logprobs=None,
                ),
            ],
            usage=lf.LMSamplingUsage(
                prompt_tokens=100, completion_tokens=100, total_tokens=200
            ),
        ),
    )


if __name__ == '__main__':
  unittest.main()
