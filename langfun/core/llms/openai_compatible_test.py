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

from typing import Any
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import openai_compatible
import pyglove as pg
import requests


def mock_chat_completion_request(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  messages = json['messages']
  if len(messages) > 1:
    system_message = f' system={messages[0]["content"]}'
  else:
    system_message = ''

  if 'response_format' in json:
    response_format = f' format={json["response_format"]["type"]}'
  else:
    response_format = ''

  choices = []
  for k in range(json['n']):
    if json.get('logprobs'):
      logprobs = dict(
          content=[
              dict(
                  token='chosen_token',
                  logprob=0.5,
                  top_logprobs=[
                      dict(
                          token=f'alternative_token_{i + 1}',
                          logprob=0.1
                      ) for i in range(3)
                  ]
              )
          ]
      )
    else:
      logprobs = None

    choices.append(dict(
        message=dict(
            content=(
                f'Sample {k} for message.{system_message}{response_format}'
            )
        ),
        logprobs=logprobs,
    ))
  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str(
      dict(
          choices=choices,
          usage=lf.LMSamplingUsage(
              prompt_tokens=100,
              completion_tokens=100,
              total_tokens=200,
          ),
      )
  ).encode()
  return response


def mock_chat_completion_request_vision(
    url: str, json: dict[str, Any], **kwargs
):
  del url, kwargs
  choices = []
  urls = [
      c['image_url']['url']
      for c in json['messages'][0]['content'] if c['type'] == 'image_url'
  ]
  for k in range(json['n']):
    choices.append(pg.Dict(
        message=pg.Dict(
            content=f'Sample {k} for message: {"".join(urls)}'
        ),
        logprobs=None,
    ))
  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str(
      dict(
          choices=choices,
          usage=lf.LMSamplingUsage(
              prompt_tokens=100,
              completion_tokens=100,
              total_tokens=200,
          ),
      )
  ).encode()
  return response


class OpenAIComptibleTest(unittest.TestCase):
  """Tests for OpenAI compatible language model."""

  def test_request_args(self):
    self.assertEqual(
        openai_compatible.OpenAICompatible(
            api_endpoint='https://test-server',
            model='test-model'
        )._request_args(
            lf.LMSamplingOptions(
                temperature=1.0, stop=['\n'], n=1, random_seed=123
            )
        ),
        dict(
            model='test-model',
            top_logprobs=None,
            n=1,
            temperature=1.0,
            stop=['\n'],
            seed=123,
        ),
    )

  def test_call_chat_completion(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_chat_completion_request
      lm = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server', model='test-model',
      )
      self.assertEqual(
          lm('hello', sampling_options=lf.LMSamplingOptions(n=2)),
          'Sample 0 for message.',
      )

  def test_call_chat_completion_with_logprobs(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_chat_completion_request
      lm = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server', model='test-model',
      )
      results = lm.sample(['hello'], logprobs=True)
      for result in results:
        result.rebind(
            {path: 0 for path in pg.query(result, '.*total_call_interval')},
            skip_notification=True,
        )
      self.assertEqual(len(results), 1)
      self.assertEqual(
          results[0],
          lf.LMSamplingResult(
              [
                  lf.LMSample(
                      response=lf.AIMessage(
                          text='Sample 0 for message.',
                          metadata={
                              'score': 0.0,
                              'logprobs': [(
                                  'chosen_token',
                                  0.5,
                                  [
                                      ('alternative_token_1', 0.1),
                                      ('alternative_token_2', 0.1),
                                      ('alternative_token_3', 0.1),
                                  ],
                              )],
                              'is_cached': False,
                              'usage': lf.LMSamplingUsage(
                                  prompt_tokens=100,
                                  completion_tokens=100,
                                  total_tokens=200,
                                  estimated_cost=None,
                              ),
                          },
                          tags=['lm-response'],
                      ),
                      logprobs=[(
                          'chosen_token',
                          0.5,
                          [
                              ('alternative_token_1', 0.1),
                              ('alternative_token_2', 0.1),
                              ('alternative_token_3', 0.1),
                          ],
                      )],
                  )
              ],
              usage=lf.LMSamplingUsage(
                  prompt_tokens=100,
                  completion_tokens=100,
                  total_tokens=200,
                  estimated_cost=None,
              ),
          ),
      )

  def test_call_chat_completion_vision(self):

    class FakeImage(lf_modalities.Image):

      @property
      def mime_type(self) -> str:
        return 'image/png'

    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_chat_completion_request_vision
      lm_1 = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server',
          model='test-model1',
      )
      lm_2 = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server',
          model='test-model2',
      )
      for lm in (lm_1, lm_2):
        self.assertEqual(
            lm(
                lf.UserMessage(
                    'hello <<[[image]]>>',
                    image=FakeImage.from_uri('https://fake/image')
                ),
                sampling_options=lf.LMSamplingOptions(n=2)
            ),
            'Sample 0 for message: https://fake/image',
        )

    class TextOnlyModel(openai_compatible.OpenAICompatible):

      class ModelInfo(lf.ModelInfo):
        input_modalities: list[str] = lf.ModelInfo.TEXT_INPUT_ONLY

      @property
      def model_info(self) -> lf.ModelInfo:
        return TextOnlyModel.ModelInfo('text-only-model')

    lm_3 = TextOnlyModel(
        api_endpoint='https://test-server', model='test-model3'
    )
    with self.assertRaisesRegex(ValueError, 'Unsupported modality'):
      lm_3(
          lf.UserMessage(
              'hello <<[[image]]>>',
              image=FakeImage.from_uri('https://fake/image')
          ),
      )

  def test_sample_chat_completion(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_chat_completion_request
      lm = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server', model='test-model'
      )
      results = lm.sample(
          ['hello', 'bye'], sampling_options=lf.LMSamplingOptions(n=3)
      )
      for result in results:
        result.rebind(
            {path: 0 for path in pg.query(result, '.*total_call_interval')},
            skip_notification=True,
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
                        is_cached=False,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66,
                            estimated_cost=None,
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
                        is_cached=False,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66,
                            estimated_cost=None,
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
                        is_cached=False,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66,
                            estimated_cost=None,
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
            ],
            usage=lf.LMSamplingUsage(
                prompt_tokens=100, completion_tokens=100, total_tokens=200,
                estimated_cost=None,
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
                        is_cached=False,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66,
                            estimated_cost=None,
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
                        is_cached=False,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66,
                            estimated_cost=None,
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
                        is_cached=False,
                        usage=lf.LMSamplingUsage(
                            prompt_tokens=33,
                            completion_tokens=33,
                            total_tokens=66,
                            estimated_cost=None,
                        ),
                        tags=[lf.Message.TAG_LM_RESPONSE],
                    ),
                    score=0.0,
                    logprobs=None,
                ),
            ],
            usage=lf.LMSamplingUsage(
                prompt_tokens=100, completion_tokens=100, total_tokens=200,
                estimated_cost=None,
            ),
        ),
    )

  def test_sample_with_contextual_options(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_chat_completion_request
      lm = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server', model='test-model'
      )
      with lf.use_settings(sampling_options=lf.LMSamplingOptions(n=2)):
        results = lm.sample(['hello'])
      for result in results:
        result.rebind(
            {path: 0 for path in pg.query(result, '.*total_call_interval')},
            skip_notification=True,
        )

    self.assertEqual(len(results), 1)
    self.assertEqual(
        results[0],
        lf.LMSamplingResult(
            [
                lf.LMSample(
                    lf.AIMessage(
                        'Sample 0 for message.',
                        score=0.0,
                        logprobs=None,
                        is_cached=False,
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
                        'Sample 1 for message.',
                        score=0.0,
                        logprobs=None,
                        is_cached=False,
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
            ],
            usage=lf.LMSamplingUsage(
                prompt_tokens=100, completion_tokens=100, total_tokens=200
            ),
        )
    )

  def test_call_with_system_message(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_chat_completion_request
      lm = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server', model='test-model'
      )
      self.assertEqual(
          lm(
              lf.UserMessage(
                  'hello',
                  system_message='hi',
              ),
              sampling_options=lf.LMSamplingOptions(n=2)
          ),
          '''Sample 0 for message. system=[{'type': 'text', 'text': 'hi'}]''',
      )

  def test_call_with_json_schema(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_chat_completion_request
      lm = openai_compatible.OpenAICompatible(
          api_endpoint='https://test-server', model='test-model'
      )
      self.assertEqual(
          lm(
              lf.UserMessage(
                  'hello',
                  json_schema={
                      'type': 'object',
                      'properties': {
                          'name': {'type': 'string'},
                      },
                      'required': ['name'],
                      'title': 'Person',
                  }
              ),
              sampling_options=lf.LMSamplingOptions(n=2)
          ),
          'Sample 0 for message. format=json_schema',
      )

    # Test bad json schema.
    with self.assertRaisesRegex(ValueError, '`json_schema` must be a dict'):
      lm(lf.UserMessage('hello', json_schema='foo'))

    with self.assertRaisesRegex(
        ValueError, 'The root of `json_schema` must have a `title` field'
    ):
      lm(lf.UserMessage('hello', json_schema={}))


if __name__ == '__main__':
  unittest.main()
