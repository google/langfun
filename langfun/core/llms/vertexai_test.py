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
"""Tests for Gemini models."""

import base64
import os
from typing import Any
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import vertexai
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
  content = json['contents'][0]['parts'][0]['text']
  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'candidates': [
          {
              'content': {
                  'role': 'model',
                  'parts': [
                      {
                          'text': (
                              f'This is a response to {content} with '
                              f'temperature={c.temperature}, '
                              f'top_p={c.topP}, '
                              f'top_k={c.topK}, '
                              f'max_tokens={c.maxOutputTokens}, '
                              f'stop={"".join(c.stopSequences)}.'
                          )
                      },
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


class VertexAITest(unittest.TestCase):
  """Tests for Vertex model with REST API."""

  def test_content_from_message_text_only(self):
    text = 'This is a beautiful day'
    model = vertexai.VertexAIGeminiPro1_5_002()
    chunks = model._content_from_message(lf.UserMessage(text))
    self.assertEqual(chunks, {'role': 'user', 'parts': [{'text': text}]})

  def test_content_from_message_mm(self):
    image = lf_modalities.Image.from_bytes(example_image)
    message = lf.UserMessage(
        'This is an <<[[image]]>>, what is it?', image=image
    )

    # Non-multimodal model.
    with self.assertRaisesRegex(lf.ModalityError, 'Unsupported modality'):
      vertexai.VertexAIGeminiPro1()._content_from_message(message)

    model = vertexai.VertexAIGeminiPro1Vision()
    content = model._content_from_message(message)
    self.assertEqual(
        content,
        {
            'role': 'user',
            'parts': [
                {'text': 'This is an'},
                {
                    'inlineData': {
                        'data': base64.b64encode(example_image).decode(),
                        'mimeType': 'image/png',
                    }
                },
                {'text': ', what is it?'},
            ],
        },
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_and_location_check(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      _ = vertexai.VertexAIGeminiPro1()._api_initialized

    with self.assertRaisesRegex(ValueError, 'Please specify `location`'):
      _ = vertexai.VertexAIGeminiPro1(project='abc')._api_initialized

    self.assertTrue(
        vertexai.VertexAIGeminiPro1(
            project='abc', location='us-central1'
        )._api_initialized
    )

    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    self.assertTrue(vertexai.VertexAIGeminiPro1()._api_initialized)
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']

  def test_generation_config(self):
    model = vertexai.VertexAIGeminiPro1()
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
    with self.assertRaisesRegex(
        ValueError, '`json_schema` must be a dict, got'
    ):
      model._generation_config(
          lf.UserMessage('hi', json_schema='not a dict'),
          lf.LMSamplingOptions(),
      )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_call_model(self):
    with mock.patch('requests.Session.post') as mock_generate:
      mock_generate.side_effect = mock_requests_post

      lm = vertexai.VertexAIGeminiPro1_5_002(
          project='abc', location='us-central1'
      )
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
      self.assertEqual(r.metadata.usage.prompt_tokens, 3)
      self.assertEqual(r.metadata.usage.completion_tokens, 4)


if __name__ == '__main__':
  unittest.main()
