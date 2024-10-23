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

import os
import unittest
from unittest import mock

from google.cloud.aiplatform import models as aiplatform_models
from vertexai import generative_models
import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import vertexai
import pyglove as pg


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


def mock_generate_content(content, generation_config, **kwargs):
  del kwargs
  c = pg.Dict(generation_config.to_dict())
  return generative_models.GenerationResponse.from_dict({
      'candidates': [
          {
              'index': 0,
              'content': {
                  'role': 'model',
                  'parts': [
                      {
                          'text': (
                              f'This is a response to {content[0]} with '
                              f'temperature={c.temperature}, '
                              f'top_p={c.top_p}, '
                              f'top_k={c.top_k}, '
                              f'max_tokens={c.max_output_tokens}, '
                              f'stop={"".join(c.stop_sequences)}.'
                          )
                      },
                  ],
              },
          },
      ]
  })


def mock_endpoint_predict(instances, **kwargs):
  del kwargs
  assert len(instances) == 1
  return aiplatform_models.Prediction(
      predictions=[
          f"This is a response to {instances[0]['prompt']} with"
          f" temperature={instances[0]['temperature']},"
          f" top_p={instances[0]['top_p']}, top_k={instances[0]['top_k']},"
          f" max_tokens={instances[0]['max_tokens']}."
      ],
      deployed_model_id='',
  )


class VertexAITest(unittest.TestCase):
  """Tests for Vertex model."""

  def test_content_from_message_text_only(self):
    text = 'This is a beautiful day'
    model = vertexai.VertexAIGeminiPro1()
    chunks = model._content_from_message(lf.UserMessage(text))
    self.assertEqual(chunks, [text])

  def test_content_from_message_mm(self):
    message = lf.UserMessage(
        'This is an <<[[image]]>>, what is it?',
        image=lf_modalities.Image.from_bytes(example_image),
    )

    # Non-multimodal model.
    with self.assertRaisesRegex(lf.ModalityError, 'Unsupported modality'):
      vertexai.VertexAIGeminiPro1()._content_from_message(message)

    model = vertexai.VertexAIGeminiPro1Vision()
    chunks = model._content_from_message(message)
    self.maxDiff = None
    self.assertEqual([chunks[0], chunks[2]], ['This is an', ', what is it?'])
    self.assertIsInstance(chunks[1], generative_models.Part)

  def test_generation_response_to_message_text_only(self):
    response = generative_models.GenerationResponse.from_dict({
        'candidates': [
            {
                'index': 0,
                'content': {
                    'role': 'model',
                    'parts': [
                        {
                            'text': 'hello world',
                        },
                    ],
                },
            },
        ],
    })
    model = vertexai.VertexAIGeminiPro1()
    message = model._generation_response_to_message(response)
    self.assertEqual(message, lf.AIMessage('hello world'))

  def test_model_hub(self):
    with mock.patch(
        'vertexai.generative_models.'
        'GenerativeModel.__init__'
    ) as mock_model_init:
      mock_model_init.side_effect = lambda *args, **kwargs: None
      model = vertexai._VERTEXAI_MODEL_HUB.get_generative_model(
          'gemini-1.0-pro'
      )
      self.assertIsNotNone(model)
      self.assertIs(
          vertexai._VERTEXAI_MODEL_HUB.get_generative_model('gemini-1.0-pro'),
          model,
      )

    with mock.patch(
        'vertexai.language_models.'
        'TextGenerationModel.from_pretrained'
    ) as mock_model_init:

      class TextGenerationModel:
        pass

      mock_model_init.side_effect = lambda *args, **kw: TextGenerationModel()
      model = vertexai._VERTEXAI_MODEL_HUB.get_text_generation_model(
          'text-bison'
      )
      self.assertIsNotNone(model)
      self.assertIs(
          vertexai._VERTEXAI_MODEL_HUB.get_text_generation_model('text-bison'),
          model,
      )

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
    config = model._generation_config(
        lf.UserMessage('hi', json_schema=json_schema),
        lf.LMSamplingOptions(
            temperature=2.0,
            top_p=1.0,
            top_k=20,
            max_tokens=1024,
            stop=['\n'],
        ),
    )
    actual = config.to_dict()
    # There is a discrepancy between the `property_ordering` in the
    # Google-internal version and the open-source version.
    actual['response_schema'].pop('property_ordering', None)
    self.assertEqual(
        actual,
        dict(
            temperature=2.0,
            top_p=1.0,
            top_k=20.0,
            max_output_tokens=1024,
            stop_sequences=['\n'],
            response_mime_type='application/json',
            response_schema={
                'type_': 'OBJECT',
                'properties': {
                    'name': {'type_': 'STRING'}
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

  def test_call_generative_model(self):
    with mock.patch(
        'vertexai.generative_models.'
        'GenerativeModel.__init__'
    ) as mock_model_init:
      mock_model_init.side_effect = lambda *args, **kwargs: None

      with mock.patch(
          'vertexai.generative_models.'
          'GenerativeModel.generate_content'
      ) as mock_generate:
        mock_generate.side_effect = mock_generate_content

        lm = vertexai.VertexAIGeminiPro1(project='abc', location='us-central1')
        self.assertEqual(
            lm(
                'hello',
                temperature=2.0,
                top_p=1.0,
                top_k=20,
                max_tokens=1024,
                stop='\n',
            ).text,
            (
                'This is a response to hello with temperature=2.0, '
                'top_p=1.0, top_k=20.0, max_tokens=1024, stop=\n.'
            ),
        )

  def test_call_text_generation_model(self):
    with mock.patch(
        'vertexai.language_models.'
        'TextGenerationModel.from_pretrained'
    ) as mock_model_init:

      class TextGenerationModel:

        def predict(self, prompt, **kwargs):
          c = pg.Dict(kwargs)
          return pg.Dict(
              text=(
                  f'This is a response to {prompt} with '
                  f'temperature={c.temperature}, '
                  f'top_p={c.top_p}, '
                  f'top_k={c.top_k}, '
                  f'max_tokens={c.max_output_tokens}, '
                  f'stop={"".join(c.stop_sequences)}.'
              )
          )

      mock_model_init.side_effect = lambda *args, **kw: TextGenerationModel()
      lm = vertexai.VertexAIPalm2(project='abc', location='us-central1')
      self.assertEqual(
          lm(
              'hello',
              temperature=2.0,
              top_p=1.0,
              top_k=20,
              max_tokens=1024,
              stop='\n',
          ).text,
          (
              'This is a response to hello with temperature=2.0, '
              'top_p=1.0, top_k=20, max_tokens=1024, stop=\n.'
          ),
      )

  def test_call_endpoint_model(self):
    with mock.patch(
        'google.cloud.aiplatform.models.Endpoint.__init__'
    ) as mock_model_init:
      mock_model_init.side_effect = lambda *args, **kwargs: None
      with mock.patch(
          'google.cloud.aiplatform.models.Endpoint.predict'
      ) as mock_model_predict:

        mock_model_predict.side_effect = mock_endpoint_predict
        lm = vertexai.VertexAI(
            'custom',
            endpoint_name='123',
            project='abc',
            location='us-central1',
        )
        self.assertEqual(
            lm(
                'hello',
                temperature=2.0,
                top_p=1.0,
                top_k=20,
                max_tokens=50,
            ),
            'This is a response to hello with temperature=2.0, top_p=1.0,'
            ' top_k=20, max_tokens=50.',
        )


if __name__ == '__main__':
  unittest.main()
