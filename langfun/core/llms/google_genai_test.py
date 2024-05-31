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

from google import generativeai as genai
import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import google_genai
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


def mock_get_model(model_name, *args, **kwargs):
  del args, kwargs
  if 'gemini' in model_name:
    method = 'generateContent'
  elif 'chat' in model_name:
    method = 'generateMessage'
  else:
    method = 'generateText'
  return pg.Dict(supported_generation_methods=[method])


def mock_generate_text(*, model, prompt, **kwargs):
  return pg.Dict(
      candidates=[pg.Dict(output=f'{prompt} to {model} with {kwargs}')]
  )


def mock_chat(*, model, messages, **kwargs):
  return pg.Dict(
      candidates=[pg.Dict(content=f'{messages} to {model} with {kwargs}')]
  )


def mock_generate_content(content, generation_config, **kwargs):
  del kwargs
  c = generation_config
  return genai.types.GenerateContentResponse(
      done=True,
      iterator=None,
      chunks=[],
      result=pg.Dict(
          prompt_feedback=pg.Dict(block_reason=None),
          candidates=[
              pg.Dict(
                  content=pg.Dict(
                      parts=[
                          pg.Dict(
                              text=(
                                  f'This is a response to {content[0]} with '
                                  f'n={c.candidate_count}, '
                                  f'temperature={c.temperature}, '
                                  f'top_p={c.top_p}, '
                                  f'top_k={c.top_k}, '
                                  f'max_tokens={c.max_output_tokens}, '
                                  f'stop={c.stop_sequences}.'
                              )
                          )
                      ]
                  ),
              ),
          ],
      ),
  )


class GenAITest(unittest.TestCase):
  """Tests for Google GenAI model."""

  def test_content_from_message_text_only(self):
    text = 'This is a beautiful day'
    model = google_genai.GeminiPro()
    chunks = model._content_from_message(lf.UserMessage(text))
    self.assertEqual(chunks, [text])

  def test_content_from_message_mm(self):
    message = lf.UserMessage(
        'This is an <<[[image]]>>, what is it?',
        image=lf_modalities.Image.from_bytes(example_image),
    )

    # Non-multimodal model.
    with self.assertRaisesRegex(lf.ModalityError, 'Unsupported modality'):
      google_genai.GeminiPro()._content_from_message(message)

    model = google_genai.GeminiProVision()
    chunks = model._content_from_message(message)
    self.maxDiff = None
    self.assertEqual(
        chunks,
        [
            'This is an',
            genai.types.BlobDict(mime_type='image/png', data=example_image),
            ', what is it?',
        ],
    )

  def test_response_to_result_text_only(self):
    response = genai.types.GenerateContentResponse(
        done=True,
        iterator=None,
        chunks=[],
        result=pg.Dict(
            prompt_feedback=pg.Dict(block_reason=None),
            candidates=[
                pg.Dict(
                    content=pg.Dict(
                        parts=[pg.Dict(text='This is response 1.')]
                    ),
                ),
                pg.Dict(
                    content=pg.Dict(parts=[pg.Dict(text='This is response 2.')])
                ),
            ],
        ),
    )
    model = google_genai.GeminiProVision()
    result = model._response_to_result(response)
    self.assertEqual(
        result,
        lf.LMSamplingResult([
            lf.LMSample(lf.AIMessage('This is response 1.'), score=0.0),
            lf.LMSample(lf.AIMessage('This is response 2.'), score=0.0),
        ]),
    )

  def test_model_hub(self):
    orig_get_model = genai.get_model
    genai.get_model = mock_get_model

    model = google_genai._GOOGLE_GENAI_MODEL_HUB.get('gemini-pro')
    self.assertIsNotNone(model)
    self.assertIs(google_genai._GOOGLE_GENAI_MODEL_HUB.get('gemini-pro'), model)

    genai.get_model = orig_get_model

  def test_api_key_check(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      _ = google_genai.GeminiPro()._api_initialized

    self.assertTrue(google_genai.GeminiPro(api_key='abc')._api_initialized)
    os.environ['GOOGLE_API_KEY'] = 'abc'
    self.assertTrue(google_genai.GeminiPro()._api_initialized)
    del os.environ['GOOGLE_API_KEY']

  def test_call(self):
    with mock.patch(
        'google.generativeai.GenerativeModel.generate_content',
    ) as mock_generate:
      orig_get_model = genai.get_model
      genai.get_model = mock_get_model
      mock_generate.side_effect = mock_generate_content

      lm = google_genai.GeminiPro(api_key='test_key')
      self.maxDiff = None
      self.assertEqual(
          lm('hello', temperature=2.0, top_k=20, max_tokens=1024).text,
          (
              'This is a response to hello with n=1, temperature=2.0, '
              'top_p=None, top_k=20, max_tokens=1024, stop=None.'
          ),
      )
      genai.get_model = orig_get_model

  def test_call_with_legacy_completion_model(self):
    orig_get_model = genai.get_model
    genai.get_model = mock_get_model
    orig_generate_text = genai.generate_text
    genai.generate_text = mock_generate_text

    lm = google_genai.Palm2(api_key='test_key')
    self.maxDiff = None
    self.assertEqual(
        lm('hello', temperature=2.0, top_k=20).text,
        (
            "hello to models/text-bison-001 with {'temperature': 2.0, "
            "'top_k': 20, 'top_p': None, 'candidate_count': 1, "
            "'max_output_tokens': None, 'stop_sequences': None}"
        ),
    )
    genai.get_model = orig_get_model
    genai.generate_text = orig_generate_text

  def test_call_with_legacy_chat_model(self):
    orig_get_model = genai.get_model
    genai.get_model = mock_get_model
    orig_chat = genai.chat
    genai.chat = mock_chat

    lm = google_genai.Palm2_IT(api_key='test_key')
    self.maxDiff = None
    self.assertEqual(
        lm('hello', temperature=2.0, top_k=20).text,
        (
            "hello to models/chat-bison-001 with {'temperature': 2.0, "
            "'top_k': 20, 'top_p': None, 'candidate_count': 1}"
        ),
    )
    genai.get_model = orig_get_model
    genai.chat = orig_chat


if __name__ == '__main__':
  unittest.main()
