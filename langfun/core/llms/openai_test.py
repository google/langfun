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
"""Tests for openai models."""

import unittest
from unittest import mock

import langfun.core as lf
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
  return pg.Dict(choices=choices, usage=openai.Usage(
      prompt_tokens=100,
      completion_tokens=100,
      total_tokens=200,
  ))


def mock_chat_completion_query(messages, *, n=1, **kwargs):
  del messages, kwargs
  choices = []
  for k in range(n):
    choices.append(pg.Dict(
        message=pg.Dict(
            content=f'Sample {k} for message.'
        )
    ))
  return pg.Dict(choices=choices, usage=openai.Usage(
      prompt_tokens=100,
      completion_tokens=100,
      total_tokens=200,
  ))


class OpenaiTest(unittest.TestCase):
  """Tests for OpenAI language model."""

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

  def test_sample_completion(self):
    with mock.patch('openai.Completion.create') as mock_completion:
      mock_completion.side_effect = mock_completion_query
      lm = openai.OpenAI(api_key='test_key', model='text-davinci-003')
      results = lm.sample(
          ['hello', 'bye'], sampling_options=lf.LMSamplingOptions(n=3)
      )

    self.assertEqual(len(results), 2)
    self.assertEqual(results[0], openai.LMSamplingResult([
        lf.LMSample('Sample 0 for prompt 0.', score=0.0),
        lf.LMSample('Sample 1 for prompt 0.', score=0.1),
        lf.LMSample('Sample 2 for prompt 0.', score=0.2),
    ], usage=openai.Usage(
        prompt_tokens=100, completion_tokens=100, total_tokens=200)))

    self.assertEqual(results[1], openai.LMSamplingResult([
        lf.LMSample('Sample 0 for prompt 1.', score=0.0),
        lf.LMSample('Sample 1 for prompt 1.', score=0.1),
        lf.LMSample('Sample 2 for prompt 1.', score=0.2),
    ]))

  def test_sample_chat_completion(self):
    with mock.patch('openai.ChatCompletion.create') as mock_chat_completion:
      mock_chat_completion.side_effect = mock_chat_completion_query
      lm = openai.OpenAI(api_key='test_key', model='gpt-4')
      results = lm.sample(
          ['hello', 'bye'], sampling_options=lf.LMSamplingOptions(n=3)
      )

    self.assertEqual(len(results), 2)
    self.assertEqual(results[0], openai.LMSamplingResult([
        lf.LMSample('Sample 0 for message.', score=0.0),
        lf.LMSample('Sample 1 for message.', score=0.0),
        lf.LMSample('Sample 2 for message.', score=0.0),
    ], usage=openai.Usage(
        prompt_tokens=100, completion_tokens=100, total_tokens=200)))
    self.assertEqual(results[1], openai.LMSamplingResult([
        lf.LMSample('Sample 0 for message.', score=0.0),
        lf.LMSample('Sample 1 for message.', score=0.0),
        lf.LMSample('Sample 2 for message.', score=0.0),
    ], usage=openai.Usage(
        prompt_tokens=100, completion_tokens=100, total_tokens=200)))

  def test_sample_with_contextual_options(self):
    with mock.patch('openai.Completion.create') as mock_completion:
      mock_completion.side_effect = mock_completion_query
      lm = openai.OpenAI(api_key='test_key', model='text-davinci-003')
      with lf.use_settings(sampling_options=lf.LMSamplingOptions(n=2)):
        results = lm.sample(['hello'])

    self.assertEqual(len(results), 1)
    self.assertEqual(results[0], openai.LMSamplingResult([
        lf.LMSample('Sample 0 for prompt 0.', score=0.0),
        lf.LMSample('Sample 1 for prompt 0.', score=0.1),
    ], usage=openai.Usage(
        prompt_tokens=100, completion_tokens=100, total_tokens=200)))


if __name__ == '__main__':
  unittest.main()
