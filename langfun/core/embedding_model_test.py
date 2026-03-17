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
import asyncio
import unittest

import langfun.core as lf
from langfun.core import embedding_model


class SimpleEmbeddingModel(lf.EmbeddingModel):

  def _embed(self, message):
    return lf.EmbeddingResult(
        embedding=[float(len(message.text))],
        usage=lf.EmbeddingUsage(prompt_tokens=len(message.text)),
    )


class EmbeddingUsageTest(unittest.TestCase):

  def test_defaults(self):
    usage = embedding_model.EmbeddingUsage()
    self.assertEqual(usage.prompt_tokens, 0)
    self.assertEqual(usage.total_tokens, 0)
    self.assertEqual(usage.num_requests, 1)
    self.assertIsNone(usage.estimated_cost)

  def test_custom_values(self):
    usage = embedding_model.EmbeddingUsage(
        prompt_tokens=10, total_tokens=10, num_requests=1, estimated_cost=0.01
    )
    self.assertEqual(usage.prompt_tokens, 10)
    self.assertEqual(usage.total_tokens, 10)
    self.assertEqual(usage.estimated_cost, 0.01)


class EmbeddingResultTest(unittest.TestCase):

  def test_with_embedding(self):
    result = embedding_model.EmbeddingResult(embedding=[0.1, 0.2, 0.3])
    self.assertEqual(result.embedding, [0.1, 0.2, 0.3])
    self.assertIsInstance(result.usage, embedding_model.EmbeddingUsage)

  def test_with_usage(self):
    usage = embedding_model.EmbeddingUsage(prompt_tokens=5)
    result = embedding_model.EmbeddingResult(embedding=[1.0], usage=usage)
    self.assertEqual(result.usage.prompt_tokens, 5)


class EmbeddingModelTest(unittest.TestCase):

  def test_embed_with_string(self):
    em = SimpleEmbeddingModel()
    result = em('hello')
    self.assertEqual(result.embedding, [5.0])
    self.assertEqual(result.usage.prompt_tokens, 5)

  def test_embed_with_message(self):
    em = SimpleEmbeddingModel()
    msg = lf.UserMessage('hi')
    result = em(msg)
    self.assertEqual(result.embedding, [2.0])

  def test_acall(self):
    em = SimpleEmbeddingModel()
    result = asyncio.run(em.acall('test'))
    self.assertEqual(result.embedding, [4.0])

  def test_model_defaults(self):
    em = SimpleEmbeddingModel()
    self.assertIsNone(em.model)
    self.assertIsNone(em.max_concurrency)
    self.assertEqual(em.timeout, 120.0)
    self.assertEqual(em.max_attempts, 5)
    self.assertEqual(em.retry_interval, (5, 60))
    self.assertTrue(em.exponential_backoff)
    self.assertEqual(em.max_retry_interval, 300)

  def test_embedding_options_defaults(self):
    em = SimpleEmbeddingModel()
    self.assertIsNone(em.embedding_options.task_type)
    self.assertIsNone(em.embedding_options.output_dimensionality)

  def test_embedding_options_passthrough(self):
    em = SimpleEmbeddingModel(
        task_type='RETRIEVAL_QUERY',
        output_dimensionality=256,
    )
    self.assertEqual(em.embedding_options.task_type, 'RETRIEVAL_QUERY')
    self.assertEqual(em.embedding_options.output_dimensionality, 256)

  def test_embedding_options_explicit(self):
    options = lf.EmbeddingOptions(task_type='CLUSTERING')
    em = SimpleEmbeddingModel(embedding_options=options)
    self.assertEqual(em.embedding_options.task_type, 'CLUSTERING')
    self.assertIsNone(em.embedding_options.output_dimensionality)

  def test_embed_time_override(self):
    em = SimpleEmbeddingModel(task_type='RETRIEVAL_QUERY')
    self.assertEqual(em.embedding_options.task_type, 'RETRIEVAL_QUERY')
    em('test', task_type='CLUSTERING')
    self.assertEqual(em.embedding_options.task_type, 'RETRIEVAL_QUERY')


class ModelRegistryTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._original_factory = lf.EmbeddingModel._MODEL_FACTORY.copy()
    lf.EmbeddingModel._MODEL_FACTORY.clear()

  def tearDown(self):
    super().tearDown()
    lf.EmbeddingModel._MODEL_FACTORY = self._original_factory

  def test_register_and_get(self):
    lf.EmbeddingModel.register('test-model', SimpleEmbeddingModel)
    em = lf.EmbeddingModel.get('test-model')
    self.assertIsInstance(em, SimpleEmbeddingModel)
    self.assertEqual(em.model, 'test-model')

  def test_get_not_found(self):
    with self.assertRaisesRegex(ValueError, 'Model not found'):
      lf.EmbeddingModel.get('nonexistent-model')

  def test_get_with_kwargs(self):
    lf.EmbeddingModel.register('test-model', SimpleEmbeddingModel)
    em = lf.EmbeddingModel.get('test-model', timeout=60.0)
    self.assertEqual(em.timeout, 60.0)

  def test_get_with_query_string(self):
    lf.EmbeddingModel.register('test-model', SimpleEmbeddingModel)
    em = lf.EmbeddingModel.get('test-model?timeout=30')
    self.assertEqual(em.timeout, 30)

  def test_get_kwargs_override_query_string(self):
    lf.EmbeddingModel.register('test-model', SimpleEmbeddingModel)
    em = lf.EmbeddingModel.get('test-model?timeout=30', timeout=60.0)
    self.assertEqual(em.timeout, 60.0)

  def test_dir_empty(self):
    self.assertEqual(lf.EmbeddingModel.dir(), [])

  def test_dir_sorted(self):
    lf.EmbeddingModel.register('b-model', SimpleEmbeddingModel)
    lf.EmbeddingModel.register('a-model', SimpleEmbeddingModel)
    self.assertEqual(lf.EmbeddingModel.dir(), ['a-model', 'b-model'])

  def test_dir_with_regex(self):
    lf.EmbeddingModel.register('text-embedding-small', SimpleEmbeddingModel)
    lf.EmbeddingModel.register('text-embedding-large', SimpleEmbeddingModel)
    lf.EmbeddingModel.register('gemini-embedding', SimpleEmbeddingModel)
    result = lf.EmbeddingModel.dir('text-.*')
    self.assertEqual(result, ['text-embedding-large', 'text-embedding-small'])

  def test_parse_model_str_simple(self):
    model_id, kwargs = lf.EmbeddingModel._parse_model_str('my-model')
    self.assertEqual(model_id, 'my-model')
    self.assertEqual(kwargs, {})

  def test_parse_model_str_with_kwargs(self):
    model_id, kwargs = lf.EmbeddingModel._parse_model_str(
        'my-model?timeout=30&max_attempts=3'
    )
    self.assertEqual(model_id, 'my-model')
    self.assertEqual(kwargs, {'timeout': 30, 'max_attempts': 3})

  def test_parse_model_str_with_bool(self):
    _, kwargs = lf.EmbeddingModel._parse_model_str(
        'my-model?exponential_backoff=true'
    )
    self.assertTrue(kwargs['exponential_backoff'])

  def test_parse_model_str_invalid(self):
    with self.assertRaisesRegex(ValueError, 'Invalid model string'):
      lf.EmbeddingModel._parse_model_str('a?b?c')

  def test_parse_model_str_invalid_kwargs_format(self):
    with self.assertRaisesRegex(ValueError, 'Invalid kwargs in model string'):
      lf.EmbeddingModel._parse_model_str('my-model?invalid_format')

  def test_parse_model_str_with_float(self):
    _, kwargs = lf.EmbeddingModel._parse_model_str(
        'my-model?some_float=3.14'
    )
    self.assertEqual(kwargs['some_float'], 3.14)

  def test_parse_model_str_with_string(self):
    _, kwargs = lf.EmbeddingModel._parse_model_str(
        'my-model?some_string=hello'
    )
    self.assertEqual(kwargs['some_string'], 'hello')

  def test_get_with_regex_match(self):
    # Register with a regex pattern
    lf.EmbeddingModel.register('text-embedding-.*', SimpleEmbeddingModel)
    em = lf.EmbeddingModel.get('text-embedding-v1')
    self.assertIsInstance(em, SimpleEmbeddingModel)
    self.assertEqual(em.model, 'text-embedding-v1')

  def test_get_with_multiple_regex_matches(self):
    lf.EmbeddingModel.register('text-.*', SimpleEmbeddingModel)
    lf.EmbeddingModel.register('text-embedding-.*', SimpleEmbeddingModel)
    with self.assertRaisesRegex(ValueError, 'Multiple models found'):
      lf.EmbeddingModel.get('text-embedding-v1')


if __name__ == '__main__':
  unittest.main()
