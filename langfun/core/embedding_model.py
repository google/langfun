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
"""Interface for embedding model."""

import abc
import re
from typing import Annotated, Any, Callable, ClassVar, Literal

from langfun.core import async_support
from langfun.core import component
from langfun.core import message as message_lib
import pyglove as pg


class EmbeddingUsage(pg.Object):
  """Usage information for an embedding request."""

  prompt_tokens: int = 0
  total_tokens: int = 0
  num_requests: int = 1
  estimated_cost: Annotated[
      float | None,
      'Estimated cost in US dollars. If None, not supported.',
  ] = None


class EmbeddingResult(pg.Object):
  """Result from an embedding model."""

  embedding: Annotated[
      list[float],
      'The embedding vector.',
  ]

  usage: Annotated[
      EmbeddingUsage,
      'Usage information.',
  ] = EmbeddingUsage()


class EmbeddingOptions(component.Component):
  """Options for embedding generation."""

  output_dimensionality: Annotated[
      int | None,
      (
          'The number of dimensions of the output embedding. '
          'If None, uses the model default.'
      ),
  ] = None

  task_type: Annotated[
      Literal[
          'RETRIEVAL_QUERY',
          'RETRIEVAL_DOCUMENT',
          'SEMANTIC_SIMILARITY',
          'CLASSIFICATION',
          'CLUSTERING',
          'QUESTION_ANSWERING',
          'FACT_VERIFICATION',
          'CODE_RETRIEVAL_QUERY',
      ]
      | None,
      (
          'The type of task for which the embedding is generated. '
          'If None, defaults to the model default. '
          'Applicable only for VertexAI models.'
      ),
  ] = None


class EmbeddingModel(component.Component):
  """Interface for embedding model.

  `lf.EmbeddingModel` provides a consistent interface for interacting with
  various embedding models, such as Gemini Embedding, OpenAI Embedding, etc.
  It abstracts away provider-specific details, allowing users to switch
  between models seamlessly.

  **Key Features:**

  *   **Unified API**: Provides `__call__` and `acall` methods across all
      supported models.
  *   **Concurrency**: Manages concurrency to respect API rate limits via
      `max_concurrency`.
  *   **Retries**: Automatic retries with exponential backoff for transient
      errors via `max_attempts` and `retry_interval`.

  **1. Creating an Embedding Model:**

  ```python
  em = lf.embedding.VertexAI(
      model='gemini-embedding-002',
      project='my-project', location='us-central1'
  )
  ```

  **2. Embedding:**

  ```python
  result = em('hello world')
  print(result.embedding)
  # Output: [0.01, -0.02, ...]
  ```
  """

  model: Annotated[
      str | None,
      'Model ID.',
  ] = None

  max_concurrency: Annotated[
      int | None,
      (
          'Max concurrent requests being sent to the server. '
          'If None, there is no limit.'
      ),
  ] = None

  timeout: Annotated[
      float | None, 'Timeout in seconds. If None, there is no timeout.'
  ] = 120.0

  max_attempts: Annotated[
      int,
      'A number of max attempts to request the embedding model if fails.',
  ] = 5

  retry_interval: Annotated[
      int | tuple[int, int],
      (
          'An integer as a constant wait time in seconds before next retry, '
          'or a tuple of two integers representing the range of wait time.'
      ),
  ] = (5, 60)

  exponential_backoff: Annotated[
      bool,
      'If True, the wait time will exponentially grow among retries.',
  ] = True

  max_retry_interval: Annotated[
      int,
      'The max retry interval in seconds.',
  ] = 300

  embedding_options: EmbeddingOptions = EmbeddingOptions()

  _MODEL_FACTORY: ClassVar[dict[str, Callable[..., 'EmbeddingModel']]] = {}

  @classmethod
  def register(
      cls,
      model_id_or_prefix: str, factory: Callable[..., 'EmbeddingModel']
  ) -> None:
    """Registers a factory function for a model ID."""
    cls._MODEL_FACTORY[model_id_or_prefix] = factory

  @classmethod
  def get(cls, model_str: str, *args, **kwargs) -> 'EmbeddingModel':
    """Creates an embedding model instance from a model str.

    Args:
      model_str: A string that identifies the model. It can be a model ID or a
        model ID with kwargs.
        For example, "text-embedding-3-small?dimensions=256" will create a model
        with output_dimensionality set to 256.
      *args: Additional arguments to pass to the model factory.
      **kwargs: Additional keyword arguments to pass to the model factory.
        kwargs provided here will take precedence over kwargs parsed from
        model_str.

    Returns:
      An embedding model instance.
    """
    model_id, model_kwargs = cls._parse_model_str(model_str)
    model_kwargs.update(kwargs)

    factory = cls._MODEL_FACTORY.get(model_id)
    if factory is None:
      factories = []
      for k, v in cls._MODEL_FACTORY.items():
        if re.match(k, model_id):
          factories.append((k, v))
      if not factories:
        raise ValueError(f'Model not found: {model_id!r}.')
      elif len(factories) > 1:
        raise ValueError(
            f'Multiple models found for {model_id!r}: '
            f'{[x[0] for x in factories]}. '
            'Please specify a more specific model ID.'
        )
      factory = factories[0][1]
    return factory(model_id, *args, **model_kwargs)

  @classmethod
  def _parse_model_str(cls, model_str: str) -> tuple[str, dict[str, Any]]:
    """Parses a model string into model ID and kwargs."""
    parts = model_str.split('?')
    if len(parts) == 1:
      return model_str, {}
    elif len(parts) == 2:
      model_id, kwargs_str = parts
      kwargs = {}
      for kv in kwargs_str.split('&'):
        kv_parts = kv.split('=')
        if len(kv_parts) != 2:
          raise ValueError(f'Invalid kwargs in model string: {model_str!r}.')
        k, v = kv_parts
        if v.isnumeric():
          v = int(v)
        elif v.lower() in ('true', 'false'):
          v = v.lower() == 'true'
        else:
          v = v.strip()
          try:
            v = float(v)
          except ValueError:
            pass
        kwargs[k] = v
      return model_id, kwargs
    else:
      raise ValueError(f'Invalid model string: {model_str!r}.')

  @classmethod
  def dir(cls, regex: str | None = None):
    """Returns a list of model IDs that match the given regex."""
    if regex is None:
      return sorted(list(EmbeddingModel._MODEL_FACTORY.keys()))
    return sorted(
        [
            k
            for k in EmbeddingModel._MODEL_FACTORY.keys()
            if re.match(regex, k)
        ]
    )

  @pg.explicit_method_override
  def __init__(self, *args, **kwargs) -> None:
    embedding_options = kwargs.pop(
        'embedding_options',
        pg.clone(self.__schema__.fields['embedding_options'].default_value),
    )
    embedding_options_delta = {}

    for k, v in kwargs.items():
      if EmbeddingOptions.__schema__.get_field(k) is not None:
        embedding_options_delta[k] = v

    if embedding_options_delta:
      embedding_options.rebind(embedding_options_delta)

    for k in embedding_options_delta:
      del kwargs[k]

    super().__init__(*args, embedding_options=embedding_options, **kwargs)

  def __call__(
      self,
      message: str | message_lib.Message,
      **kwargs,
  ) -> EmbeddingResult:
    """Embeds the given message into a vector."""
    if isinstance(message, str):
      message = message_lib.UserMessage.from_value(message)
    with component.context(override_attrs=True, **kwargs):
      return self._embed(message)

  async def acall(
      self,
      message: str | message_lib.Message,
      **kwargs,
  ) -> EmbeddingResult:
    """Async version of __call__."""
    return await async_support.invoke_async(self, message, **kwargs)

  @abc.abstractmethod
  def _embed(
      self,
      message: message_lib.Message,
  ) -> EmbeddingResult:
    """Subclass should override."""
