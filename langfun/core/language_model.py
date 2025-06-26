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
"""Interface for language model."""

import abc
import contextlib
import dataclasses
import datetime
import enum
import functools
import math
import re
import threading
import time
from typing import Annotated, Any, Callable, ClassVar, Iterator, Literal, Optional, Sequence, Tuple, Type, Union, final
from langfun.core import component
from langfun.core import concurrent
from langfun.core import console
from langfun.core import message as message_lib

import pyglove as pg

#
# Common errors during calling language models.
#


class LMError(RuntimeError):
  """Base class for language model errors."""


class LMInputError(LMError):
  """Base class for errors with bad input."""


class ContextLimitError(LMInputError):
  """Error for context limit exceeded."""


class RetryableLMError(LMError):
  """Base class for LLM errors that can be solved by retrying."""


class RateLimitError(RetryableLMError):
  """Error for rate limit reached."""


class TemporaryLMError(RetryableLMError):
  """Error for temporary service issues that can be retried."""


#
# Language model information.
#


class ModelInfo(pg.Object):
  """Common information for a language model."""

  # Constant for modalities.
  TEXT_INPUT_ONLY = []

  model_id: Annotated[
      str,
      'A global unique identifier of the language model. ',
  ]

  alias_for: Annotated[
      str | None,
      'The fixed-version model ID that this model is aliased for.',
  ] = None

  #
  # Basic information.
  #

  model_type: Annotated[
      Literal['unknown', 'pretrained', 'instruction-tuned', 'thinking'],
      'The type of the model.'
  ] = 'unknown'

  provider: Annotated[
      str | None,
      (
          'The service provider (host) of the LLM. E.g. VertexAI, Microsoft, '
          'etc.'
      )
  ] = None

  description: Annotated[
      str | None,
      'An optional description of the model family.'
  ] = None

  url: Annotated[
      str | None,
      'The URL of the model.'
  ] = None

  release_date: Annotated[
      datetime.date | None,
      'The release date of the model. '
  ] = None

  in_service: Annotated[
      bool,
      'If True, the model is in service.'
  ] = True

  #
  # LLM capabilities.
  #

  input_modalities: Annotated[
      list[str] | None,
      (
          'Supported MIME types as model inputs. '
          'If None, this information is unknown, so Langfun allows all '
          'modalities from the input to be passed to the model.'
      )
  ] = None

  class ContextLength(pg.Object):
    """Context length information."""

    max_input_tokens: Annotated[
        int | None,
        (
            'The maximum number of input tokens of the language model. '
            'If None, there is no limit or this information is unknown.'
        )
    ] = None

    max_output_tokens: Annotated[
        int | None,
        (
            'The maximum number of output tokens of the language model. '
            'If None, there is no limit or this information is unknown.'
        )
    ] = None

    max_cot_tokens: Annotated[
        int | None,
        (
            'The maximum number of Chain-of-Thought tokens to generate. '
            'If None, there is not limit or not applicable.'
        )
    ] = None

  context_length: Annotated[
      ContextLength | None,
      (
          'Context length information of the model. '
          'If None, this information is unknown.'
      )
  ] = None

  #
  # Common pricing information.
  #

  class Pricing(pg.Object):
    """Pricing information."""

    cost_per_1m_cached_input_tokens: Annotated[
        float | None,
        (
            'The cost per 1M cached input tokens in US dollars. '
            'If None, this information is unknown.'
        )
    ] = None

    cost_per_1m_input_tokens: Annotated[
        float | None,
        (
            'The cost per 1M input tokens in US dollars. '
            'If None, this information is unknown.'
        )
    ] = None

    cost_per_1m_output_tokens: Annotated[
        float | None,
        (
            'The cost per 1M output tokens in US dollars. '
            'If None, this information is unknown.'
        )
    ] = None

    def estimate_cost(self, usage: 'LMSamplingUsage') -> float | None:
      """Estimates the cost of using the model. Subclass could override.

      Args:
        usage: The usage information of the model.

      Returns:
        The estimated cost in US dollars. If None, cost estimating is not
        supported on the model.
      """
      # NOTE(daiyip): supported cached tokens accounting in future.
      if (self.cost_per_1m_input_tokens is None
          or self.cost_per_1m_output_tokens is None):
        return None
      return (
          self.cost_per_1m_input_tokens * usage.prompt_tokens
          + self.cost_per_1m_output_tokens
          * (usage.total_tokens - usage.prompt_tokens)
      ) / 1000_000

  pricing: Annotated[
      Pricing | None,
      (
          'Pricing information. If None, this information is unknown.'
      )
  ] = None

  #
  # Rate limits.
  #

  class RateLimits(pg.Object):
    """Preset rate limits."""

    max_requests_per_minute: Annotated[
        int | None,
        (
            'The max number of requests per minute.'
            'If None, there is no limit.'
        )
    ] = None

    max_tokens_per_minute: Annotated[
        int | None,
        (
            'The max number of tokens per minute.'
            'If None, there is no limit.'
        )
    ] = None

  rate_limits: Annotated[
      RateLimits | None,
      (
          'Rate limits. If None, this information is unknown.'
      )
  ] = None

  #
  # Additional information.
  #

  metadata: Annotated[
      dict[str, Any],
      (
          'Model metadata. This could be used to store model-specific '
          'information, which could be consumed by modules that need to '
          'apply model-specific logic.'
      ),
  ] = {}

  #
  # Common model protocols.
  #

  @final
  def estimate_cost(self, usage: 'LMSamplingUsage') -> float | None:
    """Estimates the cost of using the model."""
    if self.pricing is None:
      return None
    return self.pricing.estimate_cost(usage)

  def supports_input(self, mime_type: str) -> bool:
    """Returns True if an input MIME type is supported.

    Subclass could override.

    Args:
      mime_type: The MIME type of the input.

    Returns:
      True if the input MIME type is supported.
    """
    if self._input_modalities is None:
      return True
    return mime_type.lower() in self._input_modalities

  @property
  def resource_id(self) -> str:
    """Returns the resource ID of the LLM. Subclass could override."""
    canonical_model_id = self.alias_for or self.model_id
    if self.provider is None or not pg.is_deterministic(self.provider):
      return canonical_model_id
    provider = self.provider.lower().replace(' ', '_')
    return f'{provider}://{canonical_model_id}'

  def _on_bound(self):
    super()._on_bound()
    self._input_modalities = set(
        [mime_type.lower() for mime_type in self.input_modalities]
    ) if self.input_modalities is not None else None


#
# Language model input/output interfaces.
#


class LMSample(pg.Object):
  """Response candidate."""

  response: pg.typing.Annotated[
      pg.typing.Object(
          message_lib.Message,
          # Allowing automatic conversion from text to AIMessage.
          transform=message_lib.AIMessage.from_value,
      ),
      'The natural language response of LM.',
  ]

  score: Annotated[
      float, 'The score of sampled response. The larger is better'
  ] = 0.0

  logprobs: Annotated[
      list[tuple[str, float, list[tuple[str, float]]]] | None,
      '(token, log prob, top tokens and their probs).',
  ] = None


class RetryStats(pg.Object):
  """Retry stats, which is aggregated across multiple retry entries."""

  num_occurences: Annotated[
      int,
      'Total number of retry attempts on LLM (excluding the first attempt).',
  ] = 0

  total_wait_interval: Annotated[
      float, 'Total wait interval in seconds due to retry.'
  ] = 0

  total_call_interval: Annotated[
      float, 'Total LLM call interval in seconds.'
  ] = 0

  errors: Annotated[
      dict[str, int],
      'A Counter of error types encountered during the retry attempts.',
  ] = {}

  @classmethod
  def from_retry_entries(
      cls, retry_entries: Sequence[concurrent.RetryEntry]
  ) -> 'RetryStats':
    """Creates a RetryStats from a sequence of RetryEntry."""
    if not retry_entries:
      return RetryStats()
    errors = {}
    for retry in retry_entries:
      if retry.error is not None:
        errors[retry.error.__class__.__name__] = (
            errors.get(retry.error.__class__.__name__, 0) + 1
        )
    return RetryStats(
        num_occurences=len(retry_entries) - 1,
        total_wait_interval=sum(e.wait_interval for e in retry_entries),
        total_call_interval=sum(e.call_interval for e in retry_entries),
        errors=errors,
    )

  def __add__(self, other: 'RetryStats') -> 'RetryStats':
    errors = self.errors.copy()
    for error, count in other.errors.items():
      errors[error] = errors.get(error, 0) + count
    return RetryStats(
        num_occurences=self.num_occurences + other.num_occurences,
        total_wait_interval=self.total_wait_interval
        + other.total_wait_interval,
        total_call_interval=self.total_call_interval
        + other.total_call_interval,
        errors=errors,
    )

  def __radd__(self, other: 'RetryStats') -> 'RetryStats':
    return self + other


class LMSamplingUsage(pg.Object):
  """Usage information per completion."""

  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  num_requests: int = 1
  estimated_cost: Annotated[
      float | None,
      (
          'Estimated cost in US dollars. If None, cost estimating is not '
          'suppported on the model being queried.'
      ),
  ] = None
  retry_stats: RetryStats = RetryStats()
  completion_tokens_details: dict[str, Any] | None = None

  def __bool__(self) -> bool:
    return self.num_requests > 0

  @property
  def average_prompt_tokens(self) -> int:
    """Returns the average prompt tokens per request."""
    return self.prompt_tokens // self.num_requests

  @property
  def average_completion_tokens(self) -> int:
    """Returns the average completion tokens per request."""
    return self.completion_tokens // self.num_requests

  @property
  def average_total_tokens(self) -> int:
    """Returns the average total tokens per request."""
    return self.total_tokens // self.num_requests

  @property
  def average_estimated_cost(self) -> float | None:
    """Returns the average estimated cost per request."""
    if self.estimated_cost is None:
      return None
    return self.estimated_cost / self.num_requests

  def __add__(self, other: Optional['LMSamplingUsage']) -> 'LMSamplingUsage':
    if other is None:
      return self
    if self.estimated_cost is None:
      estimated_cost = other.estimated_cost
    elif other.estimated_cost is None:
      estimated_cost = self.estimated_cost
    else:
      estimated_cost = self.estimated_cost + other.estimated_cost
    return LMSamplingUsage(
        prompt_tokens=self.prompt_tokens + other.prompt_tokens,
        completion_tokens=self.completion_tokens + other.completion_tokens,
        total_tokens=self.total_tokens + other.total_tokens,
        num_requests=self.num_requests + other.num_requests,
        estimated_cost=estimated_cost,
        retry_stats=self.retry_stats + other.retry_stats,
    )

  def __radd__(self, other: Optional['LMSamplingUsage']) -> 'LMSamplingUsage':
    return self + other


class UsageNotAvailable(LMSamplingUsage):
  """Usage information not available."""
  prompt_tokens: pg.typing.Int(0).freeze()       # pytype: disable=invalid-annotation
  completion_tokens: pg.typing.Int(0).freeze()   # pytype: disable=invalid-annotation
  total_tokens: pg.typing.Int(0).freeze()        # pytype: disable=invalid-annotation
  estimated_cost: pg.typing.Float(default=None, is_noneable=True).freeze()    # pytype: disable=invalid-annotation

  def __add__(self, other: Optional['LMSamplingUsage']) -> 'UsageNotAvailable':
    if other is None:
      return self
    return UsageNotAvailable(
        num_requests=self.num_requests + other.num_requests
    )

  def __radd__(self, other: Optional['LMSamplingUsage']) -> 'UsageNotAvailable':
    return self + other


class LMSamplingResult(pg.Object):
  """Language model response."""

  samples: Annotated[
      list[LMSample],
      (
          'Multiple samples of the same prompt, sorted by their score. '
          'The first candidate has the highest score.'
      ),
  ] = []

  usage: Annotated[
      LMSamplingUsage,
      'Usage information. Currently only OpenAI models are supported.',
  ] = UsageNotAvailable()

  is_cached: Annotated[
      bool,
      'Whether the result is from cache or not.'
  ] = False


class LMSamplingOptions(component.Component):
  """Language model sampling options."""

  temperature: Annotated[
      float | None,
      (
          'Model temperature, which is usually between 0 and 1.0. '
          'OpenAI models have temperature range from 0.0 to 2.0. '
          'If None (default), honor the model\'s default behavior. '
      )
  ] = None

  max_tokens: Annotated[
      int | None,
      (
          'Per example max tokens to generate. '
          'If None, use the model default.'
      )
  ] = None

  n: Annotated[int | None, 'Max number of samples to return.'] = 1

  top_k: Annotated[
      int | None,
      (
          'Top k tokens to sample the next token. '
          'Not applicable to OpenAI models.'
      )
  ] = 40

  top_p: Annotated[
      float | None,
      (
          'Only sample the next token from top N tokens whose accumulated '
          'probability // mass <= p. For OpenAI models, set `temperature` or '
          '`top_p` but not both.'
      ),
  ] = None

  stop: Annotated[
      list[str] | None,
      (
          'A list of stop sequences that prevent LLMs from outputting '
          'more tokens. For example, when `stop` is set to ["User:", "Model:"] '
          'LLMs will stop to emit more tokens when `User:` or '
          '`Model:` is reached.'
      ),
  ] = None

  random_seed: Annotated[
      int | None, 'A fixed random seed used during model inference.'
  ] = None

  logprobs: Annotated[
      bool,
      (
          'Whether to return log probabilities of the output tokens or not. If '
          'true, returns the log probabilities of each output token returned '
          'in the content of message.'
      ),
  ] = False

  top_logprobs: Annotated[
      int | None,
      (
          'An integer between 0 and 5 specifying the number of most likely '
          'tokens to return at each token position, each with an associated '
          'log probability. logprobs must be set to true if this parameter is '
          'used.'
      ),
  ] = None

  max_thinking_tokens: Annotated[
      int | None, 'Number of max thinking tokens.'
  ] = None

  reasoning_effort: Annotated[
      Literal['low', 'medium', 'high'] | None,
      (
          'This parameter is used by OpenAI reasoning models (e.g. O3, O4 mini)'
          ' to guides how many reasoning tokens to generate before creating a'
          ' response to the prompt.'
      ),
  ] = None

  def cache_key(self) -> tuple[Any, ...]:
    """Returns a tuple of current values as cache key."""
    return (
        self.temperature,
        self.max_tokens,
        self.n,
        self.top_k,
        self.top_p,
        self.random_seed
    )


class LMScoringResult(pg.Object):
  """Language model scoring result."""

  score: Annotated[
      float,
      'The log likelyhood of the requested completion towards the prompt.',
  ]
  gradients: Annotated[
      Any | None,
      '(Optional) gradients from the score method, w.r.t.' +
      ' prompt.metadata.weights.',
  ] = None


class LMCache(pg.Object):
  """Interface for LM cache."""

  @dataclasses.dataclass
  class Stats:
    """Cache stats."""

    num_queries: int = 0
    num_hits: int = 0
    num_hit_expires: int = 0
    num_misses: int = 0
    num_updates: int = 0
    num_deletes: int = 0

  @abc.abstractmethod
  def get(
      self, lm: 'LanguageModel', prompt: message_lib.Message, seed: int
  ) -> LMSamplingResult | None:
    """Gets the cached result of a prompt generated by a language model."""

  @abc.abstractmethod
  def put(
      self,
      lm: 'LanguageModel',
      prompt: message_lib.Message,
      result: LMSamplingResult,
      seed: int,
  ) -> None:
    """Puts the result of a prompt generated by a language model in cache."""

  @abc.abstractmethod
  def delete(
      self,
      lm: 'LanguageModel',
      prompt: message_lib.Message,
      seed: int,
  ) -> bool:
    """Deletes the result of a prompt generated by a language model in cache."""

  @property
  @abc.abstractmethod
  def stats(self) -> Stats:
    """Returns cache stats."""


class LMDebugMode(enum.IntFlag):
  """Sets debugging mode for a language model.

  INFO toggles whether information about the LM will be printed.
  PROMPT toggles whether the prompts sent to the LM will be printed.
  RESPONSE toggles whether the responses from the LM will be printed.
  """
  NONE = 0

  INFO = enum.auto()
  PROMPT = enum.auto()
  RESPONSE = enum.auto()

  @classmethod
  @property
  def ALL(cls) -> 'LMDebugMode':  # pylint: disable=invalid-name
    return LMDebugMode.INFO | LMDebugMode.PROMPT | LMDebugMode.RESPONSE


class LanguageModel(component.Component):
  """Interface of a language model.

  Language models are at the center of LLM-based agents. ``LanguageModel``
  is the interface to interact with different language modles.

  In langfun, users can use different language models with the same agents,
  allowing fast prototype, as well as side-by-side comparisons.
  """

  sampling_options: LMSamplingOptions = LMSamplingOptions()

  cache: Annotated[
      LMCache | None,
      (
          'Sampling cache. If None, no cache will be used.'
      )
  ] = component.contextual(default=None)

  max_concurrency: Annotated[
      int | None,
      (
          'Max concurrent requests being sent to the server. '
          'If None, there is no limit. '
          'Please note that the concurrency control is based on the '
          '`info.resource_id` property, meaning that model instances shared '
          'the same resource ID will be accounted under the same concurrency '
          'control key. This allows a process-level concurrency control '
          'for specific models regardless the number of LM (client) instances '
          'created by the program. Subclasses could override the '
          '`max_concurrency` property to allow dynamic concurrency control.'
      ),
  ] = None

  timeout: Annotated[
      float | None, 'Timeout in seconds. If None, there is no timeout.'
  ] = 120.0

  max_attempts: Annotated[
      int,
      (
          'A number of max attempts to request the LM if fails.'
          'The retry wait time is determined per LM serivice.'
      ),
  ] = 5

  retry_interval: Annotated[
      int | tuple[int, int],
      (
          'An integer as a constant wait time in seconds before next retry, '
          'or a tuple of two integers representing the range of wait time, '
          'based on which the next wait time will be randmly chosen.'
      )
  ] = (5, 60)

  exponential_backoff: Annotated[
      bool,
      (
          'If True, the wait time among multiple attempts will exponentially '
          'grow. If `retry_interval` is an integer, the wait time for the '
          'k\'th attempt will be `retry_interval * 2 ^ (k - 1)` seconds. If '
          '`retry_interval` is a tuple, the wait time range for the k\'th '
          'attempt will be `(retry_interval[0] * 2 ^ (k - 1), '
          'retry_interval[1] * 2 ^ (k - 1)`) seconds.'
      )
  ] = True

  max_retry_interval: Annotated[
      int,
      (
          'The max retry interval in seconds. This is useful when the retry '
          'interval is exponential, to avoid the wait time to grow '
          'exponentially.'
      )
  ] = 300

  debug: Annotated[
      bool | LMDebugMode,
      (
          'If True, the prompt and the response will be output to stdout. '
          'Specific debugging fields (info, prompt, response) can be specified '
          'using the LMDebugMode flags.'
      ),
  ] = False

  _MODEL_FACTORY: ClassVar[dict[str, Callable[..., 'LanguageModel']]] = {}

  @classmethod
  def register(
      cls,
      model_id_or_prefix: str, factory: Callable[..., 'LanguageModel']
  ) -> None:
    """Registers a factory function for a model ID."""
    cls._MODEL_FACTORY[model_id_or_prefix] = factory

  @classmethod
  def get(cls, model_str: str, *args, **kwargs) -> 'LanguageModel':
    """Creates a language model instance from a model str.

    Args:
      model_str: A string that identifies the model. It can be a model ID or a
        model ID with kwargs.
        For example, "gpt-o3?temperature=0.1&n=2" will create a GPT-o3 model
        with temperature set to 0.1 and n set to 2.
      *args: Additional arguments to pass to the model factory.
      **kwargs: Additional keyword arguments to pass to the model factory.
        kwargs provided here will take precedence over kwargs parsed from
        model_str.

    Returns:
      A language model instance.
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
      return sorted(list(LanguageModel._MODEL_FACTORY.keys()))
    return sorted(
        [k for k in LanguageModel._MODEL_FACTORY.keys() if re.match(regex, k)]
    )

  @pg.explicit_method_override
  def __init__(self, *args, **kwargs) -> None:
    """Overrides __init__ to pass through **kwargs to sampling options."""

    sampling_options = kwargs.pop(
        'sampling_options',
        pg.clone(self.__schema__.fields['sampling_options'].default_value)
    )
    sampling_options_delta = {}

    for k, v in kwargs.items():
      if LMSamplingOptions.__schema__.get_field(k) is not None:
        sampling_options_delta[k] = v

    if sampling_options_delta:
      sampling_options.rebind(sampling_options_delta)

    for k in sampling_options_delta:
      del kwargs[k]

    super().__init__(*args, sampling_options=sampling_options, **kwargs)

  def _on_bound(self):
    super()._on_bound()
    self._call_counter = 0
    self.__dict__.pop('model_info', None)

  @functools.cached_property
  def model_info(self) -> ModelInfo:
    """Returns the specification of the model."""
    return ModelInfo(model_id='unknown')

  #
  # Shortcut properties/methods from `model_info`.
  # If these behaviors need to be changed, please override the corresponding
  # methods in the ModelInfo subclasses instead of these properties/methods.
  #

  @final
  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model_info.alias_for or self.model_info.model_id

  @final
  @property
  def resource_id(self) -> str:
    """Resource ID for performing request parallism control."""
    return self.model_info.resource_id

  @final
  @property
  def context_length(self) -> ModelInfo.ContextLength | None:
    """Returns the context length of the model."""
    return self.model_info.context_length

  @final
  @property
  def pricing(self) -> ModelInfo.Pricing | None:
    """Returns the pricing information of the model."""
    return self.model_info.pricing

  @final
  @property
  def rate_limits(self) -> ModelInfo.RateLimits | None:
    """Returns the rate limits to the model."""
    return self.model_info.rate_limits

  @final
  def supports_input(self, mime_type: str):
    """Returns True if an input type is supported. Subclasses can override."""
    return self.model_info.supports_input(mime_type)

  @final
  def estimate_cost(self, usage: LMSamplingUsage) -> float | None:
    """Returns the estimated cost of a usage. Subclasses can override."""
    return self.model_info.estimate_cost(usage)

  #
  # Language model operations.
  #

  def sample(
      self,
      prompts: list[str | message_lib.Message],
      *,
      cache_seed: int = 0,
      **kwargs,
  ) -> list[LMSamplingResult]:
    """Samples one or multiple prompts."""
    # Internal usage logging.

    prompts = [message_lib.UserMessage.from_value(p) for p in prompts]

    with component.context(override_attrs=True, **kwargs):
      if self.cache is None:
        results = self._sample(prompts)
      else:
        results = self._sample_with_cache_lookup(prompts, cache_seed)

      for prompt, result in zip(prompts, results):

        # Tag LM input.
        prompt.tag(message_lib.Message.TAG_LM_INPUT)

        for sample in result.samples:
          # Update metadata for response message.

          response = sample.response
          response.metadata.score = sample.score
          response.metadata.logprobs = sample.logprobs
          response.metadata.is_cached = result.is_cached

          # Update estimated cost.
          usage = result.usage
          estimated_cost = self.estimate_cost(usage)
          if estimated_cost is not None:
            usage.rebind(
                estimated_cost=estimated_cost, skip_notification=True
            )

          # NOTE(daiyip): Current usage is computed at per-result level,
          # which is accurate when n=1. For n > 1, we average the usage across
          # multiple samples.
          if len(result.samples) == 1 or isinstance(usage, UsageNotAvailable):
            response.metadata.usage = usage
          else:
            n = len(result.samples)
            response.metadata.usage = LMSamplingUsage(
                prompt_tokens=usage.prompt_tokens // n,
                completion_tokens=usage.completion_tokens // n,
                total_tokens=usage.total_tokens // n,
                estimated_cost=(
                    usage.estimated_cost / n if usage.estimated_cost else None
                ),
                retry_stats=RetryStats(
                    num_occurences=usage.retry_stats.num_occurences // n,
                    total_wait_interval=usage.retry_stats.total_wait_interval
                    / n,
                    total_call_interval=usage.retry_stats.total_call_interval
                    / n,
                    errors={
                        error: count // n
                        for error, count in usage.retry_stats.errors.items()
                    },
                ),
            )

          # Track usage.
          trackers = component.context_value('__usage_trackers__', [])
          if trackers:
            model_id = self.model_id
            for tracker in trackers:
              tracker.track(model_id, usage, result.is_cached)

          # Track the prompt for corresponding response.
          response.source = prompt

          # Tag LM response.
          response.tag(message_lib.Message.TAG_LM_RESPONSE)
      return results

  def _sample_with_cache_lookup(
      self, prompts: list[str | message_lib.Message], cache_seed: int
  ) -> list[LMSamplingResult]:
    """Sample with cache lookup."""
    assert self.cache is not None

    results = [None] * len(prompts)
    requests, request_to_result_index = [], {}

    # Perform cache lookup and figure out sampling requests to make.
    for i, prompt in enumerate(prompts):
      r = None
      # Query cache if cache_seed is not None.
      if cache_seed is not None:
        r = self.cache.get(self, prompt, seed=cache_seed)

      if r is None:
        request_to_result_index[len(requests)] = i
        requests.append(prompt)
      else:
        result = r.clone()
        assert result.is_cached, result
        results[i] = result

    # Sample non-cache-hit prompts.
    if requests:
      requested_results = self._sample(requests)
      assert len(requested_results) == len(requests), (
          requests, requested_results)

      # Combine cached results and newly requested results.
      for i, (prompt, result) in enumerate(zip(requests, requested_results)):
        results[request_to_result_index[i]] = result

        # Carry the cache seed in response message.
        for sample in result.samples:
          sample.response.set('cache_seed', cache_seed)

        if cache_seed is not None:
          self.cache.put(
              self,
              prompt,
              result.clone(override=dict(is_cached=True)),
              seed=cache_seed
          )
    return results  # pytype: disable=bad-return-type

  @abc.abstractmethod
  def _sample(
      self,
      prompt: list[message_lib.Message],
  ) -> list[LMSamplingResult]:
    """Subclass should override."""

  def _parallel_execute_with_currency_control(
      self,
      action: Callable[..., LMSamplingResult],
      inputs: Sequence[Any],
      retry_on_errors: Union[
          None,
          Union[Type[BaseException], Tuple[Type[BaseException], str]],
          Sequence[Union[Type[BaseException], Tuple[Type[BaseException], str]]],
      ] = RetryableLMError,
  ) -> list[Any]:
    """Helper method for subclasses for implementing _sample."""
    if self.max_concurrency is None:
      execute = action
      executor = None
      max_workers = len(inputs)
    else:
      execute = lambda x: _ConcurrencyControl.get(
          self.resource_id, self.max_concurrency)(action, x)
      executor = self.resource_id if len(inputs) > 1 else None
      max_workers = self.max_concurrency

    executed_jobs = concurrent.concurrent_execute(
        execute,
        inputs,
        executor=executor,
        max_workers=max_workers,
        retry_on_errors=retry_on_errors,
        max_attempts=self.max_attempts,
        retry_interval=self.retry_interval,
        exponential_backoff=self.exponential_backoff,
        max_retry_interval=self.max_retry_interval,
        return_jobs=True,
    )
    for job in executed_jobs:
      if isinstance(job.result, LMSamplingResult):
        job.result.usage.rebind(
            retry_stats=RetryStats.from_retry_entries(job.retry_entries),
            skip_notification=True,
        )
    return [job.result for job in executed_jobs]

  def __call__(
      self, prompt: message_lib.Message, *, cache_seed: int = 0, **kwargs
  ) -> message_lib.Message:
    """Returns the first candidate."""
    prompt = message_lib.UserMessage.from_value(prompt)
    with component.context(override_attrs=True, **kwargs):
      sampling_options = self.sampling_options
      if sampling_options.n != 1:
        sampling_options = sampling_options.clone(override=dict(n=1))

      call_counter = self._call_counter
      self._call_counter += 1
      request_start = time.time()
      result = self.sample(
          [prompt], sampling_options=sampling_options, cache_seed=cache_seed
      )[0]
      elapse = time.time() - request_start
      response = result.samples[0].response
      self._debug(prompt, response, call_counter, result.usage, elapse)
      return response

  def _debug(
      self,
      prompt: message_lib.Message,
      response: message_lib.Message,
      call_counter: int,
      usage: LMSamplingUsage,
      elapse: float,
  ) -> None:
    """Outputs debugging information."""
    debug = self.debug
    if isinstance(debug, bool):
      debug = LMDebugMode.ALL if debug else LMDebugMode.NONE

    if debug & LMDebugMode.INFO:
      self._debug_model_info(call_counter, usage)

    if debug & LMDebugMode.PROMPT:
      self._debug_prompt(prompt, call_counter, usage)

    if debug & LMDebugMode.RESPONSE:
      self._debug_response(response, call_counter, usage, elapse)

  def _debug_model_info(
      self, call_counter: int, usage: LMSamplingUsage) -> None:
    """Outputs debugging information about the model."""
    title_suffix = ''
    if usage.total_tokens != 0:
      title_suffix = pg.colored(
          f' (total {usage.total_tokens} tokens)', 'red'
      )

    console.write(
        self.format(compact=True, use_inferred=True),
        title=f'[{call_counter}] LM INFO{title_suffix}:',
        color='magenta',
    )

  def _debug_prompt(
      self,
      prompt: message_lib.Message,
      call_counter: int,
      usage: LMSamplingUsage,
  ) -> None:
    """Outputs debugging information about the prompt."""
    title_suffix = ''
    if usage.prompt_tokens != 0:
      title_suffix = pg.colored(f' ({usage.prompt_tokens} tokens)', 'red')

    console.write(
        # We use metadata 'formatted_text' for scenarios where the prompt text
        # is formatted by the LM.
        prompt.get('formatted_text', prompt.text),
        title=f'\n[{call_counter}] PROMPT SENT TO LM{title_suffix}:',
        color='green',
    )
    referred_modalities = prompt.referred_modalities()
    if referred_modalities:
      console.write(
          pg.object_utils.kvlist_str(
              [(k, repr(v), None) for k, v in referred_modalities.items()]
          ),
          title=f'\n[{call_counter}] MODALITY OBJECTS SENT TO LM:',
          color='green',
      )

  def _debug_response(
      self,
      response: message_lib.Message,
      call_counter: int,
      usage: LMSamplingUsage,
      elapse: float
  ) -> None:
    """Outputs debugging information about the response."""
    title_suffix = ' ('
    if usage.completion_tokens != 0:
      title_suffix += f'{usage.completion_tokens} tokens '
    title_suffix += f'in {elapse:.2f} seconds)'
    title_suffix = pg.colored(title_suffix, 'red')

    console.write(
        str(response) + '\n',
        title=f'\n[{call_counter}] LM RESPONSE{title_suffix}:',
        color='blue',
    )

  def score(
      self,
      prompt: str | message_lib.Message | list[message_lib.Message],
      completions: list[str | message_lib.Message],
      **kwargs,
  ) -> list[LMScoringResult]:
    """Scores the given prompt."""
    if isinstance(prompt, list):
      if len(prompt) != len(completions):
        raise ValueError(
            'prompt and completions must have the same length.'
        )
      prompt = [message_lib.UserMessage.from_value(p) for p in prompt]
    else:
      prompt = message_lib.UserMessage.from_value(prompt)
    completions = [message_lib.UserMessage.from_value(c) for c in completions]

    call_counter = self._call_counter
    self._call_counter += 1
    request_start = time.time()

    with component.context(override_attrs=True, **kwargs):
      scoring_results = self._score(prompt, completions)
      elapse = time.time() - request_start
      self._debug_score(
          prompt, completions, scoring_results, call_counter, elapse
      )
      return scoring_results

  def _score(
      self, prompt: message_lib.Message | list[message_lib.Message],
      completions: list[message_lib.Message]
  ) -> list[LMScoringResult]:
    """Subclass to implement."""
    raise NotImplementedError(
        f'{self.__class__.__name__} does not support scoring.'
    )

  def _debug_score(
      self,
      prompt: message_lib.Message | list[message_lib.Message],
      completions: list[message_lib.Message],
      scoring_results: list[LMScoringResult],
      call_counter: int,
      elapse: float,
  ):
    debug = self.debug
    if isinstance(debug, bool):
      debug = LMDebugMode.ALL if debug else LMDebugMode.NONE

    if debug & LMDebugMode.INFO:
      self._debug_model_info(call_counter, UsageNotAvailable())

    if debug & LMDebugMode.PROMPT:
      console.write(
          prompt,
          title=f'\n[{call_counter}] SCORING LM WITH PROMPT:',
          color='green',
      )
      if isinstance(prompt, list):
        referred_modalities_lst = [p.referred_modalities() for p in prompt]
      else:
        referred_modalities_lst = [prompt.referred_modalities(),]
      if referred_modalities_lst:
        for referred_modalities in referred_modalities_lst:
          console.write(
              pg.object_utils.kvlist_str(
                  [(k, repr(v), None) for k, v in referred_modalities.items()]
              ),
              title=f'\n[{call_counter}] MODALITY OBJECTS SENT TO LM:',
              color='green',
          )

    if debug & LMDebugMode.RESPONSE:
      console.write(
          '',
          title=(
              f'\n[{call_counter}] SCORING COMPLETED (in {elapse:.2f} seconds):'
          ),
          color='blue',
      )
      for i, (c, r) in enumerate(zip(completions, scoring_results)):
        console.write(
            c,
            title=f'COMPLETION #{i}',
            color='green',
        )
        console.write(
            f'score: {r.score}',
            color='blue',
        )

  def tokenize(
      self,
      prompt: str | message_lib.Message,
      **kwargs,
  ) -> list[tuple[str | bytes, int]]:
    """Tokenizes the given prompt."""
    prompt = message_lib.UserMessage.from_value(prompt)
    call_counter = self._call_counter
    self._call_counter += 1

    with component.context(override_attrs=True, **kwargs):
      request_start = time.time()
      tokens = self._tokenize(prompt)
      elapse = time.time() - request_start
      self._debug_tokenize(prompt, tokens, call_counter, elapse)
      return tokens

  def _tokenize(
      self, prompt: message_lib.Message
  ) -> list[tuple[str | bytes, int]]:
    """Subclass to implement."""
    raise NotImplementedError(
        f'{self.__class__.__name__} does not support tokenization.'
    )

  def _debug_tokenize(
      self,
      prompt: message_lib.Message,
      tokens: list[tuple[str | bytes, int]],
      call_counter: int,
      elapse: float,
  ):
    debug = self.debug
    if isinstance(debug, bool):
      debug = LMDebugMode.ALL if debug else LMDebugMode.NONE

    if debug & LMDebugMode.INFO:
      self._debug_model_info(call_counter, UsageNotAvailable())

    if debug & LMDebugMode.PROMPT:
      console.write(
          prompt,
          title=f'\n[{call_counter}] PROMPT TO TOKENIZE:',
          color='green',
      )
      referred_modalities_lst = [prompt.referred_modalities(),]
      if referred_modalities_lst:
        for referred_modalities in referred_modalities_lst:
          console.write(
              pg.object_utils.kvlist_str(
                  [(k, repr(v), None) for k, v in referred_modalities.items()]
              ),
              title=f'\n[{call_counter}] MODALITY OBJECTS SENT TO LM:',
              color='green',
          )

    if debug & LMDebugMode.RESPONSE:
      console.write(
          tokens,
          title=(
              f'\n[{call_counter}] {len(tokens)} TOKENS RETURNED '
              f'(in {elapse:.2f} seconds):'
          ),
          color='blue',
      )

  @classmethod
  def estimate_max_concurrency(
      cls,
      max_tokens_per_minute: int | None,
      max_requests_per_minute: int | None,
      average_tokens_per_request: int = 250
  ) -> int | None:
    """Estimates max concurrency concurrency based on the rate limits."""
    # NOTE(daiyip): max concurrency is estimated based on the rate limit.
    # We assume each request has approximately 250 tokens, and each request
    # takes 1 second to complete. This might not be accurate for all models.
    if max_tokens_per_minute is not None:
      return max(
          int(max_tokens_per_minute / average_tokens_per_request / 60), 1
      )
    elif max_requests_per_minute is not None:
      return max(int(max_requests_per_minute / 60), 1)
    return None


class _ConcurrencyControl:
  """Controls the max concurrent LLM calls for a given model."""

  _MODEL_CONCURRENCY: ClassVar[dict[str, '_ConcurrencyControl']] = {}

  def __init__(self, max_concurrency: int):
    self.max_concurrency = max_concurrency
    self._concurrency = 0

  @property
  def concurrency(self) -> int:
    """Returns the current concurrency."""
    return self._concurrency

  def __call__(self, fn: Callable[..., Any], *args, **kwargs):
    """Calls the function with concurrency control."""
    while self._concurrency >= self.max_concurrency:
      time.sleep(0.01)

    try:
      # Increment/decrement is atomic in Python, so we don't need to protect it
      # with a lock.
      self._concurrency += 1
      return fn(*args, **kwargs)
    finally:
      self._concurrency -= 1

  @classmethod
  def get(
      cls, model_id: str, max_concurrency: int | None = None
  ) -> '_ConcurrencyControl':
    """Returns the concurrency control for the given model ID."""
    control = cls._MODEL_CONCURRENCY.get(model_id, None)
    if control is None:
      assert max_concurrency is not None
      control = cls(max_concurrency)
      cls._MODEL_CONCURRENCY[model_id] = control
    return control


class UsageSummary(pg.Object, pg.views.HtmlTreeView.Extension):
  """Usage sumary."""

  class AggregatedUsage(pg.Object):
    """Aggregated usage."""

    total: LMSamplingUsage = LMSamplingUsage(0, 0, 0, 0, 0.0)
    breakdown: dict[str, LMSamplingUsage] = {}

    def __bool__(self) -> bool:
      """Returns True if the usage is non-empty."""
      return bool(self.breakdown)

    def add(
        self,
        model_id: str,
        usage: LMSamplingUsage,
    ) -> None:
      """Adds an entry to the breakdown."""
      aggregated = self.breakdown.get(model_id, None)
      with pg.notify_on_change(False):
        self.breakdown[model_id] = usage + aggregated
        self.rebind(
            total=self.total + usage,
            raise_on_no_change=False
        )

    def merge(self, other: 'UsageSummary.AggregatedUsage') -> None:
      """Merges the usage summary."""
      with pg.notify_on_change(False):
        for model_id, usage in other.breakdown.items():
          self.add(model_id, usage)

  def _on_bound(self):
    super()._on_bound()
    self._usage_badge = None
    self._lock = threading.Lock()

  @property
  def total(self) -> LMSamplingUsage:
    return self.cached.total + self.uncached.total

  def add(self, model_id: str, usage: LMSamplingUsage, is_cached: bool):
    """Updates the usage summary."""
    with self._lock:
      if is_cached:
        usage.rebind(estimated_cost=0.0, skip_notification=True)
        self.cached.add(model_id, usage)
      else:
        self.uncached.add(model_id, usage)
      self._update_view()

  def merge(self, other: 'UsageSummary', as_cached: bool = False) -> None:
    """Aggregates the usage summary.

    Args:
      other: The usage summary to merge.
      as_cached: Whether to merge the usage summary as cached.
    """
    with self._lock:
      self.cached.merge(other.cached)
      if as_cached:
        self.cached.merge(other.uncached)
      else:
        self.uncached.merge(other.uncached)
      self._update_view()

  def _sym_nondefault(self) -> dict[str, Any]:
    """Overrides nondefault values so volatile values are not included."""
    return dict()

  #
  # Html views for the usage summary.
  #

  def _update_view(self):
    if self._usage_badge is not None:
      self._usage_badge.update(
          self._badge_text(),
          tooltip=pg.format(
              self, verbose=False, custom_format=self._tooltip_format,
              hide_default_values=True,
          ),
          styles=dict(color=self._badge_color()),
      )

  def _badge_text(self) -> str:
    if self.total.estimated_cost is not None:
      return f'{self.total.estimated_cost:.3f}'
    return '0.000'

  def _badge_color(self) -> str | None:
    if self.total.estimated_cost is None or self.total.estimated_cost < 1.0:
      return None

    # Step 1: The normal cost range is around 1e-3 to 1e5.
    # Therefore we normalize the log10 value from [-3, 5] to [0, 1].
    normalized_value = (math.log10(self.total.estimated_cost) + 3) / (5 + 3)

    # Step 2: Interpolate between green and red
    red = int(255 * normalized_value)
    green = int(255 * (1 - normalized_value))
    return f'rgb({red}, {green}, 0)'

  def _tooltip_format(self, v, root_indent):
    del root_indent
    if isinstance(v, int):
      return f'{v:,}'
    if isinstance(v, float):
      return f'{v:,.3f}'
    return None

  def _html_tree_view(
      self,
      *,
      view: pg.views.HtmlTreeView,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ) -> pg.Html:
    extra_flags = extra_flags or {}
    as_badge = extra_flags.pop('as_badge', False)
    interactive = extra_flags.get('interactive', True)
    if as_badge:
      usage_badge = self._usage_badge
      if usage_badge is None:
        usage_badge = pg.views.html.controls.Badge(
            self._badge_text(),
            tooltip=pg.format(
                self, custom_format=self._tooltip_format, verbose=False,
                hide_default_values=True,
            ),
            css_classes=['usage-summary'],
            styles=dict(color=self._badge_color()),
            interactive=True,
        )
        if interactive:
          self._usage_badge = usage_badge
      return usage_badge.to_html()
    return super()._html_tree_view(
        view=view,
        extra_flags=extra_flags,
        **kwargs
    )

  @classmethod
  @functools.cache
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .usage-summary.label {
            display: inline-flex;
            border-radius: 5px;
            padding: 5px;
            background-color: #f1f1f1;
            color: #CCC;
        }
        .usage-summary.label::before {
            content: '$';
        }
        """
    ]

pg.members(
    dict(
        cached=(
            pg.typing.Object(
                UsageSummary.AggregatedUsage,
                default=UsageSummary.AggregatedUsage()
            ),
            'Aggregated usages for cached LLM calls.'
        ),
        uncached=(
            pg.typing.Object(
                UsageSummary.AggregatedUsage,
                default=UsageSummary.AggregatedUsage()
            ),
            'Aggregated usages for uncached LLM calls.'
        ),
    )
)(UsageSummary)


class _UsageTracker:
  """Usage tracker."""

  def __init__(self, model_ids: set[str] | None):
    self.model_ids = model_ids
    self.usage_summary = UsageSummary()

  def track(self, model_id: str, usage: LMSamplingUsage, is_cached: bool):
    if self.model_ids is None or model_id in self.model_ids:
      self.usage_summary.add(model_id, usage, is_cached)


@contextlib.contextmanager
def track_usages(
    *lm: Union[str, LanguageModel]
) -> Iterator[UsageSummary]:
  """Context manager to track the usages of all language models in scope.

  `lf.track_usages` works with threads spawned by `lf.concurrent_map` and
  `lf.concurrent_execute`.

  Example:
    ```
    lm = lf.llms.GeminiPro1()
    with lf.track_usages() as usages:
      # invoke any code that will call LLMs.

    print(usages.uncached.breakdown[lm.model_id])
    ```

  Args:
    *lm: The language model(s) to track. If None, track all models in scope.

  Yields:
    A dictionary of model ID to usage. If a model does not supports usage
    counting, the dict entry will be None.
  """
  if not lm:
    model_ids = None
  else:
    model_ids = [m.model_id if isinstance(m, LanguageModel) else m for m in lm]

  trackers = component.context_value('__usage_trackers__', [])
  tracker = _UsageTracker(set(model_ids) if model_ids else None)
  with component.context(__usage_trackers__=trackers + [tracker]):
    try:
      yield tracker.usage_summary
    finally:
      pass
