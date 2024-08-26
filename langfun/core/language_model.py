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
import enum
import threading
import time
from typing import Annotated, Any, Callable, Iterator, Sequence, Tuple, Type, Union
from langfun.core import component
from langfun.core import concurrent
from langfun.core import console
from langfun.core import message as message_lib

import pyglove as pg

TOKENS_PER_REQUEST = 250  # Estimated num tokens for a single request
DEFAULT_MAX_CONCURRENCY = 1  # Use this as max concurrency if no RPM or TPM data


#
# Common errors during calling language models.
#


class LMError(RuntimeError):
  """Base class for language model errors."""


class RetryableLMError(LMError):
  """Base class for LLM errors that can be solved by retrying."""


class RateLimitError(RetryableLMError):
  """Error for rate limit reached."""


class TemporaryLMError(RetryableLMError):
  """Error for temporary service issues that can be retried."""


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


class LMSamplingUsage(pg.Object):
  """Usage information per completion."""

  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  num_requests: int = 1

  def __add__(self, other: 'LMSamplingUsage') -> 'LMSamplingUsage':
    return LMSamplingUsage(
        prompt_tokens=self.prompt_tokens + other.prompt_tokens,
        completion_tokens=self.completion_tokens + other.completion_tokens,
        total_tokens=self.total_tokens + other.total_tokens,
        num_requests=self.num_requests + other.num_requests,
    )


class UsageNotAvailable(LMSamplingUsage):
  """Usage information not available."""
  prompt_tokens: pg.typing.Int(0).freeze()       # pytype: disable=invalid-annotation
  completion_tokens: pg.typing.Int(0).freeze()   # pytype: disable=invalid-annotation
  total_tokens: pg.typing.Int(0).freeze()        # pytype: disable=invalid-annotation
  num_requests: pg.typing.Int(1).freeze()        # pytype: disable=invalid-annotation

  def __bool__(self) -> bool:
    return False


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
          '`resource_id` property, meaning that model instances shared '
          'the same resource ID will be accounted under the same concurrency '
          'control key. This allows a process-level concurrency control '
          'for specific models regardless the number of LM (client) instances '
          'created by the program. Subclasses could override this number or '
          'replace it with a `max_concurrency` property to allow dynamic '
          'concurrency control.'
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

  debug: Annotated[
      bool | LMDebugMode,
      (
          'If True, the prompt and the response will be output to stdout. '
          'Specific debugging fields (info, prompt, response) can be specified '
          'using the LMDebugMode flags.'
      ),
  ] = False

  @pg.explicit_method_override
  def __init__(self, *args, **kwargs) -> None:
    """Overrides __init__ to pass through **kwargs to sampling options."""

    sampling_options = kwargs.pop('sampling_options', LMSamplingOptions())
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

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.__class__.__name__

  @property
  def resource_id(self) -> str:
    """Resource ID for performing request parallism control."""
    return self.model_id

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

          # NOTE(daiyip): Current usage is computed at per-result level,
          # which is accurate when n=1. For n > 1, we average the usage across
          # multiple samples.
          usage = result.usage
          if len(result.samples) == 1 or not usage:
            response.metadata.usage = usage
          else:
            n = len(result.samples)
            response.metadata.usage = LMSamplingUsage(
                prompt_tokens=usage.prompt_tokens // n,
                completion_tokens=usage.completion_tokens // n,
                total_tokens=usage.total_tokens // n,
            )

          # Track usage.
          trackers = component.context_value('__usage_trackers__', [])
          if trackers:
            model_id = self.model_id
            for tracker in trackers:
              tracker.track(model_id, usage)

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
        results[i] = r.clone()

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
          self.cache.put(self, prompt, result.clone(), seed=cache_seed)

    return results  # pytype: disable=bad-return-type

  @abc.abstractmethod
  def _sample(
      self,
      prompt: list[message_lib.Message],
  ) -> list[LMSamplingResult]:
    """Subclass should override."""

  def _parallel_execute_with_currency_control(
      self,
      action: Callable[..., Any],
      inputs: Sequence[Any],
      retry_on_errors: Union[
          None,
          Union[Type[Exception], Tuple[Type[Exception], str]],
          Sequence[Union[Type[Exception], Tuple[Type[Exception], str]]],
      ] = RetryableLMError,
  ) -> Any:
    """Helper method for subclasses for implementing _sample."""
    return concurrent.concurrent_execute(
        action,
        inputs,
        executor=self.resource_id if self.max_concurrency else None,
        max_workers=self.max_concurrency or len(inputs),
        retry_on_errors=retry_on_errors,
        max_attempts=self.max_attempts,
        retry_interval=self.retry_interval,
        exponential_backoff=self.exponential_backoff,
    )

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
      title_suffix = console.colored(
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
      title_suffix = console.colored(f' ({usage.prompt_tokens} tokens)', 'red')

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
    title_suffix = console.colored(title_suffix, 'red')

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

  def rate_to_max_concurrency(
      self, requests_per_min: float = 0, tokens_per_min: float = 0
  ) -> int:
    """Converts a rate to a max concurrency."""
    if tokens_per_min > 0:
      return max(int(tokens_per_min / TOKENS_PER_REQUEST / 60), 1)
    elif requests_per_min > 0:
      return max(int(requests_per_min / 60), 1)  # Max concurrency can't be zero
    else:
      return DEFAULT_MAX_CONCURRENCY  # Default of 1


class _UsageTracker:
  """Usage tracker."""

  def __init__(self, model_ids: set[str] | None):
    self.model_ids = model_ids
    self._lock = threading.Lock()
    self.usages = {
        m: LMSamplingUsage(0, 0, 0, 0) for m in model_ids
    } if model_ids else {}

  def track(self, model_id: str, usage: LMSamplingUsage):
    if self.model_ids is not None and model_id not in self.model_ids:
      return
    with self._lock:
      if not isinstance(usage, UsageNotAvailable) and model_id in self.usages:
        self.usages[model_id] += usage
      else:
        self.usages[model_id] = usage


@contextlib.contextmanager
def track_usages(
    *lm: Union[str, LanguageModel]
) -> Iterator[dict[str, LMSamplingUsage]]:
  """Context manager to track the usages of all language models in scope.

  `lf.track_usages` works with threads spawned by `lf.concurrent_map` and
  `lf.concurrent_execute`.

  Example:
    ```
    lm = lf.llms.GeminiPro1()
    with lf.track_usages() as usages:
      # invoke any code that will call LLMs.

    print(usages[lm.model_id])
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
      yield tracker.usages
    finally:
      pass
