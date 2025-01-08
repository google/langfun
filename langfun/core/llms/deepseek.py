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
"""Language models from DeepSeek."""

import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import rest
import pyglove as pg

SUPPORTED_MODELS_AND_SETTINGS = {
    # pylint: disable=g-line-too-long
    # TODO(yifenglu): The RPM and TPM are arbitrary numbers. Update them once DeepSeek provides concrete guidelines.
    # DeepSeek doesn't control the rate limit at the moment: https://api-docs.deepseek.com/quick_start/rate_limit
    # The cost is based on: https://api-docs.deepseek.com/quick_start/pricing
    'deepseek-chat': pg.Dict(
        in_service=True,
        rpm=100,
        tpm=1000000,
        cost_per_1k_input_tokens=0.00014,
        cost_per_1k_output_tokens=0.00028,
    ),
}


# DeepSeek API uses an API format compatible with OpenAI.
# Reference: https://api-docs.deepseek.com/
@lf.use_init_args(['model'])
class DeepSeek(rest.REST):
  """DeepSeek model."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      'The name of the model to use.',
  ]

  api_endpoint: str = 'https://api.deepseek.com/chat/completions'

  multimodal: Annotated[bool, 'Whether this model has multimodal support.'] = (
      False
  )

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'DEEPSEEK_API_KEY'."
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None

  def _initialize(self):
    api_key = self.api_key or os.environ.get('DEEPSEEK_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `DEEPSEEK_API_KEY` with your DeepSeek API key.'
      )
    self._api_key = api_key

  @property
  def headers(self) -> dict[str, Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self._api_key}',
    }
    return headers

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'DeepSeek({self.model})'

  @property
  def max_concurrency(self) -> int:
    rpm = SUPPORTED_MODELS_AND_SETTINGS[self.model].get('rpm', 0)
    tpm = SUPPORTED_MODELS_AND_SETTINGS[self.model].get('tpm', 0)
    return self.rate_to_max_concurrency(
        requests_per_min=rpm, tokens_per_min=tpm
    )

  def estimate_cost(
      self, num_input_tokens: int, num_output_tokens: int
  ) -> float | None:
    """Estimate the cost based on usage."""
    cost_per_1k_input_tokens = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_input_tokens', None
    )
    cost_per_1k_output_tokens = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_output_tokens', None
    )
    if cost_per_1k_output_tokens is None or cost_per_1k_input_tokens is None:
      return None
    return (
        cost_per_1k_input_tokens * num_input_tokens
        + cost_per_1k_output_tokens * num_output_tokens
    ) / 1000

  @classmethod
  def dir(cls):
    return [k for k, v in SUPPORTED_MODELS_AND_SETTINGS.items() if v.in_service]

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    # Reference:
    # https://platform.openai.com/docs/api-reference/completions/create
    # NOTE(daiyip): options.top_k is not applicable.
    args = dict(
        model=self.model,
        n=options.n,
        top_logprobs=options.top_logprobs,
    )
    if options.logprobs:
      args['logprobs'] = options.logprobs

    if options.temperature is not None:
      args['temperature'] = options.temperature
    if options.max_tokens is not None:
      args['max_completion_tokens'] = options.max_tokens
    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.stop:
      args['stop'] = options.stop
    if options.random_seed is not None:
      args['seed'] = options.random_seed
    return args

  def _content_from_message(self, message: lf.Message):
    """Returns a OpenAI content object from a Langfun message."""

    def _uri_from(chunk: lf.Modality) -> str:
      if chunk.uri and chunk.uri.lower().startswith(
          ('http:', 'https:', 'ftp:')
      ):
        return chunk.uri
      return chunk.content_uri

    content = []
    for chunk in message.chunk():
      if isinstance(chunk, str):
        item = dict(type='text', text=chunk)
      elif isinstance(chunk, lf_modalities.Image) and self.multimodal:
        item = dict(type='image_url', image_url=dict(url=_uri_from(chunk)))
      else:
        raise ValueError(f'Unsupported modality: {chunk!r}.')
      content.append(item)
    return content

  def request(
      self, prompt: lf.Message, sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request_args = self._request_args(sampling_options)

    # Users could use `metadata_json_schema` to pass additional
    # request arguments.
    json_schema = prompt.metadata.get('json_schema')
    if json_schema is not None:
      if not isinstance(json_schema, dict):
        raise ValueError(f'`json_schema` must be a dict, got {json_schema!r}.')
      if 'title' not in json_schema:
        raise ValueError(
            'The root of `json_schema` must have a `title` field, '
            f'got {json_schema!r}.'
        )
      request_args.update(
          response_format=dict(
              type='json_schema',
              json_schema=dict(
                  schema=json_schema,
                  name=json_schema['title'],
                  strict=True,
              ),
          )
      )
      prompt.metadata.formatted_text = (
          prompt.text
          + '\n\n [RESPONSE FORMAT (not part of prompt)]\n'
          + pg.to_json_str(request_args['response_format'], json_indent=2)
      )

    # Prepare messages.
    messages = []
    # Users could use `metadata_system_message` to pass system message.
    system_message = prompt.metadata.get('system_message')
    if system_message:
      system_message = lf.SystemMessage.from_value(system_message)
      messages.append(
          dict(
              role='system', content=self._content_from_message(system_message)
          )
      )
    messages.append(
        dict(role='user', content=self._content_from_message(prompt))
    )
    request = dict()
    request.update(request_args)
    request['messages'] = messages
    return request

  def _parse_choice(self, choice: dict[str, Any]) -> lf.LMSample:
    # Reference:
    # https://platform.openai.com/docs/api-reference/chat/object
    logprobs = None
    choice_logprobs = choice.get('logprobs')
    if choice_logprobs:
      logprobs = [
          (
              t['token'],
              t['logprob'],
              [(tt['token'], tt['logprob']) for tt in t['top_logprobs']],
          )
          for t in choice_logprobs['content']
      ]
    return lf.LMSample(
        choice['message']['content'],
        score=0.0,
        logprobs=logprobs,
    )

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    usage = json['usage']
    return lf.LMSamplingResult(
        samples=[self._parse_choice(choice) for choice in json['choices']],
        usage=lf.LMSamplingUsage(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'],
            estimated_cost=self.estimate_cost(
                num_input_tokens=usage['prompt_tokens'],
                num_output_tokens=usage['completion_tokens'],
            ),
        ),
    )


class DeepSeekChat(DeepSeek):
  """DeepSeek Chat model.

  Currently, it is powered by DeepSeek-V3 model, 64K input contenxt window and
  8k max output tokens.
  """

  model = 'deepseek-chat'
