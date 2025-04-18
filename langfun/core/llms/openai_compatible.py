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
"""Base for OpenAI compatible models (including OpenAI)."""

from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.data.conversion import openai as openai_conversion  # pylint: disable=unused-import
from langfun.core.llms import rest
import pyglove as pg


@lf.use_init_args(['api_endpoint', 'model'])
class OpenAICompatible(rest.REST):
  """Base for OpenAI compatible models."""

  model: Annotated[
      str, 'The name of the model to use.',
  ] = ''

  @property
  def headers(self) -> dict[str, Any]:
    return {
        'Content-Type': 'application/json'
    }

  def _request_args(
      self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # Reference:
    # https://platform.openai.com/docs/api-reference/completions/create
    # NOTE(daiyip): options.top_k is not applicable.
    args = dict(
        n=options.n,
        top_logprobs=options.top_logprobs,
    )
    if self.model:
      args['model'] = self.model
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
    if options.reasoning_effort is not None:
      args['reasoning_effort'] = options.reasoning_effort
    return args

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request_args = self._request_args(sampling_options)

    # Users could use `metadata_json_schema` to pass additional
    # request arguments.
    json_schema = prompt.metadata.get('json_schema')
    if json_schema is not None:
      if not isinstance(json_schema, dict):
        raise ValueError(
            f'`json_schema` must be a dict, got {json_schema!r}.'
        )
      if 'title' not in json_schema:
        raise ValueError(
            f'The root of `json_schema` must have a `title` field, '
            f'got {json_schema!r}.'
        )
      request_args.update(
          response_format=dict(
              type='json_schema',
              json_schema=dict(
                  schema=json_schema,
                  name=json_schema['title'],
                  strict=True,
              )
          )
      )
      prompt.metadata.formatted_text = (
          prompt.text
          + '\n\n [RESPONSE FORMAT (not part of prompt)]\n'
          + pg.to_json_str(request_args['response_format'], json_indent=2)
      )

    # Prepare messages.
    messages = []

    def modality_check(chunk: str | lf.Modality) -> Any:
      if (isinstance(chunk, lf_modalities.Mime)
          and not self.supports_input(chunk.mime_type)):
        raise ValueError(
            f'Unsupported modality: {chunk!r}.'
        )
      return chunk

    # Users could use `metadata_system_message` to pass system message.
    system_message = prompt.metadata.get('system_message')
    if system_message:
      assert isinstance(system_message, lf.SystemMessage), type(system_message)
      messages.append(
          system_message.as_format(
              'openai', chunk_preprocessor=modality_check
          )
      )
    messages.append(
        prompt.as_format('openai', chunk_preprocessor=modality_check)
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
        lf.Message.from_value(choice['message'], format='openai'),
        score=0.0,
        logprobs=logprobs,
    )

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    """Returns a LMSamplingResult from a JSON response."""
    usage = json['usage']
    return lf.LMSamplingResult(
        samples=[self._parse_choice(choice) for choice in json['choices']],
        usage=lf.LMSamplingUsage(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'],
            completion_tokens_details=usage.get(
                'completion_tokens_details', None
            ),
        ),
    )

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if (status_code == 413
        or (status_code == 400 and b'string_above_max_length' in content)):
      return lf.ContextLimitError(f'{status_code}: {content}')
    return super()._error(status_code, content)
