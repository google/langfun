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
class OpenAIChatCompletionAPI(rest.REST):
  """Base class for models compatible with OpenAI's Chat Completion API.

  This class provides a common interface for language models that adhere to
  the OpenAI Chat Completion API format, which is used by providers like
  Groq, DeepSeek, and others. It standardizes request formatting and
  response parsing for these models.

  **References:**

  *   https://platform.openai.com/docs/api-reference/chat
  """

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
    args = {}

    if self.model:
      args['model'] = self.model
    if options.n != 1:
      args['n'] = options.n
    if options.top_logprobs is not None:
      args['top_logprobs'] = options.top_logprobs
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
    if options.extras:
      args.update(options.extras)
    return args

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request_args = self._request_args(sampling_options)

    # Handle structured output.
    output_schema = self._structure_output_schema(prompt)
    if output_schema is not None:
      request_args.update(
          response_format=dict(
              type='json_schema',
              json_schema=output_schema,
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
              'openai_chat_completion_api', chunk_preprocessor=modality_check
          )
      )
    messages.append(
        prompt.as_format(
            'openai_chat_completion_api',
            chunk_preprocessor=modality_check
        )
    )
    request = dict()
    request.update(request_args)
    request['messages'] = messages
    return request

  def _structure_output_schema(
      self, prompt: lf.Message
  ) -> dict[str, Any] | None:
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
      return dict(
          schema=json_schema,
          name=json_schema['title'],
          strict=True,
      )
    return None

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
        lf.Message.from_value(
            choice['message'],
            format='openai_chat_completion_api'
        ),
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


class OpenAIResponsesAPI(OpenAIChatCompletionAPI):
  """Base class for models compatible with OpenAI's Responses API.

  This class provides a common interface for language models that adhere to
  the new OpenAI Responses API format. It standardizes request formatting
  and response parsing for these models, including handling instructions
  (system messages) and structured outputs.

  **References:**

  *   https://platform.openai.com/docs/api-reference/responses
  """

  def _request_args(
      self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    if options.logprobs:
      raise ValueError('logprobs is not supported on Responses API.')
    if options.n != 1:
      raise ValueError('n must be 1 for Responses API.')
    return super()._request_args(options)

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request_args = self._request_args(sampling_options)

    # Handle structured output.
    output_schema = self._structure_output_schema(prompt)
    if output_schema is not None:
      output_schema['type'] = 'json_schema'
      request_args.update(text=dict(format=output_schema))
      prompt.metadata.formatted_text = (
          prompt.text
          + '\n\n [RESPONSE FORMAT (not part of prompt)]\n'
          + pg.to_json_str(request_args['text'], json_indent=2)
      )

    request = dict()
    request.update(request_args)

    # Users could use `metadata_system_message` to pass system message.
    system_message = prompt.metadata.get('system_message')
    if system_message:
      assert isinstance(system_message, lf.SystemMessage), type(system_message)
      request['instructions'] = system_message.text

    # Prepare input.
    def modality_check(chunk: str | lf.Modality) -> Any:
      if (isinstance(chunk, lf_modalities.Mime)
          and not self.supports_input(chunk.mime_type)):
        raise ValueError(
            f'Unsupported modality: {chunk!r}.'
        )
      return chunk

    request['input'] = [
        prompt.as_format(
            'openai_responses_api',
            chunk_preprocessor=modality_check
        )
    ]
    return request

  def _parse_output(self, output: dict[str, Any]) -> lf.LMSample:
    for item in output:
      if isinstance(item, dict) and item.get('type') == 'message':
        return lf.LMSample(
            lf.Message.from_value(item, format='openai_responses_api'),
            score=0.0,
        )
    raise ValueError('No message found in output.')

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    """Returns a LMSamplingResult from a JSON response."""
    usage = json['usage']
    return lf.LMSamplingResult(
        samples=[self._parse_output(json['output'])],
        usage=lf.LMSamplingUsage(
            prompt_tokens=usage['input_tokens'],
            completion_tokens=usage['output_tokens'],
            total_tokens=usage['total_tokens'],
            completion_tokens_details=usage.get(
                'output_tokens_details', None
            ),
        ),
    )
