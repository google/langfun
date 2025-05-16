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
"""Query LLM for structured output."""

import contextlib
import dataclasses
import functools
import inspect
import time
from typing import Annotated, Any, Callable, ClassVar, Iterator, Type, Union
import uuid

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


@lf.use_init_args(['schema', 'default', 'examples'])
class LfQuery(mapping.Mapping):
  """Base class for different implementations of `lf.query`.

  By subclassing this class, users could create variations of prompts for
  `lf.query` and associated them with specific protocols and versions.

  For example:

  ```
  class _MyLfQuery(LFQuery):
    protocol = 'my_format'
    version = '1.0'

    template_str = inspect.cleandoc(
      '''
      ...
      '''
    )
    mapping_template = lf.Template(
      '''
      ...
      '''
    )

  lf.query(..., protocol='my_format:1.0')
  ```

  (THIS IS NOT A TEMPLATE)
  """

  context_title = 'CONTEXT'
  input_title = 'INPUT_OBJECT'

  # Mark schema as required.
  schema: pg.typing.Annotated[
      schema_lib.schema_spec(), 'Required schema for parsing.'
  ]

  # A map from (protocol, version) to the query structure class.
  # This is used to map different protocols/versions to different templates.
  # So users can use `lf.query(..., protocol='<protocol>:<version>')` to use
  # a specific version of the prompt. We use this feature to support variations
  # of prompts and maintain backward compatibility.
  _OOP_PROMPT_MAP: ClassVar[
      dict[
          str,        # protocol.
          dict[
              str,    # version.
              Type['LfQuery']
          ]
      ]
  ] = {}

  # This the flag to update default protocol version.
  _DEFAULT_PROTOCOL_VERSIONS: ClassVar[dict[str, str]] = {
      'python': '2.0',
      'json': '1.0',
  }

  def __init_subclass__(cls) -> Any:
    super().__init_subclass__()
    if not inspect.isabstract(cls):
      protocol = cls.__schema__['protocol'].default_value
      version_dict = cls._OOP_PROMPT_MAP.get(protocol)
      if version_dict is None:
        version_dict = {}
        cls._OOP_PROMPT_MAP[protocol] = version_dict
      dest_cls = version_dict.get(cls.version)
      if dest_cls is not None and dest_cls.__type_name__ != cls.__type_name__:
        raise ValueError(
            f'Version {cls.version} is already registered for {dest_cls!r} '
            f'under protocol {protocol!r}. Please use a different version.'
        )
      version_dict[cls.version] = cls

  @classmethod
  def from_protocol(cls, protocol: str) -> Type['LfQuery']:
    """Returns a query structure from the given protocol and version."""
    if ':' in protocol:
      protocol, version = protocol.split(':')
    else:
      version = cls._DEFAULT_PROTOCOL_VERSIONS.get(protocol)
      if version is None:
        version_dict = cls._OOP_PROMPT_MAP.get(protocol)
        if version_dict is None:
          raise ValueError(
              f'Protocol {protocol!r} is not supported. Available protocols: '
              f'{sorted(cls._OOP_PROMPT_MAP.keys())}.'
          )
        elif len(version_dict) == 1:
          version = list(version_dict.keys())[0]
        else:
          raise ValueError(
              f'Multiple versions found for protocol {protocol!r}, please '
              f'specify a version with "{protocol}:<version>".'
          )

    version_dict = cls._OOP_PROMPT_MAP.get(protocol)
    if version_dict is None:
      raise ValueError(
          f'Protocol {protocol!r} is not supported. Available protocols: '
          f'{sorted(cls._OOP_PROMPT_MAP.keys())}.'
      )
    dest_cls = version_dict.get(version)
    if dest_cls is None:
      raise ValueError(
          f'Version {version!r} is not supported for protocol {protocol!r}. '
          f'Available versions: {sorted(version_dict.keys())}.'
      )
    return dest_cls


class _LfQueryJsonV1(LfQuery):
  """Query a structured value using JSON as the protocol."""

  preamble = """
      Please respond to the last {{ input_title }} with {{ output_title}} according to {{ schema_title }}:

      INSTRUCTIONS:
        1. If the schema has `_type`, carry it over to the JSON output.
        2. If a field from the schema cannot be extracted from the response, use null as the JSON value.

      {{ input_title }}:
        1 + 1 =

      {{ schema_title }}:
        {"result": {"_type": "langfun.core.structured.query.Answer", "final_answer": int}}

      {{ output_title}}:
        {"result": {"_type": "langfun.core.structured.query.Answer", "final_answer": 2}}
      """

  version = '1.0'
  protocol = 'json'
  schema_title = 'SCHEMA'
  output_title = 'JSON'


class _LfQueryPythonV1(LfQuery):
  """Query a structured value using Python as the protocol."""

  preamble = """
      Please respond to the last {{ input_title }} with {{ output_title }} according to {{ schema_title }}.

      {{ input_title }}:
        1 + 1 =

      {{ schema_title }}:
        Answer

        ```python
        class Answer:
          final_answer: int
        ```

      {{ output_title }}:
        ```python
        Answer(
          final_answer=2
        )
        ```
      """
  version = '1.0'
  protocol = 'python'
  schema_title = 'OUTPUT_TYPE'
  output_title = 'OUTPUT_OBJECT'
  mapping_template = lf.Template(
      """
      {%- if example.context -%}
      {{ context_title}}:
      {{ example.context | indent(2, True)}}

      {% endif -%}

      {{ input_title }}:
      {{ example.input_repr(protocol, compact=False) | indent(2, True) }}

      {% if example.schema -%}
      {{ schema_title }}:
      {{ example.schema_repr(protocol) | indent(2, True) }}

      {% endif -%}

      {{ output_title }}:
      {%- if example.has_output %}
      {{ example.output_repr(protocol, compact=False) | indent(2, True) }}
      {% endif -%}
      """
  )


class _LfQueryPythonV2(LfQuery):
  """Query a structured value using Python as the protocol."""

  preamble = """
      Please respond to the last {{ input_title }} with {{ output_title }} only according to {{ schema_title }}.

      {{ input_title }}:
        1 + 1 =

      {{ schema_title }}:
        Answer

        ```python
        class Answer:
          final_answer: int
        ```

      {{ output_title }}:
        ```python
        output = Answer(
          final_answer=2
        )
        ```
      """
  version = '2.0'
  protocol = 'python'
  input_title = 'REQUEST'
  schema_title = 'OUTPUT PYTHON TYPE'
  output_title = 'OUTPUT PYTHON OBJECT'
  mapping_template = lf.Template(
      """
      {%- if example.context -%}
      {{ context_title}}:
      {{ example.context | indent(2, True)}}

      {% endif -%}

      {{ input_title }}:
      {{ example.input_repr(protocol, compact=False) | indent(2, True) }}

      {% if example.schema -%}
      {{ schema_title }}:
      {{ example.schema_repr(protocol) | indent(2, True) }}

      {% endif -%}

      {{ output_title }}:
      {%- if example.has_output %}
      {{ example.output_repr(protocol, compact=False, assign_to_var='output') | indent(2, True) }}
      {% endif -%}
      """
  )


def query(
    prompt: Union[str, lf.Template, Any],
    schema: schema_lib.SchemaType |  None = None,
    default: Any = lf.RAISE_IF_HAS_ERROR,
    *,
    lm: lf.LanguageModel | list[lf.LanguageModel] | None = None,
    num_samples: int | list[int] = 1,
    system_message: str | lf.Template | None = None,
    examples: list[mapping.MappingExample] | None = None,
    cache_seed: int | None = 0,
    response_postprocess: Callable[[str], str] | None = None,
    autofix: int = 0,
    autofix_lm: lf.LanguageModel | None = None,
    protocol: str | None = None,
    returns_message: bool = False,
    skip_lm: bool = False,
    invocation_id: str | None = None,
    **kwargs,
) -> Any:
  """Query one or more language models for structured or unstructured outputs.

  This is the primary API in Langfun for interacting with language models,
  supporting natural language prompts, structured inputs, and multiple advanced
  features.

  Key Features:

    - **Input**: Accepts natural language strings, structured inputs (e.g.,
      `pg.Object`), and templates (`lf.Template`) with modality objects.

    - **Output**: Returns structured outputs when `schema` is specified;
      otherwise, outputs raw natural language (as a string).

    - **Few-shot examples**: Supports structured few-shot examples with the
      `examples` argument.

    - **Multi-LM fan-out**: Sends queries to multiple language models with in
      multiple samples in parallel,  returning a list of outputs.

  Examples:

    Case 1: Regular natural language-based LLM query:

    ```
    lf.query('1 + 1 = ?', lm=lf.llms.Gpt4Turbo())

    # Outptut: '2'
    ```

    Case 2: Query with structured output.

    ```
    lf.query('1 + 1 = ?', int, lm=lf.llms.Gpt4Turbo())

    # Output: 2
    ```

    Case 3: Query with structured input.

    ```
    class Sum(pg.Object):
      a: int
      b: int

    lf.query(Sum(1, 1), int, lm=lf.llms.Gpt4Turbo())

    # Output: 2
    ```

    Case 4: Query with input of mixed modalities.

    ```
    class Animal(pg.Object):
      pass

    class Dog(Animal):
      pass

    class Entity(pg.Object):
      name: str

    lf.query(
        'What is in this {{image}} and {{objects}}?'
        list[Entity],
        lm=lf.llms.Gpt4Turbo()
        image=lf.Image(path='/path/to/a/airplane.png'),
        objects=[Dog()],
    )

    # Output: [Entity(name='airplane'), Entity(name='dog')]
    ```

    Case 5: Query with structured few-shot examples.
    ```
    lf.query(
        'What is in this {{image}} and {{objects}}?'
        list[Entity],
        lm=lf.llms.Gpt4Turbo()
        image=lf.Image(path='/path/to/a/dinasaur.png'),
        objects=[Dog()],
        examples=[
            lf.MappingExample(
                input=lf.Template(
                    'What is the object near the house in this {{image}}?',
                    image=lf.Image(path='/path/to/image.png'),
                ),
                schema=Entity,
                output=Entity('cat'),
            ),
        ],
    )

    # Output: [Entity(name='dinasaur'), Entity(name='dog')]
    ```

    Case 6: Multiple queries to multiple models.
    ```
    lf.query(
        '1 + 1 = ?',
        int,
        lm=[
            lf.llms.Gpt4Turbo(),
            lf.llms.Gemini1_5Pro(),
        ],
        num_samples=[1, 2],
    )
    # Output: [2, 2, 2]
    ```

  Args:
    prompt: The input query. Can be:
      - A natural language string (supports templating with `{{}}`),
      - A `pg.Object` object for structured input,
      - An `lf.Template` for mixed or template-based inputs.
    schema: Type annotation or `lf.Schema` object for the expected output. 
      If `None` (default), the response will be a natural language string.
    default: Default value to return if parsing fails. If not specified, an
      error will be raised.
    lm: The language model(s) to query. Can be:
      - A single `LanguageModel`,
      - A list of `LanguageModel`s for multi-model fan-out.
      If `None`, the LM from `lf.context` will be used.
    num_samples: Number of samples to generate. If a list is provided, its
      length must match the number of models in `lm`.
    system_message: System instructions to guide the model output. If None,
      no system message will be used.
    examples: Few-shot examples to guide the model output. Defaults to `None`.
    cache_seed: Seed for caching the query. Queries with the same
      `(lm, prompt, cache_seed)` will use cached responses. If `None`,
      caching is disabled.
    response_postprocess: A post-processing function for the raw LM response.
      If `None`, no post-processing occurs.
    autofix: Number of attempts for auto-fixing code errors. Set to `0` to
      disable auto-fixing. Not supported with the `'json'` protocol.
    autofix_lm: The LM to use for auto-fixing. Defaults to the `autofix_lm`
      from `lf.context` or the main `lm`.
    protocol: Format for schema representation. Builtin choices are `'json'` or
      `'python'`, users could extend with their own protocols by subclassing
      `lf.structured.LfQuery'. Also protocol could be specified with a version
      in the format of 'protocol:version', e.g., 'python:1.0', so users could
      use a specific version of the prompt based on the protocol. Please see the
      documentation of `LfQuery` for more details. If None, the protocol from
      context manager `lf.query_protocol` will be used, or 'python' if not
      specified.
    returns_message:  If `True`, returns an `lf.Message` object instead of
      the final parsed result.
    skip_lm: If `True`, skips the LLM call and returns the rendered 
      prompt as a `UserMessage` object.
    invocation_id: The ID of the query invocation, which will be passed to
      `lf.QueryInvocation` when `lf.trackIf `None`, a unique ID will
      be generated.
    **kwargs: Additional keyword arguments for:
      - Rendering templates (e.g., `template_str`, `preamble`),
      - Configuring `lf.structured.Mapping`.

  Returns:
    The result of the query:
    - A single output or a list of outputs if multiple models/samples are used.
    - Each output is a parsed object matching `schema`, an `lf.Message` (if 
      `returns_message=True`), or a natural language string (default).
  """
    # Internal usage logging.

  if protocol is None:
    protocol = lf.context_value('__query_protocol__', 'python')

  def _invocation_id():
    return invocation_id or f'query@{uuid.uuid4().hex[-7:]}'

  # Multiple quries will be issued when `lm` is a list or `num_samples` is
  # greater than 1.
  if isinstance(lm, list) or num_samples != 1:
    def _single_query(inputs):
      i, (lm, example_i) = inputs
      return query(
          prompt,
          schema,
          default=default,
          lm=lm,
          system_message=system_message,
          examples=examples,
          # Usually num_examples should not be large, so we multiple the user
          # provided cache seed by 100 to avoid collision.
          cache_seed=(
              None if cache_seed is None else cache_seed * 100 + example_i
          ),
          response_postprocess=response_postprocess,
          autofix=autofix,
          autofix_lm=autofix_lm,
          protocol=protocol,
          returns_message=returns_message,
          skip_lm=skip_lm,
          invocation_id=f'{_invocation_id()}:{i}',
          **kwargs,
      )
    lm_list = lm if isinstance(lm, list) else [lm]
    num_samples_list = (
        num_samples if isinstance(num_samples, list)
        else [num_samples] * len(lm_list)
    )
    assert len(lm_list) == len(num_samples_list), (
        'Expect the length of `num_samples` to be the same as the '
        f'the length of `lm`. Got {num_samples} and {lm_list}.'
    )
    query_inputs = []
    total_queries = 0
    for lm, num_samples in zip(lm_list, num_samples_list):
      query_inputs.extend([(lm, i) for i in range(num_samples)])
      total_queries += num_samples

    samples = []
    for _, output, error in lf.concurrent_map(
        _single_query, enumerate(query_inputs),
        max_workers=max(64, total_queries),
        ordered=True,
    ):
      if error is None:
        samples.append(output)
    return samples

  # Normalize query schema.
  # When `lf.query` is used for symbolic completion, schema is automatically
  # inferred when it is None.
  if isinstance(prompt, pg.Symbolic) and prompt.sym_partial and schema is None:
    schema = prompt.__class__

  # Attach system message as input template metadata, which will be passed
  # through to the rendered message metadata under key `system_message`.
  if system_message is not None:
    kwargs['metadata_system_message'] = lf.Template.from_value(
        system_message
    ).render(message_cls=lf.SystemMessage)

  # Normalize query input.
  if isinstance(prompt, (lf.Message, str)):
    # Query with structured output.
    prompt_kwargs = kwargs.copy()
    prompt_kwargs.pop('template_str', None)
    query_input = lf.Template.from_value(prompt, **prompt_kwargs)
  elif isinstance(prompt, lf.Template):
    # Create a copy of the prompt if it has a parent object, so all child
    # modality objects could be referred by path relative to the prompt.
    query_input = prompt.clone() if prompt.sym_parent is not None else prompt

    # Attach template metadata from kwargs. This is used to pass through fields
    # from kwargs to the rendered message.
    template_metadata = {
        k: v for k, v in kwargs.items() if k.startswith('metadata_')
    }
    query_input.rebind(
        template_metadata, skip_notification=True, raise_on_no_change=False
    )
  elif pg.MISSING_VALUE == prompt:
    query_input = lf.UserMessage('')
  else:
    query_input = schema_lib.mark_missing(prompt)

  # Determine query class.
  if schema in (None, str):
    # Non-structured query.
    query_cls = None
  else:
    # Query with structured output.
    query_cls = LfQuery.from_protocol(protocol)
    if ':' not in protocol:
      protocol = f'{protocol}:{query_cls.version}'

  # `skip_lm`` is True when `lf.query_prompt` is called.
  # and `prompt` is `pg.MISSING_VALUE` when `lf.query_output` is called.
  # In these cases, we do not track the query invocation.
  if skip_lm or pg.MISSING_VALUE == prompt:
    trackers = []
  else:
    trackers = lf.context_value('__query_trackers__', [])

  # Mark query start with trackers.
  # NOTE: prompt is MISSING_VALUE when `lf.query_output` is called.
  # We do not track the query invocation in this case.
  if trackers:
    invocation = QueryInvocation(
        id=_invocation_id(),
        input=pg.Ref(query_input),
        schema=(
            schema_lib.Schema.from_value(schema)
            if schema not in (None, str) else None
        ),
        default=default,
        lm=pg.Ref(lm),
        examples=pg.Ref(examples) if examples else [],
        protocol=protocol,
        kwargs={k: pg.Ref(v) for k, v in kwargs.items()},
        start_time=time.time(),
    )
    for i, tracker in enumerate(trackers):
      if i == 0 or tracker.include_child_scopes:
        tracker.track(invocation)
  else:
    invocation = None

  def _mark_query_completed(output_message, error, usage_summary):
    # Mark query completion with trackers.
    if not trackers:
      return

    if output_message is not None:
      # To minimize payload for serialization, we remove the result and usage
      # fields from the metadata. They will be computed on the fly when the
      # invocation is rendered.
      metadata = dict(output_message.metadata)
      metadata.pop('result', None)
      metadata.pop('usage', None)
      lm_response = lf.AIMessage(output_message.text, metadata=metadata)
    else:
      lm_response = None

    assert invocation is not None
    invocation.mark_completed(
        lm_response=lm_response, error=error, usage_summary=usage_summary,
    )
    for i, tracker in enumerate(trackers):
      if i == 0 or tracker.include_child_scopes:
        tracker.mark_completed(invocation)

  with lf.track_usages() as usage_summary:
    try:
      if query_cls is None:
        # Query with natural language output.
        output_message = lf.LangFunc.from_value(query_input, **kwargs)(
            lm=lm, cache_seed=cache_seed, skip_lm=skip_lm
        )
        if response_postprocess:
          processed_text = response_postprocess(output_message.text)
          if processed_text != output_message.text:
            output_message = lf.AIMessage(processed_text, source=output_message)
      else:
        # Query with structured output.
        output_message = query_cls(
            input=(
                query_input.render(lm=lm)
                if isinstance(query_input, lf.Template)
                else query_input
            ),
            schema=schema,
            examples=examples,
            response_postprocess=response_postprocess,
            autofix=autofix if protocol.startswith('python:') else 0,
            **kwargs,
        )(
            lm=lm,
            autofix_lm=autofix_lm or lm,
            cache_seed=cache_seed,
            skip_lm=skip_lm,
        )
      _mark_query_completed(output_message, None, usage_summary)
    except mapping.MappingError as e:
      _mark_query_completed(
          e.lm_response, pg.ErrorInfo.from_exception(e), usage_summary
      )
      if lf.RAISE_IF_HAS_ERROR == default:
        raise e
      output_message = e.lm_response
      output_message.result = default
    except BaseException as e:
      _mark_query_completed(
          None, pg.ErrorInfo.from_exception(e), usage_summary
      )
      raise e

  if returns_message:
    return output_message
  return output_message.text if schema in (None, str) else output_message.result


@contextlib.contextmanager
def query_protocol(protocol: str) -> Iterator[None]:
  """Context manager for setting the query protocol for the scope."""
  with lf.context(__query_protocol__=protocol):
    try:
      yield
    finally:
      pass

#
# Helper function for map-reduce style querying.
#


def query_and_reduce(
    prompt: Union[str, lf.Template, Any],
    schema: schema_lib.SchemaType | None = None,
    *,
    reduce: Callable[[list[Any]], Any],
    lm: lf.LanguageModel | list[lf.LanguageModel] | None = None,
    num_samples: int | list[int] = 1,
    **kwargs,
) -> Any:
  """Issues multiple `lf.query` calls in parallel and reduce the outputs.
  
  Args:
    prompt: A str (may contain {{}} as template) as natural language input, or a
      `pg.Symbolic` object as structured input as prompt to LLM.
    schema: A type annotation as the schema for output object. If str (default),
      the response will be a str in natural language.
    reduce: A function to reduce the outputs of multiple `lf.query` calls. It
      takes a list of outputs and returns the final object.
    lm: The language model to use. If not specified, the language model from
      `lf.context` context manager will be used.
    num_samples: The number of samples to obtain from each language model being
      requested. If a list is provided, it should have the same length as `lm`.
    **kwargs: Additional arguments to pass to `lf.query`.

  Returns:
    The reduced output from multiple `lf.query` calls.
  """
  results = query(prompt, schema, lm=lm, num_samples=num_samples, **kwargs)
  if isinstance(results, list):
    results = reduce(results)
  return results


#
# Functions for decomposing `lf.query` into pre-llm and post-llm operations.
#


def query_prompt(
    prompt: Union[str, lf.Template, Any],
    schema: schema_lib.SchemaType | None = None,
    **kwargs,
) -> lf.Message:
  """Returns the final prompt sent to LLM for `lf.query`."""
  kwargs.pop('returns_message', None)
  kwargs.pop('skip_lm', None)
  return query(prompt, schema, skip_lm=True, returns_message=True, **kwargs)


def query_output(
    response: Union[str, lf.Message],
    schema: schema_lib.SchemaType | None = None,
    **kwargs,
) -> Any:
  """Returns the final output of `lf.query` from a provided LLM response."""
  # Delay import to avoid circular dependency in Colab.
  # llms > data/conversion > structured > querying
  from langfun.core.llms import fake  # pylint: disable=g-import-not-at-top

  kwargs.pop('prompt', None)
  kwargs.pop('lm', None)
  return query(
      pg.MISSING_VALUE, schema, lm=fake.StaticResponse(response), **kwargs
  )


#
# Functions for computing reward of an LLM response based on a mapping example.
#


def query_reward(
    mapping_example: Union[str, mapping.MappingExample],
    response: Union[str, lf.Message],
) -> float | None:
  """Returns the reward of an LLM response based on an mapping example."""
  if isinstance(mapping_example, str):
    mapping_example = pg.from_json_str(mapping_example)
    assert isinstance(mapping_example, mapping.MappingExample), mapping_example
  schema = mapping_example.schema

  if schema and isinstance(schema.spec, pg.typing.Object):
    output_cls = schema.spec.cls
  elif schema is None and isinstance(mapping_example.output, pg.Object):
    output_cls = mapping_example.output.__class__
  else:
    output_cls = None

  reward_fn = _reward_fn(output_cls)
  if reward_fn is None:
    return None

  return reward_fn(
      query_output(response, output_cls),
      mapping_example.input,
      mapping_example.output,
      mapping_example.metadata,
  )


@functools.cache
def _reward_fn(cls) -> Callable[
    [
        pg.Object,    # Actual output object.
        Any,          # Input object.
        pg.Object,    # Expected output object.
        pg.Dict       # User metadata.
    ], float] | None:
  """Returns the reward function for a class that is being queried."""
  if not callable(getattr(cls, '__reward__', None)):
    return None

  signature = pg.typing.signature(cls.__reward__)
  num_args = len(signature.args)
  if num_args < 2 or num_args > 4:
    raise TypeError(
        f'`{cls.__type_name__}.__reward__` should have signature: '
        '`__reward__(self, input, [expected_output], [expected_metadata])`.'
    )
  def _reward(self, input, expected_output, metadata):  # pylint: disable=redefined-builtin
    args = [self, input, expected_output, metadata]
    return cls.__reward__(*args[:num_args])
  return _reward


#
# Functions for tracking `lf.query` invocations.
#


class QueryInvocation(pg.Object, pg.views.HtmlTreeView.Extension):
  """A class to represent the invocation of `lf.query`."""

  #
  # Query input.
  #

  id: Annotated[
      str,
      'The ID of the query invocation.'
  ]

  input: Annotated[
      Union[lf.Template, pg.Symbolic],
      'Mapping input of `lf.query`.'
  ]

  schema: pg.typing.Annotated[
      schema_lib.schema_spec(noneable=True),
      'Schema of `lf.query`.'
  ]

  default: Annotated[
      Any,
      'Default value of `lf.query`.'
  ] = lf.RAISE_IF_HAS_ERROR

  lm: Annotated[
      lf.LanguageModel,
      'Language model used for `lf.query`.'
  ]

  examples: Annotated[
      list[mapping.MappingExample],
      'Fewshot exemplars for `lf.query`.'
  ]

  protocol: Annotated[
      str,
      'Protocol of `lf.query`.'
  ] = 'python'

  kwargs: Annotated[
      dict[str, Any],
      'Kwargs of `lf.query`.'
  ] = {}

  #
  # Query output.
  #

  lm_response: Annotated[
      lf.Message | None,
      'Raw LM response. If None, query is not completed yet or failed.'
  ] = None

  error: Annotated[
      pg.ErrorInfo | None,
      'Error info if the query failed.'
  ] = None

  #
  # Execution details.
  #

  start_time: Annotated[
      float,
      'Start time of query.'
  ]

  end_time: Annotated[
      float | None,
      'End time of query. If None, query is not completed yet.'
  ] = None

  usage_summary: Annotated[
      lf.UsageSummary,
      'Usage summary of the query.'
  ] = lf.UsageSummary()

  @functools.cached_property
  def lm_request(self) -> lf.Message:
    return query_prompt(
        self.input, self.schema, examples=self.examples or None,
        protocol=self.protocol,
        **self.kwargs
    )

  @property
  def output(self) -> Any:
    """The output of `lf.query`. If it failed, returns None."""
    return self._output

  @property
  def has_error(self) -> bool:
    """Returns True if the query failed to generate a valid output."""
    return self.error is not None

  @property
  def has_oop_error(self) -> bool:
    """Returns True if the query failed due to out of memory error."""
    return self.error is not None and self.error.tag.startswith('MappingError')

  @property
  def elapse(self) -> float:
    """Returns query elapse in seconds."""
    if self.end_time is None:
      return time.time() - self.start_time
    return self.end_time - self.start_time

  def as_mapping_example(
      self,
      metadata: dict[str, Any] | None = None
  ) -> mapping.MappingExample:
    """Returns a `MappingExample` object for this query invocation."""
    return mapping.MappingExample(
        input=self.input,
        schema=self.schema,
        output=self.lm_response.text if self.has_oop_error else self.output,
        metadata=metadata or {},
    )

  def _on_bound(self):
    super()._on_bound()
    self._tab_control = None
    self._output = None
    self.__dict__.pop('lm_request', None)

  @property
  def is_completed(self) -> bool:
    """Returns True if the query is completed."""
    return self.end_time is not None

  def mark_completed(
      self,
      lm_response: lf.Message | None,
      error: pg.ErrorInfo | None = None,
      usage_summary: lf.UsageSummary | None = None) -> None:
    """Marks the query as completed."""
    assert self.end_time is None, 'Query is already completed.'

    if error is None:
      # Autofix could lead to a successful `lf.query`, however, the initial
      # lm_response may not be valid. When Error is None, we always try to parse
      # the lm_response into the output. If the output is not valid, the error
      # will be updated accordingly. This logic could be optimized in future by
      # returning attempt information when autofix is enabled.
      if self.schema is not None:
        try:
          output = query_output(
              lm_response, self.schema,
              default=self.default, protocol=self.protocol
          )
        except mapping.MappingError as e:
          output = None
          error = pg.ErrorInfo.from_exception(e)
        self._output = output
      else:
        assert lm_response is not None
        self._output = lm_response.text
    elif (error.tag.startswith('MappingError')
          and self.default != lf.RAISE_IF_HAS_ERROR):
      self._output = self.default

    self.rebind(
        lm_response=lm_response,
        error=error,
        end_time=time.time(),
        skip_notification=True,
    )
    if usage_summary is not None:
      self.usage_summary.merge(usage_summary)

    # Refresh the tab control.
    if self._tab_control is None:
      return

    self._tab_control.insert(
        'schema',
        pg.views.html.controls.Tab(   # pylint: disable=g-long-ternary
            'output',
            pg.view(self.output, collapse_level=None),
            name='output',
        ),
    )
    if self.error is not None:
      self._tab_control.insert(
          'schema',
          pg.views.html.controls.Tab(
              'error',
              pg.view(self.error, collapse_level=None),
              name='error',
          )
      )
    if self.lm_response is not None:
      self._tab_control.append(
          pg.views.html.controls.Tab(
              'lm_response',
              pg.view(
                  self.lm_response,
                  extra_flags=dict(include_message_metadata=True)
              ),
              name='lm_response',
          )
      )
    self._tab_control.select(['error', 'output', 'lm_response'])

  def _html_tree_view_summary(
      self,
      *,
      view: pg.views.HtmlTreeView,
      **kwargs: Any
  ) -> pg.Html | None:
    kwargs.pop('title', None)
    kwargs.pop('enable_summary_tooltip', None)
    return view.summary(
        value=self,
        title=pg.Html.element(
            'div',
            [
                pg.views.html.controls.Label(
                    'lf.query',
                    tooltip=f'[{self.id}] Query invocation',
                    css_classes=['query-invocation-type-name']
                ),
                pg.views.html.controls.Badge(
                    f'lm={self.lm.model_id}',
                    pg.format(
                        self.lm,
                        verbose=False,
                        python_format=True,
                        hide_default_values=True
                    ),
                    css_classes=['query-invocation-lm']
                ),
                pg.views.html.controls.Badge(
                    f'{int(self.elapse)} seconds',
                    css_classes=['query-invocation-time']
                ),
                self.usage_summary.to_html(
                    extra_flags=dict(as_badge=True)
                ),
            ],
            css_classes=['query-invocation-title']
        ),
        enable_summary_tooltip=False,
        **kwargs
    )

  def _html_tree_view_content(
      self,
      *,
      view: pg.views.HtmlTreeView,
      extra_flags: dict[str, Any] | None = None,
      **kwargs: Any
  ) -> pg.Html:
    extra_flags = extra_flags or {}
    interactive = extra_flags.get('interactive', True)
    tab_control = pg.views.html.controls.TabControl([
        pg.views.html.controls.Tab(
            'input',
            pg.view(self.input, collapse_level=None),
            name='input',
        ),
        pg.views.html.controls.Tab(   # pylint: disable=g-long-ternary
            'output',
            pg.view(self.output, collapse_level=None),
            name='output',
        ) if self.is_completed else None,
        pg.views.html.controls.Tab(   # pylint: disable=g-long-ternary
            'error',
            pg.view(self.error, collapse_level=None),
            name='error',
        ) if self.has_error else None,
        pg.views.html.controls.Tab(
            'schema',
            pg.view(self.schema),
            name='schema',
        ),
        pg.views.html.controls.Tab(
            'lm_request',
            pg.view(
                self.lm_request,
                extra_flags=dict(include_message_metadata=False),
            ),
            name='lm_request',
        ),
        pg.views.html.controls.Tab(  # pylint: disable=g-long-ternary
            'lm_response',
            pg.view(
                self.lm_response,
                extra_flags=dict(include_message_metadata=True)
            ),
            name='lm_response',
        ) if self.is_completed else None,
    ], tab_position='top', selected=1)
    if interactive:
      self._tab_control = tab_control
    return tab_control.to_html(extra_flags=extra_flags)

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .query-invocation-title {
          display: inline-block;
          font-weight: normal;
        }
        .query-invocation-type-name {
          color: #888;
        }
        .query-invocation-lm.badge {
          margin-left: 5px;
          margin-right: 5px;
          color: white;
          background-color: mediumslateblue;
        }
        .query-invocation-time.badge {
          margin-left: 5px;
          border-radius: 0px;
          font-weight: bold;
          background-color: aliceblue;
        }
        .query-invocation-title .usage-summary.label {
          border-radius: 0px;
          color: #AAA;
        }
        """
    ]


@dataclasses.dataclass
class _QueryTracker:
  """Query tracker for `track_queries`."""

  include_child_scopes: Annotated[
      bool,
      (
          'If True, the queries made in nested `track_queries` contexts will '
          'be tracked by this tracker. Otherwise, only the queries made in the '
          'current scope will be included.'
      )
  ] = True

  start_callabck: Annotated[
      Callable[[QueryInvocation], None] | None,
      (
          'A callback function to be called when a query is started.'
      )
  ] = None

  end_callabck: Annotated[
      Callable[[QueryInvocation], None] | None,
      (
          'A callback function to be called when a query is completed.'
      )
  ] = None

  tracked_queries: Annotated[
      list[QueryInvocation],
      (
          'The list of queries tracked by this tracker.'
      )
  ] = dataclasses.field(default_factory=list)

  def track(self, invocation: QueryInvocation) -> None:
    self.tracked_queries.append(invocation)
    if self.start_callabck is not None:
      self.start_callabck(invocation)

  def mark_completed(self, invocation: QueryInvocation) -> None:
    assert invocation in self.tracked_queries, invocation
    if self.end_callabck is not None:
      self.end_callabck(invocation)


@contextlib.contextmanager
def track_queries(
    include_child_scopes: bool = True,
    *,
    start_callabck: Callable[[QueryInvocation], None] | None = None,
    end_callabck: Callable[[QueryInvocation], None] | None = None,
) -> Iterator[list[QueryInvocation]]:
  """Track all queries made during the context.

  Example:

    ```
    with lf.track_queries() as queries:
      lf.query('hi', lm=lm)
      lf.query('What is this {{image}}?', lm=lm, image=image)

    print(queries)
    ```

  Args:
    include_child_scopes: If True, the queries made in child scopes will be
      included in the returned list. Otherwise, only the queries made in the
      current scope will be included.
    start_callabck: A callback function to be called when a query is started.
    end_callabck: A callback function to be called when a query is completed.

  Yields:
    A list of `QueryInvocation` objects representing the queries made during
    the context.
  """
  trackers = lf.context_value('__query_trackers__', [])
  tracker = _QueryTracker(
      include_child_scopes=include_child_scopes,
      start_callabck=start_callabck,
      end_callabck=end_callabck
  )

  with lf.context(
      __query_trackers__=[tracker] + trackers
  ):
    try:
      yield tracker.tracked_queries
    finally:
      pass
