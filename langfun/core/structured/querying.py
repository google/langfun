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
import functools
import time
from typing import Annotated, Any, Callable, Iterator, Type, Union

import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


@lf.use_init_args(['schema', 'default', 'examples'])
class _QueryStructure(mapping.Mapping):
  """Query an object out from a natural language text."""

  context_title = 'CONTEXT'
  input_title = 'INPUT_OBJECT'

  # Mark schema as required.
  schema: pg.typing.Annotated[
      schema_lib.schema_spec(), 'Required schema for parsing.'
  ]


class _QueryStructureJson(_QueryStructure):
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

  protocol = 'json'
  schema_title = 'SCHEMA'
  output_title = 'JSON'


class _QueryStructurePython(_QueryStructure):
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
  protocol = 'python'
  schema_title = 'OUTPUT_TYPE'
  output_title = 'OUTPUT_OBJECT'


def _query_structure_cls(
    protocol: schema_lib.SchemaProtocol,
) -> Type[_QueryStructure]:
  if protocol == 'json':
    return _QueryStructureJson
  elif protocol == 'python':
    return _QueryStructurePython
  else:
    raise ValueError(f'Unknown protocol: {protocol!r}.')


def query(
    prompt: Union[str, lf.Template, Any],
    schema: schema_lib.SchemaType |  None = None,
    default: Any = lf.RAISE_IF_HAS_ERROR,
    *,
    lm: lf.LanguageModel | list[lf.LanguageModel] | None = None,
    num_samples: int | list[int] = 1,
    examples: list[mapping.MappingExample] | None = None,
    cache_seed: int | None = 0,
    response_postprocess: Callable[[str], str] | None = None,
    autofix: int = 0,
    autofix_lm: lf.LanguageModel | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    returns_message: bool = False,
    skip_lm: bool = False,
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
    protocol: Format for schema representation. Choices are `'json'` or
      `'python'`. Default is `'python'`.
    returns_message:  If `True`, returns an `lf.Message` object instead of
      the final parsed result.
    skip_lm: If `True`, skips the LLM call and returns the rendered 
      prompt as a `UserMessage` object.
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

  # Multiple quries will be issued when `lm` is a list or `num_samples` is
  # greater than 1.
  if isinstance(lm, list) or num_samples != 1:
    def _single_query(inputs):
      lm, example_i = inputs
      return query(
          prompt,
          schema,
          default=default,
          lm=lm,
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
        _single_query, query_inputs, max_workers=max(64, total_queries),
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

  with lf.track_usages() as usage_summary:
    start_time = time.time()
    if schema in (None, str):
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
      output_message = _query_structure_cls(protocol)(
          input=(
              query_input.render(lm=lm)
              if isinstance(query_input, lf.Template)
              else query_input
          ),
          schema=schema,
          default=default,
          examples=examples,
          response_postprocess=response_postprocess,
          autofix=autofix if protocol == 'python' else 0,
          **kwargs,
      )(
          lm=lm,
          autofix_lm=autofix_lm or lm,
          cache_seed=cache_seed,
          skip_lm=skip_lm,
      )
    end_time = time.time()

  def _result(message: lf.Message):
    return message.text if schema in (None, str) else message.result

  # Track the query invocations.
  if pg.MISSING_VALUE != prompt and not skip_lm:
    trackers = lf.context_value('__query_trackers__', [])
    if trackers:
      # To minimize payload for serialization, we remove the result and usage
      # fields from the metadata. They will be computed on the fly when the
      # invocation is rendered.
      metadata = dict(output_message.metadata)
      metadata.pop('result', None)
      metadata.pop('usage', None)

      invocation = QueryInvocation(
          input=pg.Ref(query_input),
          schema=(
              schema_lib.Schema.from_value(schema)
              if schema not in (None, str) else None
          ),
          lm=pg.Ref(lm),
          examples=pg.Ref(examples) if examples else [],
          lm_response=lf.AIMessage(output_message.text, metadata=metadata),
          usage_summary=usage_summary,
          start_time=start_time,
          end_time=end_time,
      )
      for i, (tracker, include_child_scopes) in enumerate(trackers):
        if i == 0 or include_child_scopes:
          tracker.append(invocation)
  return output_message if returns_message else _result(output_message)


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

  input: Annotated[
      Union[lf.Template, pg.Symbolic],
      'Mapping input of `lf.query`.'
  ]
  schema: pg.typing.Annotated[
      schema_lib.schema_spec(noneable=True),
      'Schema of `lf.query`.'
  ]
  lm_response: Annotated[
      lf.Message,
      'Raw LM response.'
  ]
  lm: Annotated[
      lf.LanguageModel,
      'Language model used for `lf.query`.'
  ]
  examples: Annotated[
      list[mapping.MappingExample],
      'Fewshot exemplars for `lf.query`.'
  ]
  usage_summary: Annotated[
      lf.UsageSummary,
      'Usage summary for `lf.query`.'
  ]
  start_time: Annotated[
      float,
      'Start time of query.'
  ]
  end_time: Annotated[
      float,
      'End time of query.'
  ]

  @functools.cached_property
  def lm_request(self) -> lf.Message:
    return query_prompt(self.input, self.schema, examples=self.examples or None)

  @functools.cached_property
  def output(self) -> Any:
    """The output of `lf.query`. If it failed, returns the `MappingError`."""
    try:
      return query_output(self.lm_response, self.schema)
    except mapping.MappingError as e:
      return e

  @property
  def has_error(self) -> bool:
    """Returns True if the query failed to generate a valid output."""
    return isinstance(self.output, BaseException)

  @property
  def elapse(self) -> float:
    """Returns query elapse in seconds."""
    return self.end_time - self.start_time

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('lm_request', None)
    self.__dict__.pop('output', None)

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
                self.usage_summary.to_html(extra_flags=dict(as_badge=True))
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
      **kwargs: Any
  ) -> pg.Html:
    return pg.views.html.controls.TabControl([
        pg.views.html.controls.Tab(
            'input',
            pg.view(self.input, collapse_level=None),
        ),
        pg.views.html.controls.Tab(
            'output',
            pg.view(self.output, collapse_level=None),
        ),
        pg.views.html.controls.Tab(
            'schema',
            pg.view(self.schema),
        ),
        pg.views.html.controls.Tab(
            'lm_request',
            pg.view(
                self.lm_request,
                extra_flags=dict(include_message_metadata=False),
            ),
        ),
        pg.views.html.controls.Tab(
            'lm_response',
            pg.view(
                self.lm_response,
                extra_flags=dict(include_message_metadata=False)
            ),
        ),
    ], tab_position='top', selected=1).to_html()

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


@contextlib.contextmanager
def track_queries(
    include_child_scopes: bool = True
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

  Yields:
    A list of `QueryInvocation` objects representing the queries made during
    the context.
  """
  trackers = lf.context_value('__query_trackers__', [])
  tracker = []

  with lf.context(
      __query_trackers__=[(tracker, include_child_scopes)] + trackers
  ):
    try:
      yield tracker
    finally:
      pass
