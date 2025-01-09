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
"""The base of symbolic mapping methods."""

import functools
import io
from typing import Annotated, Any, Callable
import langfun.core as lf
from langfun.core.structured import schema as schema_lib
import pyglove as pg


class MappingError(Exception):  # pylint: disable=g-bad-exception-name
  """Mapping error."""

  def __init__(self, lm_response: lf.Message, cause: Exception):
    self._lm_response = lm_response
    self._cause = cause

  @property
  def lm_response(self) -> lf.Message:
    """Returns the LM response that failed to be mapped."""
    return self._lm_response

  @property
  def cause(self) -> Exception:
    """Returns the cause of the error."""
    return self._cause

  def __str__(self) -> str:
    return self.format(include_lm_response=True)

  def format(self, include_lm_response: bool = True) -> str:
    """Formats the mapping error."""
    r = io.StringIO()
    error_message = str(self.cause).rstrip()
    r.write(
        pg.colored(
            f'{self.cause.__class__.__name__}: {error_message}', 'magenta'
        )
    )
    if include_lm_response:
      r.write('\n\n')
      r.write(pg.colored('[LM Response]', 'blue', styles=['bold']))
      r.write('\n')
      r.write(pg.colored(self.lm_response.text, 'blue'))
    return r.getvalue()


@pg.use_init_args(['input', 'output', 'schema', 'context'])
class MappingExample(lf.NaturalLanguageFormattable,
                     lf.Component,
                     pg.views.HtmlTreeView.Extension):
  """Mapping example between text, schema and structured value."""

  input: pg.typing.Annotated[
      pg.typing.Any(transform=schema_lib.mark_missing),
      (
          'The input object of the mapping. It could be either a natural '
          'language-based string, or a Python object.'
      ),
  ]

  output: Annotated[
      Any,
      (
          'The output object of the mapping. It could be either a natural '
          'language-based string, or a Python object.'
      ),
  ] = schema_lib.MISSING

  schema: pg.typing.Annotated[
      # Automatic conversion from annotation to schema.
      schema_lib.schema_spec(noneable=True),
      (
          'A `lf.structured.Schema` object that constrains target value '
          'If None, the target is expected to be a natural language-based '
          'response returned from LMs.'
      ),
  ] = lf.contextual(default=None)

  context: Annotated[
      str | None,
      'The natural language context for this mapping. ',
  ] = None

  metadata: Annotated[
      dict[str, Any],
      (
          'The metadata associated with the mapping example, '
          'which chould carry structured data, such as tool function input. '
          'It is a `pg.Dict` object whose keys can be accessed by attributes.'
      ),
  ] = pg.Dict()

  def schema_repr(
      self, protocol: schema_lib.SchemaProtocol = 'python', **kwargs
  ) -> str:
    """Returns the string representation of schema based on protocol."""
    if self.schema is None:
      return ''
    return self.schema.schema_str(protocol, **kwargs)

  @property
  def has_output(self) -> bool:
    """Returns True if the mapping output is present."""
    return self.output != schema_lib.MISSING

  @classmethod
  def value_repr(
      cls,
      value: Any,
      protocol: schema_lib.SchemaProtocol = 'python',
      use_modality_ref: bool = False,
      **kwargs
  ) -> str:
    if isinstance(value, str):
      return value
    if isinstance(value, lf.Modality):
      with lf.modality.format_modality_as_ref():
        return str(value)

    # Placehold modalities if they are present.
    if use_modality_ref and pg.contains(value, type=lf.Modality):
      value = lf.ModalityRef.placehold(value)
    return schema_lib.value_repr(protocol).repr(value, **kwargs)

  def input_repr(
      self,
      protocol: schema_lib.SchemaProtocol = 'python',
      compact: bool = False,
      verbose: bool = True,
      **kwargs
  ) -> str:
    """Returns the string representation of the input object."""
    return self.value_repr(
        self.input, protocol, compact=compact, verbose=verbose, **kwargs
    )

  def output_repr(
      self,
      protocol: schema_lib.SchemaProtocol = 'python',
      compact: bool = False,
      verbose: bool = True,
      **kwargs
  ) -> str:
    """Returns the string representation of the output object."""
    return self.value_repr(
        self.output, protocol, compact=compact, verbose=verbose, **kwargs
    )

  def natural_language_format(self) -> str:
    result = io.StringIO()
    if self.context:
      result.write(pg.colored('[CONTEXT]\n', styles=['bold']))
      result.write(pg.colored(self.context, color='magenta'))
      result.write('\n\n')

    result.write(pg.colored('[INPUT]\n', styles=['bold']))
    result.write(pg.colored(self.input_repr(), color='green'))

    if self.schema is not None:
      result.write('\n\n')
      result.write(pg.colored('[SCHEMA]\n', styles=['bold']))
      result.write(pg.colored(self.schema_repr(), color='red'))

    if schema_lib.MISSING != self.output:
      result.write('\n\n')
      result.write(pg.colored('[OUTPUT]\n', styles=['bold']))
      result.write(pg.colored(self.output_repr(), color='blue'))

    if self.metadata:
      result.write('\n\n')
      result.write(pg.colored('[METADATA]\n', styles=['bold']))
      result.write(pg.colored(str(self.metadata), color='cyan'))
    return result.getvalue().strip()

  @classmethod
  @functools.cache
  def _html_tree_view_config(cls) -> dict[str, Any]:

    def render_value(view, *, value, **kwargs):
      if isinstance(value, lf.Template):
        # Make a shallow copy to make sure modalities are rooted by
        # the input.
        value = value.clone().render()
      if value is None:
        return None
      return view.render(value, **kwargs)

    return pg.views.HtmlTreeView.get_kwargs(
        super()._html_tree_view_config(),
        dict(
            include_keys=['input', 'output', 'context', 'schema', 'metadata'],
            extra_flags=dict(
                render_value_fn=render_value,
            ),
            child_config=dict(
                input=dict(
                    collapse_level=1,
                ),
                output=dict(
                    css_classes=['lf-example-output'],
                    collapse_level=1,
                ),
                schema=dict(
                    css_classes=['lf-example-schema'],
                    collapse_level=1,
                ),
                metadata=dict(
                    css_classes=['lf-example-metadata'],
                    collapse_level=1,
                ),
            ),
        )
    )

  @classmethod
  @functools.cache
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .lf-example-output {
            color: dodgerblue;
        }
        .lf-example-schema {
            color: blue;
        }
        """
    ]


class Mapping(lf.LangFunc):
  """Base class for mapping.

  {{ preamble }}

  {% if examples -%}
  {% for example in examples -%}
  {{ mapping_template.render(example=example) }}

  {% endfor %}
  {% endif -%}
  {{ mapping_template.render(example=mapping_request) }}
  """

  #
  # Input for mapping.
  #

  input: Annotated[
      pg.Symbolic,
      (
          'The mapping input. It could be `lf.Message` (a pg.Symbolic '
          'subclass) as natural language input, or other symbolic object '
          'as structured input.'
      ),
  ]

  context: Annotated[
      str | None, 'The mapping context. A string as natural language '
  ] = None

  schema: pg.typing.Annotated[
      # Automatic conversion from annotation to schema.
      schema_lib.schema_spec(noneable=True),
      'A `lf.structured.Schema` object that constrains mapping output ',
  ] = None

  @property
  def mapping_request(self) -> MappingExample:
    """Returns a MappingExample as the mapping request."""
    if isinstance(self.input, lf.Message):
      input_value = self.input.text
    else:
      input_value = pg.Ref(self.input)
    return MappingExample(
        input=input_value,
        schema=pg.Ref(self.schema),
        context=self.context,
    )

  #
  # Customizable in child classes.
  #

  preamble: Annotated[
      lf.Template,
      'Preamble used as mapping instructions.',
  ]

  mapping_template: Annotated[
      lf.Template | None,
      (
          'Template for demonstrating current mapping based on a '
          'MappingExample object. When the output of the mapping example is '
          'absent, the demonstration represents a mapping request.'
      ),
  ] = lf.Template(
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

  input_title: Annotated[str, 'The section title for input.'] = 'INPUT'

  output_title: Annotated[str, 'The section title for output.'] = 'OUTPUT'

  context_title: Annotated[str, 'The section title for context.'] = 'CONTEXT'

  schema_title: Annotated[str, 'The section title for schema.'] = 'SCHEMA'

  protocol: Annotated[
      schema_lib.SchemaProtocol,
      'The protocol for representing the schema and value.',
  ] = 'python'

  #
  # Other user-provided flags.
  #

  examples: Annotated[
      list[MappingExample] | None,
      'Fewshot examples for improving the quality of mapping.',
  ] = lf.contextual(default=None)

  autofix: Annotated[
      int,
      (
          'Max attempts for LLM-based code correction. '
          'If 0 (default), there is no automatic correction. '
          'This flag is effective only when the output needs to be structured.'
      ),
  ] = 0

  autofix_lm: Annotated[
      lf.LanguageModel,
      (
          'Language model used for code correction. '
          'If None, `lm` will be used. This flag is effective only when the '
          'output needs to be structured.'
      ),
  ] = lf.contextual(default=None)

  default: Annotated[
      Any,
      (
          'The default value to use if the LM response is not a valid code '
          'based on the schema (after autofix). '
          'If unspecified, error will be raisen.'
      ),
  ] = lf.RAISE_IF_HAS_ERROR

  response_postprocess: Annotated[
      Callable[[str], str] | None,
      (
          'A callable object that post process the raw LLM response before '
          'parsing it into the output Python object.'
      )
  ] = None

  #
  # Key methods for implementing specific mappings.
  #

  def transform_input(self, lm_input: lf.Message) -> lf.Message:
    # Find modalities to fill the input message.
    lm_input.metadata.update(
        examples=pg.Ref(self.examples),
        input=pg.Ref(self.input),
        schema=pg.Ref(self.schema) if self.schema is not None else None,
    )
    if isinstance(self.input, lf.Message):
      lm_input.source = self.input
    return lm_input

  def transform_output(self, lm_output: lf.Message) -> lf.Message:
    """Transforms LM response into structure if schema is present."""
    try:
      lm_output = self.postprocess_response(lm_output)
      lm_output.result = self.postprocess_result(self.parse_result(lm_output))
    except Exception as e:  # pylint: disable=broad-exception-caught
      if (self.lm.cache is not None
          and lm_output.lm_input.cache_seed is not None):
        success = self.lm.cache.delete(
            self.lm, lm_output.lm_input, lm_output.lm_input.cache_seed
        )
        assert success
      if self.default == lf.RAISE_IF_HAS_ERROR:
        raise MappingError(lm_output, e) from e
      lm_output.result = self.default
    return lm_output

  def parse_result(self, lm_output: lf.Message) -> Any:
    """Parse result from LLM response."""
    schema = self.mapping_request.schema
    if schema is None:
      return None
    return schema.parse(
        lm_output.text,
        protocol=self.protocol,
        additional_context=self.globals(),
        autofix=self.autofix,
        autofix_lm=self.autofix_lm or self.lm,
    )

  def postprocess_response(self, response: lf.Message) -> lf.Message:
    """Post process LLM response."""
    if self.response_postprocess is not None:
      postprocessed_text = self.response_postprocess(response.text)
      if postprocessed_text != response.text:
        return lf.AIMessage(postprocessed_text, source=response)
    return response

  def postprocess_result(self, result: Any) -> Any:
    """Post process structured output."""
    return result

  def globals(self) -> dict[str, Any]:
    """Gets additional symbol definitions besides schema as globals."""
    return {'ModalityRef': lf.modality.ModalityRef}

