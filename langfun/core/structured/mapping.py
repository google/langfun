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
"""Mapping interfaces."""

import abc
import io
from typing import Annotated, Any, Type
import langfun.core as lf
from langfun.core.structured import schema as schema_lib
import pyglove as pg


# NOTE(daiyip): We put `schema` at last as it could inherit from the parent
# objects.
@pg.use_init_args(['nl_context', 'nl_text', 'value', 'schema'])
class MappingExample(lf.NaturalLanguageFormattable, lf.Component):
  """Mapping example between text, schema and structured value."""

  nl_context: Annotated[
      str | None,
      (
          'The natural language user prompt. It is either used for directly '
          'prompting the LM to generate `schema`/`value`, or used as the '
          'context information when LM outputs the natural language '
          'representations of `schema`/`value` via `nl_text`. For the former '
          'case, it is optional.'
      ),
  ] = None

  nl_text: Annotated[
      str | None,
      (
          'The natural language representation of the object or schema. '
          '`nl_text` is the representation for `value` if `value` is not None; '
          'Otherwise it is the representation for `schema`. '
          'When it is None, LM directly maps the `nl_context` to `schema` / '
          '`value`. In such case, the natural language representation of '
          '`schema` or `value` is not present. '
          '`nl_text` is used as input when we map natural language to value '
          'or to generate the schema. Also it could be the output when we '
          'map the `schema` and `value` back to natural language.'
      )
  ] = None

  schema: pg.typing.Annotated[
      # Automatic conversion from annotation to schema.
      schema_lib.schema_spec(noneable=True),
      (
          'A `lf.structured.Schema` object that constrains the structured '
          'value. It could be used as input when we map `nl_context`/`nl_text` '
          'to `value`, or could be used as output when we want to directly '
          'extract `schema` from `nl_context`/`nl_text`, in which case `value` '
          "will be None. A mapping's schema could be None when we map a "
          '`value` to natural language.'
      ),
  ] = lf.contextual(default=None)

  value: pg.typing.Annotated[
      pg.typing.Any(transform=schema_lib.mark_missing),
      (
          'The structured representation for `nl_text` (or directly prompted '
          'from `nl_context`). It should align with schema.'
          '`value` could be used as input when we map a structured value to '
          'natural language, or as output when we map it reversely.'
      ),
  ] = schema_lib.MISSING

  def schema_str(
      self, protocol: schema_lib.SchemaProtocol = 'json', **kwargs
  ) -> str:
    """Returns the string representation of schema based on protocol."""
    if self.schema is None:
      return ''
    return self.schema.schema_str(protocol, **kwargs)

  @property
  def has_value(self) -> bool:
    """Returns True if structured value is provided."""
    return self.value != schema_lib.MISSING

  def value_str(
      self, protocol: schema_lib.SchemaProtocol = 'json', **kwargs
  ) -> str:
    """Returns the string representation of value based on protocol."""
    return schema_lib.value_repr(
        protocol).repr(self.value, self.schema, **kwargs)

  def natural_language_format(self) -> str:
    result = io.StringIO()
    if self.nl_context:
      result.write(lf.colored('[CONTEXT]\n', styles=['bold']))
      result.write(lf.colored(self.nl_context, color='magenta'))
      result.write('\n\n')

    if self.nl_text:
      result.write(lf.colored('[TEXT]\n', styles=['bold']))
      result.write(lf.colored(self.nl_text, color='green'))
      result.write('\n\n')

    if self.schema is not None:
      result.write(lf.colored('[SCHEMA]\n', styles=['bold']))
      result.write(lf.colored(self.schema_str(), color='red'))
      result.write('\n\n')

    if schema_lib.MISSING != self.value:
      result.write(lf.colored('[VALUE]\n', styles=['bold']))
      result.write(lf.colored(self.value_str(), color='blue'))
    return result.getvalue().strip()


class Mapping(lf.LangFunc):
  """Base class for mapping.

  {{ preamble }}

  {% if examples -%}
  {% for example in examples -%}
  {{ demo_mapping(example) }}

  {% endfor %}
  {% endif -%}
  {{ demo_mapping(mapping_request) }}
  """

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
  ] = None

  examples: Annotated[
      list[MappingExample] | None,
      'Fewshot examples for improving the quality of mapping.',
  ] = lf.contextual(default=None)

  protocol: Annotated[
      schema_lib.SchemaProtocol,
      'The protocol for representing the schema and value.',
  ] = 'python'

  autofix: Annotated[
      int,
      (
          'Max attempts for LLM-based code correction. '
          'If 0 (default), there is no automatic correction. '
          'This flag is effective only when the output needs to be structured.'
      ),
  ] = 3

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

  @property
  @abc.abstractmethod
  def mapping_request(self) -> MappingExample:
    """Returns a MappingExample as the mapping request."""

  def demo_mapping(self, example: MappingExample) -> str:
    """Demonstrates mapping based on the example."""
    assert self.mapping_template is not None
    return self.mapping_template.render(example=example)

  def globals(self) -> dict[str, Any]:
    """Gets additional symbol definitions besides schema as globals."""
    return {'ModalityRef': lf.modality.ModalityRef}

  def postprocess_result(self, result: Any) -> Any:
    """Post process structured output."""
    return result

  def transform_output(self, lm_output: lf.Message) -> lf.Message:
    """Transforms LM response into structure if schema is present."""
    schema = self.mapping_request.schema
    if schema is None:
      return lm_output

    try:
      result = schema.parse(
          lm_output.text,
          protocol=self.protocol,
          additional_context=self.globals(),
          autofix=self.autofix,
          autofix_lm=self.autofix_lm or self.lm,
      )
      lm_output.result = self.postprocess_result(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
      if self.default == lf.RAISE_IF_HAS_ERROR:
        raise e
      lm_output.result = self.default
    return lm_output

  #
  # Helper methods.
  #

  def value_str(self, value: Any, markdown: bool = True) -> str:
    if self.has_modalities(value):
      value = lf.ModalityRef.placehold(value)
    return schema_lib.value_repr('python').repr(
        value, compact=False, verbose=True, markdown=markdown
    )

  def has_modalities(self, value: Any) -> bool:
    """Returns true if the value has modalities."""
    return pg.contains(value, type=lf.Modality)

  def modalities(
      self, value: Any, root_path: pg.KeyPath | None = None
  ) -> dict[str, lf.Modality]:
    return lf.Modality.from_value(value, root_path)

  def modality_refs_str(self, value: Any) -> str:
    with lf.modality.format_modality_as_ref(True):
      return pg.format(
          self.modalities(value),
          compact=False,
          verbose=False,
          python_format=True,
      )

  def missing_type_dependencies(self, value: Any) -> list[Type[Any]]:
    value_specs = tuple(
        [v.value_spec for v in schema_lib.Missing.find_missing(value).values()]
    )
    return schema_lib.class_dependencies(value_specs, include_subclasses=True)

  def type_definitions_str(self, value: Any) -> str | None:
    return schema_lib.class_definitions(
        self.missing_type_dependencies(value), markdown=True
    )


class NaturalLanguageToStructure(Mapping):
  """LangFunc for converting natural language text to structured value."""

  mapping_template = lf.Template("""
      {%- if example.nl_context -%}
      {{ nl_context_title}}:
      {{ example.nl_context | indent(2, True)}}

      {% endif -%}
      {%- if example.nl_text -%}
      {{ nl_text_title }}:
      {{ example.nl_text | indent(2, True) }}

      {% endif -%}
      {{ schema_title }}:
      {{ example.schema_str(protocol) | indent(2, True) }}

      {{ value_title }}:
      {%- if example.has_value %}
      {{ example.value_str(protocol) | indent(2, True) }}
      {% endif -%}
      """)

  schema: pg.typing.Annotated[
      # Automatic conversion from annotation to schema.
      schema_lib.schema_spec(),
      'A `lf.structured.Schema` that constrains the structured value.',
  ]

  nl_context_title: Annotated[str, 'The section title for nl_context.'] = (
      'USER_REQUEST'
  )

  nl_text_title: Annotated[str, 'The section title for nl_text.'] = (
      'LM_RESPONSE'
  )

  schema_title: Annotated[str, 'The section title for schema.']

  value_title: Annotated[str, 'The section title for schema.']

  protocol: Annotated[
      schema_lib.SchemaProtocol,
      'The protocol for representing the schema and value.',
  ]

  @property
  def mapping_request(self) -> MappingExample:
    """Mapping request."""
    return MappingExample(
        nl_context=self.nl_context,
        nl_text=self.nl_text,
        schema=pg.Ref(self.schema),
    )

  @property
  @abc.abstractmethod
  def nl_context(self) -> str | None:
    """Returns the natural language context for obtaining the response.

    Returns:
      The natural language context (prompt) for obtaining the response (either
      in natural language or directly to structured protocol). If None,
      `nl_text` must be provided.
    """

  @property
  @abc.abstractmethod
  def nl_text(self) -> str | None:
    """Returns the natural language text to map.

    Returns:
      The natural language text (in LM response) to map to object. If None,
      the LM directly outputs structured protocol instead of natural language.
      If None, `nl_context` must be provided.
    """


class StructureToNaturalLanguage(Mapping):
  """LangFunc for converting a structured value to natural language."""

  mapping_template = lf.Template("""
      {%- if example.nl_context -%}
      {{ nl_context_title}}:
      {{ example.nl_context | indent(2, True)}}

      {% endif -%}
      {{ value_title}}:
      {{ value_str(example.value, markdown=False) | indent(2, True) }}

      {{ nl_text_title }}:
      {%- if example.nl_text %}
      {{ example.nl_text | indent(2, True) }}
      {% endif -%}
      """)

  input_value: Annotated[
      pg.Symbolic, 'A symbolic object with `lf.MISSING` values to complete.'
  ] = lf.contextual()

  nl_context: Annotated[
      str | None, 'The natural language context for describing the object.'
  ] = lf.contextual(default=None)

  nl_context_title: Annotated[str, 'The section title for nl_context.'] = (
      'CONTEXT_FOR_DESCRIPTION'
  )

  nl_text_title: Annotated[str, 'The section title for nl_text.'] = (
      'NATURAL_LANGUAGE_TEXT'
  )

  value_title: Annotated[str, 'The section title for schema.'] = 'PYTHON_OBJECT'

  @property
  def mapping_request(self) -> MappingExample:
    return MappingExample(
        nl_context=self.nl_context,
        value=pg.Ref(self.input_value),
    )


class Pair(pg.Object):
  """Value pair used for expressing structure-to-structure mapping."""

  left: pg.typing.Annotated[
      pg.typing.Any(transform=schema_lib.mark_missing), 'The left-side value.'
  ]
  right: pg.typing.Annotated[
      pg.typing.Any(transform=schema_lib.mark_missing), 'The right-side value.'
  ]

  @property
  def has_right(self) -> bool:
    """Returns True if the right value is present."""
    return self.right != schema_lib.MISSING


class StructureToStructure(Mapping):
  """Base class for structure-to-structure mapping."""

  input_value: Annotated[
      pg.Symbolic, 'A symbolic object with `lf.MISSING` values to complete.'
  ] = lf.contextual()

  type_definitions_title: Annotated[
      str, 'The section title for type definitions.'
  ] = 'CLASS_DEFINITIONS'

  modality_refs_title: Annotated[
      str, 'The section title for modality objects.'
  ] = 'MODALITY_REFS'

  input_value_title: Annotated[str, 'The section title for input value.']
  output_value_title: Annotated[str, 'The section title for output value.']

  def globals(self):
    context = super().globals()
    classes = schema_lib.class_dependencies(self.input_value)
    context.update({cls.__name__: cls for cls in classes})
    return context

  def transform_input(self, lm_input: lf.Message) -> lf.Message:
    # Find modalities to fill the input message.
    modalities = self.modalities(self.input_value)
    modalities.update(
        self.modalities(self.examples, root_path=pg.KeyPath('examples'))
    )
    if modalities:
      lm_input.metadata.update(pg.object_utils.canonicalize(modalities))
    return lm_input

  def postprocess_result(self, result: Any) -> Any:
    """Postprocess result."""
    # Try restore modality objects from the input value to output value.
    modalities = self.modalities(self.input_value)
    if modalities:
      result.rebind(modalities)
    return result
