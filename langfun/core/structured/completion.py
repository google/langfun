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
"""Symbolic completion."""

from typing import Annotated, Any, Type

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


class CompleteStructure(mapping.Mapping):
  """Complete structure by filling the missing fields."""

  input: Annotated[
      pg.Symbolic, 'A symbolic object with `lf.MISSING` values to complete.'
  ] = lf.contextual()

  mapping_template = lf.Template("""
      {{ input_title }}:
      {{ example.input_repr(use_modality_ref=True) | indent(2, True) }}

      {%- if missing_type_dependencies(example.input) %}

      {{ schema_title }}:
      {{ class_defs_repr(example.input) | indent(2, True) }}
      {%- endif %}
      {%- if has_modality_refs(example.input) %}

      {{ modality_refs_title }}:
      {{ modality_refs_repr(example.input) | indent(2, True) }}
      {%- endif %}

      {{ output_title }}:
      {%- if example.has_output %}
      {{ example.output_repr(use_modality_ref=True) | indent(2, True) }}
      {% endif -%}
      """)

  input_title = 'INPUT_OBJECT'
  output_title = 'OUTPUT_OBJECT'
  schema_title = 'CLASS_DEFINITIONS'
  modality_refs_title: Annotated[
      str, 'The section title for modality refs.'
  ] = 'MODALITY_REFERENCES'

  preamble = lf.LangFunc(
      """
      Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

      INSTRUCTIONS:
      1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
      2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

      {{input_title}}:
        ```python
        Answer(
          question='1 + 1 =',
          answer=MISSING(int)
        )
        ```

      {{output_title}}:
        ```python
        Answer(
          question='1 + 1 =',
          answer=2
        )
        ```
      """,
      input_title=input_title,
      output_title=output_title,
  )

  @property
  def mapping_request(self) -> mapping.MappingExample:
    """Returns a MappingExample as the mapping request."""
    return mapping.MappingExample(
        input=pg.Ref(self.input),
        # Use the schema of input object.
        schema=pg.Ref(schema_lib.Schema.from_value(self.input.__class__)),
        context=self.context,
    )

  def transform_input(self, lm_input: lf.Message) -> lf.Message:
    if not pg.contains(self.input, type=schema_lib.Missing):
      raise ValueError(
          'The input of `lf.complete` must contain a least one '
          f'missing value. Encountered: {self.input}.'
      )
    return super().transform_input(lm_input)

  def missing_type_dependencies(self, value: Any) -> list[Type[Any]]:
    value_specs = tuple(
        [v.value_spec for v in schema_lib.Missing.find_missing(value).values()]
    )
    return schema_lib.class_dependencies(value_specs, include_subclasses=True)

  def class_defs_repr(self, value: Any) -> str | None:
    return schema_lib.class_definitions(
        self.missing_type_dependencies(value),
        markdown=True,
        allowed_dependencies=set()
    )

  def postprocess_result(self, result: Any) -> Any:
    """Postprocess result."""
    # Try restore modality objects from the input value to output value.
    modalities = self.modalities(self.input)
    if modalities:
      # Remove the `input` prefix for all entries.
      modalities = pg.object_utils.flatten(
          pg.object_utils.canonicalize(modalities)['input']
      )
      result.rebind(modalities)
    return result

  def globals(self):
    context = super().globals()

    # Add class dependencies from the input value to the globals.
    classes = schema_lib.class_dependencies(self.input)
    context.update({cls.__name__: cls for cls in classes})

    # NOTE(daiyip): since `lf.complete` could have fields of Any type, which
    # could be user provided objects. For LLMs to restores these objects, we
    # need to expose their types to the code evaluation context.
    context.update(self._input_value_dependencies())
    return context

  def _input_value_dependencies(self) -> dict[str, Any]:
    """Returns the class dependencies from input value."""
    context = {}
    def _visit(k, v, p):
      del k, p
      if isinstance(v, pg.Object):
        cls = v.__class__
        context[cls.__name__] = cls
    pg.traverse(self.input, _visit)
    return context

  #
  # Helper methods for handling modalities.
  #

  def has_modality_refs(self, value: Any) -> bool:
    """Returns true if the value has modalities."""
    return not isinstance(value, lf.Modality) and pg.contains(
        value, type=lf.Modality
    )

  def modalities(self, value: Any) -> dict[str, lf.Modality]:
    return lf.Modality.from_value(value)

  def modality_refs_repr(self, value: Any) -> str:
    with lf.modality.format_modality_as_ref(True):
      return pg.format(
          self.modalities(value),
          compact=False,
          verbose=False,
          python_format=True,
      )


def complete(
    input_value: pg.Symbolic,
    default: Any = lf.RAISE_IF_HAS_ERROR,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    cache_seed: int | None = 0,
    autofix: int = 0,
    autofix_lm: lf.LanguageModel | None = None,
    returns_message: bool = False,
    **kwargs,
) -> Any:
  """Complete a symbolic value by filling its missing fields.

  Examples:

    ```
    class FlightDuration:
      hours: int
      minutes: int

    class Flight(pg.Object):
      airline: str
      flight_number: str
      departure_airport_code: str
      arrival_airport_code: str
      departure_time: str
      arrival_time: str
      duration: FlightDuration
      stops: int
      price: float

    prompt = '''
      Information about flight UA2631.
      '''

    r = lf.query(prompt, Flight)
    assert isinstance(r, Flight)
    assert r.airline == 'United Airlines'
    assert r.departure_airport_code == 'SFO'
    assert r.duration.hour = 7
    ```

  Args:
    input_value: A symbolic value that may contain missing values.
    default: The default value if parsing failed. If not specified, error will
      be raised.
    lm: The language model to use. If not specified, the language model from
      `lf.context` context manager will be used.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    cache_seed: Seed for computing cache key. The cache key is determined by a
      tuple of (lm, prompt, cache seed). If None, cache will be disabled for
      the query even cache is configured by the LM.
    autofix: Number of attempts to auto fix the generated code. If 0, autofix is
      disabled.
    autofix_lm: The language model to use for autofix. If not specified, the
      `autofix_lm` from `lf.context` context manager will be used. Otherwise it
      will use `lm`.
    returns_message: If True, returns `lf.Message` as the output, instead of
      returning the structured `message.result`.
    **kwargs: Keyword arguments passed to the
      `lf.structured.NaturalLanguageToStructureed` transform.

  Returns:
    The result based on the schema.
  """
  t = CompleteStructure(
      input=schema_lib.mark_missing(input_value),
      default=default,
      examples=examples,
      autofix=autofix,
      **kwargs,
  )

  output = t(lm=lm, cache_seed=cache_seed, autofix_lm=autofix_lm or lm)
  return output if returns_message else output.result
