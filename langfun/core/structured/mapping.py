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

import io
from typing import Annotated, Any
import langfun.core as lf
import langfun.core.llms as lf_llms
from langfun.core.structured import schema as schema_lib
import pyglove as pg


class MappingError(ValueError):   # pylint: disable=g-bad-exception-name
  """Mappingg error."""

  def __eq__(self, other):
    return isinstance(other, MappingError) and self.args == other.args

  def __ne__(self, other):
    return not self.__eq__(other)


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
      pg.typing.Object(
          schema_lib.Schema, transform=schema_lib.JsonSchema.from_value
      ).noneable(),
      (
          'A `lf.structured.Schema` object that constrains the structured '
          'value. It could be used as input when we map `nl_context`/`nl_text` '
          'to `value`, or could be used as output when we want to directly '
          'extract `schema` from `nl_context`/`nl_text`, in which case `value` '
          "will be None. A mapping's schema could be None when we map a "
          '`value` to natural language.'
      ),
  ] = lf.contextual()

  value: Annotated[
      Any,
      (
          'The structured representation for `nl_text` (or directly prompted '
          'from `nl_context`). It should align with schema.'
          '`value` could be used as input when we map a structured value to '
          'natural language, or as output when we map it reversely.'
      )
  ] = None

  @property
  def schema_repr(self) -> str:
    return self.schema.schema_repr()

  @property
  def value_repr(self) -> str:
    return self.schema.value_repr(self.value)

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

    result.write(lf.colored('[SCHEMA]\n', styles=['bold']))
    result.write(lf.colored(self.schema_repr, color='red'))
    result.write('\n\n')

    if pg.MISSING_VALUE != self.value:
      result.write(lf.colored('[VALUE]\n', styles=['bold']))
      result.write(lf.colored(self.value_repr, color='blue'))
    return result.getvalue()


class Mapping(lf.LangFunc):
  """Base class for mapping."""

  message: Annotated[
      lf.Message,
      'The input message.'
  ] = lf.contextual()

  examples: Annotated[
      list[MappingExample] | None,
      'Fewshot examples for improving the quality of mapping.'
  ] = lf.contextual(default=None)

  # The default model for generating the mappings.
  lm = lf_llms.Gpt35(temperature=0.0)
