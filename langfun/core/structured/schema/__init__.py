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
"""Schema and prompting protocols for structured data."""

# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member
from langfun.core.structured.schema.base import Schema
from langfun.core.structured.schema.base import SchemaError
from langfun.core.structured.schema.base import SchemaType

from langfun.core.structured.schema.base import class_dependencies
from langfun.core.structured.schema.base import schema_spec
from langfun.core.structured.schema.base import annotation

from langfun.core.structured.schema.base import PromptingProtocol
from langfun.core.structured.schema.base import schema_repr
from langfun.core.structured.schema.base import value_repr
from langfun.core.structured.schema.base import parse_value

from langfun.core.structured.schema.base import mark_missing
from langfun.core.structured.schema.base import Missing
from langfun.core.structured.schema.base import MISSING
from langfun.core.structured.schema.base import Unknown
from langfun.core.structured.schema.base import UNKNOWN

# JSON protocol.
from langfun.core.structured.schema.json import JsonError
from langfun.core.structured.schema.json import JsonPromptingProtocol
from langfun.core.structured.schema.json import cleanup_json

# Python protocol.
from langfun.core.structured.schema.python import PythonPromptingProtocol
from langfun.core.structured.schema.python import source_form
from langfun.core.structured.schema.python import structure_from_python

from langfun.core.structured.schema.python import class_definition
from langfun.core.structured.schema.python import class_definitions
from langfun.core.structured.schema.python import include_method_in_prompt
