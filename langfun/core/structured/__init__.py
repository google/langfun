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
"""Common langfun templates."""

# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member

from langfun.core.structured.schema import include_method_in_prompt

from langfun.core.structured.schema import Missing
from langfun.core.structured.schema import MISSING
from langfun.core.structured.schema import Unknown
from langfun.core.structured.schema import UNKNOWN

from langfun.core.structured.schema import Schema
from langfun.core.structured.schema import SchemaProtocol
from langfun.core.structured.schema import schema_spec

from langfun.core.structured.schema import SchemaError
from langfun.core.structured.schema import JsonError

from langfun.core.structured.schema import class_dependencies
from langfun.core.structured.schema import class_definition
from langfun.core.structured.schema import class_definitions
from langfun.core.structured.schema import annotation
from langfun.core.structured.schema import structure_from_python

from langfun.core.structured.schema import schema_repr
from langfun.core.structured.schema import source_form
from langfun.core.structured.schema import value_repr

from langfun.core.structured.schema_generation import generate_class
from langfun.core.structured.schema_generation import classgen_example
from langfun.core.structured.schema_generation import default_classgen_examples

from langfun.core.structured.function_generation import function_gen

from langfun.core.structured.mapping import Mapping
from langfun.core.structured.mapping import MappingError
from langfun.core.structured.mapping import MappingExample

from langfun.core.structured.parsing import parse
from langfun.core.structured.parsing import call

from langfun.core.structured.querying import track_queries
from langfun.core.structured.querying import QueryInvocation
from langfun.core.structured.querying import query
from langfun.core.structured.querying import query_and_reduce

from langfun.core.structured.querying import query_prompt
from langfun.core.structured.querying import query_output
from langfun.core.structured.querying import query_reward

from langfun.core.structured.description import describe
from langfun.core.structured.completion import complete

from langfun.core.structured.scoring import score

from langfun.core.structured.tokenization import tokenize

# Expose default examples for structured operations so users could refer to
# them.
from langfun.core.structured.parsing import default_parse_examples
from langfun.core.structured.description import default_describe_examples

# Default examples.


# pylint: enable=g-importing-member
# pylint: enable=g-bad-import-order
