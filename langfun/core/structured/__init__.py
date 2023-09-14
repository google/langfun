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

from langfun.core.structured.schema import Schema

from langfun.core.structured.schema import SchemaRepr
from langfun.core.structured.schema import SchemaJsonRepr
from langfun.core.structured.schema import SchemaPythonRepr
from langfun.core.structured.schema import ValueRepr
from langfun.core.structured.schema import ValueJsonRepr
from langfun.core.structured.schema import ValuePythonRepr
from langfun.core.structured.schema import schema_repr
from langfun.core.structured.schema import value_repr

from langfun.core.structured.mapping import Mapping
from langfun.core.structured.mapping import MappingExample
from langfun.core.structured.mapping import MappingError

from langfun.core.structured.nl2structure import NaturalLanguageToStructure
from langfun.core.structured.nl2structure import ParseStructure
from langfun.core.structured.nl2structure import ParseStructureJson
from langfun.core.structured.nl2structure import ParseStructurePython
from langfun.core.structured.nl2structure import QueryStructure
from langfun.core.structured.nl2structure import QueryStructureJson
from langfun.core.structured.nl2structure import QueryStructurePython
from langfun.core.structured.nl2structure import parse
from langfun.core.structured.nl2structure import query

from langfun.core.structured.structure2nl import StructureToNaturalLanguage
from langfun.core.structured.structure2nl import DescribeStructure
from langfun.core.structured.structure2nl import describe

# pylint: enable=g-importing-member
# pylint: enable=g-bad-import-order
