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
"""Natural language text to structured value."""

import abc
import inspect
from typing import Annotated, Any, Callable, Type, Union

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


class Pair(pg.Objet):
  """Value pairs."""
  x: Any
  y: Any


class StructureToStructure(mapping.Mapping):
  """LangFunc for mapping one structured value to another structured value."""


class CompleteStructure(StructureToStructure):
  pass


def complete():
  pass

class ConflateStructure(StructureToStructure):
  pass


def conflate():
  pass