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
"""Langfun."""

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
from langfun.core import *
from langfun.core import structured

Schema = structured.Schema
MISSING = structured.MISSING
UNKNOWN = structured.UNKNOWN

MappingExample = structured.MappingExample

call = structured.call
parse = structured.parse
query = structured.query
describe = structured.describe
complete = structured.complete
score = structured.score
generate_class = structured.generate_class

# Helper functions for input/output transformations based on
# `lf.query` (e.g. jax-on-beam could use these for batch processing)
query_prompt = structured.query_prompt
query_output = structured.query_output

source_form = structured.source_form
function_gen = structured.function_gen

from langfun.core import eval  # pylint: disable=redefined-builtin
from langfun.core import templates
from langfun.core import coding

PythonCode = coding.PythonCode
PythonFunction = coding.PythonFunction

from langfun.core import llms
lm_cache = llms.cache.lm_cache

from langfun.core import memories

from langfun.core import modalities

Image = modalities.Image
Video = modalities.Video
PDF = modalities.PDF

# Error types.
MappingError = structured.MappingError
SchemaError = structured.SchemaError
JsonError = structured.JsonError
CodeError = coding.CodeError

# Placeholder for Google-internal imports.

# pylint: enable=unused-import
# pylint: enable=g-import-not-at-top
# pylint: enable=g-bad-import-order

__version__ = "0.0.2"
