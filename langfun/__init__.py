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

parse = structured.parse
query = structured.query
describe = structured.describe


from langfun.core import templates
from langfun.core import transforms
from langfun.core import coding

PythonCode = coding.PythonCode

from langfun.core import llms
from langfun.core import memories

# Placeholder for Google-internal imports.

# pylint: enable=unused-import
# pylint: enable=g-import-not-at-top
# pylint: enable=g-bad-import-order

__version__ = "0.0.1"
