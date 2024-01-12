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
"""langfun features on Python code parsing and execution."""

# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member

from langfun.core.coding.python.errors import CodeError

from langfun.core.coding.python.permissions import CodePermission
from langfun.core.coding.python.permissions import permission
from langfun.core.coding.python.permissions import get_permission

from langfun.core.coding.python.parsing import PythonCodeParser

from langfun.core.coding.python.execution import context
from langfun.core.coding.python.execution import get_context
from langfun.core.coding.python.execution import evaluate
from langfun.core.coding.python.execution import sandbox_call
from langfun.core.coding.python.execution import call
from langfun.core.coding.python.execution import run

from langfun.core.coding.python.generation import PythonCode
from langfun.core.coding.python.generation import PythonFunction

from langfun.core.coding.python.correction import correct
from langfun.core.coding.python.correction import run_with_correction
from langfun.core.coding.python.correction import CodeWithError

# pylint: enable=g-importing-member
# pylint: enable=g-bad-import-order
