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
"""Python code permissions."""

import contextlib
import enum
import pyglove as pg


class CodePermission(enum.Flag):
  """Permissions for code execution."""

  # Allows basic Python code: Creating objects, assignment, operations.
  BASIC = enum.auto()

  # Allows conditions.
  CONDITION = enum.auto()

  # Allows loops.
  LOOP = enum.auto()

  # Allows exception.
  EXCEPTION = enum.auto()

  # Allows class definitions.
  CLASS_DEFINITION = enum.auto()

  # Allows function definitions.
  FUNCTION_DEFINITION = enum.auto()

  # Allows import.
  IMPORT = enum.auto()

  @classmethod
  @property
  def ALL(cls) -> 'CodePermission':    # pylint: disable=invalid-name
    """Returns all permissions."""
    return (
        CodePermission.BASIC | CodePermission.CONDITION | CodePermission.LOOP |
        CodePermission.EXCEPTION | CodePermission.CLASS_DEFINITION |
        CodePermission.FUNCTION_DEFINITION | CodePermission.IMPORT)


_TLS_CODE_RUN_PERMISSION = '__code_run_permission__'


@contextlib.contextmanager
def permission(perm: CodePermission):
  """Context manager for controling the permission for code execution.

  When the `permission` context manager is nested, the outtermost permission
  will be used. This design allows users to control permission at the top level.

  Args:
    perm: Code execution permission.

  Yields:
    Actual permission applied.
  """

  outter_perm = pg.object_utils.thread_local_get(_TLS_CODE_RUN_PERMISSION, None)

  # Use the top-level permission as the actual permission
  if outter_perm is not None:
    perm = outter_perm

  pg.object_utils.thread_local_set(_TLS_CODE_RUN_PERMISSION, perm)

  try:
    yield perm
  finally:
    if outter_perm is None:
      pg.object_utils.thread_local_del(_TLS_CODE_RUN_PERMISSION)


def get_permission() -> CodePermission:
  """Gets the current permission for code execution."""
  return pg.object_utils.thread_local_get(
      _TLS_CODE_RUN_PERMISSION, CodePermission.ALL)
