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
"""Tests for Python code permissions."""

import unittest
from langfun.core.coding.python import permissions


class CodePermissionTest(unittest.TestCase):

  def assert_set(
      self,
      permission: permissions.CodePermission,
      flag: permissions.CodePermission,
  ):
    self.assertEqual(permission & flag, flag)

  def assert_not_set(
      self,
      permission: permissions.CodePermission,
      flag: permissions.CodePermission,
  ):
    self.assertFalse(permission & flag)

  def test_all(self):
    self.assert_set(
        permissions.CodePermission.ALL, permissions.CodePermission.BASIC
    )
    self.assert_set(
        permissions.CodePermission.ALL, permissions.CodePermission.CONDITION
    )
    self.assert_set(
        permissions.CodePermission.ALL, permissions.CodePermission.LOOP
    )
    self.assert_set(
        permissions.CodePermission.ALL, permissions.CodePermission.EXCEPTION
    )
    self.assert_set(
        permissions.CodePermission.ALL,
        permissions.CodePermission.CLASS_DEFINITION,
    )
    self.assert_set(
        permissions.CodePermission.ALL,
        permissions.CodePermission.FUNCTION_DEFINITION,
    )
    self.assert_set(
        permissions.CodePermission.ALL, permissions.CodePermission.IMPORT
    )

  def test_xor(self):
    self.assert_not_set(
        permissions.CodePermission.ALL ^ permissions.CodePermission.BASIC,
        permissions.CodePermission.BASIC,
    )
    self.assert_set(
        permissions.CodePermission.ALL ^ permissions.CodePermission.BASIC,
        permissions.CodePermission.CONDITION,
    )

  def test_permission_control(self):
    self.assertEqual(
        permissions.get_permission(), permissions.CodePermission.ALL
    )
    with permissions.permission(permissions.CodePermission.BASIC):
      self.assertEqual(
          permissions.get_permission(), permissions.CodePermission.BASIC
      )
      with permissions.permission(permissions.CodePermission.ALL):
        self.assertEqual(
            permissions.get_permission(), permissions.CodePermission.BASIC
        )


if __name__ == '__main__':
  unittest.main()
