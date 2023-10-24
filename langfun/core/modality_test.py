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
"""Tests for modality."""
from typing import Any
import unittest

from langfun.core import modality
import pyglove as pg


class CustomModality(modality.Modality):
  content: str

  def to_bytes(self):
    return self.content.encode()


class ModalityTest(unittest.TestCase):

  def test_basic(self):
    v = CustomModality('a')
    self.assertIsNone(v.referred_name)
    self.assertEqual(str(v), "CustomModality(\n  content = 'a'\n)")

    _ = pg.Dict(metadata=pg.Dict(x=pg.Dict(metadata=pg.Dict(y=v))))
    self.assertEqual(v.referred_name, 'x.metadata.y')
    self.assertEqual(str(v), "CustomModality(\n  content = 'a'\n)")
    with modality.format_modality_as_ref():
      self.assertEqual(str(v), '{{x.metadata.y}}')


class ModalityRefTest(unittest.TestCase):

  def test_placehold(self):
    class A(pg.Object):
      x: Any
      y: Any

    a = A(x=dict(z=CustomModality('a')), y=CustomModality('b'))
    self.assertEqual(
        modality.ModalityRef.placehold(a),
        A(x=dict(z=modality.ModalityRef('x.z')), y=modality.ModalityRef('y')),
    )


if __name__ == '__main__':
  unittest.main()
