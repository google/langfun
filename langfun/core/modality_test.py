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
    self.assertEqual(v.id, 'custom_modality:0cc175b9')
    self.assertEqual(str(v), "CustomModality(\n  content = 'a'\n)")
    self.assertEqual(v.hash, '0cc175b9')

    _ = pg.Dict(metadata=pg.Dict(x=pg.Dict(metadata=pg.Dict(y=v))))
    self.assertEqual(v.id, 'custom_modality:0cc175b9')
    self.assertEqual(str(v), "CustomModality(\n  content = 'a'\n)")
    with modality.format_modality_as_ref():
      self.assertEqual(str(v), '<<[[custom_modality:0cc175b9]]>>')

  def test_capture_rendered_modalities(self):
    x = CustomModality('a')
    y = CustomModality('b')
    z = CustomModality('b')

    with modality.capture_rendered_modalities() as rendered_modalities:
      with modality.format_modality_as_ref():
        self.assertEqual(
            f'Hello {x} {y} {z}',
            (
                'Hello <<[[custom_modality:0cc175b9]]>> '
                '<<[[custom_modality:92eb5ffe]]>> '
                '<<[[custom_modality:92eb5ffe]]>>'
            )
        )
    self.assertEqual(len(rendered_modalities), 2)
    self.assertIs(rendered_modalities['custom_modality:0cc175b9'].value, x)
    # y and z share the same content will be treated as the same object.
    self.assertIs(rendered_modalities['custom_modality:92eb5ffe'].value, z)


class ModalityRefTest(unittest.TestCase):

  def test_placehold_and_restore(self):
    class A(pg.Object):
      x: Any
      y: Any

    image_a = CustomModality('a')
    image_b = CustomModality('b')
    a = A(x=dict(z=image_a), y=image_b)
    a_placehold = modality.ModalityRef.placehold(a)
    self.assertEqual(
        a_placehold,
        A(x=dict(z=modality.ModalityRef(image_a.id)),
          y=modality.ModalityRef(image_b.id)),
    )
    a_restore = modality.ModalityRef.restore(
        a_placehold.clone(),
        {image_a.id: image_a, image_b.id: image_b},
    )
    self.assertTrue(pg.eq(a_restore, a))
    self.assertEqual(
        modality.ModalityRef.placehold(a.x),
        dict(z=modality.ModalityRef(image_a.id)),
    )
    with self.assertRaisesRegex(ValueError, 'Modality .* not found'):
      modality.ModalityRef.restore(a_placehold, {image_a.id: image_a})

  def test_from_value(self):
    class A(pg.Object):
      x: Any
      y: Any

    a = A(x=dict(z=CustomModality('a')), y=CustomModality('b'))
    self.assertTrue(
        pg.eq(
            modality.Modality.from_value(a),
            {
                'custom_modality:0cc175b9': CustomModality('a'),
                'custom_modality:92eb5ffe': CustomModality('b'),
            },
        )
    )
    self.assertTrue(
        pg.eq(
            modality.Modality.from_value(a.x.z),
            {
                'custom_modality:0cc175b9': CustomModality('a'),
            },
        )
    )


if __name__ == '__main__':
  unittest.main()
