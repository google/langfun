# Copyright 2024 The Langfun Authors
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
import unittest

from langfun.core.eval.v2 import example as example_lib
import pyglove as pg

Example = example_lib.Example


class ExampleTest(unittest.TestCase):

  def test_basic(self):
    error = pg.object_utils.ErrorInfo(
        tag='ValueError',
        description='Bad input',
        stacktrace='...',
    )
    ex = Example(id=1, execution_status={
        'evaluate': pg.object_utils.TimeIt.Status(
            name='evaluation', elapse=1.0, error=error
        )
    })
    self.assertEqual(ex.error, error)
    self.assertFalse(ex.is_processed)
    self.assertTrue(ex.has_error)
    self.assertEqual(ex.elapse, 1.0)

    ex = Example(id=2, output=1)
    self.assertTrue(ex.is_processed)
    self.assertFalse(ex.has_error)
    self.assertIsNone(ex.elapse)

  def test_json_conversion(self):
    def input_func():
      class A(pg.Object):
        x: int

      class B(pg.Object):
        x: int = 1
        y: int = 2

      return [
          pg.Dict(
              a=A,
              b=B
          )
      ]

    inputs = input_func()
    ex = Example(
        id=1,
        input=inputs[0],
        output=inputs[0].a(1),
        metadata=dict(b=inputs[0].b())
    )
    # Serialize without input.
    json_str = pg.to_json_str(ex, exclude_input=True)
    self.assertEqual(
        pg.from_json_str(
            json_str,
            example_input_by_id=lambda i: inputs[i - 1]
        ),
        ex
    )
    pg.JSONConvertible._TYPE_REGISTRY._type_to_cls_map.pop(
        inputs[0].a.__type_name__
    )
    pg.JSONConvertible._TYPE_REGISTRY._type_to_cls_map.pop(
        inputs[0].b.__type_name__
    )
    v = pg.from_json_str(json_str, auto_dict=True)
    v.output.pop('type_name')
    v.metadata.b.pop('type_name')
    self.assertEqual(
        v,
        Example(
            id=1,
            output=pg.Dict(x=1),
            metadata=dict(b=pg.Dict(x=1, y=2)),
        )
    )
    # Serialize with input.
    ex = Example(id=2, input=pg.Dict(x=1), output=pg.Dict(x=2))
    json_str = pg.to_json_str(ex, exclude_input=False)
    self.assertEqual(pg.from_json_str(json_str), ex)

  def test_html_view(self):
    ex = Example(
        id=1,
        input=pg.Dict(a=1, b=2),
        output=3,
        metadata=dict(sum=3),
        metric_metadata=dict(match=True),
    )
    self.assertNotIn(
        'next',
        ex.to_html(extra_flags=dict(num_examples=1)).content,
    )


if __name__ == '__main__':
  unittest.main()
