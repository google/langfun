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
import os
import tempfile
import unittest

from langfun.core.eval.v2 import example as example_lib
import pyglove as pg

Example = example_lib.Example


class _PlainOutput:
  """A non-symbolic (opaque) output type, picklable at module scope."""

  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return isinstance(other, _PlainOutput) and other.value == self.value


class ExampleTest(unittest.TestCase):

  def test_basic(self):
    error = pg.ErrorInfo(
        tag='ValueError',
        description='Bad input',
        stacktrace='...',
    )
    ex = Example(id=1, execution_status={
        'evaluate': pg.utils.TimeIt.Status(
            name='evaluation', elapse=1.0, error=error
        )
    })
    self.assertIsNone(ex.error)
    self.assertFalse(ex.is_processed)
    self.assertFalse(ex.has_error)
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
            example_input_by_id=lambda i: inputs[i - 1],
            load_example_metadata=True,
        ),
        ex
    )
    self.assertEqual(
        pg.from_json_str(
            json_str,
            example_input_by_id=lambda i: inputs[i - 1],
            load_example_metadata=False,
        ),
        Example(
            id=1,
            input=inputs[0],
            output=inputs[0].a(1),
            metadata={}
        )
    )
    pg.JSONConvertible._TYPE_REGISTRY._type_to_cls_map.pop(
        inputs[0].a.__type_name__
    )
    pg.JSONConvertible._TYPE_REGISTRY._type_to_cls_map.pop(
        inputs[0].b.__type_name__
    )
    v = pg.from_json_str(
        json_str,
        convert_unknown=True,
        load_example_metadata=True
    )
    self.assertEqual(
        v,
        Example(
            id=1,
            output=pg.symbolic.UnknownTypedObject(
                inputs[0].a.__type_name__, x=1
            ),
            metadata=dict(
                b=pg.symbolic.UnknownTypedObject(
                    inputs[0].b.__type_name__, x=1, y=2
                )
            ),
        )
    )
    # Serialize with input.
    ex = Example(id=2, input=pg.Dict(x=1), output=pg.Dict(x=2))
    json_str = pg.to_json_str(ex, exclude_input=False)
    self.assertEqual(pg.from_json_str(json_str), ex)

  def test_iter_ckpts_with_opaque_output(self):
    """Checkpoint warm-start round-trips non-symbolic (opaque) outputs.

    Regression test for b/511887449: `Example.output` is typed `Any`, so eval
    outputs may be plain (non-symbolic) Python objects that pyglove serializes
    through the opaque-object pickle path. With opaque pickle disabled by
    default, reloading these trusted checkpoint files must still work.
    """

    ex = Example(id=1, input=pg.Dict(x=1), output=_PlainOutput(42))
    ckpt = os.path.join(tempfile.mkdtemp(), 'opaque_ckpt.jsonl')
    with pg.io.open_sequence(ckpt, 'a') as writer:
      writer.add(pg.to_json_str(ex, exclude_input=False))

    loaded = list(example_lib.Example.iter_ckpts(ckpt))
    self.assertEqual(len(loaded), 1)
    self.assertEqual(loaded[0].output, _PlainOutput(42))

  def test_html_view(self):
    ex = Example(
        id=1,
        input=pg.Dict(a=1, b=2),
        output=3,
        metadata=dict(sum=3),
        metric_metadata=dict(match=dict(match=True)),
    )
    self.assertNotIn(
        'next',
        ex.to_html(extra_flags=dict(num_examples=1)).content,
    )


if __name__ == '__main__':
  unittest.main()
