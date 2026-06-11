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
import dataclasses
import unittest

from langfun.core.eval.v2 import example as example_lib
import pyglove as pg

Example = example_lib.Example

# The pyglove "secure-by-default" flip -- opaque-object pickle deserialization
# disabled unless a caller opts in via enable_opaque_pickle(True) -- rolls out
# to released pyglove on a schedule separate from langfun (b/511887449).
# langfun's OSS CI installs pyglove from PyPI, which may still default opaque
# pickle to ON. Gate the secure-by-default assertions on the pyglove the suite
# actually runs against, so the functional round-trip coverage stays meaningful
# on every version while the secure-default assertions only fire where that
# default is in effect (always true for the in-repo pyglove).
_OPAQUE_PICKLE_SECURE_BY_DEFAULT = (
    not pg.utils.json_conversion._opaque_pickle_enabled  # pylint: disable=protected-access
)


@dataclasses.dataclass
class _NonSymbolicOutput:
  """A plain (non-`pg.Object`) value that serializes via the opaque path."""

  value: int


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

  def test_opaque_object_round_trip(self):
    # A non-symbolic value forces pyglove to serialize via the opaque-object
    # pickle path, which is disabled by default (secure-by-default). Loading a
    # checkpoint `Example` must still succeed because `Example.from_json` opts
    # in to opaque pickle at this trusted, first-party boundary.
    ex = Example(
        id=1,
        input=pg.Dict(x=1),
        output=_NonSymbolicOutput(42),
        metadata=dict(note=_NonSymbolicOutput(7)),
    )
    json_str = pg.to_json_str(ex, exclude_input=False)
    restored = pg.from_json_str(json_str, load_example_metadata=True)
    self.assertEqual(restored.input, pg.Dict(x=1))
    self.assertEqual(restored.output, _NonSymbolicOutput(42))
    self.assertEqual(restored.metadata['note'], _NonSymbolicOutput(7))

    # The remaining assertions verify the secure-by-default contract, which
    # only holds when the installed pyglove ships the flip (b/511887449). Skip
    # them on a released pyglove that still defaults opaque pickle to ON; the
    # round-trip above already exercises the trusted-boundary opt-in.
    if _OPAQUE_PICKLE_SECURE_BY_DEFAULT:
      # The opt-in is strictly scoped: it must not leak out of `from_json` and
      # re-enable opaque pickle for the rest of the process.
      self.assertFalse(pg.utils.json_conversion._opaque_pickle_enabled)  # pylint: disable=protected-access

      # Deserializing the same opaque payload outside the trusted `Example`
      # boundary remains blocked by default.
      with self.assertRaises(TypeError):
        pg.from_json_str(pg.to_json_str(_NonSymbolicOutput(1)))

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
