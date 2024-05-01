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
"""Tests for evaluation patching."""

import unittest
from langfun.core import llms as lf_llms
from langfun.core.eval import base
from langfun.core.eval import patching
import pyglove as pg


class PatchingCommonTest(unittest.TestCase):

  def test_patch_member(self):
    class A(pg.Object):
      x: int = 1

    class B(pg.Object):
      a: A

    b = B(A())
    pg.patch(b, [patching.patch_member(A, 'x', 2)])
    self.assertEqual(b, B(A(2)))

  def test_patch_args(self):
    s = base.Suite(
        [base.Evaluation(inputs=base.as_inputs([1]))],
        additional_args=dict(x=1, y=2),
    )
    pg.patch(s, [patching.patch_additional_args(x=3, z=4)])
    self.assertTrue(
        pg.eq(
            s,
            base.Suite(
                [
                    base.Evaluation(
                        inputs=base.as_inputs([1]),
                        additional_args=dict(x=3, y=2, z=4),
                    )
                ],
                additional_args=dict(x=3, y=2, z=4),
            ),
        )
    )

  def test_patch_lm(self):
    s = base.Suite(
        [base.Evaluation(inputs=base.as_inputs([1]))],
        lm=lf_llms.Gpt35Turbo(),
    )
    pg.patch(
        s, [patching.patch_lm(pg.oneof([lf_llms.Gpt35Turbo(), lf_llms.Gpt4()]))]
    )
    self.assertTrue(
        pg.eq(
            s,
            base.Suite(
                [
                    base.Evaluation(
                        inputs=base.as_inputs([1]),
                        lm=pg.oneof([lf_llms.Gpt35Turbo(), lf_llms.Gpt4()]),
                    )
                ],
                lm=pg.oneof([lf_llms.Gpt35Turbo(), lf_llms.Gpt4()]),
            ),
        )
    )

  def test_patch_parsing_lm(self):
    s = base.Suite(
        [base.Evaluation(inputs=base.as_inputs([1]))],
        lm=lf_llms.Gpt4(),
    )
    pg.patch(s, [patching.patch_parsing_lm(lf_llms.Gpt35Turbo())])
    self.assertTrue(
        pg.eq(
            s,
            base.Suite(
                [
                    base.Evaluation(
                        inputs=base.as_inputs([1]),
                        lm=lf_llms.Gpt4(),
                        parsing_lm=lf_llms.Gpt35Turbo(),
                    )
                ],
                # NOTE(daiyip): Suite does not have `parsing_lm` as one of its
                # variable keyword fields yet, so patching does not add to it.
                # This is okay since we only care about the leaf nodes.
                lm=lf_llms.Gpt4(),
            ),
        )
    )

  def test_patch_prompt(self):
    e = base.Evaluation(inputs=base.as_inputs([1]))
    pg.patch(e, [patching.patch_prompt('Q: {{example.question}}')])
    self.assertTrue(
        pg.eq(
            e,
            base.Evaluation(
                inputs=base.as_inputs([1]),
                prompt='Q: {{example.question}}',
            ),
        )
    )

  def test_patch_inputs(self):
    e = base.Evaluation(inputs=base.as_inputs([1]))
    pg.patch(e, [patching.patch_inputs(base.as_inputs([2]))])
    self.assertTrue(
        pg.eq(
            e,
            base.Evaluation(
                inputs=base.as_inputs([2]),
            ),
        )
    )

  def test_patch_schema_fn(self):
    @pg.functor()
    def int_schema():
      return int

    e = base.Evaluation(inputs=base.as_inputs([1]))
    pg.patch(e, [patching.patch_schema_fn(int_schema())])
    self.assertTrue(
        pg.eq(
            e,
            base.Evaluation(
                inputs=base.as_inputs([1]),
                schema_fn=int_schema(),
            ),
        )
    )


class StringPatcheTest(unittest.TestCase):

  def test_lm(self):
    target = pg.patch(
        base.Evaluation(inputs=base.as_inputs([1])),
        ['lm?haiku:gpt4', 'max_tokens?1024', 'temperature?0.7'],
    )
    self.assertEqual(
        target.lm,
        pg.oneof([
            lf_llms.Claude3Haiku(temperature=0.7, max_tokens=1024),
            lf_llms.Gpt4(temperature=0.7, max_tokens=1024),
        ]),
    )
    with self.assertRaisesRegex(ValueError, 'Unknown model name'):
      pg.patch(
          base.Evaluation(inputs=base.as_inputs([1])),
          ['lm?gpt2'],
      )


if __name__ == '__main__':
  unittest.main()
