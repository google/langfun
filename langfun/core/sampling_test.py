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
"""Test sampling."""

import unittest

from langfun.core import component
from langfun.core import language_model
from langfun.core import sampling
from langfun.core.langfunc import LangFunc
import pyglove as pg


class ExcitedEchoer(language_model.LanguageModel):
  """LM for testing."""

  def _sample(self, prompts) -> list[language_model.LMSamplingResult]:
    return [
        language_model.LMSamplingResult([
            language_model.LMSample(prompt.text + '!!!')
            ]) for prompt in prompts
    ]


class SamplingTest(unittest.TestCase):

  def test_sweep(self):
    l = LangFunc('Compute {{x}} and {{y}}', x=pg.oneof([1, 2]))
    with component.context(lm=ExcitedEchoer()):
      samples = list(sampling.sweep(l, y=pg.oneof([3, 4])))
      samples = sorted(samples, key=lambda x: (x[0].x, x[0].y))

    self.assertEqual(
        samples,
        [
            ('Compute 1 and 3', 'Compute 1 and 3!!!'),
            ('Compute 1 and 4', 'Compute 1 and 4!!!'),
            ('Compute 2 and 3', 'Compute 2 and 3!!!'),
            ('Compute 2 and 4', 'Compute 2 and 4!!!'),
        ],
    )

  def test_random_sample(self):
    l = LangFunc('Compute {{x}} and {{y}}', x=pg.oneof([1, 3, 5]))
    with component.context(lm=ExcitedEchoer()):
      samples = list(
          sampling.random_sample(l, y=pg.oneof([2, 4]), num_examples=3, seed=1)
      )
      samples = sorted(samples, key=lambda x: (x[0].x, x[0].y))

    self.assertEqual(
        samples,
        [
            ('Compute 1 and 2', 'Compute 1 and 2!!!'),
            ('Compute 3 and 2', 'Compute 3 and 2!!!'),
            ('Compute 3 and 4', 'Compute 3 and 4!!!'),
        ],
    )

  def test_random_sample_with_const_space(self):
    l = LangFunc('Compute {{x}} and {{y}}', x=1)

    with component.context(lm=ExcitedEchoer()):
      samples = list(sampling.random_sample(l, y=2, num_examples=3, seed=1))
      self.assertEqual(
          samples,
          [
              ('Compute 1 and 2', 'Compute 1 and 2!!!'),
              ('Compute 1 and 2', 'Compute 1 and 2!!!'),
              ('Compute 1 and 2', 'Compute 1 and 2!!!'),
          ],
      )

  def test_random_sample_with_errors(self):
    l = LangFunc('Compute {{x}} and {{y}}', x=pg.oneof([1, 3, 5]))
    with self.assertRaisesRegex(AttributeError, '`lm` is not found'):
      _ = next(sampling.random_sample(l, y=pg.oneof([2, 4]), num_examples=1))

  def test_random_sample_with_silenced_errors(self):
    l = LangFunc('Compute {{x}} and {{y}}', x=pg.oneof([1, 3, 5]))
    samples = list(sampling.random_sample(
        l,
        y=pg.oneof([2, 4]),
        num_examples=3,
        seed=1,
        silence_on_errors=(AttributeError,),
        ignore_examples_with_errors=False))

    samples = sorted(samples, key=lambda x: (x[0].x, x[0].y))
    self.assertEqual(
        [x[0] for x in samples],
        [
            'Compute 1 and 2',
            'Compute 3 and 2',
            'Compute 3 and 4',
        ]
    )
    for x in samples:
      self.assertIsInstance(x[1], AttributeError)

  def test_random_sample_lm_input_only(self):
    l = LangFunc('Compute {{x}} and {{y}}', x=pg.oneof([1, 3, 5]))
    samples = list(
        sampling.random_sample(
            l, y=pg.oneof([2, 4]), lm_input_only=True, num_examples=3, seed=1
        )
    )
    samples = sorted(samples, key=lambda x: (x[0].x, x[0].y))
    self.assertEqual(
        [(x[0], x[1]) for x in samples],
        [
            ('Compute 1 and 2', None),
            ('Compute 3 and 2', None),
            ('Compute 3 and 4', None),
        ],
    )


if __name__ == '__main__':
  unittest.main()
