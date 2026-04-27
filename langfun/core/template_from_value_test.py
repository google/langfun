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
"""Tests for Template.from_value modality preservation."""

import unittest

from langfun.core import template as template_lib


class FromValueModalityTest(unittest.TestCase):

  def test_from_value_template_preserves_referred_modalities(self):
    """from_value(Template) must preserve _referred_modalities."""

    class SubTemplate(template_lib.Template):
      pass

    original = template_lib.Template(template_str='Hello {{x}}')
    # Simulate what multistep_agent_base._inject_modalities does:
    # it monkey-patches _referred_modalities onto the template
    fake_modalities = {'image:abc12345': object()}  # dummy modality objects
    original._referred_modalities = fake_modalities

    # from_value on a subclass will bypass the isinstance(value, cls) check
    # and enter the isinstance(value, Template) block.
    result = SubTemplate.from_value(original)
    self.assertTrue(
        hasattr(result, '_referred_modalities'),
        'from_value(Template) must preserve _referred_modalities')
    self.assertEqual(
        result._referred_modalities, fake_modalities,
        'from_value(Template) must copy _referred_modalities from source')

  def test_from_value_template_with_kwargs_preserves_referred_modalities(self):
    """from_value(Template, **kwargs) must also preserve modalities."""
    original = template_lib.Template(template_str='Hello {{x}}')
    fake_modalities = {'image:abc12345': object()}
    original._referred_modalities = fake_modalities

    # This will trigger value.clone(override=kwargs)
    result = template_lib.Template.from_value(original, x=1)
    self.assertEqual(result._referred_modalities, fake_modalities)

  def test_from_value_template_without_modalities_works(self):
    """from_value(Template) without _referred_modalities should not crash."""
    original = template_lib.Template(template_str='Hello {{x}}')
    # No _referred_modalities set — should work fine
    result = template_lib.Template.from_value(original)
    self.assertIsNotNone(result)
    self.assertEqual(result.template_str, 'Hello {{x}}')


if __name__ == '__main__':
  unittest.main()
