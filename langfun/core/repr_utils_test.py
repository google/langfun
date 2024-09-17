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
"""Tests for langfun.core.repr_utils."""

import io
import unittest

from langfun.core import repr_utils
import pyglove as pg


class SharingContentTest(unittest.TestCase):

  def test_sharing(self):
    s = io.StringIO()

    self.assertTrue(repr_utils.write_maybe_shared(s, '<hr>'))
    self.assertTrue(repr_utils.write_maybe_shared(s, '<hr>'))

    with repr_utils.share_parts() as ctx1:
      self.assertTrue(repr_utils.write_maybe_shared(s, '<style></style>'))
      self.assertFalse(repr_utils.write_maybe_shared(s, '<style></style>'))

      with repr_utils.share_parts() as ctx2:
        self.assertIs(ctx2, ctx1)
        self.assertFalse(repr_utils.write_maybe_shared(s, '<style></style>'))
        self.assertTrue(repr_utils.write_maybe_shared(s, '<style>a</style>'))
        self.assertFalse(repr_utils.write_maybe_shared(s, '<style>a</style>'))
        self.assertTrue(repr_utils.write_maybe_shared(s, '<style>b</style>'))

      with repr_utils.share_parts() as ctx3:
        self.assertIs(ctx3, ctx1)
        self.assertFalse(repr_utils.write_maybe_shared(s, '<style></style>'))
        self.assertFalse(repr_utils.write_maybe_shared(s, '<style>a</style>'))
        self.assertFalse(repr_utils.write_maybe_shared(s, '<style>a</style>'))
        self.assertFalse(repr_utils.write_maybe_shared(s, '<style>b</style>'))

    self.assertEqual(
        s.getvalue(),
        '<hr><hr><style></style><style>a</style><style>b</style>'
    )
    self.assertEqual(ctx1['<style></style>'], 4)
    self.assertEqual(ctx1['<style>b</style>'], 2)
    self.assertEqual(ctx1['<style>a</style>'], 4)

  def test_escape_quoted(self):
    self.assertEqual(
        repr_utils.escape_quoted(str('<a>')), '<a>'
    )
    self.assertEqual(
        repr_utils.escape_quoted('x=<a>, b="<a>"'),
        'x=<a>, b="&lt;a&gt;"'
    )

  def test_html(self):
    html = repr_utils.Html('<div>foo</div>')
    self.assertEqual(html.content, '<div>foo</div>')
    self.assertEqual(html._repr_html_(), '<div>foo</div>')

  def test_html_repr(self):
    class Foo(pg.Object):
      x: int

    class Bar(pg.Object):

      def _repr_html_(self):
        return '<bar>'

    html = repr_utils.html_repr(
        {'foo': pg.Ref(Foo(1)), 'bar': Bar(), 'baz': '<lf_image>'}
    )
    self.assertIn('foo</span>', html)
    self.assertIn('<bar>', html)
    self.assertIn('&lt;lf_image&gt;', html)
    self.assertNotIn('Ref', html)


if __name__ == '__main__':
  unittest.main()
