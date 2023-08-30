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
"""Contextual component and app test."""

import unittest
import weakref

from langfun.core import component as lf
import pyglove as pg


class ComponentContextTest(unittest.TestCase):
  """Tests for ComponentContext."""

  def test_override(self):
    class A(lf.Component):
      x: int

    a = A(x=1)
    with a.override(x=2, y=1):
      self.assertEqual(a.x, 2)

      # `y`` is not an attribute of `A`.
      with self.assertRaises(AttributeError):
        _ = a.y

  def test_context(self):
    class A(lf.Component):
      x: int
      y: int = lf.contextual()
      z: int = lf.contextual(default=-1)

    with self.assertRaisesRegex(TypeError, '.* missing 1 required argument'):
      _ = A()

    a = A(x=1)
    with self.assertRaisesRegex(
        AttributeError, 'p'
    ):
      _ = a.p

    with self.assertRaisesRegex(
        AttributeError, '.* is not found under its context'
    ):
      _ = a.y

    with lf.context(y=1):
      self.assertEqual(a.y, 1)

    with self.assertRaisesRegex(
        AttributeError, '.* is not found under its context'
    ):
      _ = a.y

    # Use contextual default if it's not provided.
    self.assertEqual(a.z, -1)

    a1 = A(x=1, y=2)
    self.assertEqual(a1.x, 1)
    self.assertEqual(a1.y, 2)
    self.assertEqual(a1.z, -1)

    with lf.context(x=3, y=3, z=3) as parent_override:
      self.assertEqual(
          parent_override,
          dict(
              x=lf.ContextualOverride(3, cascade=False, override_attrs=False),
              y=lf.ContextualOverride(3, cascade=False, override_attrs=False),
              z=lf.ContextualOverride(3, cascade=False, override_attrs=False),
          ),
      )
      self.assertEqual(
          lf.get_contextual_override('y'),
          lf.ContextualOverride(3, cascade=False, override_attrs=False),
      )

      # Member attributes take precedence over `lf.context`.
      self.assertEqual(a1.x, 1)
      self.assertEqual(a1.y, 2)

      # Override attributes take precedence over member attribute.
      with a1.override(y=3):
        self.assertEqual(a1.y, 3)
        with a1.override(y=4):
          self.assertEqual(a1.y, 4)
        self.assertEqual(a1.y, 3)
      self.assertEqual(a1.y, 2)

      # `lf.context` takes precedence over contextual default.
      self.assertEqual(a1.z, 3)

      # Test nested contextual override with override_attrs=True (default).
      with lf.context(y=4, z=4, override_attrs=True) as nested_override:
        self.assertEqual(
            nested_override,
            dict(
                x=lf.ContextualOverride(3, cascade=False, override_attrs=False),
                y=lf.ContextualOverride(4, cascade=False, override_attrs=True),
                z=lf.ContextualOverride(4, cascade=False, override_attrs=True),
            ),
        )

        # Member attribute is not overriden as current scope does not override
        # `x``.
        self.assertEqual(a1.x, 1)

        # Member attribute is overriden.
        self.assertEqual(a1.y, 4)

        # `lf.Component.override` takes precedence over
        # `lf.context(override_attrs=True)`.
        with a1.override(y=3):
          self.assertEqual(a1.y, 3)
        self.assertEqual(a1.y, 4)

        # Member default is overriden.
        self.assertEqual(a1.z, 4)

      self.assertEqual(a1.y, 2)
      self.assertEqual(a1.z, 3)

    self.assertEqual(a1.y, 2)
    self.assertEqual(a1.z, -1)

  def test_context_cascade(self):
    class A(lf.Component):
      x: int
      y: int = lf.contextual()
      z: int = lf.contextual(default=-1)

    a = A(1, 2)
    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z, -1)

    with lf.context(x=3, y=3, z=3, cascade=True):
      self.assertEqual(a.x, 1)
      self.assertEqual(a.y, 2)
      self.assertEqual(a.z, 3)

      # Outter `lf.force_context` takes precedence
      # over inner `lf.force_context`.
      with lf.context(y=4, z=4, cascade=True):
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)
        self.assertEqual(a.z, 3)

      with lf.context(y=4, z=4, override_attrs=True):
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)
        self.assertEqual(a.z, 3)

    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z, -1)

    with lf.context(x=3, y=3, z=3, cascade=True, override_attrs=True):
      self.assertEqual(a.x, 3)
      self.assertEqual(a.y, 3)
      self.assertEqual(a.z, 3)

      with lf.context(y=4, z=4, override_attrs=True):
        self.assertEqual(a.x, 3)
        self.assertEqual(a.y, 3)
        self.assertEqual(a.z, 3)

    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z, -1)

  def test_sym_inferred(self):
    class A(lf.Component):
      x: int = 1
      y: int = lf.contextual()

    a = A()
    with self.assertRaisesRegex(
        AttributeError, '.* is not found under its context'):
      _ = a.sym_inferred('y')
    self.assertIsNone(a.sym_inferred('y', default=None))

    with self.assertRaises(AttributeError):
      _ = a.sym_inferred('z')
    self.assertIsNone(a.sym_inferred('z', default=None))

  def test_symbolic_assignment(self):
    class A(lf.Component):
      x: int = 1

      def _on_bound(self):
        super()._on_bound()
        self.y = self.x * 2

    a = A()
    self.assertEqual(a.y, 2)
    a.x = 2
    self.assertEqual(a.y, 4)

  def test_symbolic_eq_ne_hash(self):
    class A(lf.Component):
      x: int = 1

    a = A()
    self.assertNotEqual(a, A())
    self.assertTrue(pg.eq(a, A()))
    self.assertFalse(pg.eq(a, A(2)))
    self.assertNotEqual(hash(a), hash(A()))
    self.assertEqual(pg.hash(a), pg.hash(A()))

  def test_weak_ref(self):
    class A(lf.Component):
      x: int = 1

    a = A()
    self.assertIsNotNone(weakref.ref(a))


class ContextualAttributeTest(unittest.TestCase):
  """Tests for Component."""

  def test_contextualibute_access(self):

    class A(lf.Component):
      x: int
      y: int = lf.contextual()

    # Not okay: `A.x` is required.
    with self.assertRaisesRegex(TypeError, 'missing 1 required argument'):
      _ = A()

    # Okay: `A.y` is contextual.
    a = A(1)

    # `a.y` is not yet available from the context.
    with self.assertRaises(AttributeError):
      _ = a.y

    class B(lf.Component):
      # Attributes with annotation will be treated as symbolic fields.
      p: int
      q: A = A(2)
      z: int = lf.contextual()

    class C(lf.Component):
      a: int

      # Attributes of lf type without annotation will also be
      # treated as symbolic fields.
      b = B(2)

      y: int = 1

      # Attributes of non-componentl type without annotation will not
      # be treated as symbolic fields.
      z = 2

    c = C(1)
    b = c.b
    a = b.q

    # Test symbolic attributes declared from C.
    self.assertTrue(c.sym_hasattr('a'))
    self.assertTrue(c.sym_hasattr('b'))
    self.assertTrue(c.sym_hasattr('y'))
    self.assertFalse(c.sym_hasattr('z'))

    # Contextual access to c.y from a.
    self.assertEqual(a.y, 1)
    self.assertEqual(b.z, 2)

    # 'y' is not defined as an attribute in 'B'.
    with self.assertRaises(AttributeError):
      _ = b.y

    c.rebind(y=2)
    self.assertEqual(c.y, 2)
    self.assertEqual(a.y, 2)

    c.z = 3
    self.assertEqual(c.z, 3)
    self.assertEqual(b.z, 3)


if __name__ == '__main__':
  unittest.main()
