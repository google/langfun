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
"""Test for langfun subscriptions."""

import dataclasses
from typing import Any
import unittest

from langfun.core import subscription


class SenderA:

  def foo(self):
    subscription.emit(FooEvent(self, 1))

  def bar(self):
    subscription.emit(BarEvent(self, 2))


@dataclasses.dataclass
class FooEvent(subscription.Event[SenderA]):
  x: int


@dataclasses.dataclass
class BarEvent(subscription.Event[SenderA]):
  y: int


@dataclasses.dataclass
class BaseEventHandler:
  events: list[subscription.Event[Any]] = dataclasses.field(
      default_factory=list
  )


class FooEventHandler(BaseEventHandler, subscription.EventHandler[FooEvent]):

  def on_event(self, event: FooEvent):
    self.events.append((self, event))


class BarEventHandler(BaseEventHandler, subscription.EventHandler[BarEvent]):

  def on_event(self, event: BarEvent):
    self.events.append((self, event))


class SenderB:

  def baz(self):
    subscription.emit(BazEvent(self, 'abc'))


@dataclasses.dataclass
class BazEvent(subscription.Event[SenderB]):
  z: str


class BazEventHandler(BaseEventHandler, subscription.EventHandler[BazEvent]):

  def on_event(self, event: BazEvent):
    self.events.append((self, event))


class EventHandlerTest(unittest.TestCase):

  def test_sender_type(self):
    self.assertIs(FooEventHandler.sender_type(), SenderA)
    self.assertIs(BarEventHandler.sender_type(), SenderA)
    self.assertIs(BazEventHandler.sender_type(), SenderB)

  def test_event_type(self):
    self.assertIs(FooEventHandler.event_type(), FooEvent)
    self.assertIs(BarEventHandler.event_type(), BarEvent)
    self.assertIs(BazEventHandler.event_type(), BazEvent)

  def test_accepts(self):
    self.assertTrue(FooEventHandler.accepts(FooEvent(SenderA(), 1)))
    self.assertFalse(FooEventHandler.accepts(BarEvent(SenderA(), 2)))
    self.assertFalse(FooEventHandler.accepts(BazEvent(SenderB(), 'abc')))

    self.assertFalse(BarEventHandler.accepts(FooEvent(SenderA(), 1)))
    self.assertTrue(BarEventHandler.accepts(BarEvent(SenderA(), 2)))
    self.assertFalse(BarEventHandler.accepts(BazEvent(SenderB(), 'abc')))

    self.assertFalse(BazEventHandler.accepts(FooEvent(SenderA(), 1)))
    self.assertFalse(BazEventHandler.accepts(BarEvent(SenderA(), 2)))
    self.assertTrue(BazEventHandler.accepts(BazEvent(SenderB(), 'abc')))


class EventTest(unittest.TestCase):

  def assertLen(self, value, length):
    self.assertEqual(len(value), length)  # pylint: disable=g-generic-assert

  def assertEmpty(self, value):
    self.assertLen(value, 0)  # pylint: disable=g-generic-assert

  def test_event_subscription_and_emition(self):
    a1 = SenderA()
    a2 = SenderA()
    foo_handler1 = FooEventHandler()
    foo_handler2 = FooEventHandler()
    bar_handler = BarEventHandler()

    b = SenderB()
    baz_handler = BazEventHandler()

    # Senders have no subscribers yet.
    self.assertEmpty(list(subscription.subscribers(a1)))
    self.assertEmpty(list(subscription.subscribers(a2)))
    self.assertEmpty(list(subscription.subscribers(b)))
    self.assertEmpty(list(subscription.subscribers(SenderA)))
    self.assertEmpty(list(subscription.subscribers(SenderB)))

    # Subscribers have not subscribed any senders yet.
    self.assertEmpty(list(subscription.subscriptions(foo_handler1)))
    self.assertEmpty(list(subscription.subscriptions(foo_handler2)))
    self.assertEmpty(list(subscription.subscriptions(bar_handler)))
    self.assertEmpty(list(subscription.subscriptions(baz_handler)))

    a1.foo()
    a1.bar()
    a2.foo()
    a2.bar()
    b.baz()

    self.assertEmpty(foo_handler1.events)
    self.assertEmpty(foo_handler2.events)
    self.assertEmpty(bar_handler.events)
    self.assertEmpty(baz_handler.events)

    # Subscribe to a sender.
    subscription.subscribe([foo_handler1, bar_handler], a1)
    subscription.subscribe(baz_handler)

    # Check subscribers.
    self.assertEqual(
        list(subscription.subscribers(a1)), [foo_handler1, bar_handler]
    )
    self.assertEmpty(list(subscription.subscribers(a2)))
    self.assertEqual(list(subscription.subscribers(b)), [baz_handler])
    self.assertEmpty(list(subscription.subscribers(SenderA)))
    self.assertEqual(list(subscription.subscribers(SenderB)), [baz_handler])

    # Check subscriptions.
    self.assertEqual(list(subscription.subscriptions(foo_handler1)), [a1])
    self.assertEmpty(list(subscription.subscriptions(foo_handler2)))
    self.assertEqual(list(subscription.subscriptions(bar_handler)), [a1])
    self.assertEqual(list(subscription.subscriptions(baz_handler)), [SenderB])

    a2.foo()
    a2.bar()
    b.baz()
    self.assertEmpty(foo_handler1.events)  # Not subscribed to a2.
    self.assertEmpty(foo_handler2.events)  # No subscriptions yet.
    self.assertEmpty(bar_handler.events)  # Not subscribe to a2.
    self.assertLen(baz_handler.events, 1)
    self.assertIs(baz_handler.events[-1][0], baz_handler)
    self.assertIs(baz_handler.events[-1][1].sender, b)

    a1.foo()
    self.assertLen(foo_handler1.events, 1)
    self.assertIs(foo_handler1.events[-1][0], foo_handler1)
    self.assertIs(foo_handler1.events[-1][1].sender, a1)
    self.assertEmpty(foo_handler2.events)  # No subscriptions yet.
    self.assertEmpty(bar_handler.events)  # a1.bar not called.
    self.assertLen(baz_handler.events, 1)  # a1.baz not called.

    a1.bar()
    self.assertLen(bar_handler.events, 1)
    self.assertIs(bar_handler.events[-1][0], bar_handler)
    self.assertIs(bar_handler.events[-1][1].sender, a1)
    self.assertLen(foo_handler1.events, 1)

    # Unsubscribe foo_handler1.
    subscription.unsubscribe(foo_handler1)
    self.assertEqual(list(subscription.subscribers(a1)), [bar_handler])
    self.assertEmpty(list(subscription.subscriptions(foo_handler1)))

    a1.foo()
    b.baz()
    self.assertLen(foo_handler1.events, 1)
    self.assertLen(baz_handler.events, 2)

    # Add subscription of a different subscriber.
    subscription.subscribe(foo_handler2, [a1, a2])
    self.assertEqual(
        list(subscription.subscribers(a1)), [bar_handler, foo_handler2]
    )
    self.assertEqual(list(subscription.subscribers(a2)), [foo_handler2])
    self.assertEqual(list(subscription.subscriptions(foo_handler2)), [a1, a2])

    a1.foo()
    a2.foo()
    b.baz()
    self.assertLen(foo_handler1.events, 1)
    self.assertLen(foo_handler2.events, 2)
    self.assertIs(foo_handler2.events[-1][0], foo_handler2)
    self.assertIs(foo_handler2.events[-1][1].sender, a2)
    self.assertLen(baz_handler.events, 3)

    # Clear all subscriptions.
    subscription.clear_subscriptions()

    # Senders have no subscribers yet.
    self.assertEmpty(list(subscription.subscribers(a1)))
    self.assertEmpty(list(subscription.subscribers(a2)))
    self.assertEmpty(list(subscription.subscribers(b)))
    self.assertEmpty(list(subscription.subscribers(SenderA)))
    self.assertEmpty(list(subscription.subscribers(SenderB)))

    # Subscribers have not subscribed any senders yet.
    self.assertEmpty(list(subscription.subscriptions(foo_handler1)))
    self.assertEmpty(list(subscription.subscriptions(foo_handler2)))
    self.assertEmpty(list(subscription.subscriptions(bar_handler)))
    self.assertEmpty(list(subscription.subscriptions(baz_handler)))

    a1.foo()
    a1.bar()
    a2.foo()
    a2.bar()
    b.baz()

    # No new events are triggered.
    self.assertLen(foo_handler1.events, 1)
    self.assertLen(foo_handler2.events, 2)
    self.assertLen(bar_handler.events, 1)
    self.assertLen(baz_handler.events, 3)

    # Bad cases.
    class A:
      pass

    with self.assertRaisesRegex(
        TypeError, '.* is not the expected sender type'):
      subscription.subscribe(foo_handler1, A())

    with self.assertRaisesRegex(TypeError, '.* is not an event handler'):
      subscription.subscribe(A())

    with self.assertRaisesRegex(
        ValueError, 'There is no subscription for'):
      subscription.unsubscribe(foo_handler1)

    with self.assertRaisesRegex(
        ValueError, 'There is no subscription for'):
      subscription.unsubscribe(foo_handler1, SenderA())


if __name__ == '__main__':
  unittest.main()
