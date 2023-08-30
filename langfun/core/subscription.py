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
"""langfun subscription framework."""

import abc
import collections
import dataclasses
import functools
import inspect
import typing
from typing import Any, Callable, Generic, Iterator, Sequence, Type, TypeVar, Union
import weakref


SenderType = TypeVar('SenderType')


@dataclasses.dataclass
class Event(Generic[SenderType]):
  sender: SenderType


EventType = TypeVar('EventType')


class EventHandler(Generic[EventType], metaclass=abc.ABCMeta):
  """Interface for event subscriber."""

  @classmethod
  @functools.cache
  def event_type(cls) -> Type[Event[Any]]:
    """Returns acceptable event type."""
    return _get_generic_arg(cls, EventHandler)

  @classmethod
  @functools.cache
  def sender_type(cls) -> Type[Any]:
    """Returns acceptable sender type."""
    return _get_generic_arg(cls.event_type(), Event)

  @classmethod
  def accepts(cls, event: Event[Any]) -> bool:
    """Returns True if current event handler class can accepts an event."""
    return isinstance(event, cls.event_type())

  @abc.abstractmethod
  def on_event(self, event: EventType) -> None:
    """Handles an event."""


class _EventManager:
  """Event manager."""

  def __init__(self):
    self._sender_registry = weakref.WeakKeyDictionary()
    self._sender_type_registry: dict[
        Type[Any], list[EventHandler[Any]]  # Sender type.
    ] = collections.defaultdict(list)

  def _map_sender_subscriber(
      self,
      func: Callable[
          [
              Union[Any, Type[Any], None],  # Sender object, type or None.
              EventHandler[Any],
          ],  # Subscriber (event handler).
          None,
      ],
      sender_or_senders: Union[
          Any, Type[Any], Sequence[Union[Any, Type[Any]]], None
      ],
      subscriber_or_subscribers: Union[
          EventHandler[Any], Sequence[EventHandler[Any]]
      ],
  ) -> None:
    """Maps the product of senders and subscribers as the input to a function."""
    if isinstance(subscriber_or_subscribers, (tuple, list)):
      subscriber_list = list(subscriber_or_subscribers)
    else:
      subscriber_list = [subscriber_or_subscribers]

    if isinstance(sender_or_senders, (tuple, list)):
      senders = list(sender_or_senders)
    else:
      senders = [sender_or_senders]

    for subscriber in subscriber_list:
      for sender in senders:
        func(subscriber, sender)

  def _sender_info(
      self, sender: Union[Any, Type[Any], None]
  ) -> tuple[
      Any,  # Sender or None.
      Type[Any] | None,  # Sender type or None.
      list[EventHandler[Any]],  # Subscribers.
  ]:
    """Returns the sender id, sender type and current subscribers."""
    if inspect.isclass(sender):
      return None, sender, self._sender_type_registry[sender]
    elif sender is not None:
      sender_entry = self._sender_registry.get(sender, None)
      if sender_entry is None:
        sender_entry = []
        self._sender_registry[sender] = sender_entry
      return sender, type(sender), sender_entry
    else:
      return None, None, []

  def _subscribe(
      self,
      subscriber: EventHandler[Any],
      sender: Union[Any, Type[Any], None] = None,
  ) -> None:
    """Subscribes a subscriber to a sender."""
    if not isinstance(subscriber, EventHandler):
      raise TypeError(f'{subscriber!r} is not an event handler.')

    sender, sender_type, subscriber_list = self._sender_info(sender)
    subscriber_sender_type = subscriber.sender_type()

    if sender_type is None:
      sender_type = subscriber_sender_type
      subscriber_list = self._sender_type_registry[subscriber_sender_type]
    elif not issubclass(sender_type, subscriber_sender_type):
      raise TypeError(
          f'{sender_type!r} is not the expected sender type '
          f'({subscriber_sender_type!r}) for subscriber '
          f'{subscriber.__class__.__name__!r}'
      )
    assert subscriber_list is not None
    subscriber_list.append(subscriber)

  def _unsubscribe(
      self,
      subscriber: EventHandler[Any],
      sender: Union[Any, Type[Any], None] = None,
  ) -> None:
    """Unsubscribes a subscriber from a sender."""
    # Unsubscribe.
    unsubscribed = 0
    sender, sender_type, _ = self._sender_info(sender)

    if sender is not None:
      # Unsubscribe a single sender.
      if sender in self._sender_registry:
        unsubscribed += _remove_list_items(
            self._sender_registry[sender], lambda x: x is subscriber
        )
    else:
      # Unsubscribe all subscriptions for the subscriber.
      subscriber_sender_type = subscriber.sender_type()
      sender_type = sender_type or subscriber_sender_type

      # Unsubscribe object-based subscriptions.
      for subscriber_list in self._sender_registry.values():
        unsubscribed += _remove_list_items(
            subscriber_list, lambda x: x is subscriber
        )

      # Unsubscribe type-based subscriptions.
      for st, subscriber_list in self._sender_type_registry.items():
        if issubclass(st, sender_type):
          unsubscribed += _remove_list_items(
              subscriber_list, lambda x: x is subscriber
          )

    if unsubscribed == 0:
      raise ValueError(
          f'There is no subscription for {subscriber!r} to unsubscribe.'
      )

  def subscribe(
      self,
      subscriber: Union[EventHandler[Any], Sequence[EventHandler[Any]]],
      sender: Union[
          Any, Type[Any], Sequence[Union[Any, Type[Any]]], None
      ] = None,
  ) -> None:
    """Subscribes one or a list subscribers to one or a list of senders."""
    return self._map_sender_subscriber(
        self._subscribe,
        sender_or_senders=sender,
        subscriber_or_subscribers=subscriber,
    )

  def unsubscribe(
      self,
      subscriber: Union[EventHandler[Any], Sequence[EventHandler[Any]]],
      sender: Union[
          Any, Type[Any], Sequence[Union[Any, Type[Any]]], None
      ] = None,
  ) -> None:
    """Unsubscribes one or a list subscribers from one or a list of senders."""
    return self._map_sender_subscriber(
        self._unsubscribe,
        sender_or_senders=sender,
        subscriber_or_subscribers=subscriber,
    )

  def emit(self, event: Event[Any]) -> None:
    """Emits an event."""
    for subscriber in self.subscribers(event.sender):
      if subscriber.accepts(event):
        subscriber.on_event(event)

  def subscribers(self, sender: Any | Type[Any]) -> Iterator[EventHandler[Any]]:
    """Iterates the subscribers of a sender."""
    visited = set()
    sender, sender_type, _ = self._sender_info(sender)

    # Yield instance-level subscribers.
    if sender is not None and sender in self._sender_registry:
      for subscriber in self._sender_registry[sender]:
        if id(subscriber) not in visited:
          yield subscriber
          visited.add(id(subscriber))

    # Yield type_level subscribers.
    for registered_type, subscriber_list in self._sender_type_registry.items():
      if isinstance(sender, registered_type) or (
          sender is None and issubclass(sender_type, registered_type)
      ):
        for subscriber in subscriber_list:
          if id(subscriber) not in visited:
            yield subscriber
            visited.add(id(subscriber))

  def subscriptions(self, subscriber: EventHandler[Any]) -> Iterator[Any]:
    """Returns all subscriptions of a subscriber."""
    # Yield subscriptions based on individual senders.
    for sender, subscriber_list in self._sender_registry.items():
      if any(x is subscriber for x in subscriber_list):
        yield sender

    # Yield subscriptions based on sender types.
    for sender_type, subscribers_list in self._sender_type_registry.items():
      if any(x is subscriber for x in subscribers_list):
        yield sender_type

  def clear(self) -> None:
    """Clear all subscriptions."""
    self._sender_registry.clear()
    self._sender_type_registry.clear()


_event_manager = _EventManager()


def subscribe(
    subscriber: Union[EventHandler[Any], Sequence[EventHandler[Any]]],
    sender: Union[Any, Type[Any], Sequence[Union[Any, Type[Any]]], None] = None,
) -> None:
  """Subscribes one or a list subscribers to one or a list of senders.

  Args:
    subscriber: An event subscriber or a list of event handlers whose event
      sender matches with the sender.
    sender: One or a list of sender object or sender type, whose emitted events
      should be received. If None, the expecting sender type from the subscriber
      will be used.
  """
  _event_manager.subscribe(subscriber, sender)


def unsubscribe(
    subscriber: Union[EventHandler[Any], Sequence[EventHandler[Any]]],
    sender: Union[Any, Type[Any], Sequence[Union[Any, Type[Any]]], None] = None,
) -> None:
  """Unsubscribes one or a list subscribers from one or a list of senders.

  Args:
    subscriber: An event subscriber or a list of event handlers whose event
      sender matches with the sender.
    sender: One or a list of sender object or sender type, whose emitted events
      should be received. If None, the expecting sender type from the subscriber
      will be used.
  """
  _event_manager.unsubscribe(subscriber, sender)


def emit(event: Event[Any]) -> None:
  """Emits an event."""
  _event_manager.emit(event)


def subscribers(sender: Any | Type[Any]) -> Iterator[EventHandler[Any]]:
  """Iterates the subscribers of a sender or sender type."""
  return _event_manager.subscribers(sender)


def subscriptions(subscriber: EventHandler[Any]) -> Iterator[Any]:
  """Iterates the subscriptions of a subscriber."""
  return _event_manager.subscriptions(subscriber)


def clear_subscriptions() -> None:
  """Clear all subscriptions."""
  _event_manager.clear()


def _get_generic_arg(cls, generic_cls):
  """Get the first type argument from a generic type or subtype."""
  for orig_base in getattr(cls, '__orig_bases__', ()):
    if typing.get_origin(orig_base) is generic_cls:
      return typing.get_args(orig_base)[0]
  raise TypeError(
      f'Cannot get type argument from generic class {generic_cls}` '
      f'type from {cls}'
  )


def _remove_list_items(list_values, pred) -> int:
  """Remove list items based on predicate."""
  indices_to_remove = []
  for i, v in enumerate(list_values):
    if pred(v):
      indices_to_remove.append(i)
  for i in reversed(indices_to_remove):
    list_values.pop(i)
  return len(indices_to_remove)
