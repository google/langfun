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
"""MIME type data."""

import base64
import functools
from typing import Annotated, Any, Iterable, Type, Union
import langfun.core as lf
# Placeholder for Google-internal internet access import.
import pyglove as pg
import requests   # pylint: disable=unused-import


try:
  import magic    # pylint: disable=g-import-not-at-top
  from_buffer = magic.from_buffer
except ImportError:
  def from_buffer(*unused_args, **unused_kwargs):
    raise RuntimeError(
        'Please install "langfun[mime-auto]" to enable automatic MIME support.'
    )


class Mime(lf.Modality):
  """Base for MIME data."""

  # The regular expression that describes the MIME type str.
  # If None, the MIME type is dynamic. Subclass could override.
  MIME_PREFIX = None

  uri: Annotated[str | None, 'The URI for locating the MIME data. '] = None

  content: Annotated[
      Union[str, bytes, None], 'The raw content of the MIME type.'
  ] = None

  @functools.cached_property
  def mime_type(self) -> str:
    """Returns the MIME type."""
    mime = from_buffer((self.to_bytes()), mime=True)
    if (
        self.MIME_PREFIX
        and not mime.lower().startswith(self.MIME_PREFIX)
        # NOTE(daiyip): libmagic fails to detect the MIME type of some binary
        # files.
        and mime != 'application/octet-stream'
    ):
      raise ValueError(
          f'Expected MIME type: {self.MIME_PREFIX}, Encountered: {mime}'
      )
    return mime

  @functools.cached_property
  def is_text(self) -> bool:
    return self.mime_type.startswith((
        'text/',
        'application/javascript',
        'application/json',
        'application/ld+json',
        'application/plain',
        'application/rtf',
        'application/xhtml+xml',
        'application/xml',
        'application/x-javascript',
        'application/x-python-code',
        'application/x-tex',
        'application/x-typescript',
        'application/x-yaml',
    ))

  @property
  def is_binary(self) -> bool:
    """Returns True if the MIME type is a binary type."""
    return not self.is_text

  def to_text(self) -> str:
    """Returns the text content of the MIME type."""
    if not self.is_text:
      raise lf.ModalityError(
          f'MIME type {self.mime_type!r} cannot be converted to text.'
      )
    return self.to_bytes().decode()

  def is_compatible(
      self, mime_types: str | Iterable[str]
  ) -> bool:
    """Returns True if this object is compatible to any of the MIME types."""
    if isinstance(mime_types, str):
      mime_types = {mime_types}
    return self._is_compatible(mime_types)

  def _is_compatible(self, mime_types: Iterable[str]):
    return self.mime_type in mime_types

  def make_compatible(
      self,
      mime_types: str | Iterable[str]
      ) -> Union['Mime', list['Mime']]:
    """Makes compatible MIME objects from this object."""
    if isinstance(mime_types, str):
      mime_types = {mime_types}
    if not self._is_compatible(mime_types):
      raise lf.ModalityError(
          f'MIME type {self.mime_type!r} cannot be converted to supported '
          f'types: {mime_types!r}.'
      )
    return self._make_compatible(mime_types)

  def _make_compatible(
      self,
      mime_types: Iterable[str]
  ) -> Union['Mime', list['Mime']]:
    """Makes compatbile MIME objects from this object."""
    del mime_types
    return self

  def _on_bound(self):
    super()._on_bound()
    if self.uri is None and self.content is None:
      raise ValueError('Either uri or content must be provided.')

  def to_bytes(self) -> bytes:
    if self.content is not None:
      return self.content

    self.rebind(content=self.download(self.uri), skip_notification=True)
    return self.content

  @property
  def content_uri(self) -> str:
    """Returns the URI with encoded content."""
    base64_content = base64.b64encode(self.to_bytes()).decode()
    return f'data:{self.mime_type};base64,{base64_content}'

  @property
  def embeddable_uri(self) -> str:
    """Returns the URI that can be embedded in HTML."""
    if self.uri and self.uri.lower().startswith(('http:', 'https:', 'ftp:')):
      return self.uri
    return self.content_uri

  @classmethod
  def from_uri(cls, uri: str, **kwargs) -> 'Mime':
    if uri.startswith('data:'):
      mime_type, content = cls._parse_data_uri(uri)
      return cls.class_from_mime_type(mime_type).from_bytes(content, **kwargs)

    if cls is Mime:
      content = cls.download(uri)
      mime = from_buffer(content, mime=True).lower()
      return cls.class_from_mime_type(mime)(uri=uri, content=content, **kwargs)
    return cls(uri=uri, content=None, **kwargs)

  @classmethod
  def _parse_data_uri(cls, uri: str) -> tuple[str, bytes]:
    """Returns the MIME type and content from the given data URI."""
    assert uri.startswith('data:'), uri
    mime_end_pos = uri.find(';', 0)
    if mime_end_pos == -1:
      raise ValueError(f'Invalid data URI: {uri!r}.')
    mime_type = uri[5: mime_end_pos].strip().lower()
    encoding_end_pos = uri.find(',', mime_end_pos + 1)
    if encoding_end_pos == -1:
      raise ValueError(f'Invalid data URI: {uri!r}.')
    encoding = uri[mime_end_pos + 1: encoding_end_pos].strip().lower()
    if encoding != 'base64':
      raise ValueError(f'Unsupported encoding: {encoding!r}.')
    base64_content = uri[encoding_end_pos + 1:].strip().encode()
    return mime_type, base64.b64decode(base64_content)

  @classmethod
  def from_bytes(cls, content: bytes | str, **kwargs) -> 'Mime':
    if cls is Mime:
      mime = from_buffer(content, mime=True).lower()
      return cls.class_from_mime_type(mime)(content=content, **kwargs)
    return cls(content=content, **kwargs)

  @classmethod
  def class_from_mime_type(cls, mime_type: str) -> Type['Mime']:
    """Subclass from the given MIME type."""
    for subcls in cls.__subclasses__():
      if subcls.MIME_PREFIX is not None and mime_type.startswith(
          subcls.MIME_PREFIX):
        return subcls
    return cls

  @classmethod
  def download(cls, uri: str) -> bytes | str:
    """Downloads the content of the given URI."""
    if uri.lower().startswith(('http:', 'https:', 'ftp:')):
      return requests.get(uri, headers={'User-Agent': 'Mozilla/5.0'}).content
    else:
      content = pg.io.readfile(uri, mode='rb')
      assert content is not None
      return content

  def _html_tree_view_content(
      self,
      **kwargs) -> str:
    return self._raw_html()

  def _html_tree_view(
      self,
      view: pg.views.HtmlTreeView,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ):
    extra_flags = extra_flags if extra_flags is not None else {}
    raw_mime_content = extra_flags.get('raw_mime_content', False)
    display_modality_when_hover = extra_flags.get(
        'display_modality_when_hover', False
    )
    if raw_mime_content:
      kwargs['enable_summary'] = False
    elif display_modality_when_hover:
      kwargs.update(
          enable_summary=True,
          enable_summary_tooltip=True,
      )
    return super()._html_tree_view(
        view=view, extra_flags=extra_flags, **kwargs
    )

  def _html_tree_view_summary(
      self,
      *,
      view: pg.views.HtmlTreeView,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ):
    extra_flags = extra_flags or {}
    if extra_flags.get('display_modality_when_hover', False):
      def summary_tooltip(*args, content: str | None = None, **kwargs):
        del content
        return view.tooltip(*args, content=self._raw_html(), **kwargs)
    else:
      summary_tooltip = None
    return super()._html_tree_view_summary(
        view=view,
        summary_tooltip_fn=summary_tooltip,
        extra_flags=extra_flags,
        **kwargs
    )

  def _raw_html(self) -> str:
    if self.uri and self.uri.lower().startswith(('http:', 'https:', 'ftp:')):
      uri = self.uri
    else:
      uri = self.content_uri
    return self._mime_control_for(uri)

  def _mime_control_for(self, uri) -> str:
    return f'<embed type="{self.mime_type}" src="{uri}"/>'


@pg.use_init_args(['mime', 'content', 'uri'])
class Custom(Mime):
  """Custom MIME data."""

  mime: Annotated[
      str, 'The MIME type of the data. E.g. text/plain, or image/png. '
  ]

  @property
  def mime_type(self) -> str:
    return self.mime
