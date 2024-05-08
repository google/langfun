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
"""MIME type data."""

import abc
import base64
from typing import Annotated, Union
import langfun.core as lf
import pyglove as pg
import requests


class MimeType(lf.Modality):
  """Base for MIME type data."""

  @property
  @abc.abstractmethod
  def mime_type(self) -> str:
    """Returns the MIME type."""

  uri: Annotated[str | None, 'The URI for locating the MIME data. '] = None

  content: Annotated[
      Union[str, bytes, None], 'The raw content of the MIME type.'
  ] = None

  def _on_bound(self):
    super()._on_bound()
    if self.uri is None and self.content is None:
      raise ValueError('Either uri or content must be provided.')

  def to_bytes(self) -> bytes:
    if self.content is not None:
      return self.content

    assert self.uri is not None
    if self.uri.lower().startswith(('http:', 'https:', 'ftp:')):
      content = requests.get(
          self.uri,
          headers={'User-Agent': 'Langfun'},
      ).content
    else:
      content = pg.io.readfile(self.uri, mode='rb')
    self.rebind(content=content, skip_notification=True)
    return self.content

  @property
  def content_uri(self) -> str:
    base64_content = base64.b64encode(self.to_bytes()).decode()
    return f'data:{self.mime_type};base64,{base64_content}'

  @classmethod
  def from_uri(cls, uri: str, **kwargs) -> 'MimeType':
    return cls(uri=uri, content=None, **kwargs)

  @classmethod
  def from_bytes(cls, content: bytes | str, **kwargs) -> 'MimeType':
    return cls(content=content, **kwargs)


@pg.use_init_args(['type', 'content', 'uri'])
class Custom(MimeType):
  """Custom MIME data."""

  type: Annotated[
      str, 'The MIME type of the data. E.g. text/plain, or image/png. '
  ]

  @property
  def mime_type(self) -> str:
    return self.type


class PDF(Custom):
  """PDF document."""
  type = 'application/pdf'
