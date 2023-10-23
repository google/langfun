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
"""Image modality."""

import base64
import imghdr
from typing import Annotated, cast

import langfun.core as lf
import requests


class Image(lf.Modality):
  """Base class for image."""

  @classmethod
  def from_bytes(cls, content: bytes) -> 'Image':
    """Creates an image from bytes."""
    return ImageContent(content)

  @classmethod
  def from_uri(cls, uri: str) -> 'Image':
    """Creates an image from file."""
    return ImageFile(uri)

  @property
  def image_format(self) -> str:
    iformat = imghdr.what(None, self.to_bytes())
    if iformat not in ['png', 'jpeg']:
      raise ValueError(f'Unsupported image format: {iformat!r}.')
    return cast(str, iformat)


class ImageContent(Image):
  """Raw image content."""

  content: Annotated[bytes, 'The raw content of the image.']

  def to_bytes(self) -> bytes:
    return self.content

  def _repr_html_(self) -> str:
    image_raw = base64.b64encode(self.to_bytes()).decode()
    return f'<img src="data:image/{self.image_format};base64,{image_raw}">'


class ImageFile(Image):
  """A image file."""

  uri: Annotated[str, 'The URI of the image. (e.g. https://..../a.jpg).']

  def _on_bound(self):
    super()._on_bound()
    self._bytes = None

  def to_bytes(self) -> bytes:
    if self._bytes is None:
      self._bytes = requests.get(self.uri).content
    return self._bytes

  def _repr_html_(self) -> str:
    return f'<img src="{self.uri}">'
