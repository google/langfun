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

import functools
import io
import os
from typing import Any, Iterable

from langfun.core.modalities import mime

try:
  from PIL import Image as pil_image    # pylint: disable=g-import-not-at-top
  PILImage = pil_image.Image
  pil_open = pil_image.open
except ImportError:
  PILImage = Any

  def pil_open(*unused_args, **unused_kwargs):
    raise RuntimeError(
        'Please install "langfun[mime-pil]" to enable PIL image support.'
    )


class Image(mime.Mime):
  """Represents an image for communicating with language models.

  `lf.Image` can be initialized from a URI (HTTP/HTTPS URL or local path)
  using `lf.Image.from_uri()` or from raw bytes using `lf.Image.from_bytes()`.

  **Example:**

  ```python
  import langfun as lf

  # Load image from path
  image = lf.Image.from_path('/path/to/image.png')

  # Use image in a prompt
  prompt = lf.Template('Describe this image: {{image}}', image=image)
  response = lf.query(prompt, lm=lf.llms.Gemini25Flash())
  print(response)
  ```
  """

  MIME_PREFIX = 'image'

  @functools.cached_property
  def image_format(self) -> str:
    return self.mime_type.removeprefix(self.MIME_PREFIX + '/')

  def _mime_control_for(self, uri: str) -> str:
    return f'<img src="{uri}">'

  @functools.cached_property
  def size(self) -> tuple[int, int]:
    """Returns the size (width, height) of the image in pixels."""
    img = pil_open(io.BytesIO(self.to_bytes()))
    return img.size

  def to_pil_image(self) -> PILImage:   # pytype: disable=invalid-annotation
    return pil_open(io.BytesIO(self.to_bytes()))

  def _is_compatible(self, mime_types: Iterable[str]) -> bool:
    """Returns True if this image is compatible with any of the MIME types."""
    mime_types = set(mime_types)
    if self.mime_type in mime_types:
      return True
    if self.mime_type == 'image/gif':
      return bool(mime_types & {'image/png', 'image/jpeg', 'image/webp'})
    return False

  def _make_compatible(self, mime_types: Iterable[str]) -> 'Image':
    """Converts this image to a compatible format if needed."""
    mime_types = set(mime_types)
    if self.mime_type in mime_types:
      return self
    if self.mime_type == 'image/gif':
      # Convert to first supported format
      for target_format, pil_format in [
          ('image/png', 'PNG'),
          ('image/jpeg', 'JPEG'),
          ('image/webp', 'WEBP'),
      ]:
        if target_format in mime_types:
          return self._convert_to_format(pil_format)
    return self

  def _convert_to_format(self, pil_format: str) -> 'Image':
    """Converts this image to the specified PIL format."""
    buf = io.BytesIO()
    img = self.to_pil_image()
    # JPEG doesn't support transparency, convert RGBA to RGB
    if pil_format == 'JPEG' and img.mode in ('RGBA', 'P'):
      img = img.convert('RGB')
    try:
      img.save(buf, format=pil_format)
    except OSError:
      cwd = os.getcwd()
      try:
        os.chdir('/tmp')
        img.save(buf, format=pil_format)
      finally:
        os.chdir(cwd)
    return self.from_bytes(buf.getvalue())

  @classmethod
  def from_pil_image(cls, img: PILImage) -> 'Image':  # pytype: disable=invalid-annotation
    buf = io.BytesIO()
    try:
      img.save(buf, format='PNG')
    except OSError:
      cwd = os.getcwd()
      try:
        os.chdir('/tmp')
        img.save(buf, format='PNG')
      finally:
        os.chdir(cwd)
    return cls.from_bytes(buf.getvalue())
