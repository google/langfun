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
"""Microsoft Office file types."""

import base64
import io
import os
from typing import Iterable
from langfun.core.modalities import mime
from langfun.core.modalities import pdf
import requests


class Xlsx(mime.Mime):
  """Xlsx file type."""

  MIME_PREFIX = (
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  )

  def to_html(self) -> str:
    try:
      import pandas as pd  # pylint: disable=g-import-not-at-top
      import openpyxl      # pylint: disable=g-import-not-at-top, unused-import
      df = pd.read_excel(io.BytesIO(self.to_bytes()))
      return df.to_html()
    except ImportError as e:
      raise RuntimeError(
          'Please install "langfun[mime-xlsx]" to enable XLSX support.'
      ) from e

  def _repr_html_(self) -> str:
    return self.to_html()

  def _is_compatible(self, mime_types: Iterable[str]) -> bool:
    return bool(set(mime_types).intersection([
        'text/html',
        'text/plain',
    ]))

  def _make_compatible(self, mime_types: Iterable[str]) -> mime.Mime:
    """Returns the MimeType of the converted file."""
    del mime_types
    return mime.Mime(uri=self.uri, content=self.to_html())


class Docx(mime.Mime):
  """Docx file type."""

  MIME_PREFIX = (
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  )

  def to_xml(self) -> str:
    try:
      import docx  # pylint: disable=g-import-not-at-top
      doc = docx.Document(io.BytesIO(self.to_bytes()))
      return str(doc.element.xml)
    except ImportError as e:
      raise RuntimeError(
          'Please install "langfun[mime-docx]" to enable Docx support.'
      ) from e

  def _repr_html_(self) -> str:
    return self.to_xml()

  def _is_compatible(self, mime_types: Iterable[str]) -> bool:
    return bool(set(mime_types).intersection([
        'application/xml',
        'text/xml',
        'text/plain',
    ]))

  def _make_compatible(self, mime_types: Iterable[str]) -> mime.Mime:
    """Returns the MimeType of the converted file."""
    del mime_types
    return mime.Mime(uri=self.uri, content=self.to_xml())


class Pptx(mime.Mime):
  """Pptx file type."""

  MIME_PREFIX = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
  API_URL = 'https://v2.convertapi.com/convert/pptx/to/pdf'

  def to_pdf(self, convert_api_key: str | None = None) -> pdf.PDF:
    api_key = convert_api_key or os.environ.get('CONVERT_API_KEY')
    url = f'{self.API_URL}?Secret={api_key}'

    json = {
        'Parameters': [{
            'Name': 'File',
            'FileValue': {
                'Name': os.path.basename(self.uri) if self.uri else 'tmp.pptx',
                'Data': base64.b64encode(self.to_bytes()).decode('utf-8'),
            },
        }]
    }
    response = requests.post(url, json=json).json()
    base64_pdf = response['Files'][0]['FileData']
    return pdf.PDF.from_bytes(base64.b64decode(base64_pdf))

  def _is_compatible(self, mime_types: Iterable[str]) -> bool:
    return 'application/pdf' in mime_types

  def _make_compatible(self, mime_types: Iterable[str]) -> mime.Mime:
    """Returns the MimeType of the converted file."""
    del mime_types
    return self.to_pdf()
