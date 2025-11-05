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
"""PDF modality."""

from langfun.core.modalities import mime


class PDF(mime.Mime):
  """Represents a PDF document for communicating with language models.

  `lf.PDF` can be initialized from a URI (HTTP/HTTPS URL or local path)
  using `lf.PDF.from_uri()` or from raw bytes using `lf.PDF.from_bytes()`.

  **Example:**

  ```python
  import langfun as lf

  # Load PDF from path
  pdf = lf.PDF.from_path('/path/to/document.pdf')

  # Use PDF in a prompt
  prompt = lf.Template('Summarize this document: {{pdf}}', pdf=pdf)
  response = lf.query(prompt, lm=lf.llms.Gemini25Flash())
  print(response)
  ```
  """

  MIME_PREFIX = 'application/pdf'
