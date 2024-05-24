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
"""PDF tests."""

import unittest
from langfun.core.modalities import pdf as pdf_lib


pdf_bytes = (
    b'%PDF-1.1\n%\xc2\xa5\xc2\xb1\xc3\xab\n\n1 0 obj\n'
    b'<< /Type /Catalog\n     /Pages 2 0 R\n  >>\nendobj\n\n2 0 obj\n '
    b'<< /Type /Pages\n     /Kids [3 0 R]\n     '
    b'/Count 1\n     /MediaBox [0 0 300 144]\n  '
    b'>>\nendobj\n\n3 0 obj\n  '
    b'<<  /Type /Page\n      /Parent 2 0 R\n      /Resources\n       '
    b'<< /Font\n'
    b'<< /F1\n'
    b'<< /Type /Font\n'
    b'/Subtype /Type1\n'
    b'/BaseFont /Times-Roman\n'
    b'>>\n>>\n>>\n      '
    b'/Contents 4 0 R\n  >>\nendobj\n\n4 0 obj\n  '
    b'<< /Length 55 >>\nstream\n  BT\n    /F1 18 Tf\n    0 0 Td\n    '
    b'(Hello World) Tj\n  ET\nendstream\nendobj\n\nxref\n0 5\n0000000000 '
    b'65535 f \n0000000018 00000 n \n0000000077 00000 n \n0000000178 00000 n '
    b'\n0000000457 00000 n \ntrailer\n  <<  /Root 1 0 R\n      /Size 5\n  '
    b'>>\nstartxref\n565\n%%EOF\n'
)


class PdfTest(unittest.TestCase):

  def test_pdf(self):
    pdf = pdf_lib.PDF.from_bytes(pdf_bytes)
    self.assertEqual(pdf.mime_type, 'application/pdf')

  def test_repr_html(self):
    pdf = pdf_lib.PDF.from_bytes(pdf_bytes)
    self.assertIn(
        '<embed type="application/pdf" src="data:application/pdf;base64,',
        pdf._repr_html_()
    )


if __name__ == '__main__':
  unittest.main()
