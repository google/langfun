# Copyright 2026 The Langfun Authors
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
"""Cross-checks model prices against the `llm-prices` table.

`llm-prices` (https://github.com/Mattbusel/llm-prices, MIT) re-reads the
OpenAI, Anthropic and Gemini pricing pages and records, for every row, the
page it read and the date it read it. This test compares each priced
`SUPPORTED_MODELS` entry with the matching row, using the langfun pricing in
effect on the date the row was checked, so promotional rates with an end date
compare correctly. Models the table does not list are skipped.

The test is skipped when `llm-prices` is not installed. CI pins a version, so a
price change shows up here when the pin is bumped, not on unrelated changes.
"""

import datetime
import unittest

import langfun.core as lf
from langfun.core.llms import anthropic
from langfun.core.llms import gemini
from langfun.core.llms import openai

try:
  import llm_prices  # pylint: disable=g-import-not-at-top
except ImportError:
  llm_prices = None


def _differs(ours: float | None, theirs: float | None) -> bool:
  if theirs is None:
    # The table has no price for this rate, so there is nothing to compare.
    return False
  return ours is None or abs(ours - theirs) > 1e-9


def _price_drift(
    model_info: lf.ModelInfo, row: 'llm_prices.Price'
) -> list[str]:
  """Returns the rates where `model_info` disagrees with an llm-prices row."""
  pricing = model_info.pricing.effective_on(
      datetime.date.fromisoformat(row.checked)
  )
  drift = []

  def check(name: str, ours: float | None, theirs: float | None) -> None:
    if _differs(ours, theirs):
      drift.append(f'{name}: langfun={ours} llm-prices={theirs}')

  check('input', pricing.cost_per_1m_input_tokens, row.input)
  check('output', pricing.cost_per_1m_output_tokens, row.output)
  check(
      'cached input',
      pricing.cost_per_1m_cached_input_tokens,
      row.cache_read,
  )
  if isinstance(pricing, anthropic.AnthropicModelInfo.Pricing):
    check(
        '5-minute cache write',
        pricing._cache_creation_rate(pricing.cost_per_1m_input_tokens),  # pylint: disable=protected-access
        row.rate('cache_write', '5m'),
    )
  if isinstance(pricing, gemini.GeminiModelInfo.Pricing):
    long_context = row.long_context
    long_input = pricing.cost_per_1m_input_tokens_with_prompt_longer_than_128k
    if long_context is not None:
      check('long-context input', long_input, long_context.input)
      if long_input is not None:
        if pricing.long_context_threshold != long_context.threshold_tokens:
          drift.append(
              'long-context threshold: '
              f'langfun={pricing.long_context_threshold} '
              f'llm-prices={long_context.threshold_tokens}'
          )
        long_output = (
            pricing.cost_per_1m_output_tokens_with_prompt_longer_than_128k
        )
        check(
            'long-context output',
            pricing.cost_per_1m_output_tokens
            if long_output is None else long_output,
            long_context.output,
        )
        long_cached = (
            pricing.cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k
        )
        if long_cached is not None:
          check('long-context cached input', long_cached,
                long_context.cache_read)
    elif long_input is not None:
      drift.append(
          f'long-context input: langfun={long_input} llm-prices=None'
      )
  return drift


@unittest.skipIf(llm_prices is None, 'llm-prices is not installed.')
class PricingCrossCheckTest(unittest.TestCase):

  def _check_provider(self, provider: str, supported_models) -> None:
    compared = 0
    drift = []
    for model_info in supported_models:
      if model_info.pricing is None:
        continue
      row = llm_prices.lookup_provider(provider, model_info.model_id)
      if row is None:
        continue
      compared += 1
      for d in _price_drift(model_info, row):
        drift.append(
            f'{model_info.model_id} ({row.model}, checked {row.checked} '
            f'at {row.source_url}): {d}'
        )
    self.assertGreater(compared, 0)
    self.assertEqual(drift, [], '\n'.join(drift))

  def test_openai(self):
    self._check_provider('openai', openai.SUPPORTED_MODELS)

  def test_anthropic(self):
    self._check_provider('anthropic', anthropic.SUPPORTED_MODELS)

  def test_gemini(self):
    self._check_provider('google', gemini.SUPPORTED_MODELS)


if __name__ == '__main__':
  unittest.main()
