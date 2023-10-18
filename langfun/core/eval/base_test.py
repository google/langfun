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
"""Tests for language model."""

import os
import tempfile
import unittest

import langfun.core as lf
from langfun.core.eval import base
from langfun.core.llms import fake
import langfun.core.structured as lf_structured
import pyglove as pg


# We put class definitions outside the functors just to make it easier
# to refer to them in test.
class Solution(pg.Object):
  final_answer: int


class SolutionForCompletion(pg.Object):
  question: str
  final_answer: int


@pg.functor
def answer_schema():
  return Solution


@pg.functor
def answer_schema_with_fewshot_examples():
  return Solution, [
      lf_structured.MappingExample(
          nl_text='The result of one plus two',
          schema=Solution,
          value=Solution(3)
      )
  ]


@pg.functor
def complete_schema():
  return SolutionForCompletion


def eval_set(
    eval_id: str,
    method: str,
    schema_fn,
    lm: lf.LanguageModel,
    use_cache: bool = True,
):
  """Creates an evaluation object for testing."""
  tmp_dir = tempfile.gettempdir()
  return base.Evaluation(
      id=eval_id,
      root_dir=tmp_dir,
      inputs=[
          pg.Dict(question='Compute 1 + 1'),
          pg.Dict(question='Compute 1 + 2'),
      ],
      method=method,
      prompt='{{example.question}}',
      completion_prompt_field='question',
      schema_fn=schema_fn,
      lm=lm,
      use_cache=use_cache,
      max_workers=1,
  )


class EvaluationTest(unittest.TestCase):
  """Evaluation test."""

  def setUp(self):
    super().setUp()
    pg.symbolic.set_save_handler(pg.symbolic.default_save_handler)
    pg.symbolic.set_load_handler(pg.symbolic.default_load_handler)

  def test_basics(self):
    lm = fake.StaticSequence(['two', 'Solution(final_answer=2)'])
    s = eval_set('basic_test', 'call', schema_fn=answer_schema(), lm=lm)

    self.assertEqual(s.dir, os.path.join(s.root_dir, s.id))
    self.assertEqual(s.hash, s.clone().hash)
    self.assertEqual(
        s.hash, s.clone(override={'max_workers': 2, 'lm.timeout': 20}).hash
    )
    self.assertNotEqual(
        s.hash, s.clone(override={'prompt': 'Hello {{example.question}}'}).hash
    )
    self.assertIsNone(s.parent)
    self.assertIs(s.schema.spec.cls, Solution)
    self.assertIsNone(s.fewshot_examples)

    # Test schema_fn with fewshot examples.
    s.rebind(schema_fn=answer_schema_with_fewshot_examples())
    self.assertIs(s.schema.spec.cls, Solution)
    self.assertTrue(
        pg.eq(
            s.fewshot_examples,
            [
                lf_structured.MappingExample(
                    nl_text='The result of one plus two',
                    schema=Solution,
                    value=Solution(3))
            ]
        )
    )

  def test_schema_for_completion(self):

    @pg.functor()
    def _answer_schema():

      class Solution1(pg.Object):
        final_answer: int

      return Solution1, [
          lf.structured.MappingExample(
              nl_context='The result of one plus two',
              schema=Solution1,
              value=Solution1(3)
          )
      ]

    s = eval_set(
        'schema_for_completion', 'complete',
        schema_fn=_answer_schema(), lm=fake.StaticResponse('hi'))

    fewshot_examples = s.fewshot_examples
    solution_cls = s.schema.spec.cls

    # Verify class schema get updated.
    self.assertEqual('question', list(solution_cls.__schema__.keys())[0])

    # Verify query examples are mapped to completion examples.
    self.assertTrue(
        pg.eq(
            fewshot_examples,
            [
                lf.structured.MappingExample(
                    value=lf.structured.mapping.Pair(
                        left=solution_cls.partial(
                            question='The result of one plus two'),
                        right=solution_cls('The result of one plus two', 3),
                    )
                )
            ]
        )
    )

  def test_bad_init(self):
    @pg.functor()
    def _bad_completion_schema():
      return int

    s = eval_set(
        'bad_init1', 'complete',
        schema_fn=_bad_completion_schema(), lm=fake.StaticResponse('hi'))

    with self.assertRaisesRegex(TypeError, '.*must be .*class.*'):
      _ = s.schema

  def test_dryrun(self):
    lm = fake.StaticResponse('Solution(final_answer=2)')
    s = eval_set('dryrun_test', 'query', schema_fn=answer_schema(), lm=lm)
    s.dryrun()
    self.assertEqual(
        s.dryrun_output,
        lf.AIMessage(
            text='Solution(final_answer=2)',
            result=Solution(2),
            cache_seed=0,
            score=1.0,
            tags=['lm-response', 'lm-output', 'transformed'],
        ),
    )

  def test_run(self):
    lm = fake.StaticSequence([
        'Solution(final_answer=2)',
        '3',
    ], debug=True)
    s = eval_set('run_test', 'query', schema_fn=answer_schema(), lm=lm)
    self.assertEqual(
        s.run(dryrun=False),
        dict(
            experiment_setup=dict(
                id='run_test',
                dir=s.dir,
                model='StaticSequence',
                prompt_template='{{example.question}}',
                method='query',
                schema_fn='answer_schema()'
            ),
            cache_stats=dict(
                use_cache=True,
                num_queries=2,
                num_hits=0,
                num_updates=2
            ),
            metrics=dict(
                total=2,
                failures=1,
                failure_rate=0.5
            )
        )
    )
    self.assertTrue(
        os.path.exists(os.path.join(s.dir, base.Evaluation.EXPERIMENT_JSON)))
    self.assertTrue(
        os.path.exists(os.path.join(s.dir, base.Evaluation.RESULT_JSON)))
    self.assertTrue(
        os.path.exists(os.path.join(s.dir, base.Evaluation.CACHE_JSON)))
    self.assertTrue(
        os.path.exists(os.path.join(s.dir, base.Evaluation.INDEX_HTML)))
    self.assertTrue(
        os.path.exists(os.path.join(s.dir, base.Evaluation.FAILURES_HTML)))

  def test_run_wihtout_save(self):
    lm = fake.StaticSequence([
        'Solution(final_answer=2)',
        '3',
    ], debug=True)
    s = eval_set(
        'run_without_save_test', 'query', schema_fn=answer_schema(), lm=lm)
    s.run(save=False, dryrun=False)

    # Cache will always be saved
    self.assertTrue(
        os.path.exists(os.path.join(s.dir, base.Evaluation.CACHE_JSON)))

    self.assertFalse(
        os.path.exists(os.path.join(s.dir, base.Evaluation.EXPERIMENT_JSON)))
    self.assertFalse(
        os.path.exists(os.path.join(s.dir, base.Evaluation.RESULT_JSON)))
    self.assertFalse(
        os.path.exists(os.path.join(s.dir, base.Evaluation.INDEX_HTML)))
    self.assertFalse(
        os.path.exists(os.path.join(s.dir, base.Evaluation.FAILURES_HTML)))

  def test_load(self):
    lm = fake.StaticResponse('Solution(final_answer=2)')
    s = eval_set('loas_test', 'query', schema_fn=answer_schema(), lm=lm)
    s.run()
    self.assertIsNotNone(s.result)

    s2 = base.load(s.dir)
    self.assertTrue(pg.eq(s, s2))
    self.assertIsNone(s2.result)
    s2.load_result()
    self.assertEqual(s2.result, s.result)

  def test_run_with_filter(self):
    lm = fake.StaticResponse('Solution(final_answer=2)')
    s = eval_set('run_filter_test', 'query', schema_fn=answer_schema(), lm=lm)
    self.assertEqual(
        s.run(filter=lambda x: False),
        dict(
            experiment_setup=dict(
                id='run_filter_test',
                dir=s.dir,
                model='StaticResponse',
                prompt_template='{{example.question}}',
                method='query',
                schema_fn='answer_schema()',
            ),
            cache_stats=dict(
                use_cache=False,
            ),
            metrics=dict(total=0, failures=0, failure_rate=0.0),
        ),
    )

  def test_search_space(self):
    lm = fake.StaticSequence([
        'Solution(final_answer=2)',
        '3',
    ])
    s = base.Evaluation(
        id='search_space_test',
        root_dir=tempfile.gettempdir(),
        inputs=[
            pg.Dict(question='Compute 1 + 1'),
            pg.Dict(question='Compute 1 + 2'),
        ],
        method='query',
        prompt=pg.oneof([
            lf.Template('{{example.question}}'),
            lf.Template('Hello {{example.question}}'),
        ]),
        schema_fn=answer_schema(),
        lm=lm,
        use_cache=True,
        max_workers=1,
    )
    self.assertEqual(s.children[0].id, f'{s.id}@{s.children[0].hash}')
    self.assertEqual(
        s.children[0].dir, os.path.join(s.root_dir, s.children[0].id)
    )

    self.assertEqual(
        s.run(dryrun=False, verbose=False),
        {
            s.children[0].id: dict(
                experiment_setup=dict(
                    id=s.children[0].id,
                    dir=s.children[0].dir,
                    model='StaticSequence',
                    prompt_template='{{example.question}}',
                    method='query',
                    schema_fn='answer_schema()',
                ),
                cache_stats=dict(
                    use_cache=True, num_queries=2, num_hits=0, num_updates=2
                ),
                metrics=dict(total=2, failures=1, failure_rate=0.5),
            ),
            s.children[1].id: dict(
                experiment_setup=dict(
                    id=s.children[1].id,
                    dir=s.children[1].dir,
                    model='StaticSequence',
                    prompt_template='Hello {{example.question}}',
                    method='query',
                    schema_fn='answer_schema()',
                ),
                cache_stats=dict(
                    use_cache=True, num_queries=2, num_hits=0, num_updates=2
                ),
                metrics=dict(total=2, failures=1, failure_rate=0.5),
            ),
        },
    )

  def test_call(self):
    lm = fake.StaticSequence(['two', 'Solution(final_answer=2)'])
    s = eval_set('call_test', 'call', schema_fn=answer_schema(), lm=lm)
    self.assertEqual(s.process(s.examples[0]), Solution(2))

  def test_query(self):
    lm = fake.StaticSequence(['Solution(final_answer=2)'])
    s = eval_set('query_test', 'query', schema_fn=answer_schema(), lm=lm)
    self.assertEqual(s.process(s.examples[0]), Solution(2))

    # Test query with fewshot examples.
    lm = fake.StaticSequence(['two', 'Solution(final_answer=2)'])
    s = eval_set(
        'basic_test', 'call',
        schema_fn=answer_schema_with_fewshot_examples(),
        lm=lm)
    m = s.process(s.examples[0], returns_message=True)
    self.assertIn('The result of one plus two', m.lm_input.text)

  def test_complete(self):
    lm = fake.StaticSequence(
        ["SolutionForCompletion(question='Compute 1 + 1', final_answer=2)"]
    )
    s = eval_set(
        'complete_test', 'complete', schema_fn=complete_schema(), lm=lm
    )
    self.assertEqual(
        s.process(s.examples[0]), SolutionForCompletion('Compute 1 + 1', 2)
    )

    # Testing for using a query schema for completion.

    @pg.functor()
    def _answer_schema():

      class Solution2(pg.Object):
        answer: int

      return Solution2

    lm = fake.StaticSequence(
        ["Solution2(question='Compute 1 + 1', answer=2)"],
        debug=True,
    )
    s = eval_set(
        'complete_test2', 'complete', schema_fn=_answer_schema(), lm=lm
    )
    self.assertEqual(s.process(s.examples[0]).answer, 2)


class SuiteTest(unittest.TestCase):
  """Suite test."""

  def test_run(self):
    lm = fake.StaticSequence([
        'Solution(final_answer=2)',
        '3',
    ])
    s = base.Suite(
        'suite_run_test',
        [
            eval_set('run_test_1', 'query', schema_fn=answer_schema(), lm=lm),
            eval_set('run_test_2', 'query', schema_fn=answer_schema(), lm=lm),
        ],
    )
    self.assertEqual(
        s.run(),
        {
            s.children[0].id: dict(
                experiment_setup=dict(
                    id=s.children[0].id,
                    dir=s.children[0].dir,
                    model='StaticSequence',
                    prompt_template='{{example.question}}',
                    method='query',
                    schema_fn='answer_schema()',
                ),
                cache_stats=dict(
                    use_cache=True, num_queries=2, num_hits=0, num_updates=2
                ),
                metrics=dict(total=2, failures=1, failure_rate=0.5),
            ),
            s.children[1].id: dict(
                experiment_setup=dict(
                    id=s.children[1].id,
                    dir=s.children[1].dir,
                    model='StaticSequence',
                    prompt_template='{{example.question}}',
                    method='query',
                    schema_fn='answer_schema()',
                ),
                cache_stats=dict(
                    use_cache=True, num_queries=2, num_hits=0, num_updates=2
                ),
                metrics=dict(total=2, failures=1, failure_rate=0.5),
            ),
        },
    )


if __name__ == '__main__':
  unittest.main()
