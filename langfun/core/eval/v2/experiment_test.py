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
import datetime
import os
import tempfile
import unittest

from langfun.core.eval.v2 import evaluation as evaluation_lib
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2 import metrics as metrics_lib

import pyglove as pg

Experiment = experiment_lib.Experiment
Suite = experiment_lib.Suite
Evaluation = evaluation_lib.Evaluation
Run = experiment_lib.Run
RunId = experiment_lib.RunId
Runner = experiment_lib.Runner


@pg.functor()
def sample_inputs():
  return [
      pg.Dict(x=1)
  ]


class MyEvaluation(Evaluation):
  NAME = 'my_eval'
  replica_id: int = 0
  inputs = sample_inputs()
  metrics = [metrics_lib.Match()]

  def process(self, example):
    return 1


class ExperimentTest(unittest.TestCase):

  def test_hierarchy(self):
    exp = Suite([
        Suite([
            MyEvaluation(replica_id=0)
        ]),
        MyEvaluation(replica_id=pg.oneof(range(5))),
    ])

    self.assertIsNotNone(exp.id)
    self.assertTrue(exp.id.startswith('Suite@'))
    self.assertEqual(len(exp.children), 2)
    self.assertEqual(len(exp.leaf_nodes), 6)
    self.assertEqual(len(exp.nonleaf_nodes), 3)
    self.assertFalse(exp.is_leaf)
    self.assertFalse(exp.empty())
    self.assertEqual(len(exp.nodes), 9)

    self.assertTrue(exp.children[0].children[0].id.startswith('MyEvaluation@'))
    self.assertTrue(exp.children[0].children[0].is_leaf)
    self.assertEqual(len(exp.children[0].children[0].leaf_nodes), 1)
    self.assertFalse(exp.children[1].is_leaf)
    self.assertEqual(len(exp.children[1].children), 5)
    self.assertEqual(len(exp.children[1].leaf_nodes), 5)
    self.assertEqual(exp.leaf_nodes[-1].replica_id, 4)
    self.assertNotEqual(exp.leaf_nodes[1].hash, exp.leaf_nodes[2].hash)

    self.assertIsNone(exp.parent)
    self.assertIs(exp.children[0].parent, exp)
    self.assertIs(exp.children[0].children[0].parent, exp.children[0])
    self.assertIs(exp.children[1].children[0].parent, exp.children[1])
    self.assertIs(exp.get(exp.leaf_nodes[-1].id), exp.leaf_nodes[-1])

  def test_html_view(self):
    exp = Suite([
        Suite([
            MyEvaluation(replica_id=0)
        ]),
        MyEvaluation(replica_id=pg.oneof(range(5))),
    ])
    self.assertIn(exp.id, exp.to_html().content)
    run = Run('/root', RunId.from_id('20241102_0'), pg.Ref(exp))
    self.assertIn(
        str(run.id),
        run.to_html(
            extra_flags=dict(
                current_run=run
            )
        ).content
    )

  def test_find(self):
    exp = Experiment.find('my_eval')
    self.assertIsInstance(exp, MyEvaluation)
    exp = Experiment.find('.*_eval')
    self.assertIsInstance(exp, MyEvaluation)
    exp = Experiment.find('foo')
    self.assertTrue(pg.eq(exp, Suite([])))


class RunIdTest(unittest.TestCase):

  def test_basic(self):
    rid = RunId.from_id('20241102_0')
    self.assertEqual(
        rid.dirname('/root'), os.path.join('/root', 'run_20241102_0')
    )
    self.assertEqual(str(rid), '20241102_0')
    self.assertEqual(rid.date, datetime.date(2024, 11, 2))
    self.assertEqual(rid.number, 0)

  def test_comparison(self):
    self.assertEqual(
        RunId.from_id('20241102_0'), RunId.from_id('20241102_0')
    )
    self.assertLess(
        RunId.from_id('20241102_0'), RunId.from_id('20241102_1')
    )
    self.assertLess(
        RunId.from_id('20241101_0'), RunId.from_id('20241102_1')
    )
    self.assertGreater(
        RunId.from_id('20241102_0'), RunId.from_id('20241101_0')
    )
    self.assertLessEqual(
        RunId.from_id('20241102_0'), RunId.from_id('20241102_0')
    )
    self.assertEqual(
        RunId.from_id('20241102_0').next(),
        RunId.from_id('20241102_1')
    )

  def test_get_latest(self):
    root_dir = os.path.join(tempfile.gettempdir(), 'test_eval')
    pg.io.mkdirs(os.path.join(root_dir, 'run_20241102_0'))
    pg.io.mkdirs(os.path.join(root_dir, 'run_20241101_0'))
    self.assertEqual(
        RunId.get_latest(root_dir),
        RunId.from_id('20241102_0')
    )
    self.assertIsNone(RunId.get_latest('/notexist'))
    self.assertIsNone(RunId.get_latest(tempfile.gettempdir()))

  def test_new(self):
    rid = RunId(date=datetime.date.today(), number=1)
    self.assertEqual(
        RunId.new(root_dir=os.path.join(tempfile.gettempdir(), 'test_new')),
        rid
    )
    root_dir = os.path.join(tempfile.gettempdir(), 'test_eval2')
    pg.io.mkdirs(rid.dirname(root_dir))
    self.assertEqual(RunId.new(root_dir), rid.next())

  def test_is_valid(self):
    self.assertTrue(RunId.is_valid('latest'))
    self.assertTrue(RunId.is_valid('new'))
    self.assertTrue(RunId.is_valid('20241102_0'))
    self.assertFalse(RunId.is_valid('20241102-0'))

  def test_from_id(self):
    with self.assertRaisesRegex(
        ValueError, '.* must be one of'
    ):
      RunId.from_id('abc')

    with self.assertRaisesRegex(
        ValueError, '`root_dir` must be provided'
    ):
      RunId.from_id('latest')

    with self.assertRaisesRegex(
        ValueError, '.* no previous runs'
    ):
      RunId.from_id('latest', root_dir=tempfile.gettempdir())

    self.assertEqual(
        RunId.from_id('20241102_1'),
        RunId(date=datetime.date(2024, 11, 2), number=1)
    )
    root_dir = os.path.join(tempfile.gettempdir(), 'test_eval3')
    rid = RunId.from_id('20241102_1')
    pg.io.mkdirs(rid.dirname(root_dir))
    self.assertEqual(
        RunId.from_id('latest', root_dir=root_dir), rid
    )
    self.assertEqual(
        RunId.from_id('new', root_dir=root_dir),
        RunId(datetime.date.today(), 1)
    )
    self.assertEqual(
        RunId.from_id(None, root_dir=root_dir), rid
    )


class RunTest(unittest.TestCase):

  def test_basic(self):
    run = Run(
        '/root',
        RunId.from_id('20241102_0'),
        pg.Ref(Suite([
            MyEvaluation(replica_id=0),
        ])),
    )
    self.assertEqual(run.output_root, '/root/run_20241102_0')
    self.assertEqual(run.input_root, '/root/run_20241102_0')
    self.assertEqual(
        run.output_dir(run.experiment.leaf_nodes[0]),
        (
            '/root/run_20241102_0/MyEvaluation/'
            + run.experiment.leaf_nodes[0].hash
        )
    )
    self.assertEqual(
        run.input_path_for(run.experiment, 'a.txt'),
        '/root/run_20241102_0/a.txt'
    )
    self.assertEqual(
        run.input_path_for(run.experiment.leaf_nodes[0], 'a.txt'),
        '/root/run_20241102_0/MyEvaluation/%s/a.txt' % (
            run.experiment.leaf_nodes[0].hash
        )
    )

    # With warmup_id
    run = Run(
        '/root',
        RunId.from_id('20241102_0'),
        pg.Ref(Suite([MyEvaluation(replica_id=0)])),
        warm_start_from='/root2/run_20241103_1'
    )
    self.assertEqual(run.output_root, '/root/run_20241102_0')
    self.assertEqual(run.input_root, '/root2/run_20241103_1')
    self.assertEqual(
        run.output_dir(run.experiment.leaf_nodes[0]),
        (
            '/root/run_20241102_0/MyEvaluation/'
            + run.experiment.leaf_nodes[0].hash
        )
    )
    self.assertEqual(
        run.input_dir(run.experiment.leaf_nodes[0]),
        (
            '/root2/run_20241103_1/MyEvaluation/'
            + run.experiment.leaf_nodes[0].hash
        )
    )
    self.assertEqual(
        run.input_path_for(run.experiment, 'a.txt'),
        '/root2/run_20241103_1/a.txt'
    )
    self.assertEqual(
        run.input_path_for(run.experiment.leaf_nodes[0], 'a.txt'),
        '/root2/run_20241103_1/MyEvaluation/%s/a.txt' % (
            run.experiment.leaf_nodes[0].hash
        )
    )


class RunnerTest(unittest.TestCase):

  def test_basic(self):

    class TestRunner(Runner):
      NAME = 'test'

      def run(self):
        pass

    self.assertIsInstance(
        Runner.create(
            'test',
            current_run=Run(
                '/root',
                RunId.from_id('20241102_0'), pg.Ref(Suite([])),
            )
        ),
        TestRunner
    )
    root_dir = os.path.join(tempfile.gettempdir(), 'my_eval')
    MyEvaluation(replica_id=0).run(
        root_dir, id='20241101_0', runner='test'
    )

    with self.assertRaisesRegex(
        ValueError, 'Runner class must define a NAME constant'
    ):
      class AnotherRunner(Runner):  # pylint: disable=unused-variable
        def run(self):
          pass


if __name__ == '__main__':
  unittest.main()
