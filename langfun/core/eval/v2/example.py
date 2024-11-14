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
"""Base classes for Langfun evaluation."""

import dataclasses
import inspect
from typing import Any, Callable
import langfun.core as lf
import pyglove as pg


@dataclasses.dataclass
class Example(pg.JSONConvertible, pg.views.HtmlTreeView.Extension):
  """An item for the evaluation.

  Attributes:
    id: The 1-based ID of the item in the evaluation set.
    input: An element returned from the `Evaluable.inputs` functor.
    output: The output of the `process` method. If `pg.MISSING_VALUE`, it has
      not been processed yet.
    metadata: The metadata of the item produced by the `process` method.
    metric_metadata: The dictionary returned from `Metric.audit`.
    start_time: The start time of the evaluation item.
    end_time: The end time of the evaluation item.
    usage_summary: The summary of LLM usages of the evaluation item.
    execution_status: The timeit status of the evaluation item.
  """
  id: int
  input: Any = pg.MISSING_VALUE
  output: Any = pg.MISSING_VALUE
  error: pg.object_utils.ErrorInfo | None = None
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
  metric_metadata: dict[str, Any] | None = None
  # Execution information.
  newly_processed: bool = True
  start_time: float | None = None
  end_time: float | None = None
  usage_summary: lf.UsageSummary | None = None
  execution_status: dict[str, pg.object_utils.TimeIt.Status] | None = None

  def __post_init__(self):
    if self.execution_status is not None:
      for status in self.execution_status.values():
        if status.has_error:
          self.error = status.error
          break

  @property
  def is_processed(self) -> bool:
    """Returns whether the item has been processed."""
    return pg.MISSING_VALUE != self.output

  @property
  def has_error(self) -> bool:
    """Returns whether the item has an error."""
    return self.error is not None

  @property
  def elapse(self) -> float | None:
    """Returns the elapse time of the item."""
    if self.execution_status is not None:
      return self.execution_status['evaluate'].elapse
    return None

  def to_json(self, *, exclude_input: bool = False, **kwargs):
    """Returns the JSON representation of the item."""
    return self.to_json_dict(
        fields=dict(
            id=(self.id, None),
            input=(
                self.input if not exclude_input else pg.MISSING_VALUE,
                pg.MISSING_VALUE
            ),
            output=(self.output, pg.MISSING_VALUE),
            error=(self.error, None),
            metadata=(self.metadata, {}),
            metric_metadata=(self.metric_metadata, None),
            start_time=(self.start_time, None),
            end_time=(self.end_time, None),
            usage_summary=(self.usage_summary, None),
            execution_status=(self.execution_status, None),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def from_json(
      cls,
      json_value: dict[str, Any],
      *,
      example_input_by_id: Callable[[int], Any] | None = None,
      **kwargs
  ) -> 'Example':
    """Creates an example from the JSON representation."""
    example_id = json_value.get('id')
    if example_input_by_id:
      example_input = example_input_by_id(example_id)
    else:
      example_input = json_value.pop('input', pg.MISSING_VALUE)
      if example_input is not pg.MISSING_VALUE:
        example_input = pg.from_json(example_input, **kwargs)
    json_value['input'] = example_input

    # NOTE(daiyip): We need to load the types of the examples into the
    # deserialization context, otherwise the deserialization will fail if the
    # types are not registered.
    def example_class_defs(example) -> list[type[Any]]:
      referred_types = set()
      def _visit(k, v, p):
        del k, p
        if inspect.isclass(v):
          referred_types.add(v)
        elif isinstance(v, pg.Object):
          referred_types.add(v.__class__)
        return pg.TraverseAction.ENTER
      pg.traverse(example, _visit)
      return list(referred_types)

    with pg.JSONConvertible.load_types_for_deserialization(
        *example_class_defs(example_input)
    ):
      return cls(
          **{k: pg.from_json(v, **kwargs) for k, v in json_value.items()}
      )

  #
  # HTML rendering.
  #

  def _html_tree_view_content(
      self,
      *,
      view: pg.views.HtmlTreeView,
      root_path: pg.KeyPath | None = None,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ):
    root_path = root_path or pg.KeyPath()
    extra_flags = extra_flags or {}
    num_examples = extra_flags.get('num_examples', None)

    def _metric_metadata_badge(key, value):
      if isinstance(value, bool) and bool:
        text = key
      else:
        text = f'{key}:{value}'
      return pg.views.html.controls.Badge(
          text,
          css_classes=[pg.object_utils.camel_to_snake(key, '-')],
      )

    def _render_header():
      return pg.Html.element(
          'div',
          [
              pg.Html.element(
                  'div',
                  [
                      # Previous button.
                      pg.views.html.controls.Label(   # pylint: disable=g-long-ternary
                          '◀',
                          link=f'{self.id - 1}.html',
                          css_classes=['previous'],
                      ) if self.id > 1 else None,
                      # Current example ID.
                      pg.views.html.controls.Label(
                          f'#{self.id}',
                          css_classes=['example-id'],
                      ),
                      # Next button.
                      pg.views.html.controls.Label(   # pylint: disable=g-long-ternary
                          '▶',
                          link=f'{self.id + 1}.html',
                          css_classes=['next'],
                      ) if (num_examples is None
                            or self.id < num_examples) else None,

                  ]
              ),
              pg.Html.element(
                  'div',
                  [
                      # Usage summary.
                      pg.view(  # pylint: disable=g-long-ternary
                          self.usage_summary,
                          extra_flags=dict(as_badge=True)
                      ) if self.usage_summary is not None else None,
                      # Metric metadata.
                      pg.views.html.controls.LabelGroup(
                          [   # pylint: disable=g-long-ternary
                              _metric_metadata_badge(k, v)
                              for k, v in self.metric_metadata.items()
                          ] if self.metric_metadata else []
                      ),
                  ],
                  css_classes=['example-container'],
              )
          ]
      )

    def _render_content():
      def _tab(label, key, default):
        field = getattr(self, key)
        if default == field:
          return None
        return pg.views.html.controls.Tab(
            label=label,
            content=view.render(
                field,
                root_path=root_path + key,
                collapse_level=None,
                **view.get_passthrough_kwargs(**kwargs),
            ),
        )
      tabs = [
          _tab('Input', 'input', pg.MISSING_VALUE),
          _tab('Output', 'output', pg.MISSING_VALUE),
          _tab('Output Metadata', 'metadata', {}),
          _tab('Error', 'error', None),
      ]
      tabs = [tab for tab in tabs if tab is not None]
      return pg.views.html.controls.TabControl(
          tabs,
          len(tabs) - 1,
      )

    return pg.Html.element(
        'div',
        [
            _render_header(),
            _render_content(),
        ],
        css_classes=['eval-example']
    )

  def _html_tree_view_summary(self, *, view, **kwargs):
    return None

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .example-container {
          display: block;
          padding: 10px;
        }
        .example-id {
          font-weight: bold;
          font-size: 40px;
          margin: 0 10px;
          vertical-align: middle;
        }
        a.previous, a.next {
          text-decoration: none;
          vertical-align: middle;
          display: inline-block;
          padding: 8px 8px;
          color: #DDD;
        }
        a.previous:hover, a.next:hover {
          background-color: #ddd;
          color: black;
        }
        /* Badge styles. */
        .eval-example .badge.match {
          color: green;
          background-color: #dcefbe;
        }
        .eval-example .badge.error {
          color: red;
          background-color: #fdcccc;
        }
        .eval-example .badge.mismatch {
          color: orange;
          background-color: #ffefc4;
        }
        .eval-example .badge.score {
          color: blue;
          background-color: #c4dced;
        }
        """
    ]

