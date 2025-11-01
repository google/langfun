# Copyright 2025 The Langfun Authors
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
"""Environment event logger."""

from typing import Annotated
from langfun.env import interface
import pyglove as pg


_METRIC_NAMESPACE = '/langfun/env'


class MetricWriter(pg.Object, interface.EventHandler):
  """Event handler for streamz metrics."""

  app: Annotated[
      str,
      'Application name that will be used as a metric parameter.'
  ] = ''

  def _get_counter(
      self,
      name: str,
      description: str,
      parameters: dict[str, type[str]] | None = None,
  ) -> pg.monitoring.Counter:
    return self._metric_collection.get_counter(
        name=name,
        description=description,
        parameters=parameters,
    )

  def _get_scalar(
      self,
      name: str,
      description: str,
      parameters: dict[str, type[str]] | None = None
  ) -> pg.monitoring.Metric:
    return self._metric_collection.get_scalar(
        name=name,
        description=description,
        parameters=parameters
    )

  def _get_distribution(
      self,
      name: str,
      description: str,
      parameters: dict[str, type[str]] | None = None
  ) -> pg.monitoring.Metric:
    return self._metric_collection.get_distribution(
        name=name,
        description=description,
        parameters=parameters
    )

  def _error_tag(self, error: BaseException | None) -> str:
    if error is None:
      return 'Success'
    return pg.utils.ErrorInfo.from_exception(error).tag

  def _initialize_metrics(self) -> None:
    """Initializes metrics."""

    self._metric_collection = pg.monitoring.metric_collection(_METRIC_NAMESPACE)

    #
    # Environment metrics.
    #

    self._environment_housekeep_duration_ms = self._get_distribution(
        'environment_housekeep_duration_ms',
        description='Environment housekeeping duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'error': str,
        }
    )

    #
    # Sandbox metrics.
    #

    # Sandbox counters.
    self._sandbox_start = self._get_counter(
        'sandbox_start',
        description='Sandbox start counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_shutdown = self._get_counter(
        'sandbox_shutdown',
        description='Sandbox shutdown counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str
        }
    )
    self._sandbox_count = self._get_scalar(
        'sandbox_count',
        description='Sandbox count',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'status': str,
        },
    )
    self._sandbox_housekeep = self._get_counter(
        'sandbox_housekeep',
        description='Sandbox housekeeping counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_activity = self._get_counter(
        'sandbox_activity',
        description='Sandbox activity counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'activity': str,
            'error': str,
        }
    )

    # Sandbox scalars.
    self._sandbox_lifetime_ms = self._get_distribution(
        'sandbox_lifetime_ms',
        description='Sandbox life time in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_start_duration_ms = self._get_distribution(
        'sandbox_start_duration_ms',
        description='Sandbox start duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_shutdown_duration_ms = self._get_distribution(
        'sandbox_shutdown_duration_ms',
        description='Sandbox shutdown duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_housekeep_duration_ms = self._get_distribution(
        'sandbox_housekeep_duration_ms',
        description='Sandbox housekeeping duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_status_duration_ms = self._get_distribution(
        'sandbox_status_duration_ms',
        description='Sandbox duration of specific status in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'status': str,
        }
    )
    self._sandbox_activity_duration_ms = self._get_distribution(
        'sandbox_activity_duration_ms',
        description='Sandbox activity duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'activity': str,
            'error': str,
        }
    )

    #
    # Sandbox session metrics.
    #

    self._sandbox_session_start_duration_ms = self._get_distribution(
        'sandbox_session_start_duration_ms',
        description='Sandbox session start duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_session_end_duration_ms = self._get_distribution(
        'sandbox_session_end_duration_ms',
        description='Sandbox session end duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )
    self._sandbox_session_lifetime_ms = self._get_distribution(
        'sandbox_session_lifetime_ms',
        description='Sandbox session lifetime in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'error': str,
        }
    )

    #
    # Feature metrics.
    #

    # Feature counters.
    self._feature_setup = self._get_counter(
        'feature_setup',
        description='Feature setup counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_teardown = self._get_counter(
        'feature_teardown',
        description='Feature teardown counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_setup_session = self._get_counter(
        'feature_setup_session',
        description='Feature setup session counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_teardown_session = self._get_counter(
        'feature_teardown_session',
        description='Feature teardown session counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_activity = self._get_counter(
        'feature_activity',
        description='Feature activity counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'activity': str,
            'error': str,
        }
    )
    self._feature_housekeep = self._get_counter(
        'feature_housekeep',
        description='Feature housekeeping counter',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )

    # Feature scalars.
    self._feature_setup_duration_ms = self._get_distribution(
        'feature_setup_duration_ms',
        description='Feature setup duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_teardown_duration_ms = self._get_distribution(
        'feature_teardown_duration_ms',
        description='Feature teardown duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_setup_session_duration_ms = self._get_distribution(
        'feature_setup_session_duration_ms',
        description='Feature setup session duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_teardown_session_duration_ms = self._get_distribution(
        'feature_teardown_session_duration_ms',
        description='Feature teardown session duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )
    self._feature_activity_duration_ms = self._get_distribution(
        'feature_activity_duration_ms',
        description='Feature activity duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'activity': str,
            'error': str,
        }
    )
    self._feature_housekeep_duration_ms = self._get_distribution(
        'feature_housekeep_duration_ms',
        description='Feature housekeeping duration in milliseconds',
        parameters={
            'app': str,
            'environment_id': str,
            'image_id': str,
            'feature_name': str,
            'error': str,
        }
    )

  def on_environment_starting(
      self,
      environment: interface.Environment
  ) -> None:
    """Called when the environment is starting."""
    self._initialize_metrics()

  def on_environment_housekeep(
      self,
      environment: interface.Environment,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when the environment is housekeeping."""
    self._environment_housekeep_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(environment.id),
        error=self._error_tag(error)
    )

  def on_sandbox_start(
      self,
      sandbox: interface.Sandbox,
      duration: float,
      error: BaseException | None
  ) -> None:
    self._sandbox_start.increment(
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )
    self._sandbox_start_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )

  def on_sandbox_status_change(
      self,
      sandbox: interface.Sandbox,
      old_status: interface.Sandbox.Status,
      new_status: interface.Sandbox.Status,
      span: float
  ) -> None:
    self._sandbox_status_duration_ms.record(
        int(span * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        status=old_status.value
    )
    if old_status != interface.Sandbox.Status.CREATED:
      self._sandbox_count.increment(
          delta=-1,
          app=self.app,
          environment_id=str(sandbox.environment.id),
          image_id=sandbox.image_id,
          status=old_status.value
      )
    if new_status != interface.Sandbox.Status.OFFLINE:
      self._sandbox_count.increment(
          app=self.app,
          environment_id=str(sandbox.environment.id),
          image_id=sandbox.image_id,
          status=new_status.value
      )

  def on_sandbox_shutdown(
      self,
      sandbox: interface.Sandbox,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    self._sandbox_shutdown.increment(
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )
    self._sandbox_shutdown_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )
    self._sandbox_lifetime_ms.record(
        int(lifetime * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )

  def on_sandbox_session_start(
      self,
      sandbox: interface.Sandbox,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session starts."""
    self._sandbox_session_start_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )

  def on_sandbox_session_end(
      self,
      sandbox: interface.Sandbox,
      session_id: str,
      duration: float,
      lifetime: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox session ends."""
    self._sandbox_session_end_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )
    self._sandbox_session_lifetime_ms.record(
        int(lifetime * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )

  def on_sandbox_activity(
      self,
      name: str,
      sandbox: interface.Sandbox,
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox activity is performed."""
    self._sandbox_activity.increment(
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        activity=name,
        error=self._error_tag(error)
    )
    self._sandbox_activity_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        activity=name,
        error=self._error_tag(error)
    )

  def on_sandbox_housekeep(
      self,
      sandbox: interface.Sandbox,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox feature is housekeeping."""
    self._sandbox_housekeep.increment(
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )
    self._sandbox_housekeep_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(sandbox.environment.id),
        image_id=sandbox.image_id,
        error=self._error_tag(error)
    )

  def on_feature_setup(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    image_id = feature.sandbox.image_id if feature.sandbox else ''
    self._feature_setup.increment(
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )
    self._feature_setup_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )

  def on_feature_teardown(
      self,
      feature: interface.Feature,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    image_id = feature.sandbox.image_id if feature.sandbox else ''
    self._feature_teardown.increment(
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )
    self._feature_teardown_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )

  def on_feature_setup_session(
      self,
      feature: interface.Feature,
      session_id: str | None,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is setup."""
    image_id = feature.sandbox.image_id if feature.sandbox else ''
    self._feature_setup_session.increment(
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )
    self._feature_setup_session_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )

  def on_feature_teardown_session(
      self,
      feature: interface.Feature,
      session_id: str,
      duration: float,
      error: BaseException | None
  ) -> None:
    """Called when a sandbox feature is teardown."""
    image_id = feature.sandbox.image_id if feature.sandbox else ''
    self._feature_teardown_session.increment(
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )
    self._feature_teardown_session_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )

  def on_feature_activity(
      self,
      name: str,
      feature: interface.Feature,
      session_id: str | None,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a feature activity is performed."""
    image_id = feature.sandbox.image_id if feature.sandbox else ''
    self._feature_activity.increment(
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        activity=name,
        error=self._error_tag(error)
    )
    self._feature_activity_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        activity=name,
        error=self._error_tag(error)
    )

  def on_feature_housekeep(
      self,
      feature: interface.Feature,
      counter: int,
      duration: float,
      error: BaseException | None,
      **kwargs
  ) -> None:
    """Called when a sandbox feature is housekeeping."""
    image_id = feature.sandbox.image_id if feature.sandbox else ''
    self._feature_housekeep.increment(
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )
    self._feature_housekeep_duration_ms.record(
        int(duration * 1000),
        app=self.app,
        environment_id=str(feature.environment.id),
        image_id=image_id,
        feature_name=feature.name,
        error=self._error_tag(error)
    )
