import os
import functools
from typing import Annotated, Any

import pyglove as pg
import langfun.core as lf
from langfun.core.llms import openai_compatible
from langfun.core.llms.openai import (
    OpenAIModelInfo,
    SUPPORTED_MODELS,
    _SUPPORTED_MODELS_BY_MODEL_ID,
)


@lf.use_init_args(['model', 'deployment_name'])
class AzureOpenAI(openai_compatible.OpenAICompatible):
    """Azure OpenAI model service."""

    deployment_name: Annotated[
        str,
        'The name of the Azure OpenAI deployment.'
    ] = 'gpt-4'

    api_endpoint: str = ''

    api_version: Annotated[
        str,
        'The API version for Azure OpenAI.'
    ] = '2023-05-15'

    azure_endpoint: Annotated[
        str,
        'The base URL for Azure OpenAI (e.g. "https://<your-resource>.openai.azure.com/")'
    ] = 'https://api.openai.azure.com/'

    api_key: Annotated[
        str | None,
        (
            'API key. If None, reads from environment variable '
            "'AZURE_OPENAI_API_KEY'."
        ),
    ] = None

    def _on_bound(self):
        super()._on_bound()
        self.__dict__.pop('model_info', None)
        self._api_key = None
        self._api_endpoint = None

    def _initialize(self):
        # Authentication
        self._api_key = self.api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        if not self._api_key:
            raise ValueError(
                'Azure OpenAI requires an API key. Please provide '
                'via `api_key` argument or AZURE_OPENAI_API_KEY '
                'environment variable.'
            )

        # Endpoint construction
        self._api_endpoint = (
            f"{self.azure_endpoint.rstrip('/')}/openai/deployments/"
            f"{self.deployment_name}/chat/completions"
            f"?api-version={self.api_version}"
        )

    @property
    def api_endpoint(self) -> str:
        return self._api_endpoint

    @property
    def headers(self) -> dict[str, Any]:
        headers = super().headers
        headers.update({
            'api-key': self._api_key,
        })
        return headers

    @functools.cached_property
    def model_info(self) -> OpenAIModelInfo:
        return _SUPPORTED_MODELS_BY_MODEL_ID[self.model]

    @classmethod
    def dir(cls):
        return [s.model_id for s in SUPPORTED_MODELS if s.in_service]
