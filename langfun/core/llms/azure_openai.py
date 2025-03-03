import os
from typing import Annotated, Any

import pyglove as pg
import langfun.core as lf
from langfun.core.llms.openai import OpenAI


@lf.use_init_args(['model', 'deployment_name'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class AzureOpenAI(OpenAI):
    """Azure OpenAI model service.

    This service interacts with the Azure OpenAI API to generate chat completions.
    It uses the deployment_name and API version to construct the endpoint, and authenticates
    using an API key provided via parameter or the AZURE_OPENAI_API_KEY environment variable.

    Example:
        lm = AzureOpenAI(
            model='gpt-4o',
            deployment_name='gpt-4o',
            api_version='2024-08-01-preview',
            azure_endpoint='https://trackname.openai.azure.com/',
            api_key='token'
        )
        response = lf.query(prompt="what the capital of France", lm=lm)
        print(response)
    """

    deployment_name: Annotated[
        str,
        'The name of the Azure OpenAI deployment.'
    ] = 'gpt-4'


    api_version: Annotated[
        str,
        'The API version for Azure OpenAI.'
    ] = '2023-05-15'

    azure_endpoint: Annotated[
        str,
        'The base URL for Azure OpenAI (e.g. "https://<your-resource>.openai.azure.com/")'
    ] = 'https://api.openai.azure.com/'

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

