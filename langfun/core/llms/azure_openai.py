"""
Azure OpenAI interface.

This file defines the AzureOpenAI class, which is based on the OpenAICompatible
base class. It configures endpoints and headers to route requests to Azure OpenAI.
"""

import os
from typing import Any

import langfun.core as lf
from langfun.core.llms.openai_compatible import OpenAICompatible
import pyglove as pg

@lf.use_init_args(["resource_name", "deployment"])
class AzureOpenAI(OpenAICompatible):
    """Interface for Azure OpenAI models.

    This class makes requests to Azure OpenAI by constructing the endpoint using
    the provided resource name and deployment. An API key is required (either provided
    on initialization or via the AZURE_OPENAI_API_KEY environment variable), and
    authentication is performed using the "api-key" header.

    Attributes:
      resource_name: The name of your Azure OpenAI resource (e.g., "my-resource").
      deployment: The deployment name configured in your Azure OpenAI account.
      api_version: API version string (default "2023-03-15-preview").
      api_key: Your Azure OpenAI API key.
      api_endpoint: Optional explicit endpoint; if not provided, it is constructed.
    """
    resource_name: str
    deployment: str
    api_version: str = "2023-03-15-preview"
    api_key: str | None = None
    api_endpoint: str = ""

    def _initialize(self):
        # Attempt to get the API key from the parameter or environment variable.
        self.api_key = self.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Please specify `api_key` during __init__ or set the environment "
                "variable `AZURE_OPENAI_API_KEY` with your Azure OpenAI API key."
            )
        self._api_key = self.api_key

        # Construct api_endpoint if not provided.
        if not self.api_endpoint:
            self.api_endpoint = (
                f"https://{self.resource_name}.openai.azure.com/"
                f"openai/deployments/{self.deployment}/chat/completions"
                f"?api-version={self.api_version}"
            )

    @property
    def headers(self) -> dict[str, Any]:
        # Azure requires the API key to be passed via the "api-key" header.
        headers = super().headers.copy()
        headers["api-key"] = self._api_key
        return headers

    def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
        # Build the request payload using the base implementation.
        args = super()._request_args(options)
        # Remove the "model" parameter since Azure uses the deployment in the endpoint.
        args.pop("model", None)
        return args
