import os
import unittest
import langfun.core as lf
from langfun.core.llms import azure_openai

class AzureOpenAITest(unittest.TestCase):
    def test_api_key_missing(self):
        # Ensure that ValueError is raised when API key is not provided
        os.environ.pop('AZURE_OPENAI_API_KEY', None)
        with self.assertRaisesRegex(ValueError, 'Azure OpenAI requires an API key'):
            azure = azure_openai.AzureOpenAI(model='gpt-4')
            azure._initialize()

    def test_api_endpoint(self):
        # Test that api_endpoint is properly constructed
        azure = azure_openai.AzureOpenAI(model='gpt-4', api_key='test_key')
        azure._initialize()
        expected = 'https://api.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-05-15'
        self.assertEqual(azure.api_endpoint, expected)

    def test_headers(self):
        # Test that headers contain the api-key with correct value
        azure = azure_openai.AzureOpenAI(model='gpt-4', api_key='test_key')
        azure._initialize()
        headers = azure.headers
        self.assertIn('api-key', headers)
        self.assertEqual(headers.get('api-key'), 'test_key')

if __name__ == '__main__':
    unittest.main()
