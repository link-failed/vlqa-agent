import os
import litellm

from smolagents import LiteLLMModel
from tenacity import retry, stop_after_attempt, before_sleep_log, retry_if_exception_type, wait_exponential, wait_random
import logging

# Monkey patch litellm.completion to always convert max_tokens to max_completion_tokens for Azure
original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    # Remove max_tokens and replace with max_completion_tokens for Azure compatibility
    if 'max_tokens' in kwargs:
        max_tokens_value = kwargs.pop('max_tokens')
        if 'max_completion_tokens' not in kwargs and max_tokens_value is not None:
            kwargs['max_completion_tokens'] = max_tokens_value
    
    # Ensure max_completion_tokens is always set for Azure OpenAI calls
    if 'max_completion_tokens' not in kwargs:
        # Check if this looks like an Azure OpenAI call
        model = kwargs.get('model', '')
        if model and ('azure' in model.lower() or 'o3' in model.lower()):
            kwargs['max_completion_tokens'] = 3000
    
    return original_completion(*args, **kwargs)

# Apply the monkey patch
litellm.completion = patched_completion

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class LiteLLMModelWithBackOff(LiteLLMModel):
    def __init__(self, use_azure_auth: bool = False, *args, **kwargs):
        if use_azure_auth:
            from azure.identity import ManagedIdentityCredential, DefaultAzureCredential, get_bearer_token_provider
            from openai import AzureOpenAI
            from azure.core.credentials import AccessToken

            credential = ManagedIdentityCredential()
            scope = "https://cognitiveservices.azure.com/.default"
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )

            os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
            os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"  # Additional env var for LiteLLM
            os.environ["AZURE_OPENAI_ENDPOINT"] = "https://excelllmswedencentral.openai.azure.com/"
            
            # Set Azure AD token for LiteLLM - get the actual token
            try:
                token = token_provider()
                os.environ["AZURE_OPENAI_AD_TOKEN"] = token
            except Exception as e:
                logger.warning(f"Failed to get Azure token: {e}")
                # Fallback: try to get token directly
                try:
                    from azure.identity import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    token_result = credential.get_token("https://cognitiveservices.azure.com/.default")
                    os.environ["AZURE_OPENAI_AD_TOKEN"] = token_result.token
                except Exception as fallback_e:
                    logger.error(f"Failed to get Azure token with fallback: {fallback_e}")
            
            # For Azure OpenAI, we need to format the model_id correctly
            model_id = kwargs.get('model_id', '')
            if model_id and not model_id.startswith('azure/'):
                kwargs['model_id'] = f"azure/{model_id}"
            
            # Set api_base and api_version for Azure
            kwargs['api_base'] = "https://excelllmswedencentral.openai.azure.com/"
            kwargs['api_version'] = "2024-12-01-preview"  # Explicitly set API version
            kwargs.pop('api_key', None)
            
            # Store token provider for direct Azure client usage if needed
            self.token_provider = token_provider
        
        # Remove max_tokens from kwargs before passing to parent (Azure compatibility)
        kwargs.pop('max_tokens', None)
        kwargs.pop('max_completion_tokens', None)  # Parent class doesn't accept this
        
        super().__init__(*args, **kwargs)
        
        # Override any max_tokens setting from parent class and force Azure-compatible settings
        if hasattr(self, 'max_tokens'):
            delattr(self, 'max_tokens')

    def get_azure_client(self):
        """Get an AzureOpenAI client for direct API calls if needed"""
        if hasattr(self, 'token_provider'):
            from openai import AzureOpenAI
            return AzureOpenAI(
                azure_endpoint="https://excelllmswedencentral.openai.azure.com/",
                api_version="2024-12-01-preview",
                azure_ad_token_provider=self.token_provider
            )
        return None

    def refresh_azure_token(self):
        """Refresh the Azure AD token in environment variables"""
        if hasattr(self, 'token_provider'):
            try:
                token = self.token_provider()
                os.environ["AZURE_OPENAI_AD_TOKEN"] = token
                logger.info("Azure AD token refreshed successfully")
            except Exception as e:
                logger.error(f"Failed to refresh Azure token: {e}")

    @retry(
        stop=stop_after_attempt(450),
        wait=wait_exponential(min=1, max=120, exp_base=2, multiplier=1) + wait_random(0, 5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_exception_type((
                litellm.Timeout,
                litellm.RateLimitError,
                litellm.APIConnectionError,
                litellm.InternalServerError,
                litellm.AuthenticationError
        ))
    )
    def __call__(self, *args, **kwargs):
        # Don't pass max_completion_tokens to the parent LiteLLMModel.__call__
        # The monkey patch will handle the conversion at the litellm.completion level
        kwargs.pop('max_completion_tokens', None)
        kwargs.pop('max_tokens', None)  # Remove any max_tokens that might still be passed
        
        # Log the kwargs to understand what's being passed
        logger.debug(f"LiteLLM call kwargs after cleanup: {list(kwargs.keys())}")
        
        try:
            result = super().__call__(*args, **kwargs)
            return result
        except litellm.BadRequestError as e:
            # Check if this is the max_tokens error and try to handle it
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                logger.error(f"Still getting max_tokens error despite our fixes: {e}")
                raise e
            raise e
        except litellm.AuthenticationError as e:
            # Try to refresh token if authentication fails
            if hasattr(self, 'token_provider'):
                logger.warning("Authentication failed, attempting to refresh Azure token...")
                self.refresh_azure_token()
                # Retry once more after token refresh
                return super().__call__(*args, **kwargs)
            else:
                raise e

