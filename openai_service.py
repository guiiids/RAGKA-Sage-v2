"""
OpenAIService class for handling interactions with the Azure OpenAI API
"""
import logging
from openai import AzureOpenAI
from openai_logger import log_openai_call
from openai_logger import log_openai_usage



logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Handles interactions with the Azure OpenAI API.
    
    This class is responsible for:
    - Initializing the Azure OpenAI client
    - Sending requests to the API
    - Processing responses
    - Error handling and logging
    """
    
    def __init__(self, azure_endpoint=None, api_key=None, api_version="2024-02-01", deployment_name=None):
        """
        Initialize the OpenAI service.
        
        Args:
            azure_endpoint: The Azure OpenAI endpoint URL
            api_key: The API key for authentication
            api_version: The API version to use
            deployment_name: The deployment name to use for chat completions
        """
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment_name = deployment_name
        
        # Initialize the OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        logger.debug(f"OpenAIService initialized with endpoint: {azure_endpoint}, api_version: {api_version}, deployment: {deployment_name}")
    
    def get_chat_response(
        self,
        messages,
        max_tokens=1000,
        max_completion_tokens=None
    ):
        """
        Get a response from the OpenAI chat completions API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum tokens for standard models
            max_completion_tokens: Maximum tokens for models that use the
                ``max_completion_tokens`` parameter (e.g., ``o4-mini``)
            
        Returns:
            The assistant's response text
        """
        logger.info(f"Sending request to OpenAI with {len(messages)} messages")
        logger.debug(
            f"Using max_tokens: {max_tokens}, max_completion_tokens: {max_completion_tokens}"
        )
        
        try:
            # Prepare the request
            request = {
                'model': self.deployment_name,
                'messages': messages
            }

            if max_completion_tokens is not None:
                request['max_completion_tokens'] = max_completion_tokens
            else:
                request['max_tokens'] = max_tokens
            
            # Log the first and last message for debugging
            if messages:
                logger.debug(f"First message - Role: {messages[0]['role']}")
                logger.debug(f"Last message - Role: {messages[-1]['role']}")
            
            # Send the request to the API
            response = self.client.chat.completions.create(**request)
            
            # Log the API call
            log_openai_call(request, response)
            log_openai_usage(request, response)
            
            # Extract and return the response text
            answer = response.choices[0].message.content
            logger.info(f"Received response from OpenAI (length: {len(answer)})")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Re-raise the exception to be handled by the caller
            raise

    def get_chat_response_stream(
        self,
        messages,
        max_tokens=1000,
        max_completion_tokens=None
    ):
        """
        Stream a response from the OpenAI chat completions API.

        Yields each new partial content chunk as it arrives.
        """
        logger.info(f"Streaming request to OpenAI with {len(messages)} messages")
        logger.debug(
            f"(stream) Using max_tokens: {max_tokens}, max_completion_tokens: {max_completion_tokens}"
        )
        try:
            # Prepare the request for streaming
            request = {
                'model': self.deployment_name,
                'messages': messages,
                'stream': True,
            }

            if max_completion_tokens is not None:
                request['max_completion_tokens'] = max_completion_tokens
            else:
                request['max_tokens'] = max_tokens

            # Log for debugging
            if messages:
                logger.debug(f"(stream) First message - Role: {messages[0]['role']}")
                logger.debug(f"(stream) Last message - Role: {messages[-1]['role']}")

            # Call the API with streaming
            response = self.client.chat.completions.create(**request)
            
            # Each delta in response is an event
            for chunk in response:
                try:
                    content = chunk.choices[0].delta.content
                except Exception:
                    content = None
                if content:
                    yield content

        except Exception as e:
            logger.error(f"Error with streaming OpenAI API: {e}")
            raise

    def summarize_text(self, text: str, max_tokens: int = 60) -> str:
        """Return a short summary for storage in session memory."""
        logger.debug("Generating summary (%s tokens max)", max_tokens)
        messages = [
            {"role": "system", "content": "Summarize the following text in 40 words or less."},
            {"role": "user", "content": text},
        ]
        try:
            summary = self.get_chat_response(messages, max_tokens=max_tokens)
            logger.debug("Summary generated (%s chars)", len(summary))
            return summary
        except Exception:
            logger.exception("OpenAI summary generation failed")
            return ""
