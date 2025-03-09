import os
import boto3
import json
import uuid
import time
from dotenv import load_dotenv
from botocore.config import Config
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from typing import Any, Dict, List, Optional, Union

# Load environment variables
load_dotenv()

def get_bedrock_runtime_client():
    """
    Create and return an AWS Bedrock Runtime client with the proper authentication.
    This client is used for model inference (invoking models).
    
    Returns:
        The configured AWS Bedrock Runtime client
    """
    # Configure AWS SDK
    config = Config(
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    
    # Create a Bedrock Runtime client for model inference
    bedrock_runtime_client = boto3.client(
        service_name='bedrock-runtime',
        config=config
    )
    
    return bedrock_runtime_client

def get_bedrock_management_client():
    """
    Create and return an AWS Bedrock client with the proper authentication.
    This client is used for management operations like listing models.
    
    Returns:
        The configured AWS Bedrock management client
    """
    # Configure AWS SDK
    config = Config(
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    
    # Create a Bedrock client for management operations
    bedrock_client = boto3.client(
        service_name='bedrock',
        config=config
    )
    
    return bedrock_client

# For backward compatibility - this will return the runtime client for model inference
def get_bedrock_client():
    """
    Legacy function for backward compatibility.
    Returns the Bedrock Runtime client for model inference.
    
    Returns:
        The configured AWS Bedrock Runtime client
    """
    return get_bedrock_runtime_client()

class BedrockClaudeModel(Model):
    """
    Custom model implementation for Claude 3.7 on AWS Bedrock
    """
    def __init__(
        self, 
        model_id: str = "anthropic.claude-3-7-sonnet-20250219-v1:0", 
        **kwargs
    ):
        # Set attributes directly instead of calling super().__init__
        self.model_id = model_id
        self.supports_functions = True
        self.supports_message_history = True
        self.supports_parallel_function_calling = True
        self.supports_system_prompt = True
        self.token_limit = 200000  # Claude 3.7 Sonnet has a 200k token limit
        
        # Add any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Initialize the Bedrock client
        self.bedrock_client = get_bedrock_runtime_client()
        self._model_id = model_id
        
        # Get inference profile ID/ARN if available
        self._inference_profile = os.environ.get("AWS_BEDROCK_INFERENCE_PROFILE", "")
    
    def name(self) -> str:
        """
        Return the name of the model.
        
        Returns:
            The model name
        """
        return f"Claude on Bedrock ({self._model_id})"
        
    async def agent_model(self, **kwargs) -> 'BedrockClaudeModel':
        """
        Return the model instance for the agent.
        The framework expects this method to return an object with a 'request' method.
        
        Args:
            **kwargs: Additional parameters (function_tools, allow_text_result, result_tools)
                      passed by the Pydantic AI framework.
        
        Returns:
            The model instance with the request method
        """
        return self
    
    async def request(self, messages, model_settings=None):
        """
        Request method called by the Pydantic AI agent.
        This method should communicate with the model and return a response.
        
        Args:
            messages: ModelRequest object from Pydantic AI
            model_settings: Additional model settings
            
        Returns:
            Model response and usage information
        """
        # Convert ModelRequest to a format we can use
        message_list = []
        system_prompt = ""
        user_message_added = False
        
        # Extract messages in the format we expect
        if hasattr(messages, "system_prompt") and messages.system_prompt:
            system_prompt = messages.system_prompt
            message_list.append({"role": "system", "content": system_prompt})
            
        if hasattr(messages, "user_prompt") and messages.user_prompt:
            message_list.append({"role": "user", "content": messages.user_prompt})
            user_message_added = True
            
        if hasattr(messages, "messages") and messages.messages:
            for msg in messages.messages:
                if msg.role == "system" and not system_prompt:
                    system_prompt = msg.content
                    message_list.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    user_message_added = True
                    message_list.append({"role": msg.role, "content": msg.content})
                else:
                    message_list.append({"role": msg.role, "content": msg.content})
        
        # Ensure we have at least one user message
        if not user_message_added:
            message_list.append({"role": "user", "content": "Hello Claude, can you respond to this message?"})
            
        # Now that we have the messages in a format we understand, call predict
        response = await self.predict(message_list, **(model_settings or {}))
        
        # Create a proper usage object as expected by Pydantic AI
        # We need to create an object with the attributes that will be accessed
        class Usage:
            def __init__(self):
                self.completion_tokens = 0
                self.prompt_tokens = 0
                self.total_tokens = 0
                self.requests = 1
                self.request_tokens = 0
                self.response_tokens = 0
                self.details = None
                
        return response, Usage()
    
    async def predict(
        self,
        messages: list[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a request to the Claude model on AWS Bedrock.
        
        Args:
            messages: A list of message dictionaries in the ChatML format
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            The model response as a ModelResponse-like object
        """
        # Ensure we have at least one message to send
        if not messages:
            # If no messages provided, create a default user message
            messages = [{"role": "user", "content": "Hello"}]
            
        anthropic_messages = self._convert_to_anthropic_messages(messages)
        
        # If we have no messages after conversion, add a default one
        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": "Hello"}]
        
        # Extract system prompt if present
        system_prompt = ""
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
                break
                
        # Extract tool definitions if present
        tools = kwargs.get("tools", [])
        
        # Prepare request parameters for Claude on Bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": anthropic_messages,
        }
        
        # Only add system if it's not empty
        if system_prompt:
            request_body["system"] = system_prompt
        
        # Add tools if provided
        if tools:
            request_body["tools"] = tools
            
        # Make the API call to Bedrock
        invoke_params = {
            "body": json.dumps(request_body)
        }
        
        # Use inference profile if available, otherwise use model ID
        if self._inference_profile:
            invoke_params["modelId"] = self._inference_profile
            print(f"Using inference profile: {self._inference_profile}")
        else:
            invoke_params["modelId"] = self._model_id
            print(f"Using direct model ID: {self._model_id}")
            
        response = self.bedrock_client.invoke_model(**invoke_params)
        
        # Parse and return the response
        response_body = json.loads(response['body'].read())
        
        # Convert to a format compatible with Pydantic AI
        class ModelResponse:
            def __init__(self, content):
                self.parts = [ModelResponsePart(content)]
                
        class ModelResponsePart:
            def __init__(self, content):
                self.text = content
                
        if response_body.get("content"):
            content = response_body["content"][0]["text"] if response_body.get("content") else ""
            return ModelResponse(content)
        else:
            return ModelResponse("No content returned from model")
    
    def _convert_to_anthropic_messages(self, messages: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Anthropic format for Bedrock
        
        Args:
            messages: Messages in OpenAI format
            
        Returns:
            Messages in Anthropic format
        """
        anthropic_messages = []
        has_user_message = False
        
        for msg in messages:
            if msg["role"] == "system":
                # Skip system messages as they are handled separately
                continue
                
            content = msg["content"]
            
            # Handle tool calls/returns in the content
            if msg.get("tool_calls") and msg["role"] == "assistant":
                # Convert tool calls to Anthropic format
                content = []
                if isinstance(msg["content"], str) and msg["content"]:
                    content.append({"type": "text", "text": msg["content"]})
                
                for tool_call in msg["tool_calls"]:
                    if tool_call["type"] == "function":
                        content.append({
                            "type": "tool_use",
                            "id": tool_call.get("id", "tool-" + str(uuid.uuid4())),
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"])
                        })
            
            # Handle tool results from user messages
            if msg.get("tool_call_id") and msg["role"] == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"]
                    }]
                })
                has_user_message = True
                continue
                
            # Add the message in Anthropic format
            if msg["role"] == "user":
                has_user_message = True
                
            anthropic_messages.append({
                "role": "user" if msg["role"] == "user" else "assistant",
                "content": content
            })
        
        # Ensure there's at least one user message
        if not has_user_message and not anthropic_messages:
            anthropic_messages.append({
                "role": "user", 
                "content": "Hello"
            })
            
        return anthropic_messages 