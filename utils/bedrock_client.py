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
        
        # Extract messages in the format we expect
        if hasattr(messages, "system_prompt") and messages.system_prompt:
            system_prompt = messages.system_prompt
            message_list.append({"role": "system", "content": system_prompt})
            
        if hasattr(messages, "user_prompt") and messages.user_prompt:
            message_list.append({"role": "user", "content": messages.user_prompt})
            
        if hasattr(messages, "messages") and messages.messages:
            for msg in messages.messages:
                if msg.role == "system" and not system_prompt:
                    system_prompt = msg.content
                message_list.append({"role": msg.role, "content": msg.content})
            
        # Now that we have the messages in a format we understand, call predict
        response = await self.predict(message_list, **(model_settings or {}))
        return response, {"usage": response.get("usage", {})}
    
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
            The model response as a dictionary
        """
        anthropic_messages = self._convert_to_anthropic_messages(messages)
        
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
            "system": system_prompt,
        }
        
        # Add tools if provided
        if tools:
            request_body["tools"] = tools
            
        # Make the API call to Bedrock
        response = self.bedrock_client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(request_body)
        )
        
        # Parse and return the response
        response_body = json.loads(response['body'].read())
        
        # Convert to OpenAI-like format for compatibility
        openai_format_response = {
            "id": "bedrock-" + str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_body["content"][0]["text"] if response_body.get("content") else None,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Bedrock doesn't provide this
                "completion_tokens": -1,  # Bedrock doesn't provide this
                "total_tokens": -1  # Bedrock doesn't provide this
            }
        }
        
        # Add tool calls if present
        if response_body.get("tool_use"):
            openai_format_response["choices"][0]["message"]["tool_calls"] = []
            for tool_use in response_body["tool_use"]:
                openai_format_response["choices"][0]["message"]["tool_calls"].append({
                    "id": tool_use.get("id", "bedrock-tool-" + str(uuid.uuid4())),
                    "type": "function",
                    "function": {
                        "name": tool_use["name"],
                        "arguments": json.dumps(tool_use["input"]) 
                    }
                })
            
        return openai_format_response
    
    def _convert_to_anthropic_messages(self, messages: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Anthropic format for Bedrock
        
        Args:
            messages: Messages in OpenAI format
            
        Returns:
            Messages in Anthropic format
        """
        anthropic_messages = []
        
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
                continue
                
            # Add the message in Anthropic format
            anthropic_messages.append({
                "role": "user" if msg["role"] == "user" else "assistant",
                "content": content
            })
            
        return anthropic_messages 