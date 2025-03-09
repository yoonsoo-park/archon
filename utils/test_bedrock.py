import os
import argparse
import asyncio
import json
from dotenv import load_dotenv
from utils.bedrock_client import get_bedrock_runtime_client, get_bedrock_management_client, BedrockClaudeModel
from pydantic_ai import Agent

# Load environment variables
load_dotenv()

async def test_bedrock_connection():
    """
    Test the connection to AWS Bedrock using the configured credentials.
    """
    print("Testing AWS Bedrock connection...")
    
    try:
        # Get the Bedrock management client (not runtime client) for listing models
        client = get_bedrock_management_client()
        
        # Get available models to verify connection
        response = client.list_foundation_models()
        
        print("\n✅ Successfully connected to AWS Bedrock!")
        print(f"The following Claude 3 models are available:")
        
        # Filter for Claude models
        claude_models = [model for model in response['modelSummaries'] 
                        if 'claude' in model['modelId'].lower()]
        
        if not claude_models:
            print("No Claude models found. Make sure your AWS account has access to Claude models.")
        else:
            for model in claude_models:
                print(f"  - {model['modelId']}")
                
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to connect to AWS Bedrock: {str(e)}")
        print("\nPlease check your AWS credentials and permissions.")
        print("Make sure you have the following environment variables set:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION (default: us-east-1)")
        print("\nAlso ensure your AWS IAM user has access to the Bedrock service.")
        return False

async def test_bedrock_claude_model():
    """
    Test sending a request to the Claude model on AWS Bedrock.
    """
    print("\nTesting Claude 3.7 on AWS Bedrock...")
    
    try:
        # Get the model ID from environment variables or use the default
        model_id = os.environ.get("AWS_CLAUDE_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
        inference_profile = os.environ.get("AWS_BEDROCK_INFERENCE_PROFILE", "")
        
        if inference_profile:
            print(f"\nUsing inference profile: {inference_profile}")
        else:
            print("\nNo inference profile specified. Using direct model ID.")
            print("You may encounter errors if your AWS account requires inference profiles.")
        
        # Create a BedrockClaudeModel instance
        claude_model = BedrockClaudeModel(model_id=model_id)
        
        # Print model details for debugging
        print(f"\nModel details:")
        print(f"  - name: {claude_model.name()}")
        model_obj = await claude_model.agent_model()
        print(f"  - agent_model: {type(model_obj).__name__}")
        print(f"  - model_id: {claude_model._model_id}")
        print(f"  - inference_profile: {claude_model._inference_profile or 'Not set'}")
        print(f"  - supports_functions: {claude_model.supports_functions}")
        
        # Instead of using the Agent class, directly call our predict method
        print("\nSending direct message to Claude 3.7...")
        messages = [
            {"role": "user", "content": "Hello! Can you confirm you're Claude 3.7 running on AWS Bedrock?"}
        ]
        
        response = await claude_model.predict(messages)
        
        # Check if we got a response
        if hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
            print("\n✅ Successfully called Claude 3.7 on AWS Bedrock!")
            print("\nResponse from Claude 3.7:")
            print(response.parts[0].text)
            return True
        else:
            print("\n❌ Got a response but in unexpected format.")
            print(f"Response structure: {type(response)}")
            return False
            
    except Exception as e:
        print(f"\n❌ Failed to call Claude 3.7 on AWS Bedrock: {str(e)}")
        print("\nPlease check your AWS credentials and permissions.")
        import traceback
        print(traceback.format_exc())
        return False

async def main():
    parser = argparse.ArgumentParser(description='Test AWS Bedrock Integration')
    parser.add_argument('--connection-only', action='store_true', 
                        help='Only test the connection to AWS Bedrock')
    args = parser.parse_args()
    
    # Test the connection to AWS Bedrock
    connection_success = await test_bedrock_connection()
    
    if not connection_success:
        return
    
    # If not connection_only, also test the Claude model
    if not args.connection_only:
        await test_bedrock_claude_model()

if __name__ == "__main__":
    asyncio.run(main()) 