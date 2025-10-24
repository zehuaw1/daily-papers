#!/usr/bin/env python3
"""
Simple script to test LLM API connection with minimal token usage.
Tests if the API key and configuration are working correctly.
"""
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import AI_CONFIG


def test_llm_connection():
    """Test LLM API with a minimal token prompt"""
    print("Testing LLM API connection...")
    print(f"Model: {AI_CONFIG['model']}")
    print(f"Base URL: {AI_CONFIG['base_url']}")
    print(f"API Key: {AI_CONFIG['api_key'][:10]}..." if len(AI_CONFIG['api_key']) > 10 else "API Key: [TOO SHORT]")
    print()

    # Initialize OpenAI client
    client = OpenAI(
        api_key=AI_CONFIG["api_key"],
        base_url=AI_CONFIG["base_url"]
    )

    # Minimal test prompt to use as few tokens as possible
    test_prompt = "Say 'ok'"

    try:
        print("Sending test prompt:", repr(test_prompt))
        response = client.chat.completions.create(
            model=AI_CONFIG["model"],
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
        )

        # Extract response
        content = response.choices[0].message.content.strip()
        usage = response.usage

        print("\n✅ SUCCESS! LLM API is working.")
        print(f"\nResponse: {content}")
        print(f"\nToken usage:")
        print(f"  - Prompt tokens: {usage.prompt_tokens}")
        print(f"  - Completion tokens: {usage.completion_tokens}")
        print(f"  - Total tokens: {usage.total_tokens}")

        # Calculate cost
        input_cost = (usage.prompt_tokens / 1_000_000) * AI_CONFIG["price_per_million_input_tokens"]
        output_cost = (usage.completion_tokens / 1_000_000) * AI_CONFIG["price_per_million_output_tokens"]
        total_cost = input_cost + output_cost

        print(f"\nEstimated cost: ${total_cost:.6f} USD")

        return True

    except Exception as e:
        print(f"\n❌ FAILED! LLM API error:")
        print(f"Error: {str(e)}")

        # Check for common issues
        if "API key not valid" in str(e) or "INVALID_ARGUMENT" in str(e):
            print("\n💡 Suggestion: Check that your API key is correct in config.py")
        elif "base_url" in str(e).lower():
            print("\n💡 Suggestion: Check that the base_url is correct in config.py")
        elif "model" in str(e).lower():
            print("\n💡 Suggestion: Check that the model name is correct in config.py")

        return False


if __name__ == "__main__":
    success = test_llm_connection()
    sys.exit(0 if success else 1)
