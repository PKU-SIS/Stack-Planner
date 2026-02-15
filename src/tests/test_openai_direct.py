#!/usr/bin/env python3
"""
Direct OpenAI API test for the new endpoint
"""
from openai import OpenAI
import requests
import json


def test_api_health():
    """Test basic API endpoint health"""
    print("🔍 Testing API endpoint health...")
    base_url = "http://10.1.1.212:8080"

    try:
        # Test basic connectivity
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health endpoint accessible")
        else:
            print("⚠️ Health endpoint returned non-200 status")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")

    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Models endpoint accessible")
            models_data = response.json()
            print(f"📝 Available models: {json.dumps(models_data, indent=2)}")
        else:
            print("⚠️ Models endpoint returned non-200 status")
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")


def test_openai_client():
    """Test OpenAI client with the new endpoint"""
    print("\n🤖 Testing OpenAI client...")

    try:
        # Create OpenAI client with custom endpoint
        client = OpenAI(
            base_url="http://10.1.1.212:8080/v1",
            api_key="EMPTY",  # Some endpoints accept EMPTY as api_key
        )

        print("✅ OpenAI client created successfully")

        # Test chat completion
        print("💬 Testing chat completion...")
        response = client.chat.completions.create(
            model="Qwen2.5-32B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'API test successful' to confirm the connection works.",
                }
            ],
            max_tokens=50,
            temperature=0.0,
        )

        print("✅ Chat completion successful!")
        print(f"📝 Response: {response.choices[0].message.content}")
        print(f"🔢 Usage: {response.usage}")

        return True

    except Exception as e:
        print("❌ OpenAI client test failed!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        # Try to get more details from the error
        if hasattr(e, "response"):
            print(
                f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}"
            )
            print(
                f"Response text: {e.response.text if hasattr(e.response, 'text') else 'N/A'}"
            )

        return False


def test_alternative_auth():
    """Test with alternative authentication methods"""
    print("\n🔑 Testing alternative authentication methods...")

    # Test with different api_key values
    auth_methods = ["EMPTY", "", "sk-fake-key", "test-key"]

    for auth_key in auth_methods:
        print(f"\n🔍 Testing with api_key: '{auth_key}'")
        try:
            client = OpenAI(base_url="http://10.1.1.212:8080/v1", api_key=auth_key)

            response = client.chat.completions.create(
                model="Qwen2.5-32B-Instruct",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10,
                temperature=0.0,
            )

            print(f"✅ Success with api_key: '{auth_key}'")
            print(f"📝 Response: {response.choices[0].message.content}")
            return True

        except Exception as e:
            print(f"❌ Failed with api_key '{auth_key}': {str(e)}")
            continue

    return False


if __name__ == "__main__":
    print("🚀 Starting OpenAI API direct test...\n")
    print("Target API: http://10.1.1.212:8080/v1")
    print("Model: Qwen2.5-32B-Instruct")
    print("=" * 60)

    # Test API health
    test_api_health()

    # Test OpenAI client
    success = test_openai_client()

    if not success:
        # Try alternative auth methods
        print("\n🔄 Trying alternative authentication methods...")
        success = test_alternative_auth()

    print("\n" + "=" * 60)
    if success:
        print("🎉 API test PASSED! The endpoint is working correctly.")
    else:
        print("💥 API test FAILED! Please check the endpoint configuration.")
