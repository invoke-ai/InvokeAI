#!/usr/bin/env python3
"""Test script for Google Gemini 2.5 Flash Image integration.

This script tests the Gemini cloud provider integration end-to-end:
1. Validates API key
2. Generates a test image
3. Saves the result

Usage:
    python scripts/test_gemini_integration.py

Prerequisites:
    - Set GOOGLE_API_KEY environment variable or create .env file
    - Install dependencies: pip install httpx python-dotenv
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_gemini_integration():
    """Test Gemini integration end-to-end."""
    print("=" * 80)
    print("Google Gemini 2.5 Flash Image Integration Test")
    print("=" * 80)
    print()

    # Load environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()
        print("✓ Loaded .env file")
    except ImportError:
        print("⚠ python-dotenv not installed, using environment variables only")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✗ ERROR: GOOGLE_API_KEY not found in environment")
        print("  Please set GOOGLE_API_KEY in .env file or environment variables")
        return False

    print(f"✓ Found API key: {api_key[:20]}...")
    print()

    # Import provider
    try:
        from invokeai.app.services.cloud_providers.google_gemini_provider import GoogleGeminiProvider
        from invokeai.app.services.cloud_providers.provider_base import CloudGenerationRequest

        print("✓ Successfully imported Gemini provider")
    except ImportError as e:
        print(f"✗ ERROR: Failed to import Gemini provider: {e}")
        return False

    # Create provider instance
    try:
        provider = GoogleGeminiProvider(api_key=api_key, config={})
        print("✓ Created Gemini provider instance")
    except Exception as e:
        print(f"✗ ERROR: Failed to create provider: {e}")
        return False

    # Validate credentials
    print("\nValidating API credentials...")
    try:
        is_valid = await provider.validate_credentials()
        if is_valid:
            print("✓ API credentials are valid")
        else:
            print("✗ ERROR: API credentials are invalid")
            return False
    except Exception as e:
        print(f"✗ ERROR: Credential validation failed: {e}")
        return False

    # Generate test image
    print("\nGenerating test image...")
    print("  Prompt: 'A serene mountain landscape at sunset'")
    print("  Size: 1024x1024")
    print("  Seed: 42")
    print()

    try:
        request = CloudGenerationRequest(
            prompt="A serene mountain landscape at sunset",
            width=1024,
            height=1024,
            seed=42,
            num_images=1,
        )

        print("⏳ Calling Gemini API (this may take 10-30 seconds)...")
        response = await provider.generate_image(request)
        print("✓ Successfully generated image")
        print(f"  Generated {len(response.images)} image(s)")
        print(f"  Metadata: {response.metadata}")
    except Exception as e:
        print(f"✗ ERROR: Image generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Save image
    try:
        from PIL import Image
        from io import BytesIO

        image_bytes = response.images[0]
        pil_image = Image.open(BytesIO(image_bytes))

        output_path = project_root / "outputs" / "test_gemini_output.png"
        output_path.parent.mkdir(exist_ok=True)
        pil_image.save(output_path)

        print(f"\n✓ Saved test image to: {output_path}")
        print(f"  Image size: {pil_image.size}")
        print(f"  Image mode: {pil_image.mode}")
    except Exception as e:
        print(f"⚠ Warning: Failed to save image: {e}")

    print()
    print("=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Register the Gemini model in InvokeAI")
    print("2. Use the 'Gemini 2.5 Flash - Text to Image' node in workflows")
    print()

    return True


def main():
    """Main entry point."""
    try:
        # Run async test
        result = asyncio.run(test_gemini_integration())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
