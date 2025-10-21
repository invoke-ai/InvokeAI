# Cloud Model Integration

InvokeAI now supports cloud-based image generation models alongside local diffusion models. This allows you to leverage powerful cloud APIs from Google and OpenAI directly in your workflows.

## Supported Providers

### Google Gemini 2.5 Flash Image ✅ (Implemented)
- **Model**: `gemini-2.5-flash-image`
- **Pricing**: $0.039 per image
- **Max Resolution**: 2048×2048
- **Aspect Ratios**: 1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- **Features**: Deterministic generation with seed, natural language prompts
- **Documentation**: https://ai.google.dev/gemini-api/docs/image-generation

### Google Imagen 4 Ultra 🚧 (Coming Soon)
- **Model**: `imagen-4.0-ultra-generate-001`
- **Pricing**: $0.06 per image
- **Max Resolution**: 2048×2048 (2K)
- **Features**: SynthID watermark, prompt enhancement, safety filters

### OpenAI DALL-E 3 🚧 (Coming Soon)
- **Model**: `dall-e-3`
- **Pricing**: $0.04-$0.12 per image
- **Sizes**: 1024×1024, 1792×1024, 1024×1792
- **Features**: HD quality, vivid/natural styles

---

## Quick Start

### 1. Get API Keys

#### Google Gemini
1. Visit https://ai.google.dev/
2. Click "Get API key in Google AI Studio"
3. Create a new API key
4. Copy the key (starts with `AIza...`)

#### OpenAI (when available)
1. Visit https://platform.openai.com/api-keys
2. Create a new secret key
3. Copy the key (starts with `sk-...`)

### 2. Configure Environment

Create a `.env` file in your InvokeAI root directory:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Add your keys:
```bash
# For Google Gemini
GOOGLE_API_KEY=AIzaXXXXXXXXXXXXXXXXXXXXXXXX

# For OpenAI (when available)
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX
```

⚠️ **Important**: Never commit `.env` to git! It's already in `.gitignore`.

### 3. Install Dependencies

```bash
# Using pip
pip install httpx python-dotenv google-auth

# Or using uv (recommended)
uv pip install httpx python-dotenv google-auth
```

### 4. Test Integration

Run the test script to verify everything works:

```bash
python scripts/test_gemini_integration.py
```

You should see:
```
✓ Loaded .env file
✓ Found API key: AIzaXXXXXXXXXXXXXXXX...
✓ Successfully imported Gemini provider
✓ Created Gemini provider instance
✓ API credentials are valid
✓ Successfully generated image
✓ Saved test image to: outputs/test_gemini_output.png

✓ ALL TESTS PASSED!
```

### 5. Register Cloud Model

You need to manually register cloud models in the InvokeAI database.

**Via Python Script**:

```python
from invokeai.app.services.config import get_config
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.backend.model_manager.configs.cloud_models import GeminiFlashImageConfig
from invokeai.backend.model_manager.taxonomy import CloudProviderType

# Initialize services
config = get_config()
store = ModelRecordServiceSQL(db_path=config.db_path)

# Create Gemini config
gemini_config = GeminiFlashImageConfig(
    key="gemini-2.5-flash",
    name="Gemini 2.5 Flash Image",
    description="Google's state-of-the-art image generation model",
    source="https://ai.google.dev/",
    hash="cloud-model",  # Cloud models don't have file hashes
    path="cloud://gemini-2.5-flash-image",  # Virtual path
    file_size=0,  # Cloud models have no file size
)

# Register in database
store.add_model(gemini_config)
print(f"✓ Registered Gemini model with key: {gemini_config.key}")
```

**Via API** (coming soon):
```bash
curl -X POST http://localhost:9090/api/v1/cloud_models \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "google-gemini",
    "cloud_model_id": "gemini-2.5-flash-image",
    "name": "Gemini 2.5 Flash Image"
  }'
```

### 6. Use in Workflows

Once registered, the cloud model will appear in the model selector. You can use it with the **Gemini 2.5 Flash - Text to Image** node.

**Example Workflow**:
```json
{
  "nodes": {
    "1": {
      "type": "gemini_text_to_image",
      "model": {"key": "gemini-2.5-flash"},
      "prompt": "A futuristic city at night with neon lights",
      "aspect_ratio": "16:9",
      "seed": 42
    }
  }
}
```

---

## Usage Guide

### Aspect Ratios

Gemini 2.5 Flash supports 10 aspect ratios with optimal dimensions:

| Aspect Ratio | Dimensions | Use Case |
|--------------|------------|----------|
| 1:1 | 1024×1024 | Square / Social media |
| 3:2 | 1536×1024 | Photography / Landscape |
| 2:3 | 1024×1536 | Portrait / Vertical |
| 3:4 | 1152×1536 | Classic portrait |
| 4:3 | 1536×1152 | Classic landscape |
| 4:5 | 1024×1280 | Instagram portrait |
| 5:4 | 1280×1024 | Retro monitor |
| 9:16 | 576×1024 | Mobile / Stories |
| 16:9 | 1024×576 | Widescreen / Cinematic |
| 21:9 | 1344×576 | Ultra-wide / Panoramic |

### Deterministic Generation

Use the `seed` parameter for reproducible results:

```python
# Same prompt + seed = same image
gemini_node.prompt = "A red apple on a table"
gemini_node.seed = 12345
```

### Best Practices

**Prompting**:
- Be descriptive and specific
- Gemini understands natural language well
- Include style, mood, lighting, composition
- Example: "A serene Japanese garden at sunset, cherry blossoms, soft lighting, watercolor painting style"

**Cost Optimization**:
- Gemini: $0.039 per image (very affordable!)
- Generate multiple variations with different seeds
- Use lower resolutions for drafts (coming: resolution control)

**Error Handling**:
- Cloud calls can fail (network, quota, etc.)
- Set appropriate timeouts in workflows
- Cache successful results locally

---

## Troubleshooting

### "API key not found" Error

```
ValueError: API key not found for google-gemini.
Please set GOOGLE_API_KEY in .env file or environment variables.
```

**Solution**:
1. Verify `.env` file exists in InvokeAI root
2. Check `GOOGLE_API_KEY` is set correctly
3. Restart InvokeAI after adding/changing keys
4. Test with: `python scripts/test_gemini_integration.py`

### "API credentials are invalid" Error

**Solution**:
1. Verify API key is correct (should start with `AIza`)
2. Check key has not been restricted/deleted in Google AI Studio
3. Ensure Gemini API is enabled in your Google Cloud project
4. Try creating a new API key

### "Gemini API error (HTTP 429): Rate limit exceeded"

**Solution**:
1. You're making too many requests
2. Wait a few minutes before retrying
3. Check your quota in Google AI Studio
4. Consider upgrading to higher quota tier

### "Gemini API error (HTTP 403): Permission denied"

**Solution**:
1. API key may not have access to Gemini 2.5 Flash Image
2. Enable the Generative Language API in Google Cloud Console
3. Verify billing is enabled (if required)

### Image Not Appearing in Gallery

**Solution**:
1. Check InvokeAI logs for errors
2. Verify image was generated (check workflow status)
3. Refresh gallery
4. Check file permissions on `outputs/` directory

---

## API Reference

### CloudGenerationRequest

```python
class CloudGenerationRequest(BaseModel):
    prompt: str                    # Text description
    width: int                     # Image width (256-4096)
    height: int                    # Image height (256-4096)
    seed: Optional[int]            # Random seed (optional)
    num_images: int = 1            # Number of images (1-10)
    negative_prompt: Optional[str] # Not supported by Gemini
    guidance_scale: Optional[float] # Not supported by Gemini
```

### CloudGenerationResponse

```python
class CloudGenerationResponse(BaseModel):
    images: List[bytes]            # Generated images as PNG/JPEG bytes
    metadata: dict                 # Generation metadata
    provider_response: dict        # Raw API response
```

---

## Pricing Comparison

| Provider | Model | Price per Image | Notes |
|----------|-------|-----------------|-------|
| **Google** | Gemini 2.5 Flash | $0.039 | Best value! |
| **Google** | Imagen 4 Ultra | $0.060 | Highest quality |
| **OpenAI** | DALL-E 3 Standard | $0.040 | Good quality |
| **OpenAI** | DALL-E 3 HD | $0.080-$0.120 | Premium quality |

For comparison, running local models costs:
- **GPU electricity**: ~$0.01-0.05 per image (varies by hardware)
- **Hardware amortization**: Varies widely
- **Time**: 5-30 seconds locally vs 5-15 seconds in cloud

---

## Security & Privacy

### API Key Storage
- **Development**: Store in `.env` file (gitignored)
- **Production**: Use environment variables or secret management
- **Never**: Commit keys to git, share in public, or hardcode

### Data Privacy
- **Google**: Images and prompts may be used to improve services (check ToS)
- **OpenAI**: Similar data usage policies
- **Local models**: Complete privacy, no data leaves your machine

### Best Practices
1. Rotate API keys regularly
2. Use separate keys for dev/prod
3. Monitor usage and costs
4. Set up billing alerts
5. Review provider privacy policies

---

## Roadmap

### Phase 1: Google Gemini 2.5 Flash ✅ (Completed)
- [x] Provider implementation
- [x] Model configuration
- [x] Model loader
- [x] Text-to-image invocation
- [x] Documentation

### Phase 2: Google Imagen 4 Ultra 🚧 (In Progress)
- [ ] Vertex AI authentication
- [ ] Provider implementation
- [ ] Support for batch generation (1-4 images)
- [ ] Safety filter configuration
- [ ] Prompt enhancement toggle

### Phase 3: OpenAI Integration 📋 (Planned)
- [ ] DALL-E 3 provider
- [ ] Quality/style selection (HD, vivid, natural)
- [ ] Revised prompt handling
- [ ] GPT Image 1 (when access granted)

### Phase 4: Frontend Integration 📋 (Planned)
- [ ] Cloud model registration UI
- [ ] API key management panel
- [ ] Cloud node in workflow editor
- [ ] Cost estimation display
- [ ] Provider status indicators

### Phase 5: Advanced Features 💡 (Future)
- [ ] Response caching
- [ ] Cost tracking and budgets
- [ ] Rate limiting
- [ ] Batch processing
- [ ] Image-to-image (for supported models)

---

## Contributing

Want to add support for more cloud providers? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

**Potential providers**:
- Stability AI (Stable Diffusion API)
- Midjourney (when API available)
- Adobe Firefly
- Replicate
- RunPod
- Others?

---

## Support

- **Documentation**: This file
- **Issues**: https://github.com/invoke-ai/InvokeAI/issues
- **Discord**: https://discord.gg/ZmtBAhwWhy
- **Discussions**: https://github.com/invoke-ai/InvokeAI/discussions

---

## License

Cloud integration code is licensed under Apache 2.0, same as InvokeAI.

Individual cloud providers have their own terms of service:
- Google Gemini: https://ai.google.dev/terms
- OpenAI: https://openai.com/policies/terms-of-use
