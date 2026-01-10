# Recall Parameters API

## Overview

A new REST API endpoint has been added to the InvokeAI backend that allows programmatic updates to recallable parameters from another process. This enables external applications or scripts to modify frontend parameters like prompts, models, and step counts via HTTP requests.

When parameters are updated via the API, the backend automatically broadcasts a WebSocket event to all connected frontend clients subscribed to that queue, causing them to update immediately.

## How It Works

1. **API Request**: External application sends a POST request with parameters to update
2. **Storage**: Parameters are stored in client state persistence, associated with a queue ID
3. **Broadcast**: A WebSocket event (`recall_parameters_updated`) is emitted to all frontend clients listening to that queue
4. **Frontend Update**: Connected frontend clients receive the event and can process the updated parameters
5. **Immediate Display**: The frontend UI updates automatically with the new values

This means if you have the InvokeAI frontend open in a browser, updating parameters via the API will instantly reflect on the screen without any manual action needed.

## Endpoint

**Base URL**: `http://localhost:9090/api/v1/recall/{queue_id}`

## POST - Update Recall Parameters

Updates recallable parameters for a given queue ID.

### Request

```http
POST /api/v1/recall/{queue_id}
Content-Type: application/json

{
  "positive_prompt": "a beautiful landscape",
  "negative_prompt": "blurry, low quality",
  "model": "sd-1.5",
  "steps": 20,
  "cfg_scale": 7.5,
  "width": 512,
  "height": 512,
  "seed": 12345
}
```

### Parameters

All parameters are optional. Only provide the parameters you want to update:

| Parameter | Type | Description |
|-----------|------|-------------|
| `positive_prompt` | string | Positive prompt text |
| `negative_prompt` | string | Negative prompt text |
| `model` | string | Main model name/identifier |
| `refiner_model` | string | Refiner model name/identifier |
| `vae_model` | string | VAE model name/identifier |
| `scheduler` | string | Scheduler name |
| `steps` | integer | Number of generation steps (≥1) |
| `refiner_steps` | integer | Number of refiner steps (≥0) |
| `cfg_scale` | number | CFG scale for guidance |
| `cfg_rescale_multiplier` | number | CFG rescale multiplier |
| `refiner_cfg_scale` | number | Refiner CFG scale |
| `guidance` | number | Guidance scale |
| `width` | integer | Image width in pixels (≥64) |
| `height` | integer | Image height in pixels (≥64) |
| `seed` | integer | Random seed (≥0) |
| `denoise_strength` | number | Denoising strength (0-1) |
| `refiner_denoise_start` | number | Refiner denoising start (0-1) |
| `clip_skip` | integer | CLIP skip layers (≥0) |
| `seamless_x` | boolean | Enable seamless X tiling |
| `seamless_y` | boolean | Enable seamless Y tiling |
| `refiner_positive_aesthetic_score` | number | Refiner positive aesthetic score |
| `refiner_negative_aesthetic_score` | number | Refiner negative aesthetic score |

### Response

```json
{
  "status": "success",
  "queue_id": "queue_123",
  "updated_count": 7,
  "parameters": {
    "positive_prompt": "a beautiful landscape",
    "negative_prompt": "blurry, low quality",
    "model": "sd-1.5",
    "steps": 20,
    "cfg_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": 12345
  }
}
```

## GET - Retrieve Recall Parameters

Retrieves metadata about stored recall parameters.

### Request

```http
GET /api/v1/recall/{queue_id}
```

### Response

```json
{
  "status": "success",
  "queue_id": "queue_123",
  "note": "Use the frontend to access stored recall parameters, or set specific parameters using POST"
}
```

## Usage Examples

### Using cURL

```bash
# Update prompts and model
curl -X POST http://localhost:9090/api/v1/recall/queue_123 \
  -H "Content-Type: application/json" \
  -d '{
    "positive_prompt": "a cyberpunk city at night",
    "negative_prompt": "dark, unclear",
    "model": "sd-1.5",
    "steps": 30
  }'

# Update just the seed
curl -X POST http://localhost:9090/api/v1/recall/queue_123 \
  -H "Content-Type: application/json" \
  -d '{"seed": 99999}'
```

### Using Python

```python
import requests
import json

# Configuration
API_URL = "http://localhost:9090/api/v1/recall/queue_123"

# Update multiple parameters
params = {
    "positive_prompt": "a serene forest",
    "negative_prompt": "people, buildings",
    "steps": 25,
    "cfg_scale": 7.0,
    "seed": 42
}

response = requests.post(API_URL, json=params)
result = response.json()

print(f"Status: {result['status']}")
print(f"Updated {result['updated_count']} parameters")
print(json.dumps(result['parameters'], indent=2))
```

### Using Node.js/JavaScript

```javascript
const API_URL = 'http://localhost:9090/api/v1/recall/queue_123';

const params = {
  positive_prompt: 'a beautiful sunset',
  negative_prompt: 'blurry',
  steps: 20,
  width: 768,
  height: 768,
  seed: 12345
};

fetch(API_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(params)
})
  .then(res => res.json())
  .then(data => console.log(data));
```

## Implementation Details

- Parameters are stored in the client state persistence service, using keys prefixed with `recall_`
- The parameters are associated with a `queue_id`, allowing multiple concurrent sessions to maintain separate parameter sets
- Only non-null parameters are processed and stored
- The endpoint provides validation for numeric ranges (e.g., steps ≥ 1, dimensions ≥ 64)
- All parameter values are JSON-serialized for storage

## Integration with Frontend

The stored parameters can be accessed by the frontend through the existing client state API or by implementing hooks that read from the recall parameter storage. This allows external applications to pre-populate generation parameters before the user initiates image generation.

## Error Handling

- **400 Bad Request**: Invalid parameters or parameter values
- **500 Internal Server Error**: Server-side error storing or retrieving parameters

Errors include detailed messages explaining what went wrong.
