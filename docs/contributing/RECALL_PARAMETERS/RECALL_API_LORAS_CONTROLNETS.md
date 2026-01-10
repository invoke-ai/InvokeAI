# Recall Parameters API - LoRAs, ControlNets, and IP Adapters

## Overview

The Recall Parameters API now supports recalling LoRAs, ControlNets (including T2I Adapters and Control LoRAs), and IP Adapters along with their associated weights and settings. This allows external processes to configure complex generation setups via REST API.

## Endpoints

### POST `/api/v1/recall/{queue_id}`

Updates recallable parameters for the frontend, including LoRAs, control adapters, and IP adapters.

**Path Parameters:**
- `queue_id` (string): The queue ID to associate parameters with (typically "default")

**Request Body:**

All fields are optional. Include only the parameters you want to update.

```typescript
{
  // Standard parameters
  positive_prompt?: string;
  negative_prompt?: string;
  model?: string;           // Model name or key
  steps?: number;
  cfg_scale?: number;
  width?: number;
  height?: number;
  seed?: number;
  // ... other standard parameters
  
  // LoRAs
  loras?: Array<{
    model_name: string;     // LoRA model name
    weight?: number;        // Default: 0.75, Range: -10 to 10
    is_enabled?: boolean;   // Default: true
  }>;
  
  // Control Layers (ControlNet, T2I Adapter, Control LoRA)
  control_layers?: Array<{
    model_name: string;            // Control adapter model name
    weight?: number;               // Default: 1.0, Range: -1 to 2
    begin_step_percent?: number;   // Default: 0.0, Range: 0 to 1
    end_step_percent?: number;     // Default: 1.0, Range: 0 to 1
    control_mode?: "balanced" | "more_prompt" | "more_control";  // ControlNet only
  }>;
  
  // IP Adapters
  ip_adapters?: Array<{
    model_name: string;            // IP Adapter model name
    weight?: number;               // Default: 1.0, Range: -1 to 2
    begin_step_percent?: number;   // Default: 0.0, Range: 0 to 1
    end_step_percent?: number;     // Default: 1.0, Range: 0 to 1
    method?: "full" | "style" | "composition";  // Default: "full"
  }>;
}
```

## Model Name Resolution

The backend automatically resolves model names to their internal keys:

1. **Main Models**: Resolved from the name to the model key
2. **LoRAs**: Searched in the LoRA model database
3. **Control Adapters**: Tried in order - ControlNet → T2I Adapter → Control LoRA
4. **IP Adapters**: Searched in the IP Adapter model database

Models that cannot be resolved are skipped with a warning in the logs.

## Frontend Behavior

### LoRAs
- **Fully Supported**: LoRAs are immediately added to the LoRA list in the UI
- Existing LoRAs are cleared before adding new ones
- Each LoRA's model config is fetched and applied with the specified weight
- LoRAs appear in the LoRA selector panel

### Control Layers & IP Adapters
- **Partially Supported**: Configurations are logged but not fully applied
- Reason: These require image inputs which cannot be provided via this API
- The model keys, weights, and settings are logged for reference
- Future enhancement: Could store configs for manual image attachment

## Examples

### 1. Add LoRAs Only

```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "loras": [
      {
        "model_name": "add-detail-xl",
        "weight": 0.8,
        "is_enabled": true
      },
      {
        "model_name": "sd_xl_offset_example-lora_1.0",
        "weight": 0.5,
        "is_enabled": true
      }
    ]
  }'
```

### 2. Configure Control Layers

```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "control_layers": [
      {
        "model_name": "controlnet-canny-sdxl-1.0",
        "weight": 0.75,
        "begin_step_percent": 0.0,
        "end_step_percent": 0.8,
        "control_mode": "balanced"
      }
    ]
  }'
```

### 3. Configure IP Adapters

```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "ip_adapters": [
      {
        "model_name": "ip-adapter-plus-face_sd15",
        "weight": 0.7,
        "begin_step_percent": 0.0,
        "end_step_percent": 1.0,
        "method": "composition"
      }
    ]
  }'
```

### 4. Combined Configuration

```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "positive_prompt": "masterpiece, high quality photo",
    "negative_prompt": "blurry, low quality",
    "model": "FLUX Schnell",
    "steps": 25,
    "cfg_scale": 8.0,
    "width": 1024,
    "height": 768,
    "seed": 42,
    "loras": [
      {
        "model_name": "add-detail-xl",
        "weight": 0.6
      }
    ],
    "control_layers": [
      {
        "model_name": "controlnet-depth-sdxl-1.0",
        "weight": 1.0,
        "begin_step_percent": 0.0,
        "end_step_percent": 0.7
      }
    ]
  }'
```

## Response Format

```json
{
  "status": "success",
  "queue_id": "default",
  "updated_count": 10,
  "parameters": {
    "positive_prompt": "...",
    "steps": 25,
    "loras": [
      {
        "model_key": "abc123...",
        "weight": 0.6,
        "is_enabled": true
      }
    ],
    "control_layers": [...],
    "ip_adapters": [...]
  }
}
```

## WebSocket Events

When parameters are updated, a `recall_parameters_updated` event is emitted via WebSocket to the queue room. The frontend automatically:

1. Applies standard parameters (prompts, steps, dimensions, etc.)
2. Loads and adds LoRAs to the LoRA list
3. Logs control layer and IP adapter configurations

## Limitations

1. **Control Layers & IP Adapters**: Cannot be fully applied without image inputs
   - Images must be provided separately through the UI
   - The API stores configurations but doesn't create canvas layers

2. **Model Name Case Sensitivity**: Model names must match exactly (case-sensitive)

3. **Model Availability**: Models must be installed in InvokeAI before they can be recalled

## Testing

Use the provided test script:

```bash
./test_recall_loras_controlnets.sh
```

This will test:
- LoRA addition with multiple models
- Control layer configuration
- IP adapter configuration  
- Combined parameter updates

## Logging

Frontend logs can be viewed in the browser console:
- Set `localStorage.ROARR_FILTER = 'debug'` to see all debug messages
- Look for messages from the `events` namespace
- LoRA loading, model resolution, and parameter application are logged

Backend logs show:
- Model name → key resolution
- Parameter storage
- Event emission
- Any errors or warnings

## Future Enhancements

Potential improvements:
1. Support image URLs for control layers and IP adapters
2. Pre-create canvas layers with stored configurations
3. Support for regional guidance with IP adapters
4. Batch operations for multiple queue IDs
