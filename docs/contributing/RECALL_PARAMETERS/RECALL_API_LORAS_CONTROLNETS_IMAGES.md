# Recall Parameters API - LoRAs, ControlNets, and IP Adapters with Images

## Overview

The Recall Parameters API supports recalling LoRAs, ControlNets (including T2I Adapters and Control LoRAs), and IP Adapters along with their associated weights and settings. Control Layers and IP Adapters can now include image references from the `INVOKEAI_ROOT/outputs/images` directory for fully functional control and image prompt functionality.

## Key Features

✅ **LoRAs**: Fully functional - adds to UI, queries model configs, applies weights
✅ **Control Layers**: Full support with optional images from outputs/images
✅ **IP Adapters**: Full support with optional reference images from outputs/images
✅ **Model Name Resolution**: Automatic lookup from human-readable names to internal keys
✅ **Image Validation**: Backend validates that image files exist before sending

## Endpoints

### POST `/api/v1/recall/{queue_id}`

Updates recallable parameters for the frontend, including LoRAs, control adapters, and IP adapters with optional images.

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
    image_name?: string;           // Optional image filename from outputs/images
    weight?: number;               // Default: 1.0, Range: -1 to 2
    begin_step_percent?: number;   // Default: 0.0, Range: 0 to 1
    end_step_percent?: number;     // Default: 1.0, Range: 0 to 1
    control_mode?: "balanced" | "more_prompt" | "more_control";  // ControlNet only
  }>;
  
  // IP Adapters
  ip_adapters?: Array<{
    model_name: string;            // IP Adapter model name
    image_name?: string;           // Optional reference image filename from outputs/images
    weight?: number;               // Default: 1.0, Range: -1 to 2
    begin_step_percent?: number;   // Default: 0.0, Range: 0 to 1
    end_step_percent?: number;     // Default: 1.0, Range: 0 to 1
    method?: "full" | "style" | "composition";  // Default: "full"
    influence?: "Lowest" | "Low" | "Medium" | "High" | "Highest";  // Flux Redux only; default: "highest"
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

## Image File Handling

### Image Path Resolution

When you specify an `image_name`, the backend:
1. Constructs the full path: `{INVOKEAI_ROOT}/outputs/images/{image_name}`
2. Validates that the file exists
3. Includes the image reference in the event sent to the frontend
4. Logs whether the image was found or not

### Image Naming

Images should be referenced by their filename as it appears in the outputs/images directory:
- ✅ Correct: `"image_name": "example.png"`
- ✅ Correct: `"image_name": "my_control_image_20240110.jpg"`
- ❌ Incorrect: `"image_name": "outputs/images/example.png"`  (use relative filename only)
- ❌ Incorrect: `"image_name": "/full/path/to/example.png"`   (use relative filename only)

## Frontend Behavior

### LoRAs
- **Fully Supported**: LoRAs are immediately added to the LoRA list in the UI
- Existing LoRAs are cleared before adding new ones
- Each LoRA's model config is fetched and applied with the specified weight
- LoRAs appear in the LoRA selector panel

### Control Layers with Images
- **Fully Supported**: Control layers now support images from outputs/images
- Configuration includes model, weights, step percentages, and image reference
- Image availability is logged in frontend console
- Images can be used to create actual control layers through the UI

### IP Adapters with Images
- **Fully Supported**: IP Adapters now support reference images from outputs/images
- Configuration includes model, weights, step percentages, method, and image reference
- Image availability is logged in frontend console
- Images can be used to create actual reference image layers through the UI

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

### 2. Configure Control Layers with Image

Replace `my_control_image.png` with an actual image filename from your outputs/images directory.

```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "control_layers": [
      {
        "model_name": "controlnet-canny-sdxl-1.0",
        "image_name": "my_control_image.png",
        "weight": 0.75,
        "begin_step_percent": 0.0,
        "end_step_percent": 0.8,
        "control_mode": "balanced"
      }
    ]
  }'
```

### 3. Configure IP Adapters with Reference Image

Replace `reference_face.png` with an actual image filename from your outputs/images directory.

```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "ip_adapters": [
      {
        "model_name": "ip-adapter-plus-face_sd15",
        "image_name": "reference_face.png",
        "weight": 0.7,
        "begin_step_percent": 0.0,
        "end_step_percent": 1.0,
        "method": "composition"
      }
    ]
  }'
```

### 4. Complete Configuration with All Features

```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "positive_prompt": "masterpiece, detailed photo with specific style",
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
        "weight": 0.6,
        "is_enabled": true
      }
    ],
    "control_layers": [
      {
        "model_name": "controlnet-depth-sdxl-1.0",
        "image_name": "depth_map.png",
        "weight": 1.0,
        "begin_step_percent": 0.0,
        "end_step_percent": 0.7
      }
    ],
    "ip_adapters": [
      {
        "model_name": "ip-adapter-plus-face_sd15",
        "image_name": "style_reference.png",
        "weight": 0.5,
        "begin_step_percent": 0.0,
        "end_step_percent": 1.0,
        "method": "style"
      }
    ]
  }'
```

## Response Format

```json
{
  "status": "success",
  "queue_id": "default",
  "updated_count": 15,
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
    "control_layers": [
      {
        "model_key": "controlnet-xyz...",
        "weight": 1.0,
        "image": {
          "image_name": "depth_map.png"
        }
      }
    ],
    "ip_adapters": [
      {
        "model_key": "ip-adapter-xyz...",
        "weight": 0.5,
        "image": {
          "image_name": "style_reference.png"
        }
      }
    ]
  }
}
```

## WebSocket Events

When parameters are updated, a `recall_parameters_updated` event is emitted via WebSocket to the queue room. The frontend automatically:

1. Applies standard parameters (prompts, steps, dimensions, etc.)
2. Loads and adds LoRAs to the LoRA list
3. Logs control layer and IP adapter configurations with image information
4. Makes image references available for manual canvas/reference image creation

## Logging

### Backend Logs

Backend logs show:
- Model name → key resolution (success/failure)
- Image file validation (found/not found)
- Parameter storage confirmation
- Event emission status

Example log messages:
```
INFO: Resolved ControlNet model name 'controlnet-canny-sdxl-1.0' to key 'controlnet-xyz...'
INFO: Found image file: depth_map.png
INFO: Updated 12 recall parameters for queue default
INFO: Resolved 1 LoRA(s)
INFO: Resolved 1 control layer(s)
INFO: Resolved 1 IP adapter(s)
```

### Frontend Logs

Frontend logs (check browser console):
- Set `localStorage.ROARR_FILTER = 'debug'` to see all debug messages
- Look for messages from the `events` namespace
- LoRA loading, model resolution, and parameter application are logged

Example log messages:
```
INFO: Applied 5 recall parameters to store
INFO: Received 1 control layer(s) with image support
INFO: Control layer 1: controlnet-xyz... (weight: 0.75, image: depth_map.png)
DEBUG: Control layer 1 image available at: outputs/images/depth_map.png
INFO: Received 1 IP adapter(s) with image support
INFO: IP adapter 1: ip-adapter-xyz... (weight: 0.7, image: style_reference.png)
DEBUG: IP adapter 1 image available at: outputs/images/style_reference.png
```

## Limitations

1. **Canvas Integration**: Control layers and IP adapters with images are currently logged but not automatically added to canvas layers
   - Users can view the configuration and manually create canvas layers with the provided images
   - Future enhancement: Auto-create canvas layers with stored images

2. **Model Availability**: Models must be installed in InvokeAI before they can be recalled

3. **Image Availability**: Images must exist in the outputs/images directory
   - Missing images are logged as warnings but don't fail the request
   - Other parameters are still applied even if images are missing

4. **Image URLs**: Only local filenames from outputs/images are supported
   - Remote image URLs are not currently supported

## Testing

Use the provided test script:

```bash
./test_recall_loras_controlnets.sh
```

This will test:
- LoRA addition with multiple models
- Control layer configuration with image references
- IP adapter configuration with image references
- Combined parameter updates with all features

Note: Update the image names in the test script to match actual images in your outputs/images directory.

## Troubleshooting

### Images Not Found

If you see "Image file not found" in the logs:
1. Verify the image filename matches exactly (case-sensitive)
2. Ensure the image is in `{INVOKEAI_ROOT}/outputs/images/`
3. Check that the filename doesn't include the `outputs/images/` prefix

### Models Not Found

If you see "Could not find model" messages:
1. Verify the model name matches exactly (case-sensitive)
2. Ensure the model is installed in InvokeAI
3. Check the model name using the models browser in the UI

### Event Not Received

If the frontend doesn't receive the event:
1. Check browser console for connection errors
2. Verify the queue_id matches the frontend's queue (usually "default")
3. Check backend logs for event emission errors

## Future Enhancements

Potential improvements:
1. Auto-create canvas layers with provided control layer images
2. Auto-create reference image layers with provided IP adapter images
3. Support for image URLs
4. Batch operations for multiple queue IDs
5. Image upload capability (accept base64 or file upload)
