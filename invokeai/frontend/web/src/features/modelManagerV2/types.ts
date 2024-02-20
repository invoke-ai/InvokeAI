import { z } from "zod";

export const zBaseModel = z.enum(['any', 'sd-1', 'sd-2', 'sdxl', 'sdxl-refiner']);
export const zModelType = z.enum([
    'main',
    'vae',
    'lora',
    'controlnet',
    'embedding',
    'ip_adapter',
    'clip_vision',
    't2i_adapter',
    'onnx', // TODO(psyche): Remove this when removed from backend
]);