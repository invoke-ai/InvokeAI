import {
  zControlNetConfigV2,
  zImageWithDims,
  zIPAdapterConfigV2,
  zT2IAdapterConfigV2,
} from 'features/controlLayers/util/controlAdapters';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import {
  type ParameterHeight,
  type ParameterNegativePrompt,
  type ParameterNegativeStylePromptSDXL,
  type ParameterPositivePrompt,
  type ParameterPositiveStylePromptSDXL,
  type ParameterWidth,
  zAutoNegative,
  zParameterNegativePrompt,
  zParameterPositivePrompt,
  zParameterStrength,
} from 'features/parameters/types/parameterSchemas';
import { z } from 'zod';

const zTool = z.enum(['brush', 'eraser', 'move', 'rect']);
export type Tool = z.infer<typeof zTool>;
const zDrawingTool = zTool.extract(['brush', 'eraser']);
export type DrawingTool = z.infer<typeof zDrawingTool>;

const zPoints = z.array(z.number()).refine((points) => points.length % 2 === 0, {
  message: 'Must have an even number of points',
});
const zVectorMaskLine = z.object({
  id: z.string(),
  type: z.literal('vector_mask_line'),
  tool: zDrawingTool,
  strokeWidth: z.number().min(1),
  points: zPoints,
});
export type VectorMaskLine = z.infer<typeof zVectorMaskLine>;

const zVectorMaskRect = z.object({
  id: z.string(),
  type: z.literal('vector_mask_rect'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});
export type VectorMaskRect = z.infer<typeof zVectorMaskRect>;

const zLayerBase = z.object({
  id: z.string(),
  isEnabled: z.boolean().default(true),
  isSelected: z.boolean().default(true),
});

const zRect = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});
const zRenderableLayerBase = zLayerBase.extend({
  x: z.number(),
  y: z.number(),
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
});

const zControlAdapterLayer = zRenderableLayerBase.extend({
  type: z.literal('control_adapter_layer'),
  opacity: z.number().gte(0).lte(1),
  isFilterEnabled: z.boolean(),
  controlAdapter: z.discriminatedUnion('type', [zControlNetConfigV2, zT2IAdapterConfigV2]),
});
export type ControlAdapterLayer = z.infer<typeof zControlAdapterLayer>;

const zIPAdapterLayer = zLayerBase.extend({
  type: z.literal('ip_adapter_layer'),
  ipAdapter: zIPAdapterConfigV2,
});
export type IPAdapterLayer = z.infer<typeof zIPAdapterLayer>;

const zRgbColor = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
});
const zRegionalGuidanceLayer = zRenderableLayerBase.extend({
  type: z.literal('regional_guidance_layer'),
  maskObjects: z.array(z.discriminatedUnion('type', [zVectorMaskLine, zVectorMaskRect])),
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  ipAdapters: z.array(zIPAdapterConfigV2),
  previewColor: zRgbColor,
  autoNegative: zAutoNegative,
  uploadedMaskImage: zImageWithDims.nullable(),
});
export type RegionalGuidanceLayer = z.infer<typeof zRegionalGuidanceLayer>;

const zInitialImageLayer = zRenderableLayerBase.extend({
  type: z.literal('initial_image_layer'),
  opacity: z.number().gte(0).lte(1),
  image: zImageWithDims.nullable(),
  denoisingStrength: zParameterStrength,
});
export type InitialImageLayer = z.infer<typeof zInitialImageLayer>;

export const zLayer = z.discriminatedUnion('type', [
  zRegionalGuidanceLayer,
  zControlAdapterLayer,
  zIPAdapterLayer,
  zInitialImageLayer,
]);
export type Layer = z.infer<typeof zLayer>;

export type ControlLayersState = {
  _version: 3;
  selectedLayerId: string | null;
  layers: Layer[];
  brushSize: number;
  globalMaskLayerOpacity: number;
  positivePrompt: ParameterPositivePrompt;
  negativePrompt: ParameterNegativePrompt;
  positivePrompt2: ParameterPositiveStylePromptSDXL;
  negativePrompt2: ParameterNegativeStylePromptSDXL;
  shouldConcatPrompts: boolean;
  size: {
    width: ParameterWidth;
    height: ParameterHeight;
    aspectRatio: AspectRatioState;
  };
};
