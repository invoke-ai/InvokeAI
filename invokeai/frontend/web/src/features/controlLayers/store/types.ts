import {
  zControlNetConfigV2,
  zImageWithDims,
  zIPAdapterConfigV2,
  zT2IAdapterConfigV2,
} from 'features/controlLayers/util/controlAdapters';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import type {
  ParameterHeight,
  ParameterNegativePrompt,
  ParameterNegativeStylePromptSDXL,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';
import {
  zAutoNegative,
  zParameterNegativePrompt,
  zParameterPositivePrompt,
  zParameterStrength,
} from 'features/parameters/types/parameterSchemas';
import type { IRect } from 'konva/lib/types';
import { z } from 'zod';

const zTool = z.enum(['brush', 'eraser', 'move', 'rect']);
export type Tool = z.infer<typeof zTool>;
const zDrawingTool = zTool.extract(['brush', 'eraser']);
export type DrawingTool = z.infer<typeof zDrawingTool>;

const zPoints = z.array(z.number()).refine((points) => points.length % 2 === 0, {
  message: 'Must have an even number of points',
});
const zOLD_VectorMaskLine = z.object({
  id: z.string(),
  type: z.literal('vector_mask_line'),
  tool: zDrawingTool,
  strokeWidth: z.number().min(1),
  points: zPoints,
});

const zOLD_VectorMaskRect = z.object({
  id: z.string(),
  type: z.literal('vector_mask_rect'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});

const zRgbColor = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
});
const zRgbaColor = zRgbColor.extend({
  a: z.number().min(0).max(1),
});
export type RgbaColor = z.infer<typeof zRgbaColor>;
export const DEFAULT_RGBA_COLOR: RgbaColor = { r: 255, g: 255, b: 255, a: 1 };

const zOpacity = z.number().gte(0).lte(1);

const zBrushLine = z.object({
  id: z.string(),
  type: z.literal('brush_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
  color: zRgbaColor,
});
export type BrushLine = z.infer<typeof zBrushLine>;

const zEraserline = z.object({
  id: z.string(),
  type: z.literal('eraser_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
});
export type EraserLine = z.infer<typeof zEraserline>;

const zRectShape = z.object({
  id: z.string(),
  type: z.literal('rect_shape'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
  color: zRgbaColor,
});
export type RectShape = z.infer<typeof zRectShape>;

const zEllipseShape = z.object({
  id: z.string(),
  type: z.literal('ellipse_shape'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
  color: zRgbaColor,
});
export type EllipseShape = z.infer<typeof zEllipseShape>;

const zPolygonShape = z.object({
  id: z.string(),
  type: z.literal('polygon_shape'),
  points: zPoints,
  color: zRgbaColor,
});
export type PolygonShape = z.infer<typeof zPolygonShape>;

const zImageObject = z.object({
  id: z.string(),
  type: z.literal('image'),
  image: zImageWithDims,
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});
export type ImageObject = z.infer<typeof zImageObject>;

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

const zRasterLayer = zRenderableLayerBase.extend({
  type: z.literal('raster_layer'),
  opacity: zOpacity,
  objects: z.array(
    z.discriminatedUnion('type', [zImageObject, zBrushLine, zEraserline, zRectShape, zEllipseShape, zPolygonShape])
  ),
});
export type RasterLayer = z.infer<typeof zRasterLayer>;

const zControlAdapterLayer = zRenderableLayerBase.extend({
  type: z.literal('control_adapter_layer'),
  opacity: zOpacity,
  isFilterEnabled: z.boolean(),
  controlAdapter: z.discriminatedUnion('type', [zControlNetConfigV2, zT2IAdapterConfigV2]),
});
export type ControlAdapterLayer = z.infer<typeof zControlAdapterLayer>;

const zIPAdapterLayer = zLayerBase.extend({
  type: z.literal('ip_adapter_layer'),
  ipAdapter: zIPAdapterConfigV2,
});
export type IPAdapterLayer = z.infer<typeof zIPAdapterLayer>;

const zMaskObject = z
  .discriminatedUnion('type', [zOLD_VectorMaskLine, zOLD_VectorMaskRect, zBrushLine, zEraserline, zRectShape])
  .transform((val) => {
    // Migrate old vector mask objects to new format
    if (val.type === 'vector_mask_line') {
      const { tool, ...rest } = val;
      if (tool === 'brush') {
        const asBrushline: BrushLine = {
          ...rest,
          type: 'brush_line',
          color: { r: 255, g: 255, b: 255, a: 1 },
        };
        return asBrushline;
      } else if (tool === 'eraser') {
        const asEraserLine: EraserLine = {
          ...rest,
          type: 'eraser_line',
        };
        return asEraserLine;
      }
    } else if (val.type === 'vector_mask_rect') {
      const asRectShape: RectShape = {
        ...val,
        type: 'rect_shape',
        color: { r: 255, g: 255, b: 255, a: 1 },
      };
      return asRectShape;
    } else {
      return val;
    }
  })
  .pipe(z.discriminatedUnion('type', [zBrushLine, zEraserline, zRectShape]));

const zOLD_RegionalGuidanceLayer = zRenderableLayerBase.extend({
  type: z.literal('regional_guidance_layer'),
  maskObjects: z.array(zMaskObject),
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  ipAdapters: z.array(zIPAdapterConfigV2),
  previewColor: zRgbColor,
  autoNegative: zAutoNegative,
  uploadedMaskImage: zImageWithDims.nullable(),
});
const zRegionalGuidanceLayer = zRenderableLayerBase.extend({
  type: z.literal('regional_guidance_layer'),
  objects: z.array(zMaskObject),
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  ipAdapters: z.array(zIPAdapterConfigV2),
  previewColor: zRgbColor,
  autoNegative: zAutoNegative,
  uploadedMaskImage: zImageWithDims.nullable(),
});
const zRGLayer = z
  .union([zOLD_RegionalGuidanceLayer, zRegionalGuidanceLayer])
  .transform((val) => {
    if ('maskObjects' in val) {
      const { maskObjects, ...rest } = val;
      return { ...rest, objects: maskObjects };
    } else {
      return val;
    }
  })
  .pipe(zRegionalGuidanceLayer);
export type RegionalGuidanceLayer = z.infer<typeof zRGLayer>;

const zInitialImageLayer = zRenderableLayerBase.extend({
  type: z.literal('initial_image_layer'),
  opacity: zOpacity,
  image: zImageWithDims.nullable(),
  denoisingStrength: zParameterStrength,
});
export type InitialImageLayer = z.infer<typeof zInitialImageLayer>;

export const zLayer = z.discriminatedUnion('type', [
  zRegionalGuidanceLayer,
  zControlAdapterLayer,
  zIPAdapterLayer,
  zInitialImageLayer,
  zRasterLayer,
]);
export type Layer = z.infer<typeof zLayer>;

export type ControlLayersState = {
  _version: 3;
  selectedLayerId: string | null;
  layers: Layer[];
  brushSize: number;
  brushColor: RgbaColor;
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

export type AddEraserLineArg = { layerId: string; points: [number, number, number, number] };
export type AddBrushLineArg = AddEraserLineArg & { color: RgbaColor };
export type AddPointToLineArg = { layerId: string; point: [number, number] };
export type AddRectShapeArg = { layerId: string; rect: IRect; color: RgbaColor };
