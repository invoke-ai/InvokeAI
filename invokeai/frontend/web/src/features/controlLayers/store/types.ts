import {
  zBeginEndStepPct,
  zCLIPVisionModelV2,
  zControlModeV2,
  zId,
  zImageWithDims,
  zIPMethodV2,
  zProcessorConfig,
} from 'features/controlLayers/util/controlAdapters';
import { zModelIdentifierField } from 'features/nodes/types/common';
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
} from 'features/parameters/types/parameterSchemas';
import type { IRect } from 'konva/lib/types';
import type { ImageDTO } from 'services/api/types';
import { z } from 'zod';

const zTool = z.enum(['brush', 'eraser', 'move', 'rect', 'view', 'bbox']);
export type Tool = z.infer<typeof zTool>;
const zDrawingTool = zTool.extract(['brush', 'eraser']);

const zPoints = z.array(z.number()).refine((points) => points.length % 2 === 0, {
  message: 'Must have an even number of points',
});
const zOLD_VectorMaskLine = z.object({
  id: zId,
  type: z.literal('vector_mask_line'),
  tool: zDrawingTool,
  strokeWidth: z.number().min(1),
  points: zPoints,
});

const zOLD_VectorMaskRect = z.object({
  id: zId,
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
export type RgbColor = z.infer<typeof zRgbColor>;
const zRgbaColor = zRgbColor.extend({
  a: z.number().min(0).max(1),
});
export type RgbaColor = z.infer<typeof zRgbaColor>;
export const DEFAULT_RGBA_COLOR: RgbaColor = { r: 255, g: 255, b: 255, a: 1 };

const zOpacity = z.number().gte(0).lte(1);

const zBrushLine = z.object({
  id: zId,
  type: z.literal('brush_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
  color: zRgbaColor,
});
export type BrushLine = z.infer<typeof zBrushLine>;

const zEraserline = z.object({
  id: zId,
  type: z.literal('eraser_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
});
export type EraserLine = z.infer<typeof zEraserline>;

const zRectShape = z.object({
  id: zId,
  type: z.literal('rect_shape'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
  color: zRgbaColor,
});
export type RectShape = z.infer<typeof zRectShape>;

const zEllipseShape = z.object({
  id: zId,
  type: z.literal('ellipse_shape'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
  color: zRgbaColor,
});
export type EllipseShape = z.infer<typeof zEllipseShape>;

const zPolygonShape = z.object({
  id: zId,
  type: z.literal('polygon_shape'),
  points: zPoints,
  color: zRgbaColor,
});
export type PolygonShape = z.infer<typeof zPolygonShape>;

const zImageObject = z.object({
  id: zId,
  type: z.literal('image'),
  image: zImageWithDims,
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});
export type ImageObject = z.infer<typeof zImageObject>;

const zLayerObject = z.discriminatedUnion('type', [
  zImageObject,
  zBrushLine,
  zEraserline,
  zRectShape,
  zEllipseShape,
  zPolygonShape,
]);
export type LayerObject = z.infer<typeof zLayerObject>;

const zRect = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});

const zLayerData = z.object({
  id: zId,
  type: z.literal('layer'),
  isEnabled: z.boolean(),
  x: z.number(),
  y: z.number(),
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  opacity: zOpacity,
  objects: z.array(zLayerObject),
});
export type LayerData = z.infer<typeof zLayerData>;

const zIPAdapterData = z.object({
  id: zId,
  type: z.literal('ip_adapter'),
  isEnabled: z.boolean(),
  weight: z.number().gte(-1).lte(2),
  method: zIPMethodV2,
  image: zImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  clipVisionModel: zCLIPVisionModelV2,
  beginEndStepPct: zBeginEndStepPct,
});
export type IPAdapterData = z.infer<typeof zIPAdapterData>;
export type IPAdapterConfig = Pick<
  IPAdapterData,
  'weight' | 'image' | 'beginEndStepPct' | 'model' | 'clipVisionModel' | 'method'
>;

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

const zRegionalGuidanceData = z.object({
  id: zId,
  type: z.literal('regional_guidance'),
  isEnabled: z.boolean(),
  x: z.number(),
  y: z.number(),
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  objects: z.array(zMaskObject),
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  ipAdapters: z.array(zIPAdapterData),
  fill: zRgbColor,
  autoNegative: zAutoNegative,
  imageCache: zImageWithDims.nullable(),
});
export type RegionalGuidanceData = z.infer<typeof zRegionalGuidanceData>;

const zColorFill = z.object({
  type: z.literal('color_fill'),
  color: zRgbaColor,
});
const zImageFill = z.object({
  type: z.literal('image_fill'),
  src: z.string(),
});
const zFill = z.discriminatedUnion('type', [zColorFill, zImageFill]);
const zInpaintMaskData = z.object({
  id: zId,
  type: z.literal('inpaint_mask'),
  isEnabled: z.boolean(),
  x: z.number(),
  y: z.number(),
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  maskObjects: z.array(zMaskObject),
  fill: zFill,
  imageCache: zImageWithDims.nullable(),
});
export type InpaintMaskData = z.infer<typeof zInpaintMaskData>;

const zFilter = z.enum(['none', 'lightness_to_alpha']);
export type Filter = z.infer<typeof zFilter>;

const zControlAdapterData = z.object({
  id: zId,
  type: z.literal('control_adapter'),
  isEnabled: z.boolean(),
  x: z.number(),
  y: z.number(),
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  opacity: zOpacity,
  filter: zFilter,
  weight: z.number().gte(-1).lte(2),
  image: zImageWithDims.nullable(),
  processedImage: zImageWithDims.nullable(),
  processorConfig: zProcessorConfig.nullable(),
  processorPendingBatchId: z.string().nullable().default(null),
  beginEndStepPct: zBeginEndStepPct,
  model: zModelIdentifierField.nullable(),
  controlMode: zControlModeV2.nullable(),
});
export type ControlAdapterData = z.infer<typeof zControlAdapterData>;
export type ControlAdapterConfig = Pick<
  ControlAdapterData,
  'weight' | 'image' | 'processedImage' | 'processorConfig' | 'beginEndStepPct' | 'model' | 'controlMode'
>;

const zCanvasItemIdentifier = z.object({
  type: z.enum([
    zLayerData.shape.type.value,
    zIPAdapterData.shape.type.value,
    zControlAdapterData.shape.type.value,
    zRegionalGuidanceData.shape.type.value,
    zInpaintMaskData.shape.type.value,
  ]),
  id: zId,
});
type CanvasItemIdentifier = z.infer<typeof zCanvasItemIdentifier>;

export type CanvasV2State = {
  _version: 3;
  lastSelectedItem: CanvasItemIdentifier | null;
  prompts: {
    positivePrompt: ParameterPositivePrompt;
    negativePrompt: ParameterNegativePrompt;
    positivePrompt2: ParameterPositiveStylePromptSDXL;
    negativePrompt2: ParameterNegativeStylePromptSDXL;
    shouldConcatPrompts: boolean;
  };
  tool: {
    selected: Tool;
    selectedBuffer: Tool | null;
    invertScroll: boolean;
    brush: {
      width: number;
    };
    eraser: {
      width: number;
    };
    fill: RgbaColor;
  };
  size: {
    width: ParameterWidth;
    height: ParameterHeight;
    aspectRatio: AspectRatioState;
  };
  bbox: IRect;
};

export type StageAttrs = { x: number; y: number; width: number; height: number; scale: number };
export type AddEraserLineArg = { id: string; points: [number, number, number, number]; width: number };
export type AddBrushLineArg = AddEraserLineArg & { color: RgbaColor };
export type AddPointToLineArg = { id: string; point: [number, number] };
export type AddRectShapeArg = { id: string; rect: IRect; color: RgbaColor };
export type AddImageObjectArg = { id: string; imageDTO: ImageDTO };

//#region Type guards
export const isLine = (obj: LayerObject): obj is BrushLine | EraserLine => {
  return obj.type === 'brush_line' || obj.type === 'eraser_line';
};
