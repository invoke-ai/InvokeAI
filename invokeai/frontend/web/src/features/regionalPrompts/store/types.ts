import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import type {
  ParameterAutoNegative,
  ParameterHeight,
  ParameterNegativePrompt,
  ParameterNegativeStylePromptSDXL,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';
import type { IRect } from 'konva/lib/types';
import type { RgbColor } from 'react-colorful';

export type DrawingTool = 'brush' | 'eraser';

export type Tool = DrawingTool | 'move' | 'rect';

export type VectorMaskLine = {
  id: string;
  type: 'vector_mask_line';
  tool: DrawingTool;
  strokeWidth: number;
  points: number[];
};

export type VectorMaskRect = {
  id: string;
  type: 'vector_mask_rect';
  x: number;
  y: number;
  width: number;
  height: number;
};

export type LayerBase = {
  id: string;
  isEnabled: boolean;
};

export type RenderableLayerBase = LayerBase & {
  x: number;
  y: number;
  bbox: IRect | null;
  bboxNeedsUpdate: boolean;
  isSelected: boolean;
};

export type ControlAdapterLayer = RenderableLayerBase & {
  type: 'control_adapter_layer'; // technically, also t2i adapter layer
  controlNetId: string;
  imageName: string | null;
  opacity: number;
};

export type IPAdapterLayer = LayerBase & {
  type: 'ip_adapter_layer'; // technically, also t2i adapter layer
  ipAdapterId: string;
};

export type MaskedGuidanceLayer = RenderableLayerBase & {
  type: 'masked_guidance_layer';
  maskObjects: (VectorMaskLine | VectorMaskRect)[];
  positivePrompt: ParameterPositivePrompt | null;
  negativePrompt: ParameterNegativePrompt | null; // Up to one text prompt per mask
  ipAdapterIds: string[]; // Any number of image prompts
  previewColor: RgbColor;
  autoNegative: ParameterAutoNegative;
  needsPixelBbox: boolean; // Needs the slower pixel-based bbox calculation - set to true when an there is an eraser object
};

export type Layer = MaskedGuidanceLayer | ControlAdapterLayer | IPAdapterLayer;

export type RegionalPromptsState = {
  _version: 1;
  selectedLayerId: string | null;
  layers: Layer[];
  brushSize: number;
  globalMaskLayerOpacity: number;
  isEnabled: boolean;
  positivePrompt: ParameterPositivePrompt;
  negativePrompt: ParameterNegativePrompt;
  positivePrompt2: ParameterPositiveStylePromptSDXL;
  negativePrompt2: ParameterNegativeStylePromptSDXL;
  shouldConcatPrompts: boolean;
  initialImage: string | null;
  size: {
    width: ParameterWidth;
    height: ParameterHeight;
    aspectRatio: AspectRatioState;
  };
};
