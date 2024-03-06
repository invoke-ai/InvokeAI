import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import type { IRect, Vector2d } from 'konva/lib/types';
import type { RgbaColor } from 'react-colorful';
import { z } from 'zod';

export type CanvasLayer = 'base' | 'mask';

export const LAYER_NAMES_DICT: { label: string; value: CanvasLayer }[] = [
  { label: 'Base', value: 'base' },
  { label: 'Mask', value: 'mask' },
];

const zBoundingBoxScaleMethod = z.enum(['none', 'auto', 'manual']);
export type BoundingBoxScaleMethod = z.infer<typeof zBoundingBoxScaleMethod>;
export const isBoundingBoxScaleMethod = (v: unknown): v is BoundingBoxScaleMethod =>
  zBoundingBoxScaleMethod.safeParse(v).success;

type CanvasDrawingTool = 'brush' | 'eraser';

export type CanvasTool = CanvasDrawingTool | 'move' | 'colorPicker';

export type Dimensions = {
  width: number;
  height: number;
};

export type CanvasImage = {
  kind: 'image';
  layer: 'base';
  x: number;
  y: number;
  width: number;
  height: number;
  imageName: string;
};

export type CanvasMaskLine = {
  layer: 'mask';
  kind: 'line';
  tool: CanvasDrawingTool;
  strokeWidth: number;
  points: number[];
  clip?: IRect;
};

export type CanvasBaseLine = {
  layer: 'base';
  color?: RgbaColor;
  kind: 'line';
  tool: CanvasDrawingTool;
  strokeWidth: number;
  points: number[];
  clip?: IRect;
};

type CanvasFillRect = {
  kind: 'fillRect';
  layer: 'base';
  x: number;
  y: number;
  width: number;
  height: number;
  color: RgbaColor;
};

type CanvasEraseRect = {
  kind: 'eraseRect';
  layer: 'base';
  x: number;
  y: number;
  width: number;
  height: number;
};

type CanvasObject = CanvasImage | CanvasBaseLine | CanvasMaskLine | CanvasFillRect | CanvasEraseRect;

export type CanvasLayerState = {
  objects: CanvasObject[];
  stagingArea: {
    images: CanvasImage[];
    selectedImageIndex: number;
    boundingBox?: IRect;
  };
};

// type guards
export const isCanvasMaskLine = (obj: CanvasObject): obj is CanvasMaskLine =>
  obj.kind === 'line' && obj.layer === 'mask';

export const isCanvasBaseLine = (obj: CanvasObject): obj is CanvasBaseLine =>
  obj.kind === 'line' && obj.layer === 'base';

export const isCanvasBaseImage = (obj: CanvasObject): obj is CanvasImage =>
  obj.kind === 'image' && obj.layer === 'base';

export const isCanvasFillRect = (obj: CanvasObject): obj is CanvasFillRect =>
  obj.kind === 'fillRect' && obj.layer === 'base';

export const isCanvasEraseRect = (obj: CanvasObject): obj is CanvasEraseRect =>
  obj.kind === 'eraseRect' && obj.layer === 'base';

export const isCanvasAnyLine = (obj: CanvasObject): obj is CanvasMaskLine | CanvasBaseLine => obj.kind === 'line';

export interface CanvasState {
  _version: 1;
  boundingBoxCoordinates: Vector2d;
  boundingBoxDimensions: Dimensions;
  boundingBoxScaleMethod: BoundingBoxScaleMethod;
  brushColor: RgbaColor;
  brushSize: number;
  colorPickerColor: RgbaColor;
  futureLayerStates: CanvasLayerState[];
  isMaskEnabled: boolean;
  layer: CanvasLayer;
  layerState: CanvasLayerState;
  maskColor: RgbaColor;
  pastLayerStates: CanvasLayerState[];
  scaledBoundingBoxDimensions: Dimensions;
  shouldAntialias: boolean;
  shouldAutoSave: boolean;
  shouldCropToBoundingBoxOnSave: boolean;
  shouldDarkenOutsideBoundingBox: boolean;
  shouldInvertBrushSizeScrollDirection: boolean;
  shouldLockBoundingBox: boolean;
  shouldPreserveMaskedArea: boolean;
  shouldRestrictStrokesToBox: boolean;
  shouldShowBoundingBox: boolean;
  shouldShowCanvasDebugInfo: boolean;
  shouldShowGrid: boolean;
  shouldShowIntermediates: boolean;
  shouldShowStagingImage: boolean;
  shouldShowStagingOutline: boolean;
  shouldSnapToGrid: boolean;
  stageCoordinates: Vector2d;
  stageDimensions: Dimensions;
  stageScale: number;
  generationMode?: GenerationMode;
  batchIds: string[];
  aspectRatio: AspectRatioState;
}

export type GenerationMode = 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';
