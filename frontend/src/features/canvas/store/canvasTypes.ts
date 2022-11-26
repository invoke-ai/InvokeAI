import * as InvokeAI from 'app/invokeai';
import { Vector2d } from 'konva/lib/types';
import { RgbaColor } from 'react-colorful';

export const LAYER_NAMES_DICT = [
  { key: 'Base', value: 'base' },
  { key: 'Mask', value: 'mask' },
];

export const LAYER_NAMES = ['base', 'mask'] as const;

export type CanvasLayer = typeof LAYER_NAMES[number];

export const BOUNDING_BOX_SCALES_DICT = [
  { key: 'Auto', value: 'auto' },
  { key: 'Manual', value: 'manual' },
  { key: 'None', value: 'none' },
];

export const BOUNDING_BOX_SCALES = ['none', 'auto', 'manual'] as const;

export type BoundingBoxScale = typeof BOUNDING_BOX_SCALES[number];

export type CanvasDrawingTool = 'brush' | 'eraser';

export type CanvasTool = CanvasDrawingTool | 'move' | 'colorPicker';

export type Dimensions = {
  width: number;
  height: number;
};

export type CanvasAnyLine = {
  kind: 'line';
  tool: CanvasDrawingTool;
  strokeWidth: number;
  points: number[];
};

export type CanvasImage = {
  kind: 'image';
  layer: 'base';
  x: number;
  y: number;
  width: number;
  height: number;
  image: InvokeAI.Image;
};

export type CanvasMaskLine = CanvasAnyLine & {
  layer: 'mask';
};

export type CanvasLine = CanvasAnyLine & {
  layer: 'base';
  color?: RgbaColor;
};

export type CanvasObject = CanvasImage | CanvasLine | CanvasMaskLine;

export type CanvasLayerState = {
  objects: CanvasObject[];
  stagingArea: {
    x: number;
    y: number;
    width: number;
    height: number;
    images: CanvasImage[];
    selectedImageIndex: number;
  };
};

// type guards
export const isCanvasMaskLine = (obj: CanvasObject): obj is CanvasMaskLine =>
  obj.kind === 'line' && obj.layer === 'mask';

export const isCanvasBaseLine = (obj: CanvasObject): obj is CanvasLine =>
  obj.kind === 'line' && obj.layer === 'base';

export const isCanvasBaseImage = (obj: CanvasObject): obj is CanvasImage =>
  obj.kind === 'image' && obj.layer === 'base';

export const isCanvasAnyLine = (
  obj: CanvasObject
): obj is CanvasMaskLine | CanvasLine => obj.kind === 'line';

export interface CanvasState {
  boundingBoxCoordinates: Vector2d;
  boundingBoxDimensions: Dimensions;
  boundingBoxPreviewFill: RgbaColor;
  boundingBoxScaleMethod: BoundingBoxScale;
  brushColor: RgbaColor;
  brushSize: number;
  canvasContainerDimensions: Dimensions;
  colorPickerColor: RgbaColor;
  cursorPosition: Vector2d | null;
  doesCanvasNeedScaling: boolean;
  futureLayerStates: CanvasLayerState[];
  inpaintReplace: number;
  intermediateImage?: InvokeAI.Image;
  isCanvasInitialized: boolean;
  isDrawing: boolean;
  isMaskEnabled: boolean;
  isMouseOverBoundingBox: boolean;
  isMoveBoundingBoxKeyHeld: boolean;
  isMoveStageKeyHeld: boolean;
  isMovingBoundingBox: boolean;
  isMovingStage: boolean;
  isTransformingBoundingBox: boolean;
  layer: CanvasLayer;
  layerState: CanvasLayerState;
  maskColor: RgbaColor;
  maxHistory: number;
  minimumStageScale: number;
  pastLayerStates: CanvasLayerState[];
  scaledBoundingBoxDimensions: Dimensions;
  shouldAutoSave: boolean;
  shouldCropToBoundingBoxOnSave: boolean;
  shouldDarkenOutsideBoundingBox: boolean;
  shouldLockBoundingBox: boolean;
  shouldPreserveMaskedArea: boolean;
  shouldShowBoundingBox: boolean;
  shouldShowBrush: boolean;
  shouldShowBrushPreview: boolean;
  shouldShowCanvasDebugInfo: boolean;
  shouldShowCheckboardTransparency: boolean;
  shouldShowGrid: boolean;
  shouldShowIntermediates: boolean;
  shouldShowStagingImage: boolean;
  shouldShowStagingOutline: boolean;
  shouldSnapToGrid: boolean;
  shouldUseInpaintReplace: boolean;
  stageCoordinates: Vector2d;
  stageDimensions: Dimensions;
  stageScale: number;
  tool: CanvasTool;
}
