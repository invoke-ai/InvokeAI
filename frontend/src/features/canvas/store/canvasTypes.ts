import * as InvokeAI from 'app/invokeai';
import { Vector2d } from 'konva/lib/types';
import { RgbaColor } from 'react-colorful';

export type CanvasLayer = 'base' | 'mask';

export type CanvasDrawingTool = 'brush' | 'eraser';

export type CanvasTool = CanvasDrawingTool | 'move';

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
  brushColor: RgbaColor;
  brushSize: number;
  canvasContainerDimensions: Dimensions;
  cursorPosition: Vector2d | null;
  doesCanvasNeedScaling: boolean;
  eraserSize: number;
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
  shouldAutoSave: boolean;
  shouldDarkenOutsideBoundingBox: boolean;
  shouldLockBoundingBox: boolean;
  shouldLockToInitialImage: boolean;
  shouldPreserveMaskedArea: boolean;
  shouldShowBoundingBox: boolean;
  shouldShowBrush: boolean;
  shouldShowBrushPreview: boolean;
  shouldShowCheckboardTransparency: boolean;
  shouldShowGrid: boolean;
  shouldShowIntermediates: boolean;
  shouldSnapToGrid: boolean;
  shouldUseInpaintReplace: boolean;
  stageCoordinates: Vector2d;
  stageDimensions: Dimensions;
  stageScale: number;
  tool: CanvasTool;
}
