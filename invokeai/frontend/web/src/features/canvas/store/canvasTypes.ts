import * as InvokeAI from 'app/types/invokeai';
import { IRect, Vector2d } from 'konva/lib/types';
import { RgbaColor } from 'react-colorful';
import { ImageDTO } from 'services/api';

export const LAYER_NAMES_DICT = [
  { key: 'Base', value: 'base' },
  { key: 'Mask', value: 'mask' },
];

export const LAYER_NAMES = ['base', 'mask'] as const;

export type CanvasLayer = (typeof LAYER_NAMES)[number];

export const BOUNDING_BOX_SCALES_DICT = [
  { key: 'Auto', value: 'auto' },
  { key: 'Manual', value: 'manual' },
  { key: 'None', value: 'none' },
];

export const BOUNDING_BOX_SCALES = ['none', 'auto', 'manual'] as const;

export type BoundingBoxScale = (typeof BOUNDING_BOX_SCALES)[number];

export type CanvasDrawingTool = 'brush' | 'eraser';

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
  image: ImageDTO;
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

export type CanvasFillRect = {
  kind: 'fillRect';
  layer: 'base';
  x: number;
  y: number;
  width: number;
  height: number;
  color: RgbaColor;
};

export type CanvasEraseRect = {
  kind: 'eraseRect';
  layer: 'base';
  x: number;
  y: number;
  width: number;
  height: number;
};

export type CanvasObject =
  | CanvasImage
  | CanvasBaseLine
  | CanvasMaskLine
  | CanvasFillRect
  | CanvasEraseRect;

export type CanvasLayerState = {
  objects: CanvasObject[];
  stagingArea: {
    images: CanvasImage[];
    selectedImageIndex: number;
    sessionId?: string;
    boundingBox?: IRect;
  };
};

export type CanvasSession = {
  sessionId: string;
  boundingBox: IRect;
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

export const isCanvasAnyLine = (
  obj: CanvasObject
): obj is CanvasMaskLine | CanvasBaseLine => obj.kind === 'line';

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
  shouldAntialias: boolean;
  shouldAutoSave: boolean;
  shouldCropToBoundingBoxOnSave: boolean;
  shouldDarkenOutsideBoundingBox: boolean;
  shouldLockBoundingBox: boolean;
  shouldPreserveMaskedArea: boolean;
  shouldRestrictStrokesToBox: boolean;
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
  stageCoordinates: Vector2d;
  stageDimensions: Dimensions;
  stageScale: number;
  tool: CanvasTool;
}
