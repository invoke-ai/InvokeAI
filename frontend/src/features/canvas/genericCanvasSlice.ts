import {
  createSlice,
  SliceCaseReducers,
  ValidateSliceCaseReducers,
} from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { Vector2d } from 'konva/lib/types';
import { RgbaColor } from 'react-colorful';
import * as InvokeAI from 'app/invokeai';
import _ from 'lodash';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';

export type InpaintingTool = 'brush' | 'eraser';

export type MaskLine = {
  tool: InpaintingTool;
  strokeWidth: number;
  points: number[];
};

export type MaskCircle = {
  tool: InpaintingTool;
  radius: number;
  x: number;
  y: number;
};

export type Dimensions = {
  width: number;
  height: number;
};

export type BoundingBoxPreviewType = 'overlay' | 'ants' | 'marchingAnts';

export interface GenericCanvasState {
  tool: 'brush' | 'eraser';
  brushSize: number;
  maskColor: RgbaColor;
  cursorPosition: Vector2d | null;
  stageDimensions: Dimensions;
  stageCoordinates: Vector2d;
  boundingBoxDimensions: Dimensions;
  boundingBoxCoordinates: Vector2d;
  boundingBoxPreviewFill: RgbaColor;
  shouldShowBoundingBox: boolean;
  shouldShowBoundingBoxFill: boolean;
  lines: MaskLine[];
  pastLines: MaskLine[][];
  futureLines: MaskLine[][];
  shouldShowMask: boolean;
  shouldInvertMask: boolean;
  shouldShowCheckboardTransparency: boolean;
  shouldShowBrush: boolean;
  shouldShowBrushPreview: boolean;
  imageToInpaint?: InvokeAI.Image;
  doesCanvasNeedScaling: boolean;
  stageScale: number;
  isDrawing: boolean;
  isTransformingBoundingBox: boolean;
  isMouseOverBoundingBox: boolean;
  isMovingBoundingBox: boolean;
  shouldUseInpaintReplace: boolean;
  inpaintReplace: number;
  shouldLockBoundingBox: boolean;
  isMoveBoundingBoxKeyHeld: boolean;
  isMoveStageKeyHeld: boolean;
}

export const createGenericCanvasSlice = <
  Reducers extends SliceCaseReducers<GenericCanvasState>
>({
  name = '',
  initialState,
  reducers,
}: {
  name: string;
  initialState: GenericCanvasState;
  reducers: ValidateSliceCaseReducers<GenericCanvasState, Reducers>;
}) => {
  return createSlice({
    name,
    initialState,
    reducers: {
      setTool: (state, action: PayloadAction<InpaintingTool>) => {
        state.tool = action.payload;
      },
      toggleTool: (state) => {
        state.tool = state.tool === 'brush' ? 'eraser' : 'brush';
      },
      setBrushSize: (state, action: PayloadAction<number>) => {
        state.brushSize = action.payload;
      },
      addLine: (state, action: PayloadAction<MaskLine>) => {
        state.pastLines.push(state.lines);
        state.lines.push(action.payload);
        state.futureLines = [];
      },
      addPointToCurrentLine: (state, action: PayloadAction<number[]>) => {
        state.lines[state.lines.length - 1].points.push(...action.payload);
      },
      undo: (state) => {
        if (state.pastLines.length === 0) return;
        const newLines = state.pastLines.pop();
        if (!newLines) return;
        state.futureLines.unshift(state.lines);
        state.lines = newLines;
      },
      redo: (state) => {
        if (state.futureLines.length === 0) return;
        const newLines = state.futureLines.shift();
        if (!newLines) return;
        state.pastLines.push(state.lines);
        state.lines = newLines;
      },
      clearMask: (state) => {
        state.pastLines.push(state.lines);
        state.lines = [];
        state.futureLines = [];
        state.shouldInvertMask = false;
      },
      toggleShouldInvertMask: (state) => {
        state.shouldInvertMask = !state.shouldInvertMask;
      },
      toggleShouldShowMask: (state) => {
        state.shouldShowMask = !state.shouldShowMask;
      },
      setShouldInvertMask: (state, action: PayloadAction<boolean>) => {
        state.shouldInvertMask = action.payload;
      },
      setShouldShowMask: (state, action: PayloadAction<boolean>) => {
        state.shouldShowMask = action.payload;
        if (!action.payload) {
          state.shouldInvertMask = false;
        }
      },
      setShouldShowCheckboardTransparency: (
        state,
        action: PayloadAction<boolean>
      ) => {
        state.shouldShowCheckboardTransparency = action.payload;
      },
      setShouldShowBrushPreview: (state, action: PayloadAction<boolean>) => {
        state.shouldShowBrushPreview = action.payload;
      },
      setShouldShowBrush: (state, action: PayloadAction<boolean>) => {
        state.shouldShowBrush = action.payload;
      },
      setMaskColor: (state, action: PayloadAction<RgbaColor>) => {
        state.maskColor = action.payload;
      },
      setCursorPosition: (state, action: PayloadAction<Vector2d | null>) => {
        state.cursorPosition = action.payload;
      },
      clearImageToInpaint: (state) => {
        state.imageToInpaint = undefined;
      },
      setImageToInpaint: (state, action: PayloadAction<InvokeAI.Image>) => {
        const { width: canvasWidth, height: canvasHeight } =
          state.stageDimensions;
        const { width, height } = state.boundingBoxDimensions;
        const { x, y } = state.boundingBoxCoordinates;

        const newCoordinates: Vector2d = { x, y };
        const newDimensions: Dimensions = { width, height };

        if (width + x > canvasWidth) {
          // Bounding box at least needs to be translated
          if (width > canvasWidth) {
            // Bounding box also needs to be resized
            newDimensions.width = roundDownToMultiple(canvasWidth, 64);
          }
          newCoordinates.x = canvasWidth - newDimensions.width;
        }

        if (height + y > canvasHeight) {
          // Bounding box at least needs to be translated
          if (height > canvasHeight) {
            // Bounding box also needs to be resized
            newDimensions.height = roundDownToMultiple(canvasHeight, 64);
          }
          newCoordinates.y = canvasHeight - newDimensions.height;
        }

        state.boundingBoxDimensions = newDimensions;
        state.boundingBoxCoordinates = newCoordinates;

        state.imageToInpaint = action.payload;
        state.doesCanvasNeedScaling = true;
      },
      setStageDimensions: (state, action: PayloadAction<Dimensions>) => {
        state.stageDimensions = action.payload;

        const { width: canvasWidth, height: canvasHeight } = action.payload;

        const { width: boundingBoxWidth, height: boundingBoxHeight } =
          state.boundingBoxDimensions;

        const newBoundingBoxWidth = roundDownToMultiple(
          _.clamp(boundingBoxWidth, 64, canvasWidth / state.stageScale),
          64
        );
        const newBoundingBoxHeight = roundDownToMultiple(
          _.clamp(boundingBoxHeight, 64, canvasHeight / state.stageScale),
          64
        );

        state.boundingBoxDimensions = {
          width: newBoundingBoxWidth,
          height: newBoundingBoxHeight,
        };
      },
      setBoundingBoxDimensions: (state, action: PayloadAction<Dimensions>) => {
        state.boundingBoxDimensions = action.payload;
        const { width: boundingBoxWidth, height: boundingBoxHeight } =
          action.payload;
        const { x: boundingBoxX, y: boundingBoxY } =
          state.boundingBoxCoordinates;
        const { width: canvasWidth, height: canvasHeight } =
          state.stageDimensions;

        const scaledCanvasWidth = canvasWidth / state.stageScale;
        const scaledCanvasHeight = canvasHeight / state.stageScale;

        const roundedCanvasWidth = roundDownToMultiple(scaledCanvasWidth, 64);
        const roundedCanvasHeight = roundDownToMultiple(scaledCanvasHeight, 64);

        const roundedBoundingBoxWidth = roundDownToMultiple(
          boundingBoxWidth,
          64
        );
        const roundedBoundingBoxHeight = roundDownToMultiple(
          boundingBoxHeight,
          64
        );

        const overflowX = boundingBoxX + boundingBoxWidth - scaledCanvasWidth;
        const overflowY = boundingBoxY + boundingBoxHeight - scaledCanvasHeight;

        const newBoundingBoxWidth = _.clamp(
          roundedBoundingBoxWidth,
          64,
          roundedCanvasWidth
        );

        const newBoundingBoxHeight = _.clamp(
          roundedBoundingBoxHeight,
          64,
          roundedCanvasHeight
        );

        const overflowCorrectedX =
          overflowX > 0 ? boundingBoxX - overflowX : boundingBoxX;

        const overflowCorrectedY =
          overflowY > 0 ? boundingBoxY - overflowY : boundingBoxY;

        const clampedX = _.clamp(
          overflowCorrectedX,
          state.stageCoordinates.x,
          roundedCanvasWidth - newBoundingBoxWidth
        );

        const clampedY = _.clamp(
          overflowCorrectedY,
          state.stageCoordinates.y,
          roundedCanvasHeight - newBoundingBoxHeight
        );

        state.boundingBoxDimensions = {
          width: newBoundingBoxWidth,
          height: newBoundingBoxHeight,
        };

        state.boundingBoxCoordinates = {
          x: clampedX,
          y: clampedY,
        };
      },
      setBoundingBoxCoordinates: (state, action: PayloadAction<Vector2d>) => {
        state.boundingBoxCoordinates = action.payload;
      },
      setStageCoordinates: (state, action: PayloadAction<Vector2d>) => {
        state.stageCoordinates = action.payload;
      },
      setBoundingBoxPreviewFill: (state, action: PayloadAction<RgbaColor>) => {
        state.boundingBoxPreviewFill = action.payload;
      },
      setDoesCanvasNeedScaling: (state, action: PayloadAction<boolean>) => {
        state.doesCanvasNeedScaling = action.payload;
      },
      setStageScale: (state, action: PayloadAction<number>) => {
        state.stageScale = action.payload;
        state.doesCanvasNeedScaling = false;
      },
      setShouldShowBoundingBoxFill: (state, action: PayloadAction<boolean>) => {
        state.shouldShowBoundingBoxFill = action.payload;
      },
      setIsDrawing: (state, action: PayloadAction<boolean>) => {
        state.isDrawing = action.payload;
      },
      setClearBrushHistory: (state) => {
        state.pastLines = [];
        state.futureLines = [];
      },
      setShouldUseInpaintReplace: (state, action: PayloadAction<boolean>) => {
        state.shouldUseInpaintReplace = action.payload;
      },
      setInpaintReplace: (state, action: PayloadAction<number>) => {
        state.inpaintReplace = action.payload;
      },
      setShouldLockBoundingBox: (state, action: PayloadAction<boolean>) => {
        state.shouldLockBoundingBox = action.payload;
      },
      toggleShouldLockBoundingBox: (state) => {
        state.shouldLockBoundingBox = !state.shouldLockBoundingBox;
      },
      setShouldShowBoundingBox: (state, action: PayloadAction<boolean>) => {
        state.shouldShowBoundingBox = action.payload;
      },
      setIsTransformingBoundingBox: (state, action: PayloadAction<boolean>) => {
        state.isTransformingBoundingBox = action.payload;
      },
      setIsMovingBoundingBox: (state, action: PayloadAction<boolean>) => {
        state.isMovingBoundingBox = action.payload;
      },
      setIsMouseOverBoundingBox: (state, action: PayloadAction<boolean>) => {
        state.isMouseOverBoundingBox = action.payload;
      },
      setIsMoveBoundingBoxKeyHeld: (state, action: PayloadAction<boolean>) => {
        state.isMoveBoundingBoxKeyHeld = action.payload;
      },
      setIsMoveStageKeyHeld: (state, action: PayloadAction<boolean>) => {
        state.isMoveStageKeyHeld = action.payload;
      },
      ...reducers,
    },
  });
};
