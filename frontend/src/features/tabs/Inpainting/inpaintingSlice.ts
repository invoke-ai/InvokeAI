import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { Vector2d } from 'konva/lib/types';
import { RgbaColor } from 'react-colorful';
import * as InvokeAI from '../../../app/invokeai';
import _ from 'lodash';
import { roundDownToMultiple } from '../../../common/util/roundDownToMultiple';

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

export interface InpaintingState {
  tool: 'brush' | 'eraser';
  brushSize: number;
  maskColor: RgbaColor;
  cursorPosition: Vector2d | null;
  canvasDimensions: Dimensions;
  boundingBoxDimensions: Dimensions;
  boundingBoxCoordinate: Vector2d;
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
  needsRepaint: boolean;
  stageScale: number;
  isDrawing: boolean;
  shouldUseInpaintReplace: boolean;
  inpaintReplace: number;
  shouldLockBoundingBox: boolean;
}

const initialInpaintingState: InpaintingState = {
  tool: 'brush',
  brushSize: 50,
  maskColor: { r: 255, g: 90, b: 90, a: 0.5 },
  canvasDimensions: { width: 0, height: 0 },
  boundingBoxDimensions: { width: 64, height: 64 },
  boundingBoxCoordinate: { x: 0, y: 0 },
  boundingBoxPreviewFill: { r: 0, g: 0, b: 0, a: 0.7 },
  shouldShowBoundingBox: false,
  shouldShowBoundingBoxFill: false,
  cursorPosition: null,
  lines: [],
  pastLines: [],
  futureLines: [],
  shouldShowMask: true,
  shouldInvertMask: false,
  shouldShowCheckboardTransparency: false,
  shouldShowBrush: true,
  shouldShowBrushPreview: false,
  needsRepaint: false,
  isDrawing: false,
  stageScale: 1,
  shouldUseInpaintReplace: false,
  inpaintReplace: 1,
  shouldLockBoundingBox: true,
};

const initialState: InpaintingState = initialInpaintingState;

export const inpaintingSlice = createSlice({
  name: 'inpainting',
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
      const { width: imageWidth, height: imageHeight } = action.payload;
      const { width: boundingBoxWidth, height: boundingBoxHeight } =
        state.boundingBoxDimensions;
      const { x, y } = state.boundingBoxCoordinate;

      const newBoundingBoxWidth = roundDownToMultiple(
        _.clamp(boundingBoxWidth, 64, imageWidth),
        64
      );

      const newBoundingBoxHeight = roundDownToMultiple(
        _.clamp(boundingBoxHeight, 64, imageHeight),
        64
      );

      const newBoundingBoxX = roundDownToMultiple(
        _.clamp(x, 0, imageWidth - newBoundingBoxWidth),
        64
      );

      const newBoundingBoxY = roundDownToMultiple(
        _.clamp(y, 0, imageHeight - newBoundingBoxHeight),
        64
      );

      state.boundingBoxDimensions = {
        width: newBoundingBoxWidth,
        height: newBoundingBoxHeight,
      };

      state.boundingBoxCoordinate = { x: newBoundingBoxX, y: newBoundingBoxY };

      state.canvasDimensions = {
        width: imageWidth,
        height: imageHeight,
      };

      state.imageToInpaint = action.payload;
      state.needsRepaint = true;
    },
    setCanvasDimensions: (state, action: PayloadAction<Dimensions>) => {
      state.canvasDimensions = action.payload;

      const { width: canvasWidth, height: canvasHeight } = action.payload;

      const { width: boundingBoxWidth, height: boundingBoxHeight } =
        state.boundingBoxDimensions;

      const newBoundingBoxWidth = roundDownToMultiple(
        _.clamp(boundingBoxWidth, 64, canvasWidth),
        64
      );
      const newBoundingBoxHeight = roundDownToMultiple(
        _.clamp(boundingBoxHeight, 64, canvasHeight),
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
      const { x: boundingBoxX, y: boundingBoxY } = state.boundingBoxCoordinate;
      const { width: canvasWidth, height: canvasHeight } =
        state.canvasDimensions;

      const roundedCanvasWidth = roundDownToMultiple(canvasWidth, 64);
      const roundedCanvasHeight = roundDownToMultiple(canvasHeight, 64);

      const roundedBoundingBoxWidth = roundDownToMultiple(boundingBoxWidth, 64);
      const roundedBoundingBoxHeight = roundDownToMultiple(
        boundingBoxHeight,
        64
      );

      const overflowX = boundingBoxX + boundingBoxWidth - canvasWidth;
      const overflowY = boundingBoxY + boundingBoxHeight - canvasHeight;

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
        0,
        roundedCanvasWidth - newBoundingBoxWidth
      );

      const clampedY = _.clamp(
        overflowCorrectedY,
        0,
        roundedCanvasHeight - newBoundingBoxHeight
      );

      state.boundingBoxDimensions = {
        width: newBoundingBoxWidth,
        height: newBoundingBoxHeight,
      };

      state.boundingBoxCoordinate = {
        x: clampedX,
        y: clampedY,
      };
    },
    setBoundingBoxCoordinate: (state, action: PayloadAction<Vector2d>) => {
      state.boundingBoxCoordinate = action.payload;
    },
    setBoundingBoxPreviewFill: (state, action: PayloadAction<RgbaColor>) => {
      state.boundingBoxPreviewFill = action.payload;
    },
    setNeedsRepaint: (state, action: PayloadAction<boolean>) => {
      state.needsRepaint = action.payload;
    },
    setStageScale: (state, action: PayloadAction<number>) => {
      state.stageScale = action.payload;
      state.needsRepaint = false;
    },
    setShouldShowBoundingBoxFill: (state, action: PayloadAction<boolean>) => {
      state.shouldShowBoundingBoxFill = action.payload;
    },
    setIsDrawing: (state, action: PayloadAction<boolean>) => {
      state.isDrawing = action.payload;
    },
    setShouldShowBoundingBox: (state, action: PayloadAction<boolean>) => {
      state.shouldShowBoundingBox = action.payload;
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
  },
});

export const {
  setTool,
  setBrushSize,
  addLine,
  addPointToCurrentLine,
  setShouldInvertMask,
  setShouldShowMask,
  setShouldShowCheckboardTransparency,
  setShouldShowBrushPreview,
  setMaskColor,
  clearMask,
  undo,
  redo,
  setCursorPosition,
  setCanvasDimensions,
  setImageToInpaint,
  setBoundingBoxDimensions,
  setBoundingBoxCoordinate,
  setBoundingBoxPreviewFill,
  setNeedsRepaint,
  setStageScale,
  toggleTool,
  setShouldShowBoundingBoxFill,
  setIsDrawing,
  setShouldShowBrush,
  setShouldShowBoundingBox,
  setClearBrushHistory,
  setShouldUseInpaintReplace,
  setInpaintReplace,
  setShouldLockBoundingBox,
  toggleShouldLockBoundingBox,
} = inpaintingSlice.actions;

export default inpaintingSlice.reducer;
