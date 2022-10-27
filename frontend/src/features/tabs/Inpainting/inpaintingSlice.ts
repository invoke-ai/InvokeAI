import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { Vector2d } from 'konva/lib/types';
import { RgbaColor, RgbColor } from 'react-colorful';
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
  maskColor: RgbColor;
  maskOpacity: number;
  cursorPosition: Vector2d | null;
  canvasDimensions: Dimensions;
  boundingBoxDimensions: Dimensions;
  boundingBoxCoordinate: Vector2d;
  isMovingBoundingBox: boolean;
  boundingBoxPreviewFill: RgbaColor;
  boundingBoxPreviewType: BoundingBoxPreviewType;
  lines: MaskLine[];
  pastLines: MaskLine[][];
  futureLines: MaskLine[][];
  shouldShowMask: boolean;
  shouldInvertMask: boolean;
  shouldShowCheckboardTransparency: boolean;
  shouldShowBrushPreview: boolean;
  imageToInpaint?: InvokeAI.Image;
  needsRepaint: boolean;
  stageScale: number;
}

const initialInpaintingState: InpaintingState = {
  tool: 'brush',
  brushSize: 50,
  maskColor: { r: 255, g: 90, b: 90 },
  maskOpacity: 0.5,
  canvasDimensions: { width: 0, height: 0 },
  boundingBoxDimensions: { width: 64, height: 64 },
  boundingBoxCoordinate: { x: 0, y: 0 },
  boundingBoxPreviewFill: { r: 0, g: 0, b: 0, a: 0.5 },
  boundingBoxPreviewType: 'ants',
  cursorPosition: null,
  lines: [],
  pastLines: [],
  futureLines: [],
  shouldShowMask: true,
  shouldInvertMask: false,
  shouldShowCheckboardTransparency: false,
  shouldShowBrushPreview: false,
  isMovingBoundingBox: false,
  needsRepaint: false,
  stageScale: 1,
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
    setMaskColor: (state, action: PayloadAction<RgbColor>) => {
      state.maskColor = action.payload;
    },
    setMaskOpacity: (state, action: PayloadAction<number>) => {
      state.maskOpacity = action.payload;
    },
    setCursorPosition: (state, action: PayloadAction<Vector2d | null>) => {
      state.cursorPosition = action.payload;
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
      const { width: boundingBoxWidth, height: boundingBoxHeight } =
        action.payload;
      const { x: boundingBoxX, y: boundingBoxY } = state.boundingBoxCoordinate;
      const { width: canvasWidth, height: canvasHeight } =
        state.canvasDimensions;

      const overflowX = boundingBoxX + boundingBoxWidth - canvasWidth;
      const overflowY = boundingBoxY + boundingBoxHeight - canvasHeight;

      const newBoundingBoxX = roundDownToMultiple(
        overflowX > 0 ? boundingBoxX - overflowX : boundingBoxX,
        64
      );

      const newBoundingBoxY = roundDownToMultiple(
        overflowY > 0 ? boundingBoxY - overflowY : boundingBoxY,
        64
      );

      state.boundingBoxDimensions = {
        width: roundDownToMultiple(boundingBoxWidth, 64),
        height: roundDownToMultiple(boundingBoxHeight, 64),
      };

      state.boundingBoxCoordinate = {
        x: newBoundingBoxX,
        y: newBoundingBoxY,
      };
    },
    setBoundingBoxCoordinate: (state, action: PayloadAction<Vector2d>) => {
      state.boundingBoxCoordinate = action.payload;
    },
    setIsMovingBoundingBox: (state, action: PayloadAction<boolean>) => {
      state.isMovingBoundingBox = action.payload;
    },
    toggleIsMovingBoundingBox: (state) => {
      state.isMovingBoundingBox = !state.isMovingBoundingBox;
    },
    setBoundingBoxPreviewFill: (state, action: PayloadAction<RgbaColor>) => {
      state.boundingBoxPreviewFill = action.payload;
    },
    setBoundingBoxPreviewType: (
      state,
      action: PayloadAction<BoundingBoxPreviewType>
    ) => {
      state.boundingBoxPreviewType = action.payload;
    },
    setNeedsRepaint: (state, action: PayloadAction<boolean>) => {
      state.needsRepaint = action.payload;
    },
    setStageScale: (state, action: PayloadAction<number>) => {
      state.stageScale = action.payload;
      state.needsRepaint = false;
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
  setMaskOpacity,
  undo,
  redo,
  setCursorPosition,
  setCanvasDimensions,
  setImageToInpaint,
  setBoundingBoxDimensions,
  setBoundingBoxCoordinate,
  setIsMovingBoundingBox,
  setBoundingBoxPreviewFill,
  setBoundingBoxPreviewType,
  setNeedsRepaint,
  setStageScale,
  toggleTool,
  toggleIsMovingBoundingBox,
} = inpaintingSlice.actions;

export default inpaintingSlice.reducer;
