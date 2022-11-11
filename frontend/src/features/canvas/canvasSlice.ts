import {
  createAsyncThunk,
  createSelector,
  createSlice,
} from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import type { PayloadAction } from '@reduxjs/toolkit';
import { IRect, Vector2d } from 'konva/lib/types';
import { RgbaColor } from 'react-colorful';
import * as InvokeAI from 'app/invokeai';
import _ from 'lodash';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import { RootState } from 'app/store';
import { MutableRefObject } from 'react';
import Konva from 'konva';

export interface GenericCanvasState {
  tool: CanvasTool;
  brushSize: number;
  brushColor: RgbaColor;
  eraserSize: number;
  maskColor: RgbaColor;
  cursorPosition: Vector2d | null;
  stageDimensions: Dimensions;
  stageCoordinates: Vector2d;
  boundingBoxDimensions: Dimensions;
  boundingBoxCoordinates: Vector2d;
  boundingBoxPreviewFill: RgbaColor;
  shouldShowBoundingBox: boolean;
  shouldDarkenOutsideBoundingBox: boolean;
  isMaskEnabled: boolean;
  shouldInvertMask: boolean;
  shouldShowCheckboardTransparency: boolean;
  shouldShowBrush: boolean;
  shouldShowBrushPreview: boolean;
  stageScale: number;
  isDrawing: boolean;
  isTransformingBoundingBox: boolean;
  isMouseOverBoundingBox: boolean;
  isMovingBoundingBox: boolean;
  isMovingStage: boolean;
  shouldUseInpaintReplace: boolean;
  inpaintReplace: number;
  shouldLockBoundingBox: boolean;
  isMoveBoundingBoxKeyHeld: boolean;
  isMoveStageKeyHeld: boolean;
  intermediateImage?: InvokeAI.Image;
  shouldShowIntermediates: boolean;
}

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
  image: InvokeAI.Image;
};

export type CanvasMaskLine = CanvasAnyLine & {
  layer: 'mask';
};

export type CanvasLine = CanvasAnyLine & {
  layer: 'base';
  color?: RgbaColor;
};

type CanvasObject = CanvasImage | CanvasLine | CanvasMaskLine;

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

export type OutpaintingCanvasState = GenericCanvasState & {
  layer: CanvasLayer;
  objects: CanvasObject[];
  pastObjects: CanvasObject[][];
  futureObjects: CanvasObject[][];
  shouldShowGrid: boolean;
  shouldSnapToGrid: boolean;
  shouldAutoSave: boolean;
  stagingArea: {
    images: CanvasImage[];
    selectedImageIndex: number;
  };
};

export type InpaintingCanvasState = GenericCanvasState & {
  layer: 'mask';
  objects: CanvasObject[];
  pastObjects: CanvasObject[][];
  futureObjects: CanvasObject[][];
  imageToInpaint?: InvokeAI.Image;
};

export type BaseCanvasState = InpaintingCanvasState | OutpaintingCanvasState;

export type ValidCanvasName = 'inpainting' | 'outpainting';

export interface CanvasState {
  doesCanvasNeedScaling: boolean;
  currentCanvas: ValidCanvasName;
  inpainting: InpaintingCanvasState;
  outpainting: OutpaintingCanvasState;
}

const initialGenericCanvasState: GenericCanvasState = {
  tool: 'brush',
  brushColor: { r: 90, g: 90, b: 255, a: 1 },
  brushSize: 50,
  maskColor: { r: 255, g: 90, b: 90, a: 1 },
  eraserSize: 50,
  stageDimensions: { width: 0, height: 0 },
  stageCoordinates: { x: 0, y: 0 },
  boundingBoxDimensions: { width: 512, height: 512 },
  boundingBoxCoordinates: { x: 0, y: 0 },
  boundingBoxPreviewFill: { r: 0, g: 0, b: 0, a: 0.5 },
  shouldShowBoundingBox: true,
  shouldDarkenOutsideBoundingBox: false,
  cursorPosition: null,
  isMaskEnabled: true,
  shouldInvertMask: false,
  shouldShowCheckboardTransparency: false,
  shouldShowBrush: true,
  shouldShowBrushPreview: false,
  isDrawing: false,
  isTransformingBoundingBox: false,
  isMouseOverBoundingBox: false,
  isMovingBoundingBox: false,
  stageScale: 1,
  shouldUseInpaintReplace: false,
  inpaintReplace: 0.1,
  shouldLockBoundingBox: false,
  isMoveBoundingBoxKeyHeld: false,
  isMoveStageKeyHeld: false,
  shouldShowIntermediates: true,
  isMovingStage: false,
};

const initialCanvasState: CanvasState = {
  currentCanvas: 'inpainting',
  doesCanvasNeedScaling: false,
  inpainting: {
    layer: 'mask',
    objects: [],
    pastObjects: [],
    futureObjects: [],
    ...initialGenericCanvasState,
  },
  outpainting: {
    layer: 'base',
    objects: [],
    pastObjects: [],
    futureObjects: [],
    stagingArea: {
      images: [],
      selectedImageIndex: 0,
    },
    shouldShowGrid: true,
    shouldSnapToGrid: true,
    shouldAutoSave: false,
    ...initialGenericCanvasState,
  },
};

export const canvasSlice = createSlice({
  name: 'canvas',
  initialState: initialCanvasState,
  reducers: {
    setTool: (state, action: PayloadAction<CanvasTool>) => {
      const tool = action.payload;
      state[state.currentCanvas].tool = action.payload;
      if (tool !== 'move') {
        state[state.currentCanvas].isTransformingBoundingBox = false;
        state[state.currentCanvas].isMouseOverBoundingBox = false;
        state[state.currentCanvas].isMovingBoundingBox = false;
        state[state.currentCanvas].isMovingStage = false;
      }
    },
    setLayer: (state, action: PayloadAction<CanvasLayer>) => {
      state[state.currentCanvas].layer = action.payload;
    },
    toggleTool: (state) => {
      const currentTool = state[state.currentCanvas].tool;
      if (currentTool !== 'move') {
        state[state.currentCanvas].tool =
          currentTool === 'brush' ? 'eraser' : 'brush';
      }
    },
    setMaskColor: (state, action: PayloadAction<RgbaColor>) => {
      state[state.currentCanvas].maskColor = action.payload;
    },
    setBrushColor: (state, action: PayloadAction<RgbaColor>) => {
      state[state.currentCanvas].brushColor = action.payload;
    },
    setBrushSize: (state, action: PayloadAction<number>) => {
      state[state.currentCanvas].brushSize = action.payload;
    },
    setEraserSize: (state, action: PayloadAction<number>) => {
      state[state.currentCanvas].eraserSize = action.payload;
    },
    clearMask: (state) => {
      state[state.currentCanvas].pastObjects.push(
        state[state.currentCanvas].objects
      );
      state[state.currentCanvas].objects = state[
        state.currentCanvas
      ].objects.filter((obj) => !isCanvasMaskLine(obj));
      state[state.currentCanvas].futureObjects = [];
      state[state.currentCanvas].shouldInvertMask = false;
    },
    toggleShouldInvertMask: (state) => {
      state[state.currentCanvas].shouldInvertMask =
        !state[state.currentCanvas].shouldInvertMask;
    },
    toggleShouldShowMask: (state) => {
      state[state.currentCanvas].isMaskEnabled =
        !state[state.currentCanvas].isMaskEnabled;
    },
    setShouldInvertMask: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldInvertMask = action.payload;
    },
    setIsMaskEnabled: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isMaskEnabled = action.payload;
      state[state.currentCanvas].layer = action.payload ? 'mask' : 'base';
    },
    setShouldShowCheckboardTransparency: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state[state.currentCanvas].shouldShowCheckboardTransparency =
        action.payload;
    },
    setShouldShowBrushPreview: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldShowBrushPreview = action.payload;
    },
    setShouldShowBrush: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldShowBrush = action.payload;
    },
    setCursorPosition: (state, action: PayloadAction<Vector2d | null>) => {
      state[state.currentCanvas].cursorPosition = action.payload;
    },
    clearImageToInpaint: (state) => {
      state.inpainting.imageToInpaint = undefined;
    },

    setImageToOutpaint: (state, action: PayloadAction<InvokeAI.Image>) => {
      const { width: canvasWidth, height: canvasHeight } =
        state.outpainting.stageDimensions;
      const { width, height } = state.outpainting.boundingBoxDimensions;
      const { x, y } = state.outpainting.boundingBoxCoordinates;

      const maxWidth = Math.min(action.payload.width, canvasWidth);
      const maxHeight = Math.min(action.payload.height, canvasHeight);

      const newCoordinates: Vector2d = { x, y };
      const newDimensions: Dimensions = { width, height };

      if (width + x > maxWidth) {
        // Bounding box at least needs to be translated
        if (width > maxWidth) {
          // Bounding box also needs to be resized
          newDimensions.width = roundDownToMultiple(maxWidth, 64);
        }
        newCoordinates.x = maxWidth - newDimensions.width;
      }

      if (height + y > maxHeight) {
        // Bounding box at least needs to be translated
        if (height > maxHeight) {
          // Bounding box also needs to be resized
          newDimensions.height = roundDownToMultiple(maxHeight, 64);
        }
        newCoordinates.y = maxHeight - newDimensions.height;
      }

      state.outpainting.boundingBoxDimensions = newDimensions;
      state.outpainting.boundingBoxCoordinates = newCoordinates;

      // state.outpainting.imageToInpaint = action.payload;
      state.outpainting.objects = [
        {
          kind: 'image',
          layer: 'base',
          x: 0,
          y: 0,
          image: action.payload,
        },
      ];
      state.doesCanvasNeedScaling = true;
    },
    setImageToInpaint: (state, action: PayloadAction<InvokeAI.Image>) => {
      const { width: canvasWidth, height: canvasHeight } =
        state.inpainting.stageDimensions;
      const { width, height } = state.inpainting.boundingBoxDimensions;
      const { x, y } = state.inpainting.boundingBoxCoordinates;

      const maxWidth = Math.min(action.payload.width, canvasWidth);
      const maxHeight = Math.min(action.payload.height, canvasHeight);

      const newCoordinates: Vector2d = { x, y };
      const newDimensions: Dimensions = { width, height };

      if (width + x > maxWidth) {
        // Bounding box at least needs to be translated
        if (width > maxWidth) {
          // Bounding box also needs to be resized
          newDimensions.width = roundDownToMultiple(maxWidth, 64);
        }
        newCoordinates.x = maxWidth - newDimensions.width;
      }

      if (height + y > maxHeight) {
        // Bounding box at least needs to be translated
        if (height > maxHeight) {
          // Bounding box also needs to be resized
          newDimensions.height = roundDownToMultiple(maxHeight, 64);
        }
        newCoordinates.y = maxHeight - newDimensions.height;
      }

      state.inpainting.boundingBoxDimensions = newDimensions;
      state.inpainting.boundingBoxCoordinates = newCoordinates;

      // state.inpainting.imageToInpaint = action.payload;
      state.inpainting.objects = [
        {
          kind: 'image',
          layer: 'base',
          x: 0,
          y: 0,
          image: action.payload,
        },
      ];
      state.doesCanvasNeedScaling = true;
    },
    setStageDimensions: (state, action: PayloadAction<Dimensions>) => {
      state[state.currentCanvas].stageDimensions = action.payload;

      const { width: canvasWidth, height: canvasHeight } = action.payload;

      const { width: boundingBoxWidth, height: boundingBoxHeight } =
        state[state.currentCanvas].boundingBoxDimensions;

      const newBoundingBoxWidth = roundDownToMultiple(
        _.clamp(
          boundingBoxWidth,
          64,
          canvasWidth / state[state.currentCanvas].stageScale
        ),
        64
      );
      const newBoundingBoxHeight = roundDownToMultiple(
        _.clamp(
          boundingBoxHeight,
          64,
          canvasHeight / state[state.currentCanvas].stageScale
        ),
        64
      );

      state[state.currentCanvas].boundingBoxDimensions = {
        width: newBoundingBoxWidth,
        height: newBoundingBoxHeight,
      };
    },
    setBoundingBoxDimensions: (state, action: PayloadAction<Dimensions>) => {
      state[state.currentCanvas].boundingBoxDimensions = action.payload;
      const { width: boundingBoxWidth, height: boundingBoxHeight } =
        action.payload;
      const { x: boundingBoxX, y: boundingBoxY } =
        state[state.currentCanvas].boundingBoxCoordinates;
      const { width: canvasWidth, height: canvasHeight } =
        state[state.currentCanvas].stageDimensions;

      const scaledCanvasWidth =
        canvasWidth / state[state.currentCanvas].stageScale;
      const scaledCanvasHeight =
        canvasHeight / state[state.currentCanvas].stageScale;

      const roundedCanvasWidth = roundDownToMultiple(scaledCanvasWidth, 64);
      const roundedCanvasHeight = roundDownToMultiple(scaledCanvasHeight, 64);

      const roundedBoundingBoxWidth = roundDownToMultiple(boundingBoxWidth, 64);
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
        state[state.currentCanvas].stageCoordinates.x,
        roundedCanvasWidth - newBoundingBoxWidth
      );

      const clampedY = _.clamp(
        overflowCorrectedY,
        state[state.currentCanvas].stageCoordinates.y,
        roundedCanvasHeight - newBoundingBoxHeight
      );

      state[state.currentCanvas].boundingBoxDimensions = {
        width: newBoundingBoxWidth,
        height: newBoundingBoxHeight,
      };

      state[state.currentCanvas].boundingBoxCoordinates = {
        x: clampedX,
        y: clampedY,
      };
    },
    setBoundingBoxCoordinates: (state, action: PayloadAction<Vector2d>) => {
      state[state.currentCanvas].boundingBoxCoordinates = action.payload;
    },
    setStageCoordinates: (state, action: PayloadAction<Vector2d>) => {
      state[state.currentCanvas].stageCoordinates = action.payload;
    },
    setBoundingBoxPreviewFill: (state, action: PayloadAction<RgbaColor>) => {
      state[state.currentCanvas].boundingBoxPreviewFill = action.payload;
    },
    setDoesCanvasNeedScaling: (state, action: PayloadAction<boolean>) => {
      state.doesCanvasNeedScaling = action.payload;
    },
    setStageScale: (state, action: PayloadAction<number>) => {
      state[state.currentCanvas].stageScale = action.payload;
      state.doesCanvasNeedScaling = false;
    },
    setShouldDarkenOutsideBoundingBox: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state[state.currentCanvas].shouldDarkenOutsideBoundingBox =
        action.payload;
    },
    setIsDrawing: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isDrawing = action.payload;
    },
    setClearBrushHistory: (state) => {
      state[state.currentCanvas].pastObjects = [];
      state[state.currentCanvas].futureObjects = [];
    },
    setShouldUseInpaintReplace: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldUseInpaintReplace = action.payload;
    },
    setInpaintReplace: (state, action: PayloadAction<number>) => {
      state[state.currentCanvas].inpaintReplace = action.payload;
    },
    setShouldLockBoundingBox: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldLockBoundingBox = action.payload;
    },
    toggleShouldLockBoundingBox: (state) => {
      state[state.currentCanvas].shouldLockBoundingBox =
        !state[state.currentCanvas].shouldLockBoundingBox;
    },
    setShouldShowBoundingBox: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldShowBoundingBox = action.payload;
    },
    setIsTransformingBoundingBox: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isTransformingBoundingBox = action.payload;
    },
    setIsMovingBoundingBox: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isMovingBoundingBox = action.payload;
    },
    setIsMouseOverBoundingBox: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isMouseOverBoundingBox = action.payload;
    },
    setIsMoveBoundingBoxKeyHeld: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isMoveBoundingBoxKeyHeld = action.payload;
    },
    setIsMoveStageKeyHeld: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isMoveStageKeyHeld = action.payload;
    },
    setCurrentCanvas: (state, action: PayloadAction<ValidCanvasName>) => {
      state.currentCanvas = action.payload;
    },
    addImageToOutpaintingSesion: (
      state,
      action: PayloadAction<{
        boundingBox: IRect;
        image: InvokeAI.Image;
      }>
    ) => {
      const { boundingBox, image } = action.payload;
      if (!boundingBox || !image) return;

      const { x, y } = boundingBox;

      state.outpainting.pastObjects.push([...state.outpainting.objects]);
      state.outpainting.futureObjects = [];

      state.outpainting.objects.push({
        kind: 'image',
        layer: 'base',
        x,
        y,
        image,
      });
    },
    addLine: (state, action: PayloadAction<number[]>) => {
      const currentCanvas = state[state.currentCanvas];

      const { tool, layer, brushColor, brushSize, eraserSize } = currentCanvas;

      if (tool === 'move') return;

      let newTool: CanvasDrawingTool;

      if (layer === 'mask' && currentCanvas.shouldInvertMask) {
        newTool = tool === 'eraser' ? 'brush' : 'eraser';
      } else {
        newTool = tool;
      }

      const newStrokeWidth = tool === 'brush' ? brushSize / 2 : eraserSize / 2;

      // set & then spread this to only conditionally add the "color" key
      const newColor =
        layer === 'base' && tool === 'brush' ? { color: brushColor } : {};

      currentCanvas.pastObjects.push(currentCanvas.objects);

      currentCanvas.objects.push({
        kind: 'line',
        layer,
        tool: newTool,
        strokeWidth: newStrokeWidth,
        points: action.payload,
        ...newColor,
      });

      currentCanvas.futureObjects = [];
    },
    addPointToCurrentLine: (state, action: PayloadAction<number[]>) => {
      const lastLine =
        state[state.currentCanvas].objects.findLast(isCanvasAnyLine);

      if (!lastLine) return;

      lastLine.points.push(...action.payload);
    },
    undo: (state) => {
      if (state[state.currentCanvas].objects.length === 0) return;

      const newObjects = state[state.currentCanvas].pastObjects.pop();
      if (!newObjects) return;
      state[state.currentCanvas].futureObjects.unshift(
        state[state.currentCanvas].objects
      );
      state[state.currentCanvas].objects = newObjects;
    },
    redo: (state) => {
      if (state[state.currentCanvas].futureObjects.length === 0) return;
      const newObjects = state[state.currentCanvas].futureObjects.shift();
      if (!newObjects) return;
      state[state.currentCanvas].pastObjects.push(
        state[state.currentCanvas].objects
      );
      state[state.currentCanvas].objects = newObjects;
    },
    setShouldShowGrid: (state, action: PayloadAction<boolean>) => {
      state.outpainting.shouldShowGrid = action.payload;
    },
    setIsMovingStage: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].isMovingStage = action.payload;
    },
    setShouldSnapToGrid: (state, action: PayloadAction<boolean>) => {
      state.outpainting.shouldSnapToGrid = action.payload;
    },
    setShouldAutoSave: (state, action: PayloadAction<boolean>) => {
      state.outpainting.shouldAutoSave = action.payload;
    },
    setShouldShowIntermediates: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldShowIntermediates = action.payload;
    },
    resetCanvas: (state) => {
      state[state.currentCanvas].pastObjects.push(
        state[state.currentCanvas].objects
      );

      state[state.currentCanvas].objects = [];
      state[state.currentCanvas].futureObjects = [];
    },
  },
  extraReducers: (builder) => {
    builder.addCase(uploadOutpaintingMergedImage.fulfilled, (state, action) => {
      if (!action.payload) return;
      state.outpainting.pastObjects.push([...state.outpainting.objects]);
      state.outpainting.futureObjects = [];

      state.outpainting.objects = [
        {
          kind: 'image',
          layer: 'base',
          ...action.payload,
        },
      ];
    });
  },
});

export const {
  setTool,
  setLayer,
  setBrushColor,
  setBrushSize,
  setEraserSize,
  addLine,
  addPointToCurrentLine,
  setShouldInvertMask,
  setIsMaskEnabled,
  setShouldShowCheckboardTransparency,
  setShouldShowBrushPreview,
  setMaskColor,
  clearMask,
  clearImageToInpaint,
  undo,
  redo,
  setCursorPosition,
  setStageDimensions,
  setImageToInpaint,
  setImageToOutpaint,
  setBoundingBoxDimensions,
  setBoundingBoxCoordinates,
  setBoundingBoxPreviewFill,
  setDoesCanvasNeedScaling,
  setStageScale,
  toggleTool,
  setShouldShowBoundingBox,
  setShouldDarkenOutsideBoundingBox,
  setIsDrawing,
  setShouldShowBrush,
  setClearBrushHistory,
  setShouldUseInpaintReplace,
  setInpaintReplace,
  setShouldLockBoundingBox,
  toggleShouldLockBoundingBox,
  setIsMovingBoundingBox,
  setIsTransformingBoundingBox,
  setIsMouseOverBoundingBox,
  setIsMoveBoundingBoxKeyHeld,
  setIsMoveStageKeyHeld,
  setStageCoordinates,
  setCurrentCanvas,
  addImageToOutpaintingSesion,
  resetCanvas,
  setShouldShowGrid,
  setShouldSnapToGrid,
  setShouldAutoSave,
  setShouldShowIntermediates,
  setIsMovingStage,
} = canvasSlice.actions;

export default canvasSlice.reducer;

export const uploadOutpaintingMergedImage = createAsyncThunk(
  'canvas/uploadOutpaintingMergedImage',
  async (
    canvasImageLayerRef: MutableRefObject<Konva.Layer | null>,
    thunkAPI
  ) => {
    const { getState } = thunkAPI;

    const state = getState() as RootState;
    const stageScale = state.canvas.outpainting.stageScale;

    if (!canvasImageLayerRef.current) return;
    const tempScale = canvasImageLayerRef.current.scale();

    const { x: relativeX, y: relativeY } =
      canvasImageLayerRef.current.getClientRect({
        relativeTo: canvasImageLayerRef.current.getParent(),
      });

    canvasImageLayerRef.current.scale({
      x: 1 / stageScale,
      y: 1 / stageScale,
    });

    const clientRect = canvasImageLayerRef.current.getClientRect();

    const imageDataURL = canvasImageLayerRef.current.toDataURL(clientRect);

    canvasImageLayerRef.current.scale(tempScale);

    if (!imageDataURL) return;

    const response = await fetch(window.location.origin + '/upload', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dataURL: imageDataURL,
        name: 'outpaintingmerge.png',
      }),
    });

    const data = (await response.json()) as InvokeAI.ImageUploadResponse;

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { destination, ...rest } = data;
    const image = {
      uuid: uuidv4(),
      ...rest,
    };

    return {
      image,
      x: relativeX,
      y: relativeY,
    };
  }
);

export const currentCanvasSelector = (state: RootState): BaseCanvasState =>
  state.canvas[state.canvas.currentCanvas];

export const outpaintingCanvasSelector = (
  state: RootState
): OutpaintingCanvasState => state.canvas.outpainting;

export const inpaintingCanvasSelector = (
  state: RootState
): InpaintingCanvasState => state.canvas.inpainting;

export const baseCanvasImageSelector = createSelector(
  [currentCanvasSelector],
  (currentCanvas) => {
    return currentCanvas.objects.find(isCanvasBaseImage);
  }
);
