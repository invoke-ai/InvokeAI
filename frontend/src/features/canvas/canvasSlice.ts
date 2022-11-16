import {
  createAsyncThunk,
  createSelector,
  createSlice,
  current,
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
import { tabMap } from 'features/tabs/InvokeTabs';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { mergeAndUploadCanvas } from './util/mergeAndUploadCanvas';
import { uploadImage } from 'features/gallery/util/uploadImage';
import { setInitialCanvasImage } from './canvasReducers';
import calculateScale from './util/calculateScale';
import calculateCoordinates from './util/calculateCoordinates';
import floorCoordinates from './util/floorCoordinates';

export interface GenericCanvasState {
  boundingBoxCoordinates: Vector2d;
  boundingBoxDimensions: Dimensions;
  boundingBoxPreviewFill: RgbaColor;
  brushColor: RgbaColor;
  brushSize: number;
  cursorPosition: Vector2d | null;
  eraserSize: number;
  futureLayerStates: CanvasLayerState[];
  inpaintReplace: number;
  intermediateImage?: InvokeAI.Image;
  isDrawing: boolean;
  isMaskEnabled: boolean;
  isMouseOverBoundingBox: boolean;
  isMoveBoundingBoxKeyHeld: boolean;
  isMoveStageKeyHeld: boolean;
  isMovingBoundingBox: boolean;
  isMovingStage: boolean;
  isTransformingBoundingBox: boolean;
  layerState: CanvasLayerState;
  maskColor: RgbaColor;
  maxHistory: number;
  pastLayerStates: CanvasLayerState[];
  shouldDarkenOutsideBoundingBox: boolean;
  shouldLockBoundingBox: boolean;
  shouldPreserveMaskedArea: boolean;
  shouldShowBoundingBox: boolean;
  shouldShowBrush: boolean;
  shouldShowBrushPreview: boolean;
  shouldShowCheckboardTransparency: boolean;
  shouldShowIntermediates: boolean;
  shouldUseInpaintReplace: boolean;
  stageCoordinates: Vector2d;
  stageDimensions: Dimensions;
  stageScale: number;
  tool: CanvasTool;
  minimumStageScale: number;
}

export type CanvasMode = 'inpainting' | 'outpainting';

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

export type OutpaintingCanvasState = GenericCanvasState & {
  layer: CanvasLayer;
  shouldShowGrid: boolean;
  shouldSnapToGrid: boolean;
  shouldAutoSave: boolean;
};

export type InpaintingCanvasState = GenericCanvasState & {
  layer: 'mask';
};

export type BaseCanvasState = InpaintingCanvasState | OutpaintingCanvasState;

export type ValidCanvasName = 'inpainting' | 'outpainting';

export interface CanvasState {
  doesCanvasNeedScaling: boolean;
  currentCanvas: ValidCanvasName;
  inpainting: InpaintingCanvasState;
  outpainting: OutpaintingCanvasState;
  // mode: CanvasMode;
  shouldLockToInitialImage: boolean;
  isCanvasInitialized: boolean;
  canvasContainerDimensions: Dimensions;
}

export const initialLayerState: CanvasLayerState = {
  objects: [],
  stagingArea: {
    x: -1,
    y: -1,
    width: -1,
    height: -1,
    images: [],
    selectedImageIndex: -1,
  },
};

const initialGenericCanvasState: GenericCanvasState = {
  boundingBoxCoordinates: { x: 0, y: 0 },
  boundingBoxDimensions: { width: 512, height: 512 },
  boundingBoxPreviewFill: { r: 0, g: 0, b: 0, a: 0.5 },
  brushColor: { r: 90, g: 90, b: 255, a: 1 },
  brushSize: 50,
  cursorPosition: null,
  eraserSize: 50,
  futureLayerStates: [],
  inpaintReplace: 0.1,
  isDrawing: false,
  isMaskEnabled: true,
  isMouseOverBoundingBox: false,
  isMoveBoundingBoxKeyHeld: false,
  isMoveStageKeyHeld: false,
  isMovingBoundingBox: false,
  isMovingStage: false,
  isTransformingBoundingBox: false,
  layerState: initialLayerState,
  maskColor: { r: 255, g: 90, b: 90, a: 1 },
  maxHistory: 128,
  pastLayerStates: [],
  shouldDarkenOutsideBoundingBox: false,
  shouldLockBoundingBox: false,
  shouldPreserveMaskedArea: false,
  shouldShowBoundingBox: true,
  shouldShowBrush: true,
  shouldShowBrushPreview: false,
  shouldShowCheckboardTransparency: false,
  shouldShowIntermediates: true,
  shouldUseInpaintReplace: false,
  stageCoordinates: { x: 0, y: 0 },
  stageDimensions: { width: 0, height: 0 },
  stageScale: 1,
  minimumStageScale: 1,
  tool: 'brush',
};

const initialCanvasState: CanvasState = {
  currentCanvas: 'inpainting',
  doesCanvasNeedScaling: false,
  shouldLockToInitialImage: false,
  // mode: 'outpainting',
  isCanvasInitialized: false,
  canvasContainerDimensions: { width: 0, height: 0 },
  inpainting: {
    layer: 'mask',
    ...initialGenericCanvasState,
  },
  outpainting: {
    layer: 'base',
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
      const currentCanvas = state[state.currentCanvas];
      currentCanvas.pastLayerStates.push(currentCanvas.layerState);
      currentCanvas.layerState.objects = state[
        state.currentCanvas
      ].layerState.objects.filter((obj) => !isCanvasMaskLine(obj));
      currentCanvas.futureLayerStates = [];
      currentCanvas.shouldPreserveMaskedArea = false;
    },
    toggleShouldInvertMask: (state) => {
      state[state.currentCanvas].shouldPreserveMaskedArea =
        !state[state.currentCanvas].shouldPreserveMaskedArea;
    },
    toggleShouldShowMask: (state) => {
      state[state.currentCanvas].isMaskEnabled =
        !state[state.currentCanvas].isMaskEnabled;
    },
    setShouldPreserveMaskedArea: (state, action: PayloadAction<boolean>) => {
      state[state.currentCanvas].shouldPreserveMaskedArea = action.payload;
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
      // TODO
      // state.inpainting.imageToInpaint = undefined;
    },
    setImageToOutpaint: (state, action: PayloadAction<InvokeAI.Image>) => {
      setInitialCanvasImage(state, action.payload);
    },
    setImageToInpaint: (state, action: PayloadAction<InvokeAI.Image>) => {
      setInitialCanvasImage(state, action.payload);
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
    },
    setBoundingBoxCoordinates: (state, action: PayloadAction<Vector2d>) => {
      state[state.currentCanvas].boundingBoxCoordinates = floorCoordinates(
        action.payload
      );
    },
    setStageCoordinates: (state, action: PayloadAction<Vector2d>) => {
      state.outpainting.stageCoordinates = floorCoordinates(action.payload);
    },
    setBoundingBoxPreviewFill: (state, action: PayloadAction<RgbaColor>) => {
      state[state.currentCanvas].boundingBoxPreviewFill = action.payload;
    },
    setDoesCanvasNeedScaling: (state, action: PayloadAction<boolean>) => {
      state.doesCanvasNeedScaling = action.payload;
    },
    setStageScale: (state, action: PayloadAction<number>) => {
      state[state.currentCanvas].stageScale = action.payload;
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
      state[state.currentCanvas].pastLayerStates = [];
      state[state.currentCanvas].futureLayerStates = [];
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
    addImageToStagingArea: (
      state,
      action: PayloadAction<{
        boundingBox: IRect;
        image: InvokeAI.Image;
      }>
    ) => {
      const { boundingBox, image } = action.payload;

      if (!boundingBox || !image) return;

      const currentCanvas = state.outpainting;

      currentCanvas.pastLayerStates.push(_.cloneDeep(currentCanvas.layerState));

      if (currentCanvas.pastLayerStates.length > currentCanvas.maxHistory) {
        currentCanvas.pastLayerStates.shift();
      }

      currentCanvas.layerState.stagingArea.images.push({
        kind: 'image',
        layer: 'base',
        ...boundingBox,
        image,
      });

      currentCanvas.layerState.stagingArea.selectedImageIndex =
        currentCanvas.layerState.stagingArea.images.length - 1;

      currentCanvas.futureLayerStates = [];
    },
    discardStagedImages: (state) => {
      const currentCanvas = state[state.currentCanvas];
      currentCanvas.layerState.stagingArea = {
        ...initialLayerState.stagingArea,
      };
    },
    addLine: (state, action: PayloadAction<number[]>) => {
      const currentCanvas = state[state.currentCanvas];

      const { tool, layer, brushColor, brushSize, eraserSize } = currentCanvas;

      if (tool === 'move') return;

      const newStrokeWidth = tool === 'brush' ? brushSize / 2 : eraserSize / 2;

      // set & then spread this to only conditionally add the "color" key
      const newColor =
        layer === 'base' && tool === 'brush' ? { color: brushColor } : {};

      currentCanvas.pastLayerStates.push(currentCanvas.layerState);

      if (currentCanvas.pastLayerStates.length > currentCanvas.maxHistory) {
        currentCanvas.pastLayerStates.shift();
      }

      currentCanvas.layerState.objects.push({
        kind: 'line',
        layer,
        tool,
        strokeWidth: newStrokeWidth,
        points: action.payload,
        ...newColor,
      });

      currentCanvas.futureLayerStates = [];
    },
    addPointToCurrentLine: (state, action: PayloadAction<number[]>) => {
      const lastLine =
        state[state.currentCanvas].layerState.objects.findLast(isCanvasAnyLine);

      if (!lastLine) return;

      lastLine.points.push(...action.payload);
    },
    undo: (state) => {
      const currentCanvas = state[state.currentCanvas];

      const targetState = currentCanvas.pastLayerStates.pop();

      if (!targetState) return;

      currentCanvas.futureLayerStates.unshift(currentCanvas.layerState);

      if (currentCanvas.futureLayerStates.length > currentCanvas.maxHistory) {
        currentCanvas.futureLayerStates.pop();
      }

      currentCanvas.layerState = targetState;
    },
    redo: (state) => {
      const currentCanvas = state[state.currentCanvas];

      const targetState = currentCanvas.futureLayerStates.shift();

      if (!targetState) return;

      currentCanvas.pastLayerStates.push(currentCanvas.layerState);

      if (currentCanvas.pastLayerStates.length > currentCanvas.maxHistory) {
        currentCanvas.pastLayerStates.shift();
      }

      currentCanvas.layerState = targetState;
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
      state[state.currentCanvas].pastLayerStates.push(
        state[state.currentCanvas].layerState
      );

      state[state.currentCanvas].layerState = initialLayerState;
      state[state.currentCanvas].futureLayerStates = [];
    },
    setCanvasContainerDimensions: (
      state,
      action: PayloadAction<Dimensions>
    ) => {
      state.canvasContainerDimensions = action.payload;
    },
    resizeAndScaleCanvas: (state) => {
      const { width: containerWidth, height: containerHeight } =
        state.canvasContainerDimensions;

      const initialCanvasImage =
        state.outpainting.layerState.objects.find(isCanvasBaseImage);

      if (!initialCanvasImage) return;

      const { width: imageWidth, height: imageHeight } = initialCanvasImage;

      // const { clientWidth, clientHeight, imageWidth, imageHeight } =
      //   action.payload;

      const { shouldLockToInitialImage } = state;

      const currentCanvas = state[state.currentCanvas];

      const padding = shouldLockToInitialImage ? 1 : 0.95;

      const newScale = calculateScale(
        containerWidth,
        containerHeight,
        imageWidth,
        imageHeight,
        padding
      );

      const newDimensions = {
        width: shouldLockToInitialImage
          ? Math.floor(imageWidth * newScale)
          : Math.floor(containerWidth),
        height: shouldLockToInitialImage
          ? Math.floor(imageHeight * newScale)
          : Math.floor(containerHeight),
      };

      const newCoordinates = calculateCoordinates(
        newDimensions.width,
        newDimensions.height,
        0,
        0,
        imageWidth,
        imageHeight,
        newScale
      );

      currentCanvas.stageScale = newScale;
      currentCanvas.minimumStageScale = newScale;
      currentCanvas.stageCoordinates = newCoordinates;

      currentCanvas.stageDimensions = newDimensions;
      state.isCanvasInitialized = true;
    },
    resizeCanvas: (state) => {
      const { width: containerWidth, height: containerHeight } =
        state.canvasContainerDimensions;

      const currentCanvas = state[state.currentCanvas];

      currentCanvas.stageDimensions = {
        width: Math.floor(containerWidth),
        height: Math.floor(containerHeight),
      };
    },
    resetCanvasView: (
      state,
      action: PayloadAction<{
        contentRect: IRect;
      }>
    ) => {
      const { contentRect } = action.payload;

      const currentCanvas = state[state.currentCanvas];

      const baseCanvasImage =
        currentCanvas.layerState.objects.find(isCanvasBaseImage);
      const { shouldLockToInitialImage } = state;
      if (!baseCanvasImage) return;

      const {
        stageDimensions: { width: stageWidth, height: stageHeight },
      } = currentCanvas;

      const { x, y, width, height } = contentRect;

      const padding = shouldLockToInitialImage ? 1 : 0.95;
      const newScale = calculateScale(
        stageWidth,
        stageHeight,
        width,
        height,
        padding
      );

      const newCoordinates = calculateCoordinates(
        stageWidth,
        stageHeight,
        x,
        y,
        width,
        height,
        newScale
      );

      currentCanvas.stageScale = newScale;

      currentCanvas.stageCoordinates = newCoordinates;
    },
    nextStagingAreaImage: (state) => {
      const currentIndex =
        state.outpainting.layerState.stagingArea.selectedImageIndex;
      const length = state.outpainting.layerState.stagingArea.images.length;

      state.outpainting.layerState.stagingArea.selectedImageIndex = Math.min(
        currentIndex + 1,
        length - 1
      );
    },
    prevStagingAreaImage: (state) => {
      const currentIndex =
        state.outpainting.layerState.stagingArea.selectedImageIndex;

      state.outpainting.layerState.stagingArea.selectedImageIndex = Math.max(
        currentIndex - 1,
        0
      );
    },
    commitStagingAreaImage: (state) => {
      const currentCanvas = state[state.currentCanvas];
      const { images, selectedImageIndex } =
        currentCanvas.layerState.stagingArea;

      currentCanvas.pastLayerStates.push(_.cloneDeep(currentCanvas.layerState));

      if (currentCanvas.pastLayerStates.length > currentCanvas.maxHistory) {
        currentCanvas.pastLayerStates.shift();
      }

      currentCanvas.layerState.objects.push({
        ...images[selectedImageIndex],
      });

      currentCanvas.layerState.stagingArea = {
        ...initialLayerState.stagingArea,
      };

      currentCanvas.futureLayerStates = [];
    },
    setShouldLockToInitialImage: (state, action: PayloadAction<boolean>) => {
      state.shouldLockToInitialImage = action.payload;
    },
    // setCanvasMode: (state, action: PayloadAction<CanvasMode>) => {
    //   state.mode = action.payload;
    // },
  },
  extraReducers: (builder) => {
    builder.addCase(mergeAndUploadCanvas.fulfilled, (state, action) => {
      if (!action.payload) return;
      const { image, kind, originalBoundingBox } = action.payload;

      if (kind === 'temp_merged_canvas') {
        state.outpainting.pastLayerStates.push({
          ...state.outpainting.layerState,
        });

        state.outpainting.futureLayerStates = [];

        state.outpainting.layerState.objects = [
          {
            kind: 'image',
            layer: 'base',
            ...originalBoundingBox,
            image,
          },
        ];
      }
    });

    builder.addCase(uploadImage.fulfilled, (state, action) => {
      if (!action.payload) return;
      const { image, kind, activeTabName } = action.payload;

      if (kind !== 'init') return;

      if (activeTabName === 'inpainting') {
        setInitialCanvasImage(state, image);
      } else if (activeTabName === 'outpainting') {
        setInitialCanvasImage(state, image);
      }
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
  setShouldPreserveMaskedArea,
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
  addImageToStagingArea,
  resetCanvas,
  setShouldShowGrid,
  setShouldSnapToGrid,
  setShouldAutoSave,
  setShouldShowIntermediates,
  setIsMovingStage,
  nextStagingAreaImage,
  prevStagingAreaImage,
  commitStagingAreaImage,
  discardStagedImages,
  setShouldLockToInitialImage,
  resizeAndScaleCanvas,
  resizeCanvas,
  resetCanvasView,
  setCanvasContainerDimensions,
} = canvasSlice.actions;

export default canvasSlice.reducer;

export const currentCanvasSelector = (state: RootState): BaseCanvasState =>
  state.canvas[state.canvas.currentCanvas];

export const isStagingSelector = (state: RootState): boolean =>
  state.canvas[state.canvas.currentCanvas].layerState.stagingArea.images
    .length > 0;

export const outpaintingCanvasSelector = (
  state: RootState
): OutpaintingCanvasState => state.canvas.outpainting;

export const inpaintingCanvasSelector = (
  state: RootState
): InpaintingCanvasState => state.canvas.inpainting;

export const shouldLockToInitialImageSelector = (state: RootState): boolean =>
  state.canvas.shouldLockToInitialImage;

export const baseCanvasImageSelector = createSelector(
  [currentCanvasSelector],
  (currentCanvas) => {
    return currentCanvas.layerState.objects.find(isCanvasBaseImage);
  }
);

// export const canvasClipSelector = createSelector(
//   [canvasModeSelector, baseCanvasImageSelector],
//   (canvasMode, baseCanvasImage) => {
//     return canvasMode === 'inpainting'
//       ? {
//           clipX: 0,
//           clipY: 0,
//           clipWidth: baseCanvasImage?.width,
//           clipHeight: baseCanvasImage?.height,
//         }
//       : {};
//   }
// );
