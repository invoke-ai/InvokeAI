import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { initialAspectRatioState } from 'features/parameters/components/ImageSize/constants';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { modelChanged } from 'features/parameters/store/generationSlice';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect, Vector2d } from 'konva/lib/types';
import { atom } from 'nanostores';

import type {
  CanvasV2State,
  ControlAdapterData,
  IPAdapterData,
  LayerData,
  RegionalGuidanceData,
  RgbaColor,
  StageAttrs,
  Tool,
} from './types';
import { DEFAULT_RGBA_COLOR } from './types';

const initialState: CanvasV2State = {
  _version: 3,
  lastSelectedItem: null,
  prompts: {
    positivePrompt: '',
    negativePrompt: '',
    positivePrompt2: '',
    negativePrompt2: '',
    shouldConcatPrompts: true,
  },
  tool: {
    selected: 'bbox',
    selectedBuffer: null,
    invertScroll: false,
    fill: DEFAULT_RGBA_COLOR,
    brush: {
      width: 50,
    },
    eraser: {
      width: 50,
    },
  },
  size: {
    width: 512,
    height: 512,
    aspectRatio: deepClone(initialAspectRatioState),
  },
  bbox: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },
};

export const canvasV2Slice = createSlice({
  name: 'canvasV2',
  initialState,
  reducers: {
    positivePromptChanged: (state, action: PayloadAction<string>) => {
      state.prompts.positivePrompt = action.payload;
    },
    negativePromptChanged: (state, action: PayloadAction<string>) => {
      state.prompts.negativePrompt = action.payload;
    },
    positivePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.prompts.positivePrompt2 = action.payload;
    },
    negativePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.prompts.negativePrompt2 = action.payload;
    },
    shouldConcatPromptsChanged: (state, action: PayloadAction<boolean>) => {
      state.prompts.shouldConcatPrompts = action.payload;
    },
    widthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { width, updateAspectRatio, clamp } = action.payload;
      state.size.width = clamp ? Math.max(roundDownToMultiple(width, 8), 64) : width;
      if (updateAspectRatio) {
        state.size.aspectRatio.value = state.size.width / state.size.height;
        state.size.aspectRatio.id = 'Free';
        state.size.aspectRatio.isLocked = false;
      }
    },
    heightChanged: (state, action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { height, updateAspectRatio, clamp } = action.payload;
      state.size.height = clamp ? Math.max(roundDownToMultiple(height, 8), 64) : height;
      if (updateAspectRatio) {
        state.size.aspectRatio.value = state.size.width / state.size.height;
        state.size.aspectRatio.id = 'Free';
        state.size.aspectRatio.isLocked = false;
      }
    },
    aspectRatioChanged: (state, action: PayloadAction<AspectRatioState>) => {
      state.size.aspectRatio = action.payload;
    },
    bboxChanged: (state, action: PayloadAction<IRect>) => {
      state.bbox = action.payload;
    },
    brushWidthChanged: (state, action: PayloadAction<number>) => {
      state.tool.brush.width = Math.round(action.payload);
    },
    eraserWidthChanged: (state, action: PayloadAction<number>) => {
      state.tool.eraser.width = Math.round(action.payload);
    },
    fillChanged: (state, action: PayloadAction<RgbaColor>) => {
      state.tool.fill = action.payload;
    },
    invertScrollChanged: (state, action: PayloadAction<boolean>) => {
      state.tool.invertScroll = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addCase(modelChanged, (state, action) => {
      const newModel = action.payload;
      if (!newModel || action.meta.previousModel?.base === newModel.base) {
        // Model was cleared or the base didn't change
        return;
      }
      const optimalDimension = getOptimalDimension(newModel);
      if (getIsSizeOptimal(state.size.width, state.size.height, optimalDimension)) {
        return;
      }
      const { width, height } = calculateNewSize(state.size.aspectRatio.value, optimalDimension * optimalDimension);
      state.size.width = width;
      state.size.height = height;
    });
  },
});

export const {
  positivePromptChanged,
  negativePromptChanged,
  positivePrompt2Changed,
  negativePrompt2Changed,
  shouldConcatPromptsChanged,
  widthChanged,
  heightChanged,
  aspectRatioChanged,
  bboxChanged,
  brushWidthChanged,
  eraserWidthChanged,
  fillChanged,
  invertScrollChanged,
} = canvasV2Slice.actions;

export const selectCanvasV2Slice = (state: RootState) => state.canvasV2;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

// Ephemeral interaction state
export const $isDrawing = atom(false);
export const $isMouseDown = atom(false);
export const $lastMouseDownPos = atom<Vector2d | null>(null);
export const $lastCursorPos = atom<Vector2d | null>(null);
export const $isPreviewVisible = atom(true);
export const $lastAddedPoint = atom<Vector2d | null>(null);
export const $spaceKey = atom(false);
export const $stageAttrs = atom<StageAttrs>({
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  scale: 0,
});

// Some nanostores that are manually synced to redux state to provide imperative access
// TODO(psyche):
export const $tool = atom<Tool>('brush');
export const $toolBuffer = atom<Tool | null>(null);
export const $brushWidth = atom<number>(0);
export const $brushSpacingPx = atom<number>(0);
export const $eraserWidth = atom<number>(0);
export const $eraserSpacingPx = atom<number>(0);
export const $fill = atom<RgbaColor>(DEFAULT_RGBA_COLOR);
export const $selectedLayer = atom<LayerData | null>(null);
export const $selectedRG = atom<RegionalGuidanceData | null>(null);
export const $selectedCA = atom<ControlAdapterData | null>(null);
export const $selectedIPA = atom<IPAdapterData | null>(null);
export const $invertScroll = atom(false);
export const $bbox = atom<IRect>({ x: 0, y: 0, width: 0, height: 0 });

export const canvasV2PersistConfig: PersistConfig<CanvasV2State> = {
  name: canvasV2Slice.name,
  initialState,
  migrate,
  persistDenylist: ['bbox'],
};
