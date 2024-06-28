import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import { bboxReducers } from 'features/controlLayers/store/bboxReducers';
import { compositingReducers } from 'features/controlLayers/store/compositingReducers';
import { controlAdaptersReducers } from 'features/controlLayers/store/controlAdaptersReducers';
import { inpaintMaskReducers } from 'features/controlLayers/store/inpaintMaskReducers';
import { ipAdaptersReducers } from 'features/controlLayers/store/ipAdaptersReducers';
import { layersReducers } from 'features/controlLayers/store/layersReducers';
import { lorasReducers } from 'features/controlLayers/store/lorasReducers';
import { paramsReducers } from 'features/controlLayers/store/paramsReducers';
import { regionsReducers } from 'features/controlLayers/store/regionsReducers';
import { settingsReducers } from 'features/controlLayers/store/settingsReducers';
import { stagingAreaReducers } from 'features/controlLayers/store/stagingAreaReducers';
import { toolReducers } from 'features/controlLayers/store/toolReducers';
import { initialAspectRatioState } from 'features/parameters/components/ImageSize/constants';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { atom } from 'nanostores';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';

import type { CanvasEntityIdentifier, CanvasV2State, StageAttrs } from './types';
import { RGBA_RED } from './types';

const initialState: CanvasV2State = {
  _version: 3,
  selectedEntityIdentifier: { type: 'inpaint_mask', id: 'inpaint_mask' },
  layers: { entities: [], imageCache: null },
  controlAdapters: { entities: [] },
  ipAdapters: { entities: [] },
  regions: { entities: [] },
  loras: [],
  inpaintMask: {
    id: 'inpaint_mask',
    type: 'inpaint_mask',
    bbox: null,
    bboxNeedsUpdate: false,
    fill: RGBA_RED,
    imageCache: null,
    isEnabled: true,
    objects: [],
    x: 0,
    y: 0,
  },
  tool: {
    selected: 'bbox',
    selectedBuffer: null,
    invertScroll: false,
    fill: RGBA_RED,
    brush: {
      width: 50,
    },
    eraser: {
      width: 50,
    },
  },
  document: {
    width: 512,
    height: 512,
    aspectRatio: deepClone(initialAspectRatioState),
  },
  bbox: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
    scaleMethod: 'auto',
    scaledSize: {
      width: 512,
      height: 512,
    },
  },
  settings: {
    maskOpacity: 0.3,
    // TODO(psyche): These are copied from old canvas state, need to be implemented
    autoSave: false,
    imageSmoothing: true,
    preserveMaskedArea: false,
    showHUD: true,
    clipToBbox: false,
    cropToBboxOnSave: false,
  },
  compositing: {
    maskBlur: 16,
    maskBlurMethod: 'box',
    canvasCoherenceMode: 'Gaussian Blur',
    canvasCoherenceMinDenoise: 0,
    canvasCoherenceEdgeSize: 16,
    infillMethod: 'patchmatch',
    infillTileSize: 32,
    infillPatchmatchDownscaleSize: 1,
    infillColorValue: { r: 0, g: 0, b: 0, a: 1 },
  },
  params: {
    cfgScale: 7.5,
    cfgRescaleMultiplier: 0,
    img2imgStrength: 0.75,
    iterations: 1,
    scheduler: 'euler',
    seed: 0,
    shouldRandomizeSeed: true,
    steps: 50,
    model: null,
    vae: null,
    vaePrecision: 'fp32',
    seamlessXAxis: false,
    seamlessYAxis: false,
    clipSkip: 0,
    shouldUseCpuNoise: true,
    positivePrompt: '',
    negativePrompt: '',
    positivePrompt2: '',
    negativePrompt2: '',
    shouldConcatPrompts: true,
    refinerModel: null,
    refinerSteps: 20,
    refinerCFGScale: 7.5,
    refinerScheduler: 'euler',
    refinerPositiveAestheticScore: 6,
    refinerNegativeAestheticScore: 2.5,
    refinerStart: 0.8,
  },
  stagingArea: null,
};

export const canvasV2Slice = createSlice({
  name: 'canvasV2',
  initialState,
  reducers: {
    ...layersReducers,
    ...ipAdaptersReducers,
    ...controlAdaptersReducers,
    ...regionsReducers,
    ...lorasReducers,
    ...paramsReducers,
    ...compositingReducers,
    ...settingsReducers,
    ...toolReducers,
    ...bboxReducers,
    ...inpaintMaskReducers,
    ...stagingAreaReducers,
    widthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { width, updateAspectRatio, clamp } = action.payload;
      state.document.width = clamp ? Math.max(roundDownToMultiple(width, 8), 64) : width;
      if (updateAspectRatio) {
        state.document.aspectRatio.value = state.document.width / state.document.height;
        state.document.aspectRatio.id = 'Free';
        state.document.aspectRatio.isLocked = false;
      }
    },
    heightChanged: (state, action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { height, updateAspectRatio, clamp } = action.payload;
      state.document.height = clamp ? Math.max(roundDownToMultiple(height, 8), 64) : height;
      if (updateAspectRatio) {
        state.document.aspectRatio.value = state.document.width / state.document.height;
        state.document.aspectRatio.id = 'Free';
        state.document.aspectRatio.isLocked = false;
      }
    },
    aspectRatioChanged: (state, action: PayloadAction<AspectRatioState>) => {
      state.document.aspectRatio = action.payload;
    },
    entitySelected: (state, action: PayloadAction<CanvasEntityIdentifier>) => {
      state.selectedEntityIdentifier = action.payload;
    },
    allEntitiesDeleted: (state) => {
      state.regions.entities = [];
      state.layers.entities = [];
      state.layers.imageCache = null;
      state.ipAdapters.entities = [];
      state.controlAdapters.entities = [];
    },
  },
});

export const {
  widthChanged,
  heightChanged,
  aspectRatioChanged,
  bboxChanged,
  brushWidthChanged,
  eraserWidthChanged,
  fillChanged,
  invertScrollChanged,
  toolChanged,
  toolBufferChanged,
  maskOpacityChanged,
  entitySelected,
  allEntitiesDeleted,
  scaledBboxChanged,
  bboxScaleMethodChanged,
  clipToBboxChanged,
  // layers
  layerAdded,
  layerRecalled,
  layerDeleted,
  layerReset,
  layerMovedForwardOne,
  layerMovedToFront,
  layerMovedBackwardOne,
  layerMovedToBack,
  layerIsEnabledToggled,
  layerOpacityChanged,
  layerTranslated,
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerLinePointAdded,
  layerRectAdded,
  layerImageAdded,
  layerAllDeleted,
  layerImageCacheChanged,
  // IP Adapters
  ipaAdded,
  ipaRecalled,
  ipaIsEnabledToggled,
  ipaDeleted,
  ipaAllDeleted,
  ipaImageChanged,
  ipaMethodChanged,
  ipaModelChanged,
  ipaCLIPVisionModelChanged,
  ipaWeightChanged,
  ipaBeginEndStepPctChanged,
  // Control Adapters
  caAdded,
  caBboxChanged,
  caDeleted,
  caAllDeleted,
  caIsEnabledToggled,
  caMovedBackwardOne,
  caMovedForwardOne,
  caMovedToBack,
  caMovedToFront,
  caOpacityChanged,
  caTranslated,
  caRecalled,
  caImageChanged,
  caProcessedImageChanged,
  caModelChanged,
  caControlModeChanged,
  caProcessorConfigChanged,
  caFilterChanged,
  caProcessorPendingBatchIdChanged,
  caWeightChanged,
  caBeginEndStepPctChanged,
  // Regions
  rgAdded,
  rgRecalled,
  rgReset,
  rgIsEnabledToggled,
  rgTranslated,
  rgBboxChanged,
  rgDeleted,
  rgAllDeleted,
  rgMovedForwardOne,
  rgMovedToFront,
  rgMovedBackwardOne,
  rgMovedToBack,
  rgPositivePromptChanged,
  rgNegativePromptChanged,
  rgFillChanged,
  rgImageCacheChanged,
  rgAutoNegativeChanged,
  rgIPAdapterAdded,
  rgIPAdapterDeleted,
  rgIPAdapterImageChanged,
  rgIPAdapterWeightChanged,
  rgIPAdapterBeginEndStepPctChanged,
  rgIPAdapterMethodChanged,
  rgIPAdapterModelChanged,
  rgIPAdapterCLIPVisionModelChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgLinePointAdded,
  rgRectAdded,
  // Compositing
  setInfillMethod,
  setInfillTileSize,
  setInfillPatchmatchDownscaleSize,
  setInfillColorValue,
  setMaskBlur,
  setCanvasCoherenceMode,
  setCanvasCoherenceEdgeSize,
  setCanvasCoherenceMinDenoise,
  // Parameters
  setIterations,
  setSteps,
  setCfgScale,
  setCfgRescaleMultiplier,
  setScheduler,
  setSeed,
  setImg2imgStrength,
  setSeamlessXAxis,
  setSeamlessYAxis,
  setShouldRandomizeSeed,
  vaeSelected,
  vaePrecisionChanged,
  setClipSkip,
  shouldUseCpuNoiseChanged,
  positivePromptChanged,
  negativePromptChanged,
  positivePrompt2Changed,
  negativePrompt2Changed,
  shouldConcatPromptsChanged,
  refinerModelChanged,
  setRefinerSteps,
  setRefinerCFGScale,
  setRefinerScheduler,
  setRefinerPositiveAestheticScore,
  setRefinerNegativeAestheticScore,
  setRefinerStart,
  modelChanged,
  // LoRAs
  loraAdded,
  loraRecalled,
  loraDeleted,
  loraWeightChanged,
  loraIsEnabledChanged,
  loraAllDeleted,
  // Inpaint mask
  imReset,
  imRecalled,
  imIsEnabledToggled,
  imTranslated,
  imBboxChanged,
  imFillChanged,
  imImageCacheChanged,
  imBrushLineAdded,
  imEraserLineAdded,
  imLinePointAdded,
  imRectAdded,
  // Staging
  stagingAreaInitialized,
  stagingAreaImageAdded,
  stagingAreaBatchIdAdded,
  stagingAreaImageDiscarded,
  stagingAreaImageAccepted,
  stagingAreaReset,
  stagingAreaNextImageSelected,
  stagingAreaPreviousImageSelected,
} = canvasV2Slice.actions;

export const selectCanvasV2Slice = (state: RootState) => state.canvasV2;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

// Ephemeral state that does not need to be in redux
export const $isPreviewVisible = atom(true);
export const $stageAttrs = atom<StageAttrs>({
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  scale: 0,
});
export const $shouldShowStagedImage = atom(true);
export const $lastProgressEvent = atom<InvocationDenoiseProgressEvent | null>(null);

export const canvasV2PersistConfig: PersistConfig<CanvasV2State> = {
  name: canvasV2Slice.name,
  initialState,
  migrate,
  persistDenylist: [],
};
