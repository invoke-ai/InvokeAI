import type { PayloadAction } from '@reduxjs/toolkit';
import { createAction, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { bboxReducers } from 'features/controlLayers/store/bboxReducers';
import { compositingReducers } from 'features/controlLayers/store/compositingReducers';
import { controlAdaptersReducers } from 'features/controlLayers/store/controlAdaptersReducers';
import { initialImageReducers } from 'features/controlLayers/store/initialImageReducers';
import { inpaintMaskReducers } from 'features/controlLayers/store/inpaintMaskReducers';
import { ipAdaptersReducers } from 'features/controlLayers/store/ipAdaptersReducers';
import { layersReducers } from 'features/controlLayers/store/layersReducers';
import { lorasReducers } from 'features/controlLayers/store/lorasReducers';
import { paramsReducers } from 'features/controlLayers/store/paramsReducers';
import { regionsReducers } from 'features/controlLayers/store/regionsReducers';
import { sessionReducers } from 'features/controlLayers/store/sessionReducers';
import { settingsReducers } from 'features/controlLayers/store/settingsReducers';
import { toolReducers } from 'features/controlLayers/store/toolReducers';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { initialAspectRatioState } from 'features/parameters/components/DocumentSize/constants';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { pick } from 'lodash-es';
import { atom } from 'nanostores';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';

import type { CanvasEntityIdentifier, CanvasV2State, Coordinate, StageAttrs } from './types';
import { RGBA_RED } from './types';

const initialState: CanvasV2State = {
  _version: 3,
  selectedEntityIdentifier: null,
  layers: { entities: [], imageCache: null },
  controlAdapters: { entities: [] },
  ipAdapters: { entities: [] },
  regions: { entities: [] },
  loras: [],
  initialImage: {
    id: 'initial_image',
    type: 'initial_image',
    bbox: null,
    bboxNeedsUpdate: false,
    isEnabled: true,
    imageObject: null,
  },
  inpaintMask: {
    id: 'inpaint_mask',
    type: 'inpaint_mask',
    bbox: null,
    bboxNeedsUpdate: false,
    fill: RGBA_RED,
    imageCache: null,
    isEnabled: true,
    objects: [],
    position: {
      x: 0,
      y: 0,
    },
  },
  tool: {
    selected: 'view',
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
  bbox: {
    rect: { x: 0, y: 0, width: 512, height: 512 },
    aspectRatio: deepClone(initialAspectRatioState),
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
  session: {
    isActive: false,
    isStaging: false,
    stagedImages: [],
    selectedStagedImageIndex: 0,
  },
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
    ...sessionReducers,
    ...initialImageReducers,
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
    canvasReset: (state) => {
      state.bbox = deepClone(initialState.bbox);
      const optimalDimension = getOptimalDimension(state.params.model);
      state.bbox.rect.width = optimalDimension;
      state.bbox.rect.height = optimalDimension;
      const size = pick(state.bbox.rect, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, optimalDimension);

      state.controlAdapters = deepClone(initialState.controlAdapters);
      state.ipAdapters = deepClone(initialState.ipAdapters);
      state.layers = deepClone(initialState.layers);
      state.regions = deepClone(initialState.regions);
      state.selectedEntityIdentifier = deepClone(initialState.selectedEntityIdentifier);
      state.session = deepClone(initialState.session);
      state.tool = deepClone(initialState.tool);
      state.inpaintMask = deepClone(initialState.inpaintMask);
      state.initialImage = deepClone(initialState.initialImage);
    },
  },
});

export const {
  brushWidthChanged,
  eraserWidthChanged,
  fillChanged,
  invertScrollChanged,
  toolChanged,
  toolBufferChanged,
  maskOpacityChanged,
  entitySelected,
  allEntitiesDeleted,
  clipToBboxChanged,
  canvasReset,
  // bbox
  bboxChanged,
  bboxScaledSizeChanged,
  bboxScaleMethodChanged,
  bboxWidthChanged,
  bboxHeightChanged,
  bboxAspectRatioLockToggled,
  bboxAspectRatioIdChanged,
  bboxDimensionsSwapped,
  bboxSizeOptimized,
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
  layerImageAdded,
  layerAllDeleted,
  layerImageCacheChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerRectShapeAdded,
  layerRasterized,
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
  caScaled,
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
  rgScaled,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgRectShapeAdded,
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
  imScaled,
  imBrushLineAdded,
  imEraserLineAdded,
  imRectShapeAdded,
  // Staging
  sessionStarted,
  sessionStartedStaging,
  sessionImageStaged,
  sessionStagedImageDiscarded,
  sessionStagingAreaReset,
  sessionNextStagedImageSelected,
  sessionPrevStagedImageSelected,
  // Initial image
  iiRecalled,
  iiIsEnabledToggled,
  iiReset,
  iiImageChanged,
} = canvasV2Slice.actions;

export const selectCanvasV2Slice = (state: RootState) => state.canvasV2;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

// Ephemeral state that does not need to be in redux
export const $isPreviewVisible = atom(true);
export const $stageAttrs = atom<StageAttrs>({
  position: { x: 0, y: 0 },
  dimensions: { width: 0, height: 0 },
  scale: 0,
});
export const $shouldShowStagedImage = atom(true);
export const $lastProgressEvent = atom<InvocationDenoiseProgressEvent | null>(null);
export const $isDrawing = atom<boolean>(false);
export const $isMouseDown = atom<boolean>(false);
export const $lastAddedPoint = atom<Coordinate | null>(null);
export const $lastMouseDownPos = atom<Coordinate | null>(null);
export const $lastCursorPos = atom<Coordinate | null>(null);
export const $spaceKey = atom<boolean>(false);

export const canvasV2PersistConfig: PersistConfig<CanvasV2State> = {
  name: canvasV2Slice.name,
  initialState,
  migrate,
  persistDenylist: [],
};

export const sessionRequested = createAction(`${canvasV2Slice.name}/sessionRequested`);
export const sessionStagingAreaImageAccepted = createAction<{ index: number }>(
  `${canvasV2Slice.name}/sessionStagingAreaImageAccepted`
);
export const transformationApplied = createAction<CanvasEntityIdentifier>(
  `${canvasV2Slice.name}/transformationApplied`
);
