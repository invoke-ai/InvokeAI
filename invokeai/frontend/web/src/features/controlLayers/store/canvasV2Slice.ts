import type { PayloadAction } from '@reduxjs/toolkit';
import { createAction, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { deepClone } from 'common/util/deepClone';
import { bboxReducers } from 'features/controlLayers/store/bboxReducers';
import { compositingReducers } from 'features/controlLayers/store/compositingReducers';
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
import { isEqual, pick } from 'lodash-es';
import { atom } from 'nanostores';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';
import { assert } from 'tsafe';

import type {
  CanvasEntityIdentifier,
  CanvasInpaintMaskState,
  CanvasLayerState,
  CanvasRegionalGuidanceState,
  CanvasV2State,
  Coordinate,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  FilterConfig,
  StageAttrs,
} from './types';
import { IMAGE_FILTERS, RGBA_RED } from './types';

const initialState: CanvasV2State = {
  _version: 3,
  selectedEntityIdentifier: null,
  layers: { entities: [], compositeRasterizationCache: [] },
  ipAdapters: { entities: [] },
  regions: { entities: [] },
  loras: [],
  inpaintMask: {
    id: 'inpaint_mask',
    type: 'inpaint_mask',
    fill: RGBA_RED,
    rasterizationCache: [],
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
    isStaging: false,
    stagedImages: [],
    selectedStagedImageIndex: 0,
  },
  filter: {
    autoProcess: true,
    config: IMAGE_FILTERS.canny_image_processor.buildDefaults(),
  },
};

export function selectEntity(state: CanvasV2State, { id, type }: CanvasEntityIdentifier) {
  switch (type) {
    case 'layer':
      return state.layers.entities.find((layer) => layer.id === id);
    case 'inpaint_mask':
      return state.inpaintMask;
    case 'regional_guidance':
      return state.regions.entities.find((rg) => rg.id === id);
    case 'ip_adapter':
      return state.ipAdapters.entities.find((ip) => ip.id === id);
    default:
      return;
  }
}

const invalidateCompositeRasterizationCache = (entity: CanvasLayerState, state: CanvasV2State) => {
  if (entity.controlAdapter === null) {
    state.layers.compositeRasterizationCache = [];
  }
};

const invalidateRasterizationCaches = (
  entity: CanvasLayerState | CanvasInpaintMaskState | CanvasRegionalGuidanceState,
  state: CanvasV2State
) => {
  // TODO(psyche): We can be more efficient and only invalidate caches when the entity's changes intersect with the
  // cached rect.

  // Reset the entity's rasterization cache
  entity.rasterizationCache = [];

  // When an individual layer has its cache reset, we must also reset the composite rasterization cache because the
  // layer's image data will contribute to the composite layer's image data.
  // If the layer is used as a control layer, it will not contribute to the composite layer, so we do not need to reset
  // its cache.
  if (entity.type === 'layer') {
    invalidateCompositeRasterizationCache(entity, state);
  }
};

export const canvasV2Slice = createSlice({
  name: 'canvasV2',
  initialState,
  reducers: {
    ...layersReducers,
    ...ipAdaptersReducers,
    ...regionsReducers,
    ...lorasReducers,
    ...paramsReducers,
    ...compositingReducers,
    ...settingsReducers,
    ...toolReducers,
    ...bboxReducers,
    ...inpaintMaskReducers,
    ...sessionReducers,
    entitySelected: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      state.selectedEntityIdentifier = entityIdentifier;
    },
    entityReset: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      } else if (entity.type === 'layer' || entity.type === 'inpaint_mask' || entity.type === 'regional_guidance') {
        entity.isEnabled = true;
        entity.objects = [];
        entity.position = { x: 0, y: 0 };
        invalidateRasterizationCaches(entity, state);
      } else {
        assert(false, 'Not implemented');
      }
    },
    entityIsEnabledToggled: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.isEnabled = !entity.isEnabled;
    },
    entityMoved: (state, action: PayloadAction<EntityMovedPayload>) => {
      const { entityIdentifier, position } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (entity.type === 'layer' || entity.type === 'inpaint_mask' || entity.type === 'regional_guidance') {
        entity.position = position;
        // When an entity is moved, we need to invalidate the rasterization caches.
        invalidateRasterizationCaches(entity, state);
      }
    },
    entityRasterized: (state, action: PayloadAction<EntityRasterizedPayload>) => {
      const { entityIdentifier, imageObject, rect } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (entity.type === 'layer' || entity.type === 'inpaint_mask' || entity.type === 'regional_guidance') {
        entity.objects = [imageObject];
        entity.position = { x: rect.x, y: rect.y };
        // Remove the cache for the given rect. This should never happen, because we should never rasterize the same
        // rect twice. Just in case, we remove the old cache.
        entity.rasterizationCache = entity.rasterizationCache.filter((cache) => !isEqual(cache.rect, rect));
        entity.rasterizationCache.push({ imageName: imageObject.image.image_name, rect });
      }
    },
    entityBrushLineAdded: (state, action: PayloadAction<EntityBrushLineAddedPayload>) => {
      const { entityIdentifier, brushLine } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (entity.type === 'layer' || entity.type === 'inpaint_mask' || entity.type === 'regional_guidance') {
        entity.objects.push(brushLine);
        // When adding a brush line, we need to invalidate the rasterization caches.
        invalidateRasterizationCaches(entity, state);
      }
    },
    entityEraserLineAdded: (state, action: PayloadAction<EntityEraserLineAddedPayload>) => {
      const { entityIdentifier, eraserLine } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      } else if (entity.type === 'layer' || entity.type === 'inpaint_mask' || entity.type === 'regional_guidance') {
        entity.objects.push(eraserLine);
        // When adding an eraser line, we need to invalidate the rasterization caches.
        invalidateRasterizationCaches(entity, state);
      } else {
        assert(false, 'Not implemented');
      }
    },
    entityRectAdded: (state, action: PayloadAction<EntityRectAddedPayload>) => {
      const { entityIdentifier, rect } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      } else if (entity.type === 'layer') {
        entity.objects.push(rect);
        // When adding an eraser line, we need to invalidate the rasterization caches.
        invalidateRasterizationCaches(entity, state);
      } else {
        assert(false, 'Not implemented');
      }
    },
    entityDeleted: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (entity?.type === 'layer') {
        // When a layer is deleted, we may need to invalidate the composite rasterization cache.
        invalidateCompositeRasterizationCache(entity, state);
      }
      if (entityIdentifier.type === 'layer') {
        state.layers.entities = state.layers.entities.filter((layer) => layer.id !== entityIdentifier.id);
      } else if (entityIdentifier.type === 'regional_guidance') {
        state.regions.entities = state.regions.entities.filter((rg) => rg.id !== entityIdentifier.id);
      } else {
        assert(false, 'Not implemented');
      }
    },
    entityArrangedForwardOne: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'layer') {
        moveOneToEnd(state.layers.entities, entity);
        // When arranging an entity, we may need to invalidate the composite rasterization cache.
        invalidateCompositeRasterizationCache(entity, state);
      } else if (entity.type === 'regional_guidance') {
        moveOneToEnd(state.regions.entities, entity);
      }
    },
    entityArrangedToFront: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'layer') {
        moveToEnd(state.layers.entities, entity);
        // When arranging an entity, we may need to invalidate the composite rasterization cache.
        invalidateCompositeRasterizationCache(entity, state);
      } else if (entity.type === 'regional_guidance') {
        moveToEnd(state.regions.entities, entity);
      }
    },
    entityArrangedBackwardOne: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'layer') {
        moveOneToStart(state.layers.entities, entity);
        // When arranging an entity, we may need to invalidate the composite rasterization cache.
        invalidateCompositeRasterizationCache(entity, state);
      } else if (entity.type === 'regional_guidance') {
        moveOneToStart(state.regions.entities, entity);
      }
    },
    entityArrangedToBack: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'layer') {
        moveToStart(state.layers.entities, entity);
        // When arranging an entity, we may need to invalidate the composite rasterization cache.
        invalidateCompositeRasterizationCache(entity, state);
      } else if (entity.type === 'regional_guidance') {
        moveToStart(state.regions.entities, entity);
      }
    },
    allEntitiesDeleted: (state) => {
      state.regions.entities = [];
      state.layers.entities = [];
      state.layers.compositeRasterizationCache = [];
      state.ipAdapters.entities = [];
    },
    filterSelected: (state, action: PayloadAction<{ type: FilterConfig['type'] }>) => {
      state.filter.config = IMAGE_FILTERS[action.payload.type].buildDefaults();
    },
    filterConfigChanged: (state, action: PayloadAction<{ config: FilterConfig }>) => {
      state.filter.config = action.payload.config;
    },
    rasterizationCachesInvalidated: (state) => {
      // Invalidate the rasterization caches for all entities.

      // Layers & composite layer
      state.layers.compositeRasterizationCache = [];
      for (const layer of state.layers.entities) {
        layer.rasterizationCache = [];
      }

      // Regions
      for (const region of state.regions.entities) {
        region.rasterizationCache = [];
      }

      // Inpaint mask
      state.inpaintMask.rasterizationCache = [];
    },
    canvasReset: (state) => {
      state.bbox = deepClone(initialState.bbox);
      const optimalDimension = getOptimalDimension(state.params.model);
      state.bbox.rect.width = optimalDimension;
      state.bbox.rect.height = optimalDimension;
      const size = pick(state.bbox.rect, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, optimalDimension);

      state.ipAdapters = deepClone(initialState.ipAdapters);
      state.layers = deepClone(initialState.layers);
      state.regions = deepClone(initialState.regions);
      state.selectedEntityIdentifier = deepClone(initialState.selectedEntityIdentifier);
      state.session = deepClone(initialState.session);
      state.tool = deepClone(initialState.tool);
      state.inpaintMask = deepClone(initialState.inpaintMask);
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
  allEntitiesDeleted,
  clipToBboxChanged,
  canvasReset,
  rasterizationCachesInvalidated,
  // All entities
  entitySelected,
  entityReset,
  entityIsEnabledToggled,
  entityMoved,
  entityRasterized,
  entityBrushLineAdded,
  entityEraserLineAdded,
  entityRectAdded,
  entityDeleted,
  entityArrangedForwardOne,
  entityArrangedToFront,
  entityArrangedBackwardOne,
  entityArrangedToBack,
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
  layerAllDeleted,
  layerUsedAsControlChanged,
  layerControlAdapterModelChanged,
  layerControlAdapterControlModeChanged,
  layerControlAdapterWeightChanged,
  layerControlAdapterBeginEndStepPctChanged,
  layerCompositeRasterized,
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
  // Regions
  rgAdded,
  rgRecalled,
  rgAllDeleted,
  rgPositivePromptChanged,
  rgNegativePromptChanged,
  rgFillChanged,
  rgAutoNegativeChanged,
  rgIPAdapterAdded,
  rgIPAdapterDeleted,
  rgIPAdapterImageChanged,
  rgIPAdapterWeightChanged,
  rgIPAdapterBeginEndStepPctChanged,
  rgIPAdapterMethodChanged,
  rgIPAdapterModelChanged,
  rgIPAdapterCLIPVisionModelChanged,
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
  imRecalled,
  imFillChanged,
  // Staging
  sessionStartedStaging,
  sessionImageStaged,
  sessionStagedImageDiscarded,
  sessionStagingAreaReset,
  sessionNextStagedImageSelected,
  sessionPrevStagedImageSelected,
  // Filter
  filterSelected,
  filterConfigChanged,
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
export const $isDrawing = atom<boolean>(false);
export const $isMouseDown = atom<boolean>(false);
export const $lastAddedPoint = atom<Coordinate | null>(null);
export const $lastMouseDownPos = atom<Coordinate | null>(null);
export const $lastCursorPos = atom<Coordinate | null>(null);
export const $spaceKey = atom<boolean>(false);
export const $transformingEntity = atom<CanvasEntityIdentifier | null>(null);
export const $filteringEntity = atom<CanvasEntityIdentifier | null>(null);

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
