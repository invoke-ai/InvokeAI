import type { PayloadAction } from '@reduxjs/toolkit';
import { createAction, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { deepClone } from 'common/util/deepClone';
import { bboxReducers } from 'features/controlLayers/store/bboxReducers';
import { compositingReducers } from 'features/controlLayers/store/compositingReducers';
import { controlLayersReducers } from 'features/controlLayers/store/controlLayersReducers';
import { inpaintMaskReducers } from 'features/controlLayers/store/inpaintMaskReducers';
import { ipAdaptersReducers } from 'features/controlLayers/store/ipAdaptersReducers';
import { lorasReducers } from 'features/controlLayers/store/lorasReducers';
import { paramsReducers } from 'features/controlLayers/store/paramsReducers';
import { rasterLayersReducers } from 'features/controlLayers/store/rasterLayersReducers';
import { regionsReducers } from 'features/controlLayers/store/regionsReducers';
import { sessionReducers } from 'features/controlLayers/store/sessionReducers';
import { settingsReducers } from 'features/controlLayers/store/settingsReducers';
import { toolReducers } from 'features/controlLayers/store/toolReducers';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { simplifyFlatNumbersArray } from 'features/controlLayers/util/simplify';
import { initialAspectRatioState } from 'features/parameters/components/DocumentSize/constants';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { pick } from 'lodash-es';
import { atom } from 'nanostores';
import { assert } from 'tsafe';

import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasEntityState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  CanvasV2State,
  Coordinate,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  StageAttrs,
} from './types';
import { getEntityIdentifier, isDrawableEntity } from './types';

const initialState: CanvasV2State = {
  _version: 3,
  selectedEntityIdentifier: null,
  rasterLayers: {
    entities: [],
  },
  controlLayers: {
    entities: [],
  },
  inpaintMasks: {
    entities: [],
  },
  regions: {
    entities: [],
  },
  loras: [],
  ipAdapters: { entities: [] },
  tool: {
    selected: 'view',
    selectedBuffer: null,
    invertScroll: false,
    fill: { r: 31, g: 160, b: 224, a: 1 }, // invokeBlue.500
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
    // TODO(psyche): These are copied from old canvas state, need to be implemented
    autoSave: false,
    imageSmoothing: true,
    preserveMaskedArea: false,
    showHUD: true,
    clipToBbox: false,
    cropToBboxOnSave: false,
    canvasBackgroundStyle: 'checkerboard',
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
};

export function selectEntity(state: CanvasV2State, { id, type }: CanvasEntityIdentifier) {
  switch (type) {
    case 'raster_layer':
      return state.rasterLayers.entities.find((entity) => entity.id === id);
    case 'control_layer':
      return state.controlLayers.entities.find((entity) => entity.id === id);
    case 'inpaint_mask':
      return state.inpaintMasks.entities.find((entity) => entity.id === id);
    case 'regional_guidance':
      return state.regions.entities.find((entity) => entity.id === id);
    case 'ip_adapter':
      return state.ipAdapters.entities.find((entity) => entity.id === id);
    default:
      return;
  }
}

export function selectAllEntitiesOfType(state: CanvasV2State, type: CanvasEntityState['type']): CanvasEntityState[] {
  if (type === 'raster_layer') {
    return state.rasterLayers.entities;
  } else if (type === 'control_layer') {
    return state.controlLayers.entities;
  } else if (type === 'inpaint_mask') {
    return state.inpaintMasks.entities;
  } else if (type === 'regional_guidance') {
    return state.regions.entities;
  } else if (type === 'ip_adapter') {
    return state.ipAdapters.entities;
  } else {
    assert(false, 'Not implemented');
  }
}

export const canvasV2Slice = createSlice({
  name: 'canvasV2',
  initialState,
  reducers: {
    ...rasterLayersReducers,
    ...controlLayersReducers,
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
    entityNameChanged: (state, action: PayloadAction<EntityIdentifierPayload & { name: string | null }>) => {
      const { entityIdentifier, name } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.name = name;
    },
    entityReset: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      } else if (isDrawableEntity(entity)) {
        entity.isEnabled = true;
        entity.objects = [];
        entity.position = { x: 0, y: 0 };
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

      if (isDrawableEntity(entity)) {
        entity.position = position;
      }
    },
    entityRasterized: (state, action: PayloadAction<EntityRasterizedPayload>) => {
      const { entityIdentifier, imageObject, rect, replaceObjects } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (isDrawableEntity(entity)) {
        if (replaceObjects) {
          entity.objects = [imageObject];
          entity.position = { x: rect.x, y: rect.y };
        }
      }
    },
    entityBrushLineAdded: (state, action: PayloadAction<EntityBrushLineAddedPayload>) => {
      const { entityIdentifier, brushLine } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (!isDrawableEntity(entity)) {
        assert(false, `Cannot add a brush line to a non-drawable entity of type ${entity.type}`);
      }

      entity.objects.push({ ...brushLine, points: simplifyFlatNumbersArray(brushLine.points) });
    },
    entityEraserLineAdded: (state, action: PayloadAction<EntityEraserLineAddedPayload>) => {
      const { entityIdentifier, eraserLine } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (!isDrawableEntity(entity)) {
        assert(false, `Cannot add a eraser line to a non-drawable entity of type ${entity.type}`);
      }

      entity.objects.push({ ...eraserLine, points: simplifyFlatNumbersArray(eraserLine.points) });
    },
    entityRectAdded: (state, action: PayloadAction<EntityRectAddedPayload>) => {
      const { entityIdentifier, rect } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (!isDrawableEntity(entity)) {
        assert(false, `Cannot add a rect to a non-drawable entity of type ${entity.type}`);
      }

      entity.objects.push(rect);
    },
    entityDeleted: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;

      const firstInpaintMaskEntity = state.inpaintMasks.entities[0];

      let selectedEntityIdentifier: CanvasV2State['selectedEntityIdentifier'] = firstInpaintMaskEntity
        ? getEntityIdentifier(firstInpaintMaskEntity)
        : null;

      if (entityIdentifier.type === 'raster_layer') {
        const index = state.rasterLayers.entities.findIndex((layer) => layer.id === entityIdentifier.id);
        state.rasterLayers.entities = state.rasterLayers.entities.filter((layer) => layer.id !== entityIdentifier.id);
        const nextRasterLayer = state.rasterLayers.entities[index];
        if (nextRasterLayer) {
          selectedEntityIdentifier = { type: nextRasterLayer.type, id: nextRasterLayer.id };
        }
      } else if (entityIdentifier.type === 'control_layer') {
        const index = state.controlLayers.entities.findIndex((layer) => layer.id === entityIdentifier.id);
        state.controlLayers.entities = state.controlLayers.entities.filter((rg) => rg.id !== entityIdentifier.id);
        const nextControlLayer = state.controlLayers.entities[index];
        if (nextControlLayer) {
          selectedEntityIdentifier = { type: nextControlLayer.type, id: nextControlLayer.id };
        }
      } else if (entityIdentifier.type === 'regional_guidance') {
        const index = state.regions.entities.findIndex((layer) => layer.id === entityIdentifier.id);
        state.regions.entities = state.regions.entities.filter((rg) => rg.id !== entityIdentifier.id);
        const region = state.regions.entities[index];
        if (region) {
          selectedEntityIdentifier = { type: region.type, id: region.id };
        }
      } else if (entityIdentifier.type === 'ip_adapter') {
        const index = state.ipAdapters.entities.findIndex((layer) => layer.id === entityIdentifier.id);
        state.ipAdapters.entities = state.ipAdapters.entities.filter((rg) => rg.id !== entityIdentifier.id);
        const entity = state.ipAdapters.entities[index];
        if (entity) {
          selectedEntityIdentifier = { type: entity.type, id: entity.id };
        }
      } else if (entityIdentifier.type === 'inpaint_mask') {
        const index = state.inpaintMasks.entities.findIndex((layer) => layer.id === entityIdentifier.id);
        state.inpaintMasks.entities = state.inpaintMasks.entities.filter((rg) => rg.id !== entityIdentifier.id);
        const entity = state.inpaintMasks.entities[index];
        if (entity) {
          selectedEntityIdentifier = { type: entity.type, id: entity.id };
        }
      } else {
        assert(false, 'Not implemented');
      }

      state.selectedEntityIdentifier = selectedEntityIdentifier;
    },
    entityArrangedForwardOne: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'raster_layer') {
        moveOneToEnd(state.rasterLayers.entities, entity);
      } else if (entity.type === 'control_layer') {
        moveOneToEnd(state.controlLayers.entities, entity);
      } else if (entity.type === 'regional_guidance') {
        moveOneToEnd(state.regions.entities, entity);
      } else if (entity.type === 'inpaint_mask') {
        moveOneToEnd(state.inpaintMasks.entities, entity);
      }
    },
    entityArrangedToFront: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'raster_layer') {
        moveToEnd(state.rasterLayers.entities, entity);
      } else if (entity.type === 'control_layer') {
        moveToEnd(state.controlLayers.entities, entity);
      } else if (entity.type === 'regional_guidance') {
        moveToEnd(state.regions.entities, entity);
      } else if (entity.type === 'inpaint_mask') {
        moveToEnd(state.inpaintMasks.entities, entity);
      }
    },
    entityArrangedBackwardOne: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'raster_layer') {
        moveOneToStart(state.rasterLayers.entities, entity);
      } else if (entity.type === 'control_layer') {
        moveOneToStart(state.controlLayers.entities, entity);
      } else if (entity.type === 'regional_guidance') {
        moveOneToStart(state.regions.entities, entity);
      } else if (entity.type === 'inpaint_mask') {
        moveOneToStart(state.inpaintMasks.entities, entity);
      }
    },
    entityArrangedToBack: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'raster_layer') {
        moveToStart(state.rasterLayers.entities, entity);
      } else if (entity.type === 'control_layer') {
        moveToStart(state.controlLayers.entities, entity);
      } else if (entity.type === 'regional_guidance') {
        moveToStart(state.regions.entities, entity);
      } else if (entity.type === 'inpaint_mask') {
        moveToStart(state.inpaintMasks.entities, entity);
      }
    },
    entityOpacityChanged: (state, action: PayloadAction<EntityIdentifierPayload<{ opacity: number }>>) => {
      const { entityIdentifier, opacity } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'ip_adapter') {
        return;
      }
      entity.opacity = opacity;
    },
    allEntitiesOfTypeToggled: (state, action: PayloadAction<{ type: CanvasEntityIdentifier['type'] }>) => {
      const { type } = action.payload;
      let entities: (
        | CanvasRasterLayerState
        | CanvasControlLayerState
        | CanvasInpaintMaskState
        | CanvasRegionalGuidanceState
      )[];

      switch (type) {
        case 'raster_layer':
          entities = state.rasterLayers.entities;
          break;
        case 'control_layer':
          entities = state.controlLayers.entities;
          break;
        case 'inpaint_mask':
          entities = state.inpaintMasks.entities;
          break;
        case 'regional_guidance':
          entities = state.regions.entities;
          break;
        default:
          assert(false, 'Not implemented');
      }

      const allEnabled = entities.every((entity) => entity.isEnabled);
      for (const entity of entities) {
        entity.isEnabled = !allEnabled;
      }
    },
    allEntitiesDeleted: (state) => {
      state.ipAdapters = deepClone(initialState.ipAdapters);
      state.rasterLayers = deepClone(initialState.rasterLayers);
      state.controlLayers = deepClone(initialState.controlLayers);
      state.regions = deepClone(initialState.regions);
      state.inpaintMasks = deepClone(initialState.inpaintMasks);

      state.selectedEntityIdentifier = deepClone(initialState.selectedEntityIdentifier);
    },
    canvasReset: (state) => {
      state.bbox = deepClone(initialState.bbox);
      const optimalDimension = getOptimalDimension(state.params.model);
      state.bbox.rect.width = optimalDimension;
      state.bbox.rect.height = optimalDimension;
      const size = pick(state.bbox.rect, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, optimalDimension);
      state.session = deepClone(initialState.session);
      state.tool = deepClone(initialState.tool);

      state.ipAdapters = deepClone(initialState.ipAdapters);
      state.rasterLayers = deepClone(initialState.rasterLayers);
      state.controlLayers = deepClone(initialState.controlLayers);
      state.regions = deepClone(initialState.regions);
      state.inpaintMasks = deepClone(initialState.inpaintMasks);

      state.selectedEntityIdentifier = deepClone(initialState.selectedEntityIdentifier);
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
  allEntitiesDeleted,
  clipToBboxChanged,
  canvasReset,
  canvasBackgroundStyleChanged,
  // All entities
  entitySelected,
  entityNameChanged,
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
  entityOpacityChanged,
  allEntitiesOfTypeToggled,
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
  // Raster layers
  rasterLayerAdded,
  rasterLayerRecalled,
  rasterLayerAllDeleted,
  rasterLayerConvertedToControlLayer,
  // Control layers
  controlLayerAdded,
  controlLayerRecalled,
  controlLayerAllDeleted,
  controlLayerConvertedToRasterLayer,
  controlLayerModelChanged,
  controlLayerControlModeChanged,
  controlLayerWeightChanged,
  controlLayerBeginEndStepPctChanged,
  controlLayerWithTransparencyEffectToggled,
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
  rgFillColorChanged,
  rgFillStyleChanged,
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
  inpaintMaskAdded,
  inpaintMaskRecalled,
  inpaintMaskFillColorChanged,
  inpaintMaskFillStyleChanged,
  // Staging
  sessionStartedStaging,
  sessionImageStaged,
  sessionStagedImageDiscarded,
  sessionStagingAreaReset,
  sessionNextStagedImageSelected,
  sessionPrevStagedImageSelected,
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
export const $isDrawing = atom<boolean>(false);
export const $isMouseDown = atom<boolean>(false);
export const $lastAddedPoint = atom<Coordinate | null>(null);
export const $lastMouseDownPos = atom<Coordinate | null>(null);
export const $lastCursorPos = atom<Coordinate | null>(null);
export const $spaceKey = atom<boolean>(false);
export const $transformingEntity = atom<CanvasEntityIdentifier | null>(null);

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
