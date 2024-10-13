import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig } from 'app/store/store';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { canvasReset, newSessionRequested } from 'features/controlLayers/store/actions';
import { modelChanged } from 'features/controlLayers/store/paramsSlice';
import {
  selectAllEntities,
  selectAllEntitiesOfType,
  selectEntity,
  selectRegionalGuidanceReferenceImage,
} from 'features/controlLayers/store/selectors';
import type {
  CanvasInpaintMaskState,
  CanvasMetadata,
  FillStyle,
  RegionalGuidanceReferenceImageState,
  RgbColor,
} from 'features/controlLayers/store/types';
import {
  calculateNewSize,
  getScaledBoundingBoxDimensions,
} from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { simplifyFlatNumbersArray } from 'features/controlLayers/util/simplify';
import { isMainModelBase, zModelIdentifierField } from 'features/nodes/types/common';
import { ASPECT_RATIO_MAP } from 'features/parameters/components/Bbox/constants';
import { getGridSize, getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect } from 'konva/lib/types';
import { merge, omit } from 'lodash-es';
import type { UndoableOptions } from 'redux-undo';
import type { ControlNetModelConfig, ImageDTO, IPAdapterModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import type {
  AspectRatioID,
  BoundingBoxScaleMethod,
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasRasterLayerState,
  CanvasReferenceImageState,
  CanvasRegionalGuidanceState,
  CanvasState,
  CLIPVisionModelV2,
  ControlModeV2,
  ControlNetConfig,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  IPMethodV2,
  T2IAdapterConfig,
} from './types';
import { getEntityIdentifier, isRenderableEntity } from './types';
import {
  getControlLayerState,
  getInpaintMaskState,
  getRasterLayerState,
  getReferenceImageState,
  getRegionalGuidanceState,
  imageDTOToImageWithDims,
  initialControlNet,
  initialIPAdapter,
} from './util';

const getInitialState = (): CanvasState => {
  const initialInpaintMaskState = getInpaintMaskState(getPrefixedId('inpaint_mask'));
  const initialState: CanvasState = {
    _version: 3,
    selectedEntityIdentifier: getEntityIdentifier(initialInpaintMaskState),
    bookmarkedEntityIdentifier: getEntityIdentifier(initialInpaintMaskState),
    rasterLayers: {
      isHidden: false,
      entities: [],
    },
    controlLayers: {
      isHidden: false,
      entities: [],
    },
    inpaintMasks: {
      isHidden: false,
      entities: [initialInpaintMaskState],
    },
    regionalGuidance: {
      isHidden: false,
      entities: [],
    },
    referenceImages: { entities: [] },
    bbox: {
      rect: { x: 0, y: 0, width: 512, height: 512 },
      aspectRatio: {
        id: '1:1',
        value: 1,
        isLocked: false,
      },
      scaleMethod: 'auto',
      scaledSize: {
        width: 512,
        height: 512,
      },
      modelBase: 'sd-1',
    },
  };
  return initialState;
};

const initialState = getInitialState();

export const canvasSlice = createSlice({
  name: 'canvas',
  initialState,
  reducers: {
    // undoable canvas state
    //#region Raster layers
    rasterLayerAdded: {
      reducer: (
        state,
        action: PayloadAction<{
          id: string;
          overrides?: Partial<CanvasRasterLayerState>;
          isSelected?: boolean;
          isMergingVisible?: boolean;
        }>
      ) => {
        const { id, overrides, isSelected, isMergingVisible } = action.payload;
        const entityState = getRasterLayerState(id, overrides);

        if (isMergingVisible) {
          // When merging visible, we delete all disabled layers
          state.rasterLayers.entities = state.rasterLayers.entities.filter((layer) => !layer.isEnabled);
        }

        state.rasterLayers.entities.push(entityState);

        if (isSelected) {
          state.selectedEntityIdentifier = getEntityIdentifier(entityState);
        }
      },
      prepare: (payload: {
        overrides?: Partial<CanvasRasterLayerState>;
        isSelected?: boolean;
        isMergingVisible?: boolean;
      }) => ({
        payload: { ...payload, id: getPrefixedId('raster_layer') },
      }),
    },
    rasterLayerRecalled: (state, action: PayloadAction<{ data: CanvasRasterLayerState }>) => {
      const { data } = action.payload;
      state.rasterLayers.entities.push(data);
      state.selectedEntityIdentifier = getEntityIdentifier(data);
    },
    rasterLayerConvertedToControlLayer: {
      reducer: (
        state,
        action: PayloadAction<
          EntityIdentifierPayload<{ newId: string; overrides?: Partial<CanvasControlLayerState> }, 'raster_layer'>
        >
      ) => {
        const { entityIdentifier, newId, overrides } = action.payload;
        const layer = selectEntity(state, entityIdentifier);
        if (!layer) {
          return;
        }

        // Convert the raster layer to control layer
        const controlLayerState: CanvasControlLayerState = {
          ...deepClone(layer),
          id: newId,
          type: 'control_layer',
          controlAdapter: deepClone(initialControlNet),
          withTransparencyEffect: true,
        };

        merge(controlLayerState, overrides);

        // Remove the raster layer
        state.rasterLayers.entities = state.rasterLayers.entities.filter((layer) => layer.id !== entityIdentifier.id);

        // Add the converted control layer
        state.controlLayers.entities.push(controlLayerState);

        state.selectedEntityIdentifier = { type: controlLayerState.type, id: controlLayerState.id };
      },
      prepare: (
        payload: EntityIdentifierPayload<{ overrides?: Partial<CanvasControlLayerState> } | undefined, 'raster_layer'>
      ) => ({
        payload: { ...payload, newId: getPrefixedId('control_layer') },
      }),
    },
    //#region Control layers
    controlLayerAdded: {
      reducer: (
        state,
        action: PayloadAction<{ id: string; overrides?: Partial<CanvasControlLayerState>; isSelected?: boolean }>
      ) => {
        const { id, overrides, isSelected } = action.payload;

        const entityState = getControlLayerState(id, overrides);

        state.controlLayers.entities.push(entityState);

        if (isSelected) {
          state.selectedEntityIdentifier = getEntityIdentifier(entityState);
        }
      },
      prepare: (payload: { overrides?: Partial<CanvasControlLayerState>; isSelected?: boolean }) => ({
        payload: { ...payload, id: getPrefixedId('control_layer') },
      }),
    },
    controlLayerRecalled: (state, action: PayloadAction<{ data: CanvasControlLayerState }>) => {
      const { data } = action.payload;
      state.controlLayers.entities.push(data);
      state.selectedEntityIdentifier = { type: 'control_layer', id: data.id };
    },
    controlLayerConvertedToRasterLayer: {
      reducer: (state, action: PayloadAction<EntityIdentifierPayload<{ newId: string }, 'control_layer'>>) => {
        const { entityIdentifier, newId } = action.payload;
        const layer = selectEntity(state, entityIdentifier);
        if (!layer) {
          return;
        }

        // Convert the raster layer to control layer
        const rasterLayerState: CanvasRasterLayerState = {
          ...omit(deepClone(layer), ['type', 'controlAdapter', 'withTransparencyEffect']),
          id: newId,
          type: 'raster_layer',
        };

        // Remove the control layer
        state.controlLayers.entities = state.controlLayers.entities.filter((layer) => layer.id !== entityIdentifier.id);

        // Add the new raster layer
        state.rasterLayers.entities.push(rasterLayerState);

        state.selectedEntityIdentifier = { type: rasterLayerState.type, id: rasterLayerState.id };
      },
      prepare: (payload: EntityIdentifierPayload<void, 'control_layer'>) => ({
        payload: { ...payload, newId: getPrefixedId('raster_layer') },
      }),
    },
    controlLayerModelChanged: (
      state,
      action: PayloadAction<
        EntityIdentifierPayload<
          {
            modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null;
          },
          'control_layer'
        >
      >
    ) => {
      const { entityIdentifier, modelConfig } = action.payload;
      const layer = selectEntity(state, entityIdentifier);
      if (!layer || !layer.controlAdapter) {
        return;
      }
      if (!modelConfig) {
        layer.controlAdapter.model = null;
        return;
      }
      layer.controlAdapter.model = zModelIdentifierField.parse(modelConfig);

      // We may need to convert the CA to match the model
      if (layer.controlAdapter.type === 't2i_adapter' && layer.controlAdapter.model.type === 'controlnet') {
        // Converting from T2I Adapter to ControlNet - add `controlMode`
        const controlNetConfig: ControlNetConfig = {
          ...layer.controlAdapter,
          type: 'controlnet',
          controlMode: 'balanced',
        };
        layer.controlAdapter = controlNetConfig;
      } else if (layer.controlAdapter.type === 'controlnet' && layer.controlAdapter.model.type === 't2i_adapter') {
        // Converting from ControlNet to T2I Adapter - remove `controlMode`
        const { controlMode: _, ...rest } = layer.controlAdapter;
        const t2iAdapterConfig: T2IAdapterConfig = { ...rest, type: 't2i_adapter' };
        layer.controlAdapter = t2iAdapterConfig;
      }
    },
    controlLayerControlModeChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ controlMode: ControlModeV2 }, 'control_layer'>>
    ) => {
      const { entityIdentifier, controlMode } = action.payload;
      const layer = selectEntity(state, entityIdentifier);
      if (!layer || !layer.controlAdapter || layer.controlAdapter.type !== 'controlnet') {
        return;
      }
      layer.controlAdapter.controlMode = controlMode;
    },
    controlLayerWeightChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ weight: number }, 'control_layer'>>
    ) => {
      const { entityIdentifier, weight } = action.payload;
      const layer = selectEntity(state, entityIdentifier);
      if (!layer || !layer.controlAdapter) {
        return;
      }
      layer.controlAdapter.weight = weight;
    },
    controlLayerBeginEndStepPctChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ beginEndStepPct: [number, number] }, 'control_layer'>>
    ) => {
      const { entityIdentifier, beginEndStepPct } = action.payload;
      const layer = selectEntity(state, entityIdentifier);
      if (!layer || !layer.controlAdapter) {
        return;
      }
      layer.controlAdapter.beginEndStepPct = beginEndStepPct;
    },
    controlLayerWithTransparencyEffectToggled: (
      state,
      action: PayloadAction<EntityIdentifierPayload<void, 'control_layer'>>
    ) => {
      const { entityIdentifier } = action.payload;
      const layer = selectEntity(state, entityIdentifier);
      if (!layer) {
        return;
      }
      layer.withTransparencyEffect = !layer.withTransparencyEffect;
    },
    //#region Global Reference Images
    referenceImageAdded: {
      reducer: (
        state,
        action: PayloadAction<{ id: string; overrides?: Partial<CanvasReferenceImageState>; isSelected?: boolean }>
      ) => {
        const { id, overrides, isSelected } = action.payload;
        const entityState = getReferenceImageState(id, overrides);

        state.referenceImages.entities.push(entityState);

        if (isSelected) {
          state.selectedEntityIdentifier = getEntityIdentifier(entityState);
        }
      },
      prepare: (payload?: { overrides?: Partial<CanvasReferenceImageState>; isSelected?: boolean }) => ({
        payload: { ...payload, id: getPrefixedId('reference_image') },
      }),
    },
    referenceImageRecalled: (state, action: PayloadAction<{ data: CanvasReferenceImageState }>) => {
      const { data } = action.payload;
      state.referenceImages.entities.push(data);
      state.selectedEntityIdentifier = { type: 'reference_image', id: data.id };
    },
    referenceImageIPAdapterImageChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ imageDTO: ImageDTO | null }, 'reference_image'>>
    ) => {
      const { entityIdentifier, imageDTO } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    referenceImageIPAdapterMethodChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ method: IPMethodV2 }, 'reference_image'>>
    ) => {
      const { entityIdentifier, method } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.ipAdapter.method = method;
    },
    referenceImageIPAdapterModelChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ modelConfig: IPAdapterModelConfig | null }, 'reference_image'>>
    ) => {
      const { entityIdentifier, modelConfig } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.ipAdapter.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;
    },
    referenceImageIPAdapterCLIPVisionModelChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ clipVisionModel: CLIPVisionModelV2 }, 'reference_image'>>
    ) => {
      const { entityIdentifier, clipVisionModel } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.ipAdapter.clipVisionModel = clipVisionModel;
    },
    referenceImageIPAdapterWeightChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ weight: number }, 'reference_image'>>
    ) => {
      const { entityIdentifier, weight } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.ipAdapter.weight = weight;
    },
    referenceImageIPAdapterBeginEndStepPctChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ beginEndStepPct: [number, number] }, 'reference_image'>>
    ) => {
      const { entityIdentifier, beginEndStepPct } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.ipAdapter.beginEndStepPct = beginEndStepPct;
    },
    //#region Regional Guidance
    rgAdded: {
      reducer: (
        state,
        action: PayloadAction<{ id: string; overrides?: Partial<CanvasRegionalGuidanceState>; isSelected?: boolean }>
      ) => {
        const { id, overrides, isSelected } = action.payload;

        const entityState = getRegionalGuidanceState(id, overrides);

        state.regionalGuidance.entities.push(entityState);

        if (isSelected) {
          state.selectedEntityIdentifier = getEntityIdentifier(entityState);
        }
      },
      prepare: (payload?: { overrides?: Partial<CanvasRegionalGuidanceState>; isSelected?: boolean }) => ({
        payload: { ...payload, id: getPrefixedId('regional_guidance') },
      }),
    },
    rgRecalled: (state, action: PayloadAction<{ data: CanvasRegionalGuidanceState }>) => {
      const { data } = action.payload;
      state.regionalGuidance.entities.push(data);
      state.selectedEntityIdentifier = { type: 'regional_guidance', id: data.id };
    },
    rgPositivePromptChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ prompt: string | null }, 'regional_guidance'>>
    ) => {
      const { entityIdentifier, prompt } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.positivePrompt = prompt;
    },
    rgNegativePromptChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ prompt: string | null }, 'regional_guidance'>>
    ) => {
      const { entityIdentifier, prompt } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.negativePrompt = prompt;
    },
    rgAutoNegativeToggled: (state, action: PayloadAction<EntityIdentifierPayload<void, 'regional_guidance'>>) => {
      const { entityIdentifier } = action.payload;
      const rg = selectEntity(state, entityIdentifier);
      if (!rg) {
        return;
      }
      rg.autoNegative = !rg.autoNegative;
    },
    rgIPAdapterAdded: {
      reducer: (
        state,
        action: PayloadAction<
          EntityIdentifierPayload<
            { referenceImageId: string; overrides?: Partial<RegionalGuidanceReferenceImageState> },
            'regional_guidance'
          >
        >
      ) => {
        const { entityIdentifier, overrides, referenceImageId } = action.payload;
        const entity = selectEntity(state, entityIdentifier);
        if (!entity) {
          return;
        }
        const ipAdapter = { id: referenceImageId, ipAdapter: deepClone(initialIPAdapter) };
        merge(ipAdapter, overrides);
        entity.referenceImages.push(ipAdapter);
      },
      prepare: (
        payload: EntityIdentifierPayload<
          { overrides?: Partial<RegionalGuidanceReferenceImageState> },
          'regional_guidance'
        >
      ) => ({
        payload: { ...payload, referenceImageId: getPrefixedId('regional_guidance_ip_adapter') },
      }),
    },
    rgIPAdapterDeleted: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ referenceImageId: string }, 'regional_guidance'>>
    ) => {
      const { entityIdentifier, referenceImageId } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.referenceImages = entity.referenceImages.filter((ipAdapter) => ipAdapter.id !== referenceImageId);
    },
    rgIPAdapterImageChanged: (
      state,
      action: PayloadAction<
        EntityIdentifierPayload<{ referenceImageId: string; imageDTO: ImageDTO | null }, 'regional_guidance'>
      >
    ) => {
      const { entityIdentifier, referenceImageId, imageDTO } = action.payload;
      const referenceImage = selectRegionalGuidanceReferenceImage(state, entityIdentifier, referenceImageId);
      if (!referenceImage) {
        return;
      }
      referenceImage.ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    rgIPAdapterWeightChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ referenceImageId: string; weight: number }, 'regional_guidance'>>
    ) => {
      const { entityIdentifier, referenceImageId, weight } = action.payload;
      const referenceImage = selectRegionalGuidanceReferenceImage(state, entityIdentifier, referenceImageId);
      if (!referenceImage) {
        return;
      }
      referenceImage.ipAdapter.weight = weight;
    },
    rgIPAdapterBeginEndStepPctChanged: (
      state,
      action: PayloadAction<
        EntityIdentifierPayload<{ referenceImageId: string; beginEndStepPct: [number, number] }, 'regional_guidance'>
      >
    ) => {
      const { entityIdentifier, referenceImageId, beginEndStepPct } = action.payload;
      const referenceImage = selectRegionalGuidanceReferenceImage(state, entityIdentifier, referenceImageId);
      if (!referenceImage) {
        return;
      }
      referenceImage.ipAdapter.beginEndStepPct = beginEndStepPct;
    },
    rgIPAdapterMethodChanged: (
      state,
      action: PayloadAction<
        EntityIdentifierPayload<{ referenceImageId: string; method: IPMethodV2 }, 'regional_guidance'>
      >
    ) => {
      const { entityIdentifier, referenceImageId, method } = action.payload;
      const referenceImage = selectRegionalGuidanceReferenceImage(state, entityIdentifier, referenceImageId);
      if (!referenceImage) {
        return;
      }
      referenceImage.ipAdapter.method = method;
    },
    rgIPAdapterModelChanged: (
      state,
      action: PayloadAction<
        EntityIdentifierPayload<
          {
            referenceImageId: string;
            modelConfig: IPAdapterModelConfig | null;
          },
          'regional_guidance'
        >
      >
    ) => {
      const { entityIdentifier, referenceImageId, modelConfig } = action.payload;
      const referenceImage = selectRegionalGuidanceReferenceImage(state, entityIdentifier, referenceImageId);
      if (!referenceImage) {
        return;
      }
      referenceImage.ipAdapter.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;
    },
    rgIPAdapterCLIPVisionModelChanged: (
      state,
      action: PayloadAction<
        EntityIdentifierPayload<{ referenceImageId: string; clipVisionModel: CLIPVisionModelV2 }, 'regional_guidance'>
      >
    ) => {
      const { entityIdentifier, referenceImageId, clipVisionModel } = action.payload;
      const referenceImage = selectRegionalGuidanceReferenceImage(state, entityIdentifier, referenceImageId);
      if (!referenceImage) {
        return;
      }
      referenceImage.ipAdapter.clipVisionModel = clipVisionModel;
    },
    //#region Inpaint mask
    inpaintMaskAdded: {
      reducer: (
        state,
        action: PayloadAction<{
          id: string;
          overrides?: Partial<CanvasInpaintMaskState>;
          isSelected?: boolean;
          isMergingVisible?: boolean;
        }>
      ) => {
        const { id, overrides, isSelected, isMergingVisible } = action.payload;

        const entityState = getInpaintMaskState(id, overrides);

        if (isMergingVisible) {
          // When merging visible, we delete all disabled layers
          state.inpaintMasks.entities = state.inpaintMasks.entities.filter((layer) => !layer.isEnabled);
        }

        state.inpaintMasks.entities.push(entityState);

        if (isSelected) {
          state.selectedEntityIdentifier = getEntityIdentifier(entityState);
        }
      },
      prepare: (payload?: {
        overrides?: Partial<CanvasInpaintMaskState>;
        isSelected?: boolean;
        isMergingVisible?: boolean;
      }) => ({
        payload: { ...payload, id: getPrefixedId('inpaint_mask') },
      }),
    },
    inpaintMaskRecalled: (state, action: PayloadAction<{ data: CanvasInpaintMaskState }>) => {
      const { data } = action.payload;
      state.inpaintMasks.entities = [data];
      state.selectedEntityIdentifier = { type: 'inpaint_mask', id: data.id };
    },
    //#region BBox
    bboxScaledWidthChanged: (state, action: PayloadAction<number>) => {
      const gridSize = getGridSize(state.bbox.modelBase);

      state.bbox.scaledSize.width = roundToMultiple(action.payload, gridSize);

      if (state.bbox.aspectRatio.isLocked) {
        state.bbox.scaledSize.height = roundToMultiple(
          state.bbox.scaledSize.width / state.bbox.aspectRatio.value,
          gridSize
        );
      }
    },
    bboxScaledHeightChanged: (state, action: PayloadAction<number>) => {
      const gridSize = getGridSize(state.bbox.modelBase);

      state.bbox.scaledSize.height = roundToMultiple(action.payload, gridSize);

      if (state.bbox.aspectRatio.isLocked) {
        state.bbox.scaledSize.width = roundToMultiple(
          state.bbox.scaledSize.height * state.bbox.aspectRatio.value,
          gridSize
        );
      }
    },
    bboxScaleMethodChanged: (state, action: PayloadAction<BoundingBoxScaleMethod>) => {
      state.bbox.scaleMethod = action.payload;
      syncScaledSize(state);
    },
    bboxChangedFromCanvas: (state, action: PayloadAction<IRect>) => {
      state.bbox.rect = action.payload;

      // TODO(psyche): Figure out a way to handle this without resetting the aspect ratio on every change.
      // This action is dispatched when the user resizes or moves the bbox from the canvas. For now, when the user
      // resizes the bbox from the canvas, we unlock the aspect ratio.
      state.bbox.aspectRatio.value = state.bbox.rect.width / state.bbox.rect.height;
      state.bbox.aspectRatio.id = 'Free';

      syncScaledSize(state);
    },
    bboxWidthChanged: (
      state,
      action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>
    ) => {
      const { width, updateAspectRatio, clamp } = action.payload;
      const gridSize = getGridSize(state.bbox.modelBase);
      state.bbox.rect.width = clamp ? Math.max(roundDownToMultiple(width, gridSize), 64) : width;

      if (state.bbox.aspectRatio.isLocked) {
        state.bbox.rect.height = roundToMultiple(state.bbox.rect.width / state.bbox.aspectRatio.value, gridSize);
      }

      if (updateAspectRatio || !state.bbox.aspectRatio.isLocked) {
        state.bbox.aspectRatio.value = state.bbox.rect.width / state.bbox.rect.height;
        state.bbox.aspectRatio.id = 'Free';
        state.bbox.aspectRatio.isLocked = false;
      }

      syncScaledSize(state);
    },
    bboxHeightChanged: (
      state,
      action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>
    ) => {
      const { height, updateAspectRatio, clamp } = action.payload;
      const gridSize = getGridSize(state.bbox.modelBase);
      state.bbox.rect.height = clamp ? Math.max(roundDownToMultiple(height, gridSize), 64) : height;

      if (state.bbox.aspectRatio.isLocked) {
        state.bbox.rect.width = roundToMultiple(state.bbox.rect.height * state.bbox.aspectRatio.value, gridSize);
      }

      if (updateAspectRatio || !state.bbox.aspectRatio.isLocked) {
        state.bbox.aspectRatio.value = state.bbox.rect.width / state.bbox.rect.height;
        state.bbox.aspectRatio.id = 'Free';
        state.bbox.aspectRatio.isLocked = false;
      }

      syncScaledSize(state);
    },
    bboxAspectRatioLockToggled: (state) => {
      state.bbox.aspectRatio.isLocked = !state.bbox.aspectRatio.isLocked;
      syncScaledSize(state);
    },
    bboxAspectRatioIdChanged: (state, action: PayloadAction<{ id: AspectRatioID }>) => {
      const { id } = action.payload;
      state.bbox.aspectRatio.id = id;
      if (id === 'Free') {
        state.bbox.aspectRatio.isLocked = false;
      } else {
        state.bbox.aspectRatio.isLocked = true;
        state.bbox.aspectRatio.value = ASPECT_RATIO_MAP[id].ratio;
        const { width, height } = calculateNewSize(
          state.bbox.aspectRatio.value,
          state.bbox.rect.width * state.bbox.rect.height,
          state.bbox.modelBase
        );
        state.bbox.rect.width = width;
        state.bbox.rect.height = height;
      }

      syncScaledSize(state);
    },
    bboxDimensionsSwapped: (state) => {
      state.bbox.aspectRatio.value = 1 / state.bbox.aspectRatio.value;
      if (state.bbox.aspectRatio.id === 'Free') {
        const newWidth = state.bbox.rect.height;
        const newHeight = state.bbox.rect.width;
        state.bbox.rect.width = newWidth;
        state.bbox.rect.height = newHeight;
      } else {
        const { width, height } = calculateNewSize(
          state.bbox.aspectRatio.value,
          state.bbox.rect.width * state.bbox.rect.height,
          state.bbox.modelBase
        );
        state.bbox.rect.width = width;
        state.bbox.rect.height = height;
        state.bbox.aspectRatio.id = ASPECT_RATIO_MAP[state.bbox.aspectRatio.id].inverseID;
      }

      syncScaledSize(state);
    },
    bboxSizeOptimized: (state) => {
      const optimalDimension = getOptimalDimension(state.bbox.modelBase);
      if (state.bbox.aspectRatio.isLocked) {
        const { width, height } = calculateNewSize(
          state.bbox.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.bbox.modelBase
        );
        state.bbox.rect.width = width;
        state.bbox.rect.height = height;
      } else {
        state.bbox.aspectRatio = deepClone(initialState.bbox.aspectRatio);
        state.bbox.rect.width = optimalDimension;
        state.bbox.rect.height = optimalDimension;
      }

      syncScaledSize(state);
    },
    bboxSyncedToOptimalDimension: (state) => {
      const optimalDimension = getOptimalDimension(state.bbox.modelBase);

      if (!getIsSizeOptimal(state.bbox.rect.width, state.bbox.rect.height, state.bbox.modelBase)) {
        const bboxDims = calculateNewSize(
          state.bbox.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.bbox.modelBase
        );
        state.bbox.rect.width = bboxDims.width;
        state.bbox.rect.height = bboxDims.height;
        syncScaledSize(state);
      }
    },
    //#region Shared entity
    entitySelected: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        // Cannot select a non-existent entity
        return;
      }
      state.selectedEntityIdentifier = entityIdentifier;
    },
    bookmarkedEntityChanged: (state, action: PayloadAction<{ entityIdentifier: CanvasEntityIdentifier | null }>) => {
      const { entityIdentifier } = action.payload;
      if (!entityIdentifier) {
        state.bookmarkedEntityIdentifier = null;
        return;
      }
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        // Cannot select a non-existent entity
        return;
      }
      state.bookmarkedEntityIdentifier = entityIdentifier;
    },
    entityNameChanged: (state, action: PayloadAction<EntityIdentifierPayload<{ name: string | null }>>) => {
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
      } else if (isRenderableEntity(entity)) {
        entity.isEnabled = true;
        entity.objects = [];
        entity.position = { x: 0, y: 0 };
      } else {
        assert(false, 'Not implemented');
      }
    },
    entityDuplicated: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      const newEntity = deepClone(entity);
      if (newEntity.name) {
        newEntity.name = `${newEntity.name} (Copy)`;
      }
      switch (newEntity.type) {
        case 'raster_layer':
          newEntity.id = getPrefixedId('raster_layer');
          state.rasterLayers.entities.push(newEntity);
          break;
        case 'control_layer':
          newEntity.id = getPrefixedId('control_layer');
          state.controlLayers.entities.push(newEntity);
          break;
        case 'regional_guidance':
          newEntity.id = getPrefixedId('regional_guidance');
          for (const refImage of newEntity.referenceImages) {
            refImage.id = getPrefixedId('regional_guidance_ip_adapter');
          }
          state.regionalGuidance.entities.push(newEntity);
          break;
        case 'reference_image':
          newEntity.id = getPrefixedId('reference_image');
          state.referenceImages.entities.push(newEntity);
          break;
        case 'inpaint_mask':
          newEntity.id = getPrefixedId('inpaint_mask');
          state.inpaintMasks.entities.push(newEntity);
          break;
      }

      state.selectedEntityIdentifier = getEntityIdentifier(newEntity);
    },
    entityIsEnabledToggled: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.isEnabled = !entity.isEnabled;
    },
    entityIsLockedToggled: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.isLocked = !entity.isLocked;
    },
    entityFillColorChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ color: RgbColor }, 'inpaint_mask' | 'regional_guidance'>>
    ) => {
      const { color, entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.fill.color = color;
    },
    entityFillStyleChanged: (
      state,
      action: PayloadAction<EntityIdentifierPayload<{ style: FillStyle }, 'inpaint_mask' | 'regional_guidance'>>
    ) => {
      const { style, entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      entity.fill.style = style;
    },
    entityMoved: (state, action: PayloadAction<EntityMovedPayload>) => {
      const { entityIdentifier, position } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (isRenderableEntity(entity)) {
        entity.position = position;
      }
    },
    entityRasterized: (state, action: PayloadAction<EntityRasterizedPayload>) => {
      const { entityIdentifier, imageObject, position, replaceObjects } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (isRenderableEntity(entity)) {
        if (replaceObjects) {
          entity.objects = [imageObject];
          entity.position = position;
        }
      }
    },
    entityBrushLineAdded: (state, action: PayloadAction<EntityBrushLineAddedPayload>) => {
      const { entityIdentifier, brushLine } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (!isRenderableEntity(entity)) {
        assert(false, `Cannot add a brush line to a non-drawable entity of type ${entity.type}`);
      }

      // TODO(psyche): If we add the object without splatting, the renderer will see it as the same object and not
      // re-render it (reference equality check). I don't like this behaviour.
      entity.objects.push({
        ...brushLine,
        // If the brush line is not pressure sensitive, we simplify the points to reduce the size of the state
        points: brushLine.type === 'brush_line' ? simplifyFlatNumbersArray(brushLine.points) : brushLine.points,
      });
    },
    entityEraserLineAdded: (state, action: PayloadAction<EntityEraserLineAddedPayload>) => {
      const { entityIdentifier, eraserLine } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (!isRenderableEntity(entity)) {
        assert(false, `Cannot add a eraser line to a non-drawable entity of type ${entity.type}`);
      }

      // TODO(psyche): If we add the object without splatting, the renderer will see it as the same object and not
      // re-render it (reference equality check). I don't like this behaviour.
      entity.objects.push({
        ...eraserLine,
        // If the brush line is not pressure sensitive, we simplify the points to reduce the size of the state
        points: eraserLine.type === 'eraser_line' ? simplifyFlatNumbersArray(eraserLine.points) : eraserLine.points,
      });
    },
    entityRectAdded: (state, action: PayloadAction<EntityRectAddedPayload>) => {
      const { entityIdentifier, rect } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }

      if (!isRenderableEntity(entity)) {
        assert(false, `Cannot add a rect to a non-drawable entity of type ${entity.type}`);
      }

      // TODO(psyche): If we add the object without splatting, the renderer will see it as the same object and not
      // re-render it (reference equality check). I don't like this behaviour.
      entity.objects.push({ ...rect });
    },
    entityDeleted: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;

      let selectedEntityIdentifier: CanvasState['selectedEntityIdentifier'] = null;
      const allEntities = selectAllEntities(state);
      const index = allEntities.findIndex((entity) => entity.id === entityIdentifier.id);
      const nextIndex = allEntities.length > 1 ? (index + 1) % allEntities.length : -1;
      if (nextIndex !== -1) {
        const nextEntity = allEntities[nextIndex];
        if (nextEntity) {
          selectedEntityIdentifier = getEntityIdentifier(nextEntity);
        }
      }

      switch (entityIdentifier.type) {
        case 'raster_layer':
          state.rasterLayers.entities = state.rasterLayers.entities.filter((layer) => layer.id !== entityIdentifier.id);
          break;
        case 'control_layer':
          state.controlLayers.entities = state.controlLayers.entities.filter((rg) => rg.id !== entityIdentifier.id);
          break;
        case 'regional_guidance':
          state.regionalGuidance.entities = state.regionalGuidance.entities.filter(
            (rg) => rg.id !== entityIdentifier.id
          );
          break;
        case 'reference_image':
          state.referenceImages.entities = state.referenceImages.entities.filter((rg) => rg.id !== entityIdentifier.id);
          break;
        case 'inpaint_mask':
          state.inpaintMasks.entities = state.inpaintMasks.entities.filter((rg) => rg.id !== entityIdentifier.id);
          break;
      }

      state.selectedEntityIdentifier = selectedEntityIdentifier;
    },
    entityArrangedForwardOne: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      moveOneToEnd(selectAllEntitiesOfType(state, entity.type), entity);
    },
    entityArrangedToFront: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      moveToEnd(selectAllEntitiesOfType(state, entity.type), entity);
    },
    entityArrangedBackwardOne: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      moveOneToStart(selectAllEntitiesOfType(state, entity.type), entity);
    },
    entityArrangedToBack: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      moveToStart(selectAllEntitiesOfType(state, entity.type), entity);
    },
    entityOpacityChanged: (state, action: PayloadAction<EntityIdentifierPayload<{ opacity: number }>>) => {
      const { entityIdentifier, opacity } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      if (entity.type === 'reference_image') {
        return;
      }
      entity.opacity = opacity;
    },
    allEntitiesOfTypeIsHiddenToggled: (state, action: PayloadAction<{ type: CanvasEntityIdentifier['type'] }>) => {
      const { type } = action.payload;

      switch (type) {
        case 'raster_layer':
          state.rasterLayers.isHidden = !state.rasterLayers.isHidden;
          break;
        case 'control_layer':
          state.controlLayers.isHidden = !state.controlLayers.isHidden;
          break;
        case 'inpaint_mask':
          state.inpaintMasks.isHidden = !state.inpaintMasks.isHidden;
          break;
        case 'regional_guidance':
          state.regionalGuidance.isHidden = !state.regionalGuidance.isHidden;
          break;
        case 'reference_image':
          // no-op
          break;
      }
    },
    canvasMetadataRecalled: (state, action: PayloadAction<CanvasMetadata>) => {
      const { controlLayers, inpaintMasks, rasterLayers, referenceImages, regionalGuidance } = action.payload;
      state.controlLayers.entities = controlLayers;
      state.inpaintMasks.entities = inpaintMasks;
      state.rasterLayers.entities = rasterLayers;
      state.referenceImages.entities = referenceImages;
      state.regionalGuidance.entities = regionalGuidance;
      return state;
    },
    canvasUndo: () => {},
    canvasRedo: () => {},
    canvasClearHistory: () => {},
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, (state) => {
      return resetState(state);
    });
    builder.addCase(modelChanged, (state, action) => {
      const { model } = action.payload;
      /**
       * Because the bbox depends in part on the model, it needs to be in sync with the model. However, due to
       * complications with managing undo/redo history, we need to store the model in a separate slice from the canvas
       * state.
       *
       * Unfortunately, this means we need to manually sync the model with the canvas state. We only care about the
       * model base, so we only need to update the bbox's modelBase field.
       *
       * When we do this, we also want to update the bbox's dimensions - but only if we are not staging images on the
       * canvas, during which time the bbox must stay the same.
       *
       * Unfortunately (again), the staging state is in a different slice, to prevent issues with undo/redo history.
       *
       * There's some fanagling we must do to handle this correctly:
       * - Store the model base in this slice, so that we can access it when the user changes the bbox dimensions.
       * - Avoid updating the bbox dimensions when we are staging - only update the model base.
       * - Provide a separate action that will update the bbox dimensions and be careful to not dispatch it when staging.
       */
      const base = model?.base;
      if (isMainModelBase(base) && state.bbox.modelBase !== base) {
        state.bbox.modelBase = base;
        syncScaledSize(state);
      }
    });
    builder.addMatcher(newSessionRequested, (state) => {
      return resetState(state);
    });
  },
});

const resetState = (state: CanvasState) => {
  const newState = getInitialState();

  // We need to retain the optimal dimension across resets, as it is changed only when the model changes. Copy it
  // from the old state, then recalculate the bbox size & scaled size.
  newState.bbox.modelBase = state.bbox.modelBase;
  const optimalDimension = getOptimalDimension(newState.bbox.modelBase);
  const rect = calculateNewSize(
    newState.bbox.aspectRatio.value,
    optimalDimension * optimalDimension,
    newState.bbox.modelBase
  );
  newState.bbox.rect.width = rect.width;
  newState.bbox.rect.height = rect.height;
  syncScaledSize(newState);

  return newState;
};

export const {
  canvasMetadataRecalled,
  canvasUndo,
  canvasRedo,
  canvasClearHistory,
  // All entities
  entitySelected,
  bookmarkedEntityChanged,
  entityNameChanged,
  entityReset,
  entityIsEnabledToggled,
  entityIsLockedToggled,
  entityFillColorChanged,
  entityFillStyleChanged,
  entityMoved,
  entityDuplicated,
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
  // allEntitiesDeleted, // currently unused
  allEntitiesOfTypeIsHiddenToggled,
  // bbox
  bboxChangedFromCanvas,
  bboxScaledWidthChanged,
  bboxScaledHeightChanged,
  bboxScaleMethodChanged,
  bboxWidthChanged,
  bboxHeightChanged,
  bboxAspectRatioLockToggled,
  bboxAspectRatioIdChanged,
  bboxDimensionsSwapped,
  bboxSizeOptimized,
  bboxSyncedToOptimalDimension,
  // Raster layers
  rasterLayerAdded,
  // rasterLayerRecalled,
  rasterLayerConvertedToControlLayer,
  // Control layers
  controlLayerAdded,
  // controlLayerRecalled,
  controlLayerConvertedToRasterLayer,
  controlLayerModelChanged,
  controlLayerControlModeChanged,
  controlLayerWeightChanged,
  controlLayerBeginEndStepPctChanged,
  controlLayerWithTransparencyEffectToggled,
  // IP Adapters
  referenceImageAdded,
  // referenceImageRecalled,
  referenceImageIPAdapterImageChanged,
  referenceImageIPAdapterMethodChanged,
  referenceImageIPAdapterModelChanged,
  referenceImageIPAdapterCLIPVisionModelChanged,
  referenceImageIPAdapterWeightChanged,
  referenceImageIPAdapterBeginEndStepPctChanged,
  // Regions
  rgAdded,
  // rgRecalled,
  rgPositivePromptChanged,
  rgNegativePromptChanged,
  rgAutoNegativeToggled,
  rgIPAdapterAdded,
  rgIPAdapterDeleted,
  rgIPAdapterImageChanged,
  rgIPAdapterWeightChanged,
  rgIPAdapterBeginEndStepPctChanged,
  rgIPAdapterMethodChanged,
  rgIPAdapterModelChanged,
  rgIPAdapterCLIPVisionModelChanged,
  // Inpaint mask
  inpaintMaskAdded,
  // inpaintMaskRecalled,
} = canvasSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasPersistConfig: PersistConfig<CanvasState> = {
  name: canvasSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};

const syncScaledSize = (state: CanvasState) => {
  if (state.bbox.scaleMethod === 'auto') {
    // Sync both aspect ratio and size
    const { width, height } = state.bbox.rect;
    state.bbox.scaledSize = getScaledBoundingBoxDimensions({ width, height }, state.bbox.modelBase);
  } else if (state.bbox.scaleMethod === 'manual' && state.bbox.aspectRatio.isLocked) {
    // Only sync the aspect ratio if manual & locked
    state.bbox.scaledSize = calculateNewSize(
      state.bbox.aspectRatio.value,
      state.bbox.scaledSize.width * state.bbox.scaledSize.height,
      state.bbox.modelBase
    );
  }
};

let filter = true;

export const canvasUndoableConfig: UndoableOptions<CanvasState, UnknownAction> = {
  limit: 64,
  undoType: canvasUndo.type,
  redoType: canvasRedo.type,
  clearHistoryType: canvasClearHistory.type,
  filter: (action, _state, _history) => {
    // Ignore all actions from other slices
    if (!action.type.startsWith(canvasSlice.name)) {
      return false;
    }
    // Throttle rapid actions of the same type
    filter = actionsThrottlingFilter(action);
    return filter;
  },
  // This is pretty spammy, leave commented out unless you need it
  // debug: import.meta.env.MODE === 'development',
};

const doNotGroupMatcher = isAnyOf(entityBrushLineAdded, entityEraserLineAdded, entityRectAdded);

// Store rapid actions of the same type at most once every x time.
// See: https://github.com/omnidan/redux-undo/blob/master/examples/throttled-drag/util/undoFilter.js
const THROTTLE_MS = 1000;
let ignoreRapid = false;
let prevActionType: string | null = null;
function actionsThrottlingFilter(action: UnknownAction) {
  // If the actions are of a different type, reset the throttle and allow the action
  if (action.type !== prevActionType || doNotGroupMatcher(action)) {
    ignoreRapid = false;
    prevActionType = action.type;
    return true;
  }
  // Else, if the actions are of the same type, throttle them. Ignore the action if the flag is set.
  if (ignoreRapid) {
    return false;
  }
  // We are allowing this action - set the flag and a timeout to clear it.
  ignoreRapid = true;
  window.setTimeout(() => {
    ignoreRapid = false;
  }, THROTTLE_MS);
  return true;
}
