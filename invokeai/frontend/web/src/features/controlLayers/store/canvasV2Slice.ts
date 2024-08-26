import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig } from 'app/store/store';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { bboxReducers } from 'features/controlLayers/store/bboxReducers';
import { controlLayersReducers } from 'features/controlLayers/store/controlLayersReducers';
import { inpaintMaskReducers } from 'features/controlLayers/store/inpaintMaskReducers';
import { ipAdaptersReducers } from 'features/controlLayers/store/ipAdaptersReducers';
import { modelChanged } from 'features/controlLayers/store/paramsSlice';
import { rasterLayersReducers } from 'features/controlLayers/store/rasterLayersReducers';
import { regionsReducers } from 'features/controlLayers/store/regionsReducers';
import { selectAllEntities, selectAllEntitiesOfType, selectEntity } from 'features/controlLayers/store/selectors';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { simplifyFlatNumbersArray } from 'features/controlLayers/util/simplify';
import { calculateNewSize } from 'features/parameters/components/DocumentSize/calculateNewSize';
import { initialAspectRatioState } from 'features/parameters/components/DocumentSize/constants';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { pick } from 'lodash-es';
import { assert } from 'tsafe';

import type {
  CanvasEntityIdentifier,
  CanvasV2State,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
} from './types';
import { getEntityIdentifier, isDrawableEntity } from './types';

const initialState: CanvasV2State = {
  _version: 3,
  selectedEntityIdentifier: null,
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
    entities: [],
  },
  regions: {
    isHidden: false,
    entities: [],
  },
  ipAdapters: { entities: [] },
  bbox: {
    rect: { x: 0, y: 0, width: 512, height: 512 },
    optimalDimension: 512,
    aspectRatio: deepClone(initialAspectRatioState),
    scaleMethod: 'auto',
    scaledSize: {
      width: 512,
      height: 512,
    },
  },
};

export const canvasV2Slice = createSlice({
  name: 'canvasV2',
  initialState,
  reducers: {
    // undoable canvas state
    ...rasterLayersReducers,
    ...controlLayersReducers,
    ...ipAdaptersReducers,
    ...regionsReducers,
    ...inpaintMaskReducers,
    ...bboxReducers,
    entitySelected: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;
      state.selectedEntityIdentifier = entityIdentifier;
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
      } else if (isDrawableEntity(entity)) {
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
          state.regions.entities.push(newEntity);
          break;
        case 'ip_adapter':
          newEntity.id = getPrefixedId('ip_adapter');
          state.ipAdapters.entities.push(newEntity);
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

      // TODO(psyche): If we add the object without splatting, the renderer will see it as the same object and not
      // re-render it (reference equality check). I don't like this behaviour.
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

      // TODO(psyche): If we add the object without splatting, the renderer will see it as the same object and not
      // re-render it (reference equality check). I don't like this behaviour.
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

      // TODO(psyche): If we add the object without splatting, the renderer will see it as the same object and not
      // re-render it (reference equality check). I don't like this behaviour.
      entity.objects.push({ ...rect });
    },
    entityDeleted: (state, action: PayloadAction<EntityIdentifierPayload>) => {
      const { entityIdentifier } = action.payload;

      let selectedEntityIdentifier: CanvasV2State['selectedEntityIdentifier'] = null;
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
          state.regions.entities = state.regions.entities.filter((rg) => rg.id !== entityIdentifier.id);
          break;
        case 'ip_adapter':
          state.ipAdapters.entities = state.ipAdapters.entities.filter((rg) => rg.id !== entityIdentifier.id);
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
      if (entity.type === 'ip_adapter') {
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
          state.regions.isHidden = !state.regions.isHidden;
          break;
        case 'ip_adapter':
          // no-op
          break;
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
      state.bbox.rect.width = state.bbox.optimalDimension;
      state.bbox.rect.height = state.bbox.optimalDimension;
      const size = pick(state.bbox.rect, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, state.bbox.optimalDimension);

      state.ipAdapters = deepClone(initialState.ipAdapters);
      state.rasterLayers = deepClone(initialState.rasterLayers);
      state.controlLayers = deepClone(initialState.controlLayers);
      state.regions = deepClone(initialState.regions);
      state.inpaintMasks = deepClone(initialState.inpaintMasks);

      state.selectedEntityIdentifier = deepClone(initialState.selectedEntityIdentifier);
    },
  },
  extraReducers(builder) {
    builder.addCase(modelChanged, (state, action) => {
      const { model, previousModel } = action.payload;

      // If the model base changes (e.g. SD1.5 -> SDXL), we need to change a few things
      if (model === null || previousModel?.base === model.base) {
        return;
      }

      // Update the bbox size to match the new model's optimal size
      const optimalDimension = getOptimalDimension(model);

      state.bbox.optimalDimension = optimalDimension;

      if (!getIsSizeOptimal(state.bbox.rect.width, state.bbox.rect.height, optimalDimension)) {
        const bboxDims = calculateNewSize(state.bbox.aspectRatio.value, optimalDimension * optimalDimension);
        state.bbox.rect.width = bboxDims.width;
        state.bbox.rect.height = bboxDims.height;

        if (state.bbox.scaleMethod === 'auto') {
          state.bbox.scaledSize = getScaledBoundingBoxDimensions(bboxDims, optimalDimension);
        }
      }
    });
  },
});

export const {
  canvasReset,
  // All entities
  entitySelected,
  entityNameChanged,
  entityReset,
  entityIsEnabledToggled,
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
  allEntitiesDeleted,
  allEntitiesOfTypeIsHiddenToggled,
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
  ipaAdded,
  // ipaRecalled,
  ipaImageChanged,
  ipaMethodChanged,
  ipaModelChanged,
  ipaCLIPVisionModelChanged,
  ipaWeightChanged,
  ipaBeginEndStepPctChanged,
  // Regions
  rgAdded,
  // rgRecalled,
  rgPositivePromptChanged,
  rgNegativePromptChanged,
  rgFillColorChanged,
  rgFillStyleChanged,
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
  inpaintMaskFillColorChanged,
  inpaintMaskFillStyleChanged,
} = canvasV2Slice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasV2PersistConfig: PersistConfig<CanvasV2State> = {
  name: canvasV2Slice.name,
  initialState,
  migrate,
  persistDenylist: [],
};
