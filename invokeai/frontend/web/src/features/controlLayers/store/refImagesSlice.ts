import { objectEquals } from '@observ33r/object-equals';
import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { clamp } from 'es-toolkit/compat';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CroppableImageWithDims,
  FLUXReduxImageInfluence,
  RefImagesState,
} from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { FLUXKontextModelConfig, FLUXReduxModelConfig, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import type { PartialDeep } from 'type-fest';

import type { CLIPVisionModelV2, IPMethodV2, RefImageState } from './types';
import { getInitialRefImagesState, isFLUXReduxConfig, isIPAdapterConfig, zRefImagesState } from './types';
import { getReferenceImageState, initialFluxKontextReferenceImage, initialFLUXRedux, initialIPAdapter } from './util';

type PayloadActionWithId<T = void> = T extends void
  ? PayloadAction<{ id: string }>
  : PayloadAction<
      {
        id: string;
      } & T
    >;

const slice = createSlice({
  name: 'refImages',
  initialState: getInitialRefImagesState(),
  reducers: {
    refImageAdded: {
      reducer: (state, action: PayloadActionWithId<{ overrides?: PartialDeep<RefImageState> }>) => {
        const { id, overrides } = action.payload;
        const entityState = getReferenceImageState(id, overrides);

        state.entities.push(entityState);
        state.selectedEntityId = id;
        state.isPanelOpen = true;
      },
      prepare: (payload?: { overrides?: PartialDeep<RefImageState> }) => ({
        payload: { ...payload, id: getPrefixedId('reference_image') },
      }),
    },
    refImagesRecalled: (state, action: PayloadAction<{ entities: RefImageState[]; replace: boolean }>) => {
      const { entities, replace } = action.payload;
      if (replace) {
        state.entities = entities;
        state.isPanelOpen = false;
        state.selectedEntityId = null;
      } else {
        state.entities.push(...entities);
      }
    },
    refImageImageChanged: (state, action: PayloadActionWithId<{ croppableImage: CroppableImageWithDims | null }>) => {
      const { id, croppableImage } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      entity.config.image = croppableImage;
    },
    refImageIPAdapterMethodChanged: (state, action: PayloadActionWithId<{ method: IPMethodV2 }>) => {
      const { id, method } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (!isIPAdapterConfig(entity.config)) {
        return;
      }
      entity.config.method = method;
    },
    refImageFLUXReduxImageInfluenceChanged: (
      state,
      action: PayloadActionWithId<{ imageInfluence: FLUXReduxImageInfluence }>
    ) => {
      const { id, imageInfluence } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (!isFLUXReduxConfig(entity.config)) {
        return;
      }
      entity.config.imageInfluence = imageInfluence;
    },
    refImageModelChanged: (
      state,
      action: PayloadActionWithId<{
        modelConfig: IPAdapterModelConfig | FLUXKontextModelConfig | FLUXReduxModelConfig | null;
      }>
    ) => {
      const { id, modelConfig } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }

      const oldModel = entity.config.model;

      // First set the new model
      entity.config.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;

      if (!entity.config.model) {
        return;
      }

      if (objectEquals(oldModel, entity.config.model)) {
        // Nothing changed, so we don't need to do anything
        return;
      }

      // The type of ref image depends on the model. When the user switches the model, we rebuild the ref image.
      // When we switch the model, we keep the image the same, but change the other parameters.

      if (entity.config.model.base === 'flux' && entity.config.model.name?.toLowerCase().includes('kontext')) {
        // Switching to flux-kontext ref image
        entity.config = {
          ...initialFluxKontextReferenceImage,
          image: entity.config.image,
          model: entity.config.model,
        };
        return;
      }

      if (entity.config.model.type === 'flux_redux') {
        // Switching to flux_redux
        entity.config = {
          ...initialFLUXRedux,
          image: entity.config.image,
          model: entity.config.model,
        };
        return;
      }

      if (entity.config.model.type === 'ip_adapter') {
        // Switching to ip_adapter
        entity.config = {
          ...initialIPAdapter,
          image: entity.config.image,
          model: entity.config.model,
        };
        // Ensure that the IP Adapter model is compatible with the CLIP Vision model
        if (entity.config.model?.base === 'flux') {
          entity.config.clipVisionModel = 'ViT-L';
        } else if (entity.config.clipVisionModel === 'ViT-L') {
          // Fall back to ViT-H (ViT-G would also work)
          entity.config.clipVisionModel = 'ViT-H';
        }
        return;
      }
    },
    refImageIPAdapterCLIPVisionModelChanged: (
      state,
      action: PayloadActionWithId<{ clipVisionModel: CLIPVisionModelV2 }>
    ) => {
      const { id, clipVisionModel } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (!isIPAdapterConfig(entity.config)) {
        return;
      }
      entity.config.clipVisionModel = clipVisionModel;
    },
    refImageIPAdapterWeightChanged: (state, action: PayloadActionWithId<{ weight: number }>) => {
      const { id, weight } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (!isIPAdapterConfig(entity.config)) {
        return;
      }
      entity.config.weight = weight;
    },
    refImageIPAdapterBeginEndStepPctChanged: (
      state,
      action: PayloadActionWithId<{ beginEndStepPct: [number, number] }>
    ) => {
      const { id, beginEndStepPct } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (!isIPAdapterConfig(entity.config)) {
        return;
      }
      entity.config.beginEndStepPct = beginEndStepPct;
    },
    refImageDeleted: (state, action: PayloadActionWithId) => {
      const { id } = action.payload;
      const currentIndex = state.entities.findIndex((rg) => rg.id === id);
      state.entities = state.entities.filter((rg) => rg.id !== id);
      const nextIndex = clamp(currentIndex, 0, state.entities.length - 1);
      const nextEntity = state.entities[nextIndex];
      state.selectedEntityId = nextEntity?.id ?? null;
      if (state.selectedEntityId === null) {
        state.isPanelOpen = false;
      }
    },
    refImageSelected: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (state.isPanelOpen && state.selectedEntityId === id) {
        state.isPanelOpen = false;
      } else {
        state.isPanelOpen = true;
      }
      state.selectedEntityId = id;
    },
    refImageIsEnabledToggled: (state, action: PayloadActionWithId) => {
      const { id } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      entity.isEnabled = !entity.isEnabled;
    },
    refImagesReset: () => getInitialRefImagesState(),
  },
});

export const {
  refImageSelected,
  refImageAdded,
  refImageDeleted,
  refImageImageChanged,
  refImageIPAdapterMethodChanged,
  refImageModelChanged,
  refImageIPAdapterCLIPVisionModelChanged,
  refImageIPAdapterWeightChanged,
  refImageIPAdapterBeginEndStepPctChanged,
  refImageFLUXReduxImageInfluenceChanged,
  refImageIsEnabledToggled,
  refImagesRecalled,
} = slice.actions;

export const refImagesSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zRefImagesState,
  getInitialState: getInitialRefImagesState,
  persistConfig: {
    migrate: (state) => zRefImagesState.parse(state),
    persistDenylist: ['selectedEntityId', 'isPanelOpen'],
  },
};

export const selectRefImagesSlice = (state: RootState) => state.refImages;

export const selectReferenceImageEntities = createSelector(selectRefImagesSlice, (state) => state.entities);
export const selectSelectedRefEntityId = createSelector(selectRefImagesSlice, (state) => state.selectedEntityId);
export const selectIsRefImagePanelOpen = createSelector(selectRefImagesSlice, (state) => state.isPanelOpen);
export const selectRefImageEntityIds = createMemoizedSelector(selectReferenceImageEntities, (entities) =>
  entities.map((e) => e.id)
);
export const selectRefImageEntity = (state: RefImagesState, id: string) =>
  state.entities.find((entity) => entity.id === id) ?? null;

export function selectRefImageEntityOrThrow(state: RefImagesState, id: string, caller: string): RefImageState {
  const entity = selectRefImageEntity(state, id);
  assert(entity, `Entity with id ${id} not found in ${caller}`);
  return entity;
}
