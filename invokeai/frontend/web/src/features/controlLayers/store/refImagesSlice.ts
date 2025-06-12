import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { canvasMetadataRecalled } from 'features/controlLayers/store/canvasSlice';
import type { FLUXReduxImageInfluence, RefImagesState } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { isEqual } from 'lodash-es';
import type { ApiModelConfig, FLUXReduxModelConfig, ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import type { PartialDeep } from 'type-fest';

import type { CanvasReferenceImageState, CLIPVisionModelV2, IPMethodV2 } from './types';
import { getInitialRefImagesState } from './types';
import {
  getReferenceImageState,
  imageDTOToImageWithDims,
  initialChatGPT4oReferenceImage,
  initialFLUXRedux,
  initialIPAdapter,
} from './util';

type PayloadWithId<T = void> = T extends void
  ? { id: string }
  : {
      id: string;
    } & T;

export const refImagesSlice = createSlice({
  name: 'refImages',
  initialState: getInitialRefImagesState(),
  reducers: {
    referenceImageAdded: {
      reducer: (
        state,
        action: PayloadAction<{
          id: string;
          overrides?: PartialDeep<CanvasReferenceImageState>;
          isSelected?: boolean;
        }>
      ) => {
        const { id, overrides, isSelected } = action.payload;
        const entityState = getReferenceImageState(id, overrides);

        state.entities.push(entityState);

        if (isSelected) {
          state.selectedId = entityState.id;
        }
      },
      prepare: (payload?: { overrides?: PartialDeep<CanvasReferenceImageState>; isSelected?: boolean }) => ({
        payload: { ...payload, id: getPrefixedId('reference_image') },
      }),
    },
    referenceImageRecalled: (state, action: PayloadAction<{ data: CanvasReferenceImageState }>) => {
      const { data } = action.payload;
      state.entities.push(data);
      state.selectedId = data.id;
    },
    referenceImageIPAdapterImageChanged: (
      state,
      action: PayloadAction<PayloadWithId<{ imageDTO: ImageDTO | null }>>
    ) => {
      const { id, imageDTO } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      entity.ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    referenceImageIPAdapterMethodChanged: (state, action: PayloadAction<PayloadWithId<{ method: IPMethodV2 }>>) => {
      const { id, method } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (entity.ipAdapter.type !== 'ip_adapter') {
        return;
      }
      entity.ipAdapter.method = method;
    },
    referenceImageIPAdapterFLUXReduxImageInfluenceChanged: (
      state,
      action: PayloadAction<PayloadWithId<{ imageInfluence: FLUXReduxImageInfluence }>>
    ) => {
      const { id, imageInfluence } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (entity.ipAdapter.type !== 'flux_redux') {
        return;
      }
      entity.ipAdapter.imageInfluence = imageInfluence;
    },
    referenceImageIPAdapterModelChanged: (
      state,
      action: PayloadAction<
        PayloadWithId<{ modelConfig: IPAdapterModelConfig | FLUXReduxModelConfig | ApiModelConfig | null }>
      >
    ) => {
      const { id, modelConfig } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }

      const oldModel = entity.ipAdapter.model;

      // First set the new model
      entity.ipAdapter.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;

      if (!entity.ipAdapter.model) {
        return;
      }

      if (isEqual(oldModel, entity.ipAdapter.model)) {
        // Nothing changed, so we don't need to do anything
        return;
      }

      // The type of ref image depends on the model. When the user switches the model, we rebuild the ref image.
      // When we switch the model, we keep the image the same, but change the other parameters.

      if (entity.ipAdapter.model.base === 'chatgpt-4o') {
        // Switching to chatgpt-4o ref image
        entity.ipAdapter = {
          ...initialChatGPT4oReferenceImage,
          image: entity.ipAdapter.image,
          model: entity.ipAdapter.model,
        };
        return;
      }

      if (entity.ipAdapter.model.type === 'flux_redux') {
        // Switching to flux_redux
        entity.ipAdapter = {
          ...initialFLUXRedux,
          image: entity.ipAdapter.image,
          model: entity.ipAdapter.model,
        };
        return;
      }

      if (entity.ipAdapter.model.type === 'ip_adapter') {
        // Switching to ip_adapter
        entity.ipAdapter = {
          ...initialIPAdapter,
          image: entity.ipAdapter.image,
          model: entity.ipAdapter.model,
        };
        // Ensure that the IP Adapter model is compatible with the CLIP Vision model
        if (entity.ipAdapter.model?.base === 'flux') {
          entity.ipAdapter.clipVisionModel = 'ViT-L';
        } else if (entity.ipAdapter.clipVisionModel === 'ViT-L') {
          // Fall back to ViT-H (ViT-G would also work)
          entity.ipAdapter.clipVisionModel = 'ViT-H';
        }
        return;
      }
    },
    referenceImageIPAdapterCLIPVisionModelChanged: (
      state,
      action: PayloadAction<PayloadWithId<{ clipVisionModel: CLIPVisionModelV2 }>>
    ) => {
      const { id, clipVisionModel } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (entity.ipAdapter.type !== 'ip_adapter') {
        return;
      }
      entity.ipAdapter.clipVisionModel = clipVisionModel;
    },
    referenceImageIPAdapterWeightChanged: (state, action: PayloadAction<PayloadWithId<{ weight: number }>>) => {
      const { id, weight } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (entity.ipAdapter.type !== 'ip_adapter') {
        return;
      }
      entity.ipAdapter.weight = weight;
    },
    referenceImageIPAdapterBeginEndStepPctChanged: (
      state,
      action: PayloadAction<PayloadWithId<{ beginEndStepPct: [number, number] }>>
    ) => {
      const { id, beginEndStepPct } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      if (entity.ipAdapter.type !== 'ip_adapter') {
        return;
      }
      entity.ipAdapter.beginEndStepPct = beginEndStepPct;
    },
    //#region Shared entity
    entitySelected: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        // Cannot select a non-existent entity
        return;
      }
      state.selectedId = id;
    },
    entityNameChanged: (state, action: PayloadAction<PayloadWithId<{ name: string | null }>>) => {
      const { id, name } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      entity.name = name;
    },
    entityDuplicated: (state, action: PayloadAction<PayloadWithId>) => {
      const { id } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }

      const newEntity = deepClone(entity);
      if (newEntity.name) {
        newEntity.name = `${newEntity.name} (Copy)`;
      }
      newEntity.id = getPrefixedId('reference_image');
      state.entities.push(newEntity);

      state.selectedId = newEntity.id;
    },
    entityIsEnabledToggled: (state, action: PayloadAction<PayloadWithId>) => {
      const { id } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      entity.isEnabled = !entity.isEnabled;
    },
    entityIsLockedToggled: (state, action: PayloadAction<PayloadWithId>) => {
      const { id } = action.payload;
      const entity = selectRefImageEntity(state, id);
      if (!entity) {
        return;
      }
      entity.isLocked = !entity.isLocked;
    },
    entityDeleted: (state, action: PayloadAction<PayloadWithId>) => {
      const { id } = action.payload;

      let selectedId: string | null = null;
      const entities = state.entities;
      const index = entities.findIndex((entity) => entity.id === id);
      const nextIndex = entities.length > 1 ? (index + 1) % entities.length : -1;
      if (nextIndex !== -1) {
        const nextEntity = entities[nextIndex];
        if (nextEntity) {
          selectedId = nextEntity.id;
        }
      }
      state.entities = state.entities.filter((rg) => rg.id !== id);
      state.selectedId = selectedId;
    },
    refImagesReset: () => getInitialRefImagesState(),
  },
  extraReducers(builder) {
    builder.addCase(canvasMetadataRecalled, (state, action) => {
      const { referenceImages } = action.payload;
      state.entities = referenceImages;
    });
  },
});

export const {
  referenceImageAdded,
  // referenceImageRecalled,
  referenceImageIPAdapterImageChanged,
  referenceImageIPAdapterMethodChanged,
  referenceImageIPAdapterModelChanged,
  referenceImageIPAdapterCLIPVisionModelChanged,
  referenceImageIPAdapterWeightChanged,
  referenceImageIPAdapterBeginEndStepPctChanged,
  referenceImageIPAdapterFLUXReduxImageInfluenceChanged,
} = refImagesSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const refImagesPersistConfig: PersistConfig<RefImagesState> = {
  name: refImagesSlice.name,
  initialState: getInitialRefImagesState(),
  migrate,
  persistDenylist: [],
};

export const selectRefImagesSlice = (state: RootState) => state.refImages;

export const selectReferenceImageEntities = createSelector(selectRefImagesSlice, (state) => state.entities);
export const selectActiveReferenceImageEntities = createSelector(selectReferenceImageEntities, (entities) =>
  entities.filter((e) => e.isEnabled)
);
export const selectRefImageEntityIds = createMemoizedSelector(selectReferenceImageEntities, (entities) =>
  entities.map((e) => e.id)
);
export const selectRefImageEntity = (state: RefImagesState, id: string) =>
  state.entities.find((entity) => entity.id === id) ?? null;

export function selectRefImageEntityOrThrow(
  state: RefImagesState,
  id: string,
  caller: string
): CanvasReferenceImageState {
  const entity = selectRefImageEntity(state, id);
  assert(entity, `Entity with id ${id} not found in ${caller}`);
  return entity;
}
