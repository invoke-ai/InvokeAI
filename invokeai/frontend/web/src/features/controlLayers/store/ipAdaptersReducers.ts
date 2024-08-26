import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectEntity } from 'features/controlLayers/store/selectors';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { merge } from 'lodash-es';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';

import type {
  CanvasIPAdapterState,
  CanvasV2State,
  CLIPVisionModelV2,
  EntityIdentifierPayload,
  IPMethodV2,
} from './types';
import { getEntityIdentifier, imageDTOToImageWithDims, initialIPAdapter } from './types';

export const ipAdaptersReducers = {
  ipaAdded: {
    reducer: (
      state,
      action: PayloadAction<{ id: string; overrides?: Partial<CanvasIPAdapterState>; isSelected?: boolean }>
    ) => {
      const { id, overrides, isSelected } = action.payload;
      const entity: CanvasIPAdapterState = {
        id,
        type: 'ip_adapter',
        name: null,
        isEnabled: true,
        ipAdapter: deepClone(initialIPAdapter),
      };
      merge(entity, overrides);
      state.ipAdapters.entities.push(entity);
      if (isSelected) {
        state.selectedEntityIdentifier = getEntityIdentifier(entity);
      }
    },
    prepare: (payload?: { overrides?: Partial<CanvasIPAdapterState>; isSelected?: boolean }) => ({
      payload: { ...payload, id: getPrefixedId('ip_adapter') },
    }),
  },
  ipaRecalled: (state, action: PayloadAction<{ data: CanvasIPAdapterState }>) => {
    const { data } = action.payload;
    state.ipAdapters.entities.push(data);
    state.selectedEntityIdentifier = { type: 'ip_adapter', id: data.id };
  },
  ipaImageChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ imageDTO: ImageDTO | null }, 'ip_adapter'>>
  ) => {
    const { entityIdentifier, imageDTO } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
  ipaMethodChanged: (state, action: PayloadAction<EntityIdentifierPayload<{ method: IPMethodV2 }, 'ip_adapter'>>) => {
    const { entityIdentifier, method } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.ipAdapter.method = method;
  },
  ipaModelChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ modelConfig: IPAdapterModelConfig | null }, 'ip_adapter'>>
  ) => {
    const { entityIdentifier, modelConfig } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.ipAdapter.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;
  },
  ipaCLIPVisionModelChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ clipVisionModel: CLIPVisionModelV2 }, 'ip_adapter'>>
  ) => {
    const { entityIdentifier, clipVisionModel } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.ipAdapter.clipVisionModel = clipVisionModel;
  },
  ipaWeightChanged: (state, action: PayloadAction<EntityIdentifierPayload<{ weight: number }, 'ip_adapter'>>) => {
    const { entityIdentifier, weight } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.ipAdapter.weight = weight;
  },
  ipaBeginEndStepPctChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ beginEndStepPct: [number, number] }, 'ip_adapter'>>
  ) => {
    const { entityIdentifier, beginEndStepPct } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.ipAdapter.beginEndStepPct = beginEndStepPct;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
