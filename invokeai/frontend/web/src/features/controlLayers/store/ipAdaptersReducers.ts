import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { merge } from 'lodash-es';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type { CanvasIPAdapterState, CanvasV2State, CLIPVisionModelV2, IPMethodV2 } from './types';
import { getEntityIdentifier, imageDTOToImageWithDims, initialIPAdapter } from './types';

const selectIPAdapterEntity = (state: CanvasV2State, id: string) =>
  state.ipAdapters.entities.find((ipa) => ipa.id === id);
export const selectIPAdapterEntityOrThrow = (state: CanvasV2State, id: string) => {
  const entity = selectIPAdapterEntity(state, id);
  assert(entity, `IP Adapter with id ${id} not found`);
  return entity;
};

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
  ipaImageChanged: {
    reducer: (state, action: PayloadAction<{ id: string; imageDTO: ImageDTO | null }>) => {
      const { id, imageDTO } = action.payload;
      const entity = selectIPAdapterEntity(state, id);
      if (!entity) {
        return;
      }
      entity.ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    prepare: (payload: { id: string; imageDTO: ImageDTO | null }) => ({ payload: { ...payload, objectId: uuidv4() } }),
  },
  ipaMethodChanged: (state, action: PayloadAction<{ id: string; method: IPMethodV2 }>) => {
    const { id, method } = action.payload;
    const entity = selectIPAdapterEntity(state, id);
    if (!entity) {
      return;
    }
    entity.ipAdapter.method = method;
  },
  ipaModelChanged: (state, action: PayloadAction<{ id: string; modelConfig: IPAdapterModelConfig | null }>) => {
    const { id, modelConfig } = action.payload;
    const entity = selectIPAdapterEntity(state, id);
    if (!entity) {
      return;
    }
    entity.ipAdapter.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;
  },
  ipaCLIPVisionModelChanged: (state, action: PayloadAction<{ id: string; clipVisionModel: CLIPVisionModelV2 }>) => {
    const { id, clipVisionModel } = action.payload;
    const entity = selectIPAdapterEntity(state, id);
    if (!entity) {
      return;
    }
    entity.ipAdapter.clipVisionModel = clipVisionModel;
  },
  ipaWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
    const { id, weight } = action.payload;
    const entity = selectIPAdapterEntity(state, id);
    if (!entity) {
      return;
    }
    entity.ipAdapter.weight = weight;
  },
  ipaBeginEndStepPctChanged: (state, action: PayloadAction<{ id: string; beginEndStepPct: [number, number] }>) => {
    const { id, beginEndStepPct } = action.payload;
    const entity = selectIPAdapterEntity(state, id);
    if (!entity) {
      return;
    }
    entity.ipAdapter.beginEndStepPct = beginEndStepPct;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
