import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type { CanvasV2State, CLIPVisionModelV2, IPAdapterConfig, IPAdapterData, IPMethodV2 } from './types';
import { imageDTOToImageWithDims } from './types';

export const selectIPA = (state: CanvasV2State, id: string) => state.ipAdapters.find((ipa) => ipa.id === id);
export const selectIPAOrThrow = (state: CanvasV2State, id: string) => {
  const ipa = selectIPA(state, id);
  assert(ipa, `IP Adapter with id ${id} not found`);
  return ipa;
};

export const ipAdaptersReducers = {
  ipaAdded: {
    reducer: (state, action: PayloadAction<{ id: string; config: IPAdapterConfig }>) => {
      const { id, config } = action.payload;
      const layer: IPAdapterData = {
        id,
        type: 'ip_adapter',
        isEnabled: true,
        ...config,
      };
      state.ipAdapters.push(layer);
    },
    prepare: (config: IPAdapterConfig) => ({ payload: { id: uuidv4(), config } }),
  },
  ipaRecalled: (state, action: PayloadAction<{ data: IPAdapterData }>) => {
    state.ipAdapters.push(action.payload.data);
  },
  ipaIsEnabledToggled: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const ipa = selectIPA(state, id);
    if (ipa) {
      ipa.isEnabled = !ipa.isEnabled;
    }
  },
  ipaDeleted: (state, action: PayloadAction<{ id: string }>) => {
    state.ipAdapters = state.ipAdapters.filter((ipa) => ipa.id !== action.payload.id);
  },
  ipaImageChanged: (state, action: PayloadAction<{ id: string; imageDTO: ImageDTO | null }>) => {
    const { id, imageDTO } = action.payload;
    const ipa = selectIPA(state, id);
    if (!ipa) {
      return;
    }
    ipa.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
  ipaMethodChanged: (state, action: PayloadAction<{ id: string; method: IPMethodV2 }>) => {
    const { id, method } = action.payload;
    const ipa = selectIPA(state, id);
    if (!ipa) {
      return;
    }
    ipa.method = method;
  },
  ipaModelChanged: (
    state,
    action: PayloadAction<{
      id: string;
      modelConfig: IPAdapterModelConfig | null;
    }>
  ) => {
    const { id, modelConfig } = action.payload;
    const ipa = selectIPA(state, id);
    if (!ipa) {
      return;
    }
    if (modelConfig) {
      ipa.model = zModelIdentifierField.parse(modelConfig);
    } else {
      ipa.model = null;
    }
  },
  ipaCLIPVisionModelChanged: (state, action: PayloadAction<{ id: string; clipVisionModel: CLIPVisionModelV2 }>) => {
    const { id, clipVisionModel } = action.payload;
    const ipa = selectIPA(state, id);
    if (!ipa) {
      return;
    }
    ipa.clipVisionModel = clipVisionModel;
  },
  ipaWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
    const { id, weight } = action.payload;
    const ipa = selectIPA(state, id);
    if (!ipa) {
      return;
    }
    ipa.weight = weight;
  },
  ipaBeginEndStepPctChanged: (state, action: PayloadAction<{ id: string; beginEndStepPct: [number, number] }>) => {
    const { id, beginEndStepPct } = action.payload;
    const ipa = selectIPA(state, id);
    if (!ipa) {
      return;
    }
    ipa.beginEndStepPct = beginEndStepPct;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
