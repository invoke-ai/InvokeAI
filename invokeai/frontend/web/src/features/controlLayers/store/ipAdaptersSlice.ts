import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type { CLIPVisionModelV2, IPAdapterConfig, IPAdapterData, IPMethodV2 } from './types';
import { imageDTOToImageWithDims } from './types';

type IPAdaptersState = {
  _version: 1;
  ipAdapters: IPAdapterData[];
};

const initialState: IPAdaptersState = {
  _version: 1,
  ipAdapters: [],
};

export const selectIPA = (state: IPAdaptersState, id: string) => state.ipAdapters.find((ipa) => ipa.id === id);
export const selectIPAOrThrow = (state: IPAdaptersState, id: string) => {
  const ipa = selectIPA(state, id);
  assert(ipa, `IP Adapter with id ${id} not found`);
  return ipa;
};

export const ipAdaptersSlice = createSlice({
  name: 'ipAdapters',
  initialState,
  reducers: {
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
  },
});

export const {
  ipaAdded,
  ipaRecalled,
  ipaIsEnabledToggled,
  ipaDeleted,
  ipaImageChanged,
  ipaMethodChanged,
  ipaModelChanged,
  ipaCLIPVisionModelChanged,
  ipaWeightChanged,
  ipaBeginEndStepPctChanged,
} = ipAdaptersSlice.actions;

export const selectIPAdaptersSlice = (state: RootState) => state.ipAdapters;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const ipAdaptersPersistConfig: PersistConfig<IPAdaptersState> = {
  name: ipAdaptersSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};
