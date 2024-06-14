import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { CLIPVisionModelV2, IPMethodV2 } from 'features/controlLayers/util/controlAdapters';
import { imageDTOToImageWithDims } from 'features/controlLayers/util/controlAdapters';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

import type { IPAdapterConfig, IPAdapterData } from './types';

type IPAdaptersState = {
  _version: 1;
  ipAdapters: IPAdapterData[];
};

const initialState: IPAdaptersState = {
  _version: 1,
  ipAdapters: [],
};

const selectIpa = (state: IPAdaptersState, id: string) => state.ipAdapters.find((ipa) => ipa.id === id);

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
    ipaIsEnabledChanged: (state, action: PayloadAction<{ id: string; isEnabled: boolean }>) => {
      const { id, isEnabled } = action.payload;
      const ipa = selectIpa(state, id);
      if (ipa) {
        ipa.isEnabled = isEnabled;
      }
    },
    ipaDeleted: (state, action: PayloadAction<{ id: string }>) => {
      state.ipAdapters = state.ipAdapters.filter((ipa) => ipa.id !== action.payload.id);
    },
    ipaImageChanged: (state, action: PayloadAction<{ id: string; imageDTO: ImageDTO | null }>) => {
      const { id, imageDTO } = action.payload;
      const ipa = selectIpa(state, id);
      if (!ipa) {
        return;
      }
      ipa.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    ipaMethodChanged: (state, action: PayloadAction<{ id: string; method: IPMethodV2 }>) => {
      const { id, method } = action.payload;
      const ipa = selectIpa(state, id);
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
      const ipa = selectIpa(state, id);
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
      const ipa = selectIpa(state, id);
      if (!ipa) {
        return;
      }
      ipa.clipVisionModel = clipVisionModel;
    },
    ipaWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
      const { id, weight } = action.payload;
      const ipa = selectIpa(state, id);
      if (!ipa) {
        return;
      }
      ipa.weight = weight;
    },
    ipaBeginEndStepPctChanged: (state, action: PayloadAction<{ id: string; beginEndStepPct: [number, number] }>) => {
      const { id, beginEndStepPct } = action.payload;
      const ipa = selectIpa(state, id);
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
  ipaIsEnabledChanged,
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
