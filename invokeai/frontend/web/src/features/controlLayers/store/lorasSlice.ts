import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import type { LoRA } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { LoRAModelConfig } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

import { newSessionRequested } from './actions';

type LoRAsState = {
  loras: LoRA[];
};

export const defaultLoRAConfig: Pick<LoRA, 'weight' | 'isEnabled'> = {
  weight: 0.75,
  isEnabled: true,
};

const initialState: LoRAsState = {
  loras: [],
};

const selectLoRA = (state: LoRAsState, id: string) => state.loras.find((lora) => lora.id === id);

export const lorasSlice = createSlice({
  name: 'loras',
  initialState,
  reducers: {
    loraAdded: {
      reducer: (state, action: PayloadAction<{ model: LoRAModelConfig; id: string }>) => {
        const { model, id } = action.payload;
        const parsedModel = zModelIdentifierField.parse(model);
        state.loras.push({ ...defaultLoRAConfig, model: parsedModel, id });
      },
      prepare: (payload: { model: LoRAModelConfig }) => ({ payload: { ...payload, id: uuidv4() } }),
    },
    loraRecalled: (state, action: PayloadAction<{ lora: LoRA }>) => {
      const { lora } = action.payload;
      state.loras = state.loras.filter((l) => l.model.key !== lora.model.key && l.id !== lora.id);
      state.loras.push(lora);
    },
    loraDeleted: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      state.loras = state.loras.filter((lora) => lora.id !== id);
    },
    loraWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
      const { id, weight } = action.payload;
      const lora = selectLoRA(state, id);
      if (!lora) {
        return;
      }
      lora.weight = weight;
    },
    loraIsEnabledChanged: (state, action: PayloadAction<{ id: string; isEnabled: boolean }>) => {
      const { id, isEnabled } = action.payload;
      const lora = selectLoRA(state, id);
      if (!lora) {
        return;
      }
      lora.isEnabled = isEnabled;
    },
    loraAllDeleted: (state) => {
      state.loras = [];
    },
  },
  extraReducers(builder) {
    builder.addMatcher(newSessionRequested, () => {
      // When a new session is requested, clear all LoRAs
      return deepClone(initialState);
    });
  },
});

export const { loraAdded, loraRecalled, loraDeleted, loraWeightChanged, loraIsEnabledChanged, loraAllDeleted } =
  lorasSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const lorasPersistConfig: PersistConfig<LoRAsState> = {
  name: lorasSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};

export const selectLoRAsSlice = (state: RootState) => state.loras;
export const selectAddedLoRAs = createSelector(selectLoRAsSlice, (loras) => loras.loras);
