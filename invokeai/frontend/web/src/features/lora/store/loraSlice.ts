import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ParameterLoRAModel } from 'features/parameters/types/parameterSchemas';
import type { LoRAModelConfig } from 'services/api/types';

export type LoRA = {
  model: ParameterLoRAModel;
  weight: number;
  isEnabled?: boolean;
};

export const defaultLoRAConfig: Pick<LoRA, 'weight' | 'isEnabled'> = {
  weight: 0.75,
  isEnabled: true,
};

type LoraState = {
  _version: 1;
  loras: Record<string, LoRA>;
};

const initialLoraState: LoraState = {
  _version: 1,
  loras: {},
};

export const loraSlice = createSlice({
  name: 'lora',
  initialState: initialLoraState,
  reducers: {
    loraAdded: (state, action: PayloadAction<LoRAModelConfig>) => {
      const model = zModelIdentifierField.parse(action.payload);
      state.loras[model.key] = { ...defaultLoRAConfig, model };
    },
    loraRecalled: (state, action: PayloadAction<LoRA>) => {
      state.loras[action.payload.model.key] = action.payload;
    },
    loraRemoved: (state, action: PayloadAction<string>) => {
      const key = action.payload;
      delete state.loras[key];
    },
    loraWeightChanged: (state, action: PayloadAction<{ key: string; weight: number }>) => {
      const { key, weight } = action.payload;
      const lora = state.loras[key];
      if (!lora) {
        return;
      }
      lora.weight = weight;
    },
    loraIsEnabledChanged: (state, action: PayloadAction<{ key: string; isEnabled: boolean }>) => {
      const { key, isEnabled } = action.payload;
      const lora = state.loras[key];
      if (!lora) {
        return;
      }
      lora.isEnabled = isEnabled;
    },
  },
});

export const { loraAdded, loraRemoved, loraWeightChanged, loraIsEnabledChanged, loraRecalled } = loraSlice.actions;

export const selectLoraSlice = (state: RootState) => state.lora;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateLoRAState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const loraPersistConfig: PersistConfig<LoraState> = {
  name: loraSlice.name,
  initialState: initialLoraState,
  migrate: migrateLoRAState,
  persistDenylist: [],
};
