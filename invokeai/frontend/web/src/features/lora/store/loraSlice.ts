import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { ParameterLoRAModel } from 'features/parameters/types/parameterSchemas';
import type { LoRAConfig } from 'services/api/types';

export type LoRA = ParameterLoRAModel & {
  weight: number;
  isEnabled?: boolean;
};

export const defaultLoRAConfig: Pick<LoRA, 'weight' | 'isEnabled'> = {
  weight: 0.75,
  isEnabled: true,
};

export type LoraState = {
  _version: 1;
  loras: Record<string, LoRA>;
};

export const initialLoraState: LoraState = {
  _version: 1,
  loras: {},
};

export const loraSlice = createSlice({
  name: 'lora',
  initialState: initialLoraState,
  reducers: {
    loraAdded: (state, action: PayloadAction<LoRAConfig>) => {
      const { key, base } = action.payload;
      state.loras[key] = { key, base, ...defaultLoRAConfig };
    },
    loraRecalled: (state, action: PayloadAction<LoRAConfig & { weight: number }>) => {
      const { key, base, weight } = action.payload;
      state.loras[key] = { key, base, weight, isEnabled: true };
    },
    loraRemoved: (state, action: PayloadAction<string>) => {
      const key = action.payload;
      delete state.loras[key];
    },
    lorasCleared: (state) => {
      state.loras = {};
    },
    loraWeightChanged: (state, action: PayloadAction<{ key: string; weight: number }>) => {
      const { key, weight } = action.payload;
      const lora = state.loras[key];
      if (!lora) {
        return;
      }
      lora.weight = weight;
    },
    loraWeightReset: (state, action: PayloadAction<string>) => {
      const key = action.payload;
      const lora = state.loras[key];
      if (!lora) {
        return;
      }
      lora.weight = defaultLoRAConfig.weight;
    },
    loraIsEnabledChanged: (state, action: PayloadAction<Pick<LoRA, 'key' | 'isEnabled'>>) => {
      const { key, isEnabled } = action.payload;
      const lora = state.loras[key];
      if (!lora) {
        return;
      }
      lora.isEnabled = isEnabled;
    },
  },
});

export const {
  loraAdded,
  loraRemoved,
  loraWeightChanged,
  loraWeightReset,
  loraIsEnabledChanged,
  lorasCleared,
  loraRecalled,
} = loraSlice.actions;

export const selectLoraSlice = (state: RootState) => state.lora;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateLoRAState = (state: any): any => {
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
