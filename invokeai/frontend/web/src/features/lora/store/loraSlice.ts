import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { ParameterLoRAModel } from 'features/parameters/types/parameterSchemas';
import type { LoRAModelConfigEntity } from 'services/api/endpoints/models';

export type LoRA = ParameterLoRAModel & {
  id: string;
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
    loraAdded: (state, action: PayloadAction<LoRAModelConfigEntity>) => {
      const { model_name, id, base_model } = action.payload;
      state.loras[id] = { id, model_name, base_model, ...defaultLoRAConfig };
    },
    loraRecalled: (state, action: PayloadAction<LoRAModelConfigEntity & { weight: number }>) => {
      const { model_name, id, base_model, weight } = action.payload;
      state.loras[id] = { id, model_name, base_model, weight, isEnabled: true };
    },
    loraRemoved: (state, action: PayloadAction<string>) => {
      const id = action.payload;
      delete state.loras[id];
    },
    lorasCleared: (state) => {
      state.loras = {};
    },
    loraWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
      const { id, weight } = action.payload;
      const lora = state.loras[id];
      if (!lora) {
        return;
      }
      lora.weight = weight;
    },
    loraWeightReset: (state, action: PayloadAction<string>) => {
      const id = action.payload;
      const lora = state.loras[id];
      if (!lora) {
        return;
      }
      lora.weight = defaultLoRAConfig.weight;
    },
    loraIsEnabledChanged: (state, action: PayloadAction<Pick<LoRA, 'id' | 'isEnabled'>>) => {
      const { id, isEnabled } = action.payload;
      const lora = state.loras[id];
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
