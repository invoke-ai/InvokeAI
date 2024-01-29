import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { ParameterLoRAModel } from 'features/parameters/types/parameterSchemas';
import type { LoRAModelConfigEntity } from 'services/api/endpoints/models';

export type LoRA = ParameterLoRAModel & {
  id: string;
  weight: number;
};

export const defaultLoRAConfig = {
  weight: 0.75,
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
      state.loras[id] = { id, model_name, base_model, weight };
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
  },
});

export const { loraAdded, loraRemoved, loraWeightChanged, loraWeightReset, lorasCleared, loraRecalled } =
  loraSlice.actions;

export default loraSlice.reducer;

export const selectLoraSlice = (state: RootState) => state.lora;

export const selectNonZeroWeightLoraSlice = (state: RootState) => {
  const nonZeroWeightLoras = Object.entries(state.lora.loras)
    .filter(([, value]: [string, LoRA]) => value.weight !== 0)
    .reduce((obj: { [key: string]: LoRA }, [key, value]: [string, LoRA]) => {
      obj[key] = value
      return obj
    }, {});

  return {...state.lora, loras: nonZeroWeightLoras};
}

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateLoRAState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};
