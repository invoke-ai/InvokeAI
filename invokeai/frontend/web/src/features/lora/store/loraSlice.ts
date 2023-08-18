import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { LoRAModelParam } from 'features/parameters/types/parameterSchemas';
import { LoRAModelConfigEntity } from 'services/api/endpoints/models';

export type LoRA = LoRAModelParam & {
  id: string;
  weight: number;
};

export const defaultLoRAConfig = {
  weight: 0.75,
};

export type LoraState = {
  loras: Record<string, LoRA>;
};

export const intialLoraState: LoraState = {
  loras: {},
};

export const loraSlice = createSlice({
  name: 'lora',
  initialState: intialLoraState,
  reducers: {
    loraAdded: (state, action: PayloadAction<LoRAModelConfigEntity>) => {
      const { model_name, id, base_model } = action.payload;
      state.loras[id] = { id, model_name, base_model, ...defaultLoRAConfig };
    },
    loraRemoved: (state, action: PayloadAction<string>) => {
      const id = action.payload;
      delete state.loras[id];
    },
    lorasCleared: (state) => {
      state.loras = {};
    },
    loraWeightChanged: (
      state,
      action: PayloadAction<{ id: string; weight: number }>
    ) => {
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

export const {
  loraAdded,
  loraRemoved,
  loraWeightChanged,
  loraWeightReset,
  lorasCleared,
} = loraSlice.actions;

export default loraSlice.reducer;
