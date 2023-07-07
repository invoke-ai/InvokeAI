import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { LoRAModelParam } from 'features/parameters/store/parameterZodSchemas';
import { LoRAModelConfigEntity } from 'services/api/endpoints/models';
import { BaseModelType } from 'services/api/types';

export type Lora = {
  id: string;
  base_model: BaseModelType;
  name: string;
  weight: number;
};

export const defaultLoRAConfig = {
  weight: 0.75,
};

export type LoraState = {
  loras: Record<string, LoRAModelParam & { weight: number }>;
};

export const intialLoraState: LoraState = {
  loras: {},
};

export const loraSlice = createSlice({
  name: 'lora',
  initialState: intialLoraState,
  reducers: {
    loraAdded: (state, action: PayloadAction<LoRAModelConfigEntity>) => {
      const { name, id, base_model } = action.payload;
      state.loras[id] = { id, name, base_model, ...defaultLoRAConfig };
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
      state.loras[id].weight = weight;
    },
    loraWeightReset: (state, action: PayloadAction<string>) => {
      const id = action.payload;
      state.loras[id].weight = defaultLoRAConfig.weight;
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
