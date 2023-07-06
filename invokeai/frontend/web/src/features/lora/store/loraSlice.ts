import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { LoRAModelConfigEntity } from 'services/api/endpoints/models';

export type Lora = {
  id: string;
  name: string;
  weight: number;
};

export const defaultLoRAConfig: Omit<Lora, 'id' | 'name'> = {
  weight: 0.75,
};

export type LoraState = {
  loras: Record<string, Lora>;
};

export const intialLoraState: LoraState = {
  loras: {},
};

export const loraSlice = createSlice({
  name: 'lora',
  initialState: intialLoraState,
  reducers: {
    loraAdded: (state, action: PayloadAction<LoRAModelConfigEntity>) => {
      const { name, id } = action.payload;
      state.loras[id] = { id, name, ...defaultLoRAConfig };
    },
    loraRemoved: (state, action: PayloadAction<string>) => {
      const id = action.payload;
      delete state.loras[id];
    },
    lorasCleared: (state, action: PayloadAction<>) => {
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
