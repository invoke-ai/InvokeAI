import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { LoRAModelConfigEntity } from 'services/api/endpoints/models';

export type Lora = {
  id: string;
  name: string;
  weight: number;
};

export const defaultLoRAConfig: Omit<Lora, 'id' | 'name'> = {
  weight: 1,
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
    loraWeightChanged: (
      state,
      action: PayloadAction<{ id: string; weight: number }>
    ) => {
      const { id, weight } = action.payload;
      state.loras[id].weight = weight;
    },
  },
});

export const { loraAdded, loraRemoved, loraWeightChanged } = loraSlice.actions;

export default loraSlice.reducer;
