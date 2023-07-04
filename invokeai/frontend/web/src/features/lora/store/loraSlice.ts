import { PayloadAction, createSlice } from '@reduxjs/toolkit';

export type Lora = {
  name: string;
  weight: number;
};

export const defaultLoRAConfig: Omit<Lora, 'name'> = {
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
    loraAdded: (state, action: PayloadAction<string>) => {
      const name = action.payload;
      state.loras[name] = { name, ...defaultLoRAConfig };
    },
    loraRemoved: (state, action: PayloadAction<string>) => {
      const name = action.payload;
      delete state.loras[name];
    },
    loraWeightChanged: (
      state,
      action: PayloadAction<{ name: string; weight: number }>
    ) => {
      const { name, weight } = action.payload;
      state.loras[name].weight = weight;
    },
  },
});

export const { loraAdded, loraRemoved, loraWeightChanged } = loraSlice.actions;

export default loraSlice.reducer;
