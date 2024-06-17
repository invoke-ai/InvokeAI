import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State, LoRA } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { LoRAModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export const defaultLoRAConfig: Pick<LoRA, 'weight' | 'isEnabled'> = {
  weight: 0.75,
  isEnabled: true,
};

export const selectLoRA = (state: CanvasV2State, id: string) => state.loras.find((lora) => lora.id === id);
export const selectLoRAOrThrow = (state: CanvasV2State, id: string) => {
  const lora = selectLoRA(state, id);
  assert(lora, `LoRA with id ${id} not found`);
  return lora;
};

export const lorasReducers = {
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
} satisfies SliceCaseReducers<CanvasV2State>;
