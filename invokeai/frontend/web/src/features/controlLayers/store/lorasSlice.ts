import { createSlice, isAnyOf, type PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import type { LoRA, LoRAsState } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { LoRAModelConfig } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

import { selectActiveCanvasId, selectActiveTab } from './selectors';

export const DEFAULT_LORA_WEIGHT_CONFIG = {
  initial: 0.75,
  sliderMin: -1,
  sliderMax: 2,
  numberInputMin: -10,
  numberInputMax: 10,
  fineStep: 0.01,
  coarseStep: 0.05,
};

export const getInitialLoRAsState = (): LoRAsState => ({
  loras: [],
});

export const lorasSlice = createSlice({
  name: 'loras',
  initialState: {} as LoRAsState,
  reducers: {
    loraAdded: {
      reducer: (state, action: PayloadAction<{ model: LoRAModelConfig; id: string }>) => {
        const { model, id } = action.payload;
        const parsedModel = zModelIdentifierField.parse(model);
        const defaultLoRAConfig: Pick<LoRA, 'weight' | 'isEnabled'> = {
          weight: model.default_settings?.weight ?? DEFAULT_LORA_WEIGHT_CONFIG.initial,
          isEnabled: true,
        };
        state.loras.push({ ...defaultLoRAConfig, model: parsedModel, id });
      },
      prepare: (payload: { model: LoRAModelConfig }) => ({ payload: { ...payload, id: uuidv4() } }),
    },
    loraRecalled: (state, action: PayloadAction<{ lora: LoRA }>) => {
      const { lora } = action.payload;
      state.loras = state.loras.filter((l) => l.model.key !== lora.model.key && l.id !== lora.id);
      state.loras.push(lora);
    },
    loraDeleted: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      state.loras = state.loras.filter((lora) => lora.id !== id);
    },
    loraWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
      const { id, weight } = action.payload;
      const lora = findLoRA(state, id);
      if (!lora) {
        return;
      }
      lora.weight = weight;
    },
    loraIsEnabledChanged: (state, action: PayloadAction<{ id: string; isEnabled: boolean }>) => {
      const { id, isEnabled } = action.payload;
      const lora = findLoRA(state, id);
      if (!lora) {
        return;
      }
      lora.isEnabled = isEnabled;
    },
    loraAllDeleted: (state) => {
      state.loras = [];
    },
  },
  extraReducers(builder) {
    builder.addCase(paramsReset, () => {
      // When a new session is requested, clear all LoRAs
      return getInitialLoRAsState();
    });
  },
});

export const isLoRAsStateAction = isAnyOf(...Object.values(lorasSlice.actions), paramsReset);

export const { loraAdded, loraRecalled, loraDeleted, loraWeightChanged, loraIsEnabledChanged, loraAllDeleted } =
  lorasSlice.actions;

const initialLoRAsState = getInitialLoRAsState();

const selectActiveTabLoRAs = (state: RootState) => {
  const tab = selectActiveTab(state);
  const canvasId = selectActiveCanvasId(state);

  switch (tab) {
    case 'generate':
      return state.tab.generate.loras;
    case 'canvas':
      return state.canvas.canvases[canvasId]!.params.loras;
    case 'upscaling':
      return state.tab.upscaling.loras;
    default:
      // Fallback for global controls in other tabs
      return initialLoRAsState;
  }
};

const findLoRA = (state: LoRAsState, id: string) => state.loras.find((lora) => lora.id === id);

export const selectLoRAsSlice = (state: RootState) => selectActiveTabLoRAs(state);
export const selectAddedLoRAs = (state: RootState) => selectActiveTabLoRAs(state).loras;
export const selectLoRA = (state: RootState, id: string) => {
  const loras = selectLoRAsSlice(state);
  return findLoRA(loras, id);
};
