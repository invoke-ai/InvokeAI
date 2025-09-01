import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { type LoRA, zLoRA } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/system/store/configSlice';
import type { LoRAModelConfig } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';
import z from 'zod';

const zLoRAsState = z.object({
  loras: z.array(zLoRA),
});
type LoRAsState = z.infer<typeof zLoRAsState>;

const getInitialState = (): LoRAsState => ({
  loras: [],
});

const selectLoRA = (state: LoRAsState, id: string) => state.loras.find((lora) => lora.id === id);

const slice = createSlice({
  name: 'loras',
  initialState: getInitialState(),
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
  },
  extraReducers(builder) {
    builder.addCase(paramsReset, () => {
      // When a new session is requested, clear all LoRAs
      return getInitialState();
    });
  },
});

export const { loraAdded, loraRecalled, loraDeleted, loraWeightChanged, loraIsEnabledChanged, loraAllDeleted } =
  slice.actions;

export const lorasSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zLoRAsState,
  getInitialState,
  persistConfig: {
    migrate: (state) => zLoRAsState.parse(state),
  },
};

export const selectLoRAsSlice = (state: RootState) => state.loras;
export const selectAddedLoRAs = createSelector(selectLoRAsSlice, (loras) => loras.loras);
export const buildSelectLoRA = (id: string) =>
  createSelector([selectLoRAsSlice], (loras) => {
    return selectLoRA(loras, id);
  });
