import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api';

export const CONTROLNET_MODELS = [
  'lllyasviel/sd-controlnet-canny',
  'lllyasviel/sd-controlnet-depth',
  'lllyasviel/sd-controlnet-hed',
  'lllyasviel/sd-controlnet-seg',
  'lllyasviel/sd-controlnet-openpose',
  'lllyasviel/sd-controlnet-scribble',
  'lllyasviel/sd-controlnet-normal',
  'lllyasviel/sd-controlnet-mlsd',
] as const;

export const CONTROLNET_PROCESSORS = [
  'canny',
  'contentShuffle',
  'hed',
  'lineart',
  'lineartAnime',
  'mediapipeFace',
  'midasDepth',
  'mlsd',
  'normalBae',
  'openpose',
  'pidi',
  'zoeDepth',
] as const;

export type ControlNetModel = (typeof CONTROLNET_MODELS)[number];

export const initialControlNet: Omit<ControlNet, 'controlNetId'> = {
  isEnabled: true,
  model: CONTROLNET_MODELS[0],
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
  controlImage: null,
  processedControlImage: null,
};

export type ControlNet = {
  controlNetId: string;
  isEnabled: boolean;
  model: string;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  controlImage: ImageDTO | null;
  processedControlImage: ImageDTO | null;
};

export type ControlNetState = {
  controlNets: Record<string, ControlNet>;
};

export const initialControlNetState: ControlNetState = {
  controlNets: {},
};

export const controlNetSlice = createSlice({
  name: 'controlNet',
  initialState: initialControlNetState,
  reducers: {
    controlNetAddedFromModel: (
      state,
      action: PayloadAction<{ controlNetId: string; model: ControlNetModel }>
    ) => {
      const { controlNetId, model } = action.payload;
      state.controlNets[controlNetId] = {
        ...initialControlNet,
        controlNetId,
        model,
      };
    },
    controlNetAddedFromImage: (
      state,
      action: PayloadAction<{ controlNetId: string; controlImage: ImageDTO }>
    ) => {
      const { controlNetId, controlImage } = action.payload;
      state.controlNets[controlNetId] = {
        ...initialControlNet,
        controlNetId,
        controlImage,
      };
    },
    controlNetRemoved: (state, action: PayloadAction<string>) => {
      const controlNetId = action.payload;
      delete state.controlNets[controlNetId];
    },
    controlNetToggled: (state, action: PayloadAction<string>) => {
      const controlNetId = action.payload;
      state.controlNets[controlNetId].isEnabled =
        !state.controlNets[controlNetId].isEnabled;
    },
    controlNetImageChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; controlImage: ImageDTO }>
    ) => {
      const { controlNetId, controlImage } = action.payload;
      state.controlNets[controlNetId].controlImage = controlImage;
    },
    controlNetProcessedImageChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        processedControlImage: ImageDTO | null;
      }>
    ) => {
      const { controlNetId, processedControlImage } = action.payload;
      state.controlNets[controlNetId].processedControlImage =
        processedControlImage;
    },
    controlNetModelChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; model: ControlNetModel }>
    ) => {
      const { controlNetId, model } = action.payload;
      state.controlNets[controlNetId].model = model;
    },
    controlNetWeightChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; weight: number }>
    ) => {
      const { controlNetId, weight } = action.payload;
      state.controlNets[controlNetId].weight = weight;
    },
    controlNetBeginStepPctChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; beginStepPct: number }>
    ) => {
      const { controlNetId, beginStepPct } = action.payload;
      state.controlNets[controlNetId].beginStepPct = beginStepPct;
    },
    controlNetEndStepPctChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; endStepPct: number }>
    ) => {
      const { controlNetId, endStepPct } = action.payload;
      state.controlNets[controlNetId].endStepPct = endStepPct;
    },
  },
});

export const {
  controlNetAddedFromModel,
  controlNetAddedFromImage,
  controlNetRemoved,
  controlNetImageChanged,
  controlNetProcessedImageChanged,
  controlNetToggled,
  controlNetModelChanged,
  controlNetWeightChanged,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
} = controlNetSlice.actions;

export default controlNetSlice.reducer;
