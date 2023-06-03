import { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { ImageDTO } from 'services/api';
import {
  ControlNetProcessorType,
  RequiredCannyImageProcessorInvocation,
  RequiredControlNetProcessorNode,
} from './types';
import { CONTROLNET_PROCESSORS } from './constants';
import { controlNetImageProcessed } from './actions';

export const CONTROLNET_MODELS = [
  'lllyasviel/sd-controlnet-canny',
  'lllyasviel/sd-controlnet-depth',
  'lllyasviel/sd-controlnet-hed',
  'lllyasviel/sd-controlnet-seg',
  'lllyasviel/sd-controlnet-openpose',
  'lllyasviel/sd-controlnet-scribble',
  'lllyasviel/sd-controlnet-normal',
  'lllyasviel/sd-controlnet-mlsd',
];

export type ControlNetModel = (typeof CONTROLNET_MODELS)[number];

export const initialControlNet: Omit<ControlNet, 'controlNetId'> = {
  isEnabled: true,
  model: CONTROLNET_MODELS[0],
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
  controlImage: null,
  isControlImageProcessed: false,
  processedControlImage: null,
  processorNode: CONTROLNET_PROCESSORS.canny_image_processor
    .default as RequiredCannyImageProcessorInvocation,
};

export type ControlNet = {
  controlNetId: string;
  isEnabled: boolean;
  model: ControlNetModel;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  controlImage: ImageDTO | null;
  isControlImageProcessed: boolean;
  processedControlImage: ImageDTO | null;
  processorNode: RequiredControlNetProcessorNode;
};

export type ControlNetState = {
  controlNets: Record<string, ControlNet>;
  isEnabled: boolean;
  shouldAutoProcess: boolean;
  isProcessingControlImage: boolean;
};

export const initialControlNetState: ControlNetState = {
  controlNets: {},
  isEnabled: false,
  shouldAutoProcess: true,
  isProcessingControlImage: false,
};

export const controlNetSlice = createSlice({
  name: 'controlNet',
  initialState: initialControlNetState,
  reducers: {
    isControlNetEnabledToggled: (state) => {
      state.isEnabled = !state.isEnabled;
    },
    controlNetAdded: (
      state,
      action: PayloadAction<{ controlNetId: string; controlNet?: ControlNet }>
    ) => {
      const { controlNetId, controlNet } = action.payload;
      state.controlNets[controlNetId] = {
        ...(controlNet ?? initialControlNet),
        controlNetId,
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
      action: PayloadAction<{
        controlNetId: string;
        controlImage: ImageDTO | null;
      }>
    ) => {
      const { controlNetId, controlImage } = action.payload;
      state.controlNets[controlNetId].controlImage = controlImage;
      state.controlNets[controlNetId].processedControlImage = null;
      if (state.shouldAutoProcess && controlImage !== null) {
        state.isProcessingControlImage = true;
      }
    },
    isControlNetImageProcessedToggled: (
      state,
      action: PayloadAction<string>
    ) => {
      const controlNetId = action.payload;
      state.controlNets[controlNetId].isControlImageProcessed =
        !state.controlNets[controlNetId].isControlImageProcessed;
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
      state.isProcessingControlImage = false;
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
    controlNetProcessorParamsChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        changes: Omit<
          Partial<RequiredControlNetProcessorNode>,
          'id' | 'type' | 'is_intermediate'
        >;
      }>
    ) => {
      const { controlNetId, changes } = action.payload;
      const processorNode = state.controlNets[controlNetId].processorNode;
      state.controlNets[controlNetId].processorNode = {
        ...processorNode,
        ...changes,
      };
    },
    controlNetProcessorTypeChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        processorType: ControlNetProcessorType;
      }>
    ) => {
      const { controlNetId, processorType } = action.payload;
      state.controlNets[controlNetId].processorNode = CONTROLNET_PROCESSORS[
        processorType
      ].default as RequiredControlNetProcessorNode;
    },
    shouldAutoProcessToggled: (state) => {
      state.shouldAutoProcess = !state.shouldAutoProcess;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(controlNetImageProcessed, (state, action) => {
      if (
        state.controlNets[action.payload.controlNetId].controlImage !== null
      ) {
        state.isProcessingControlImage = true;
      }
    });
  },
});

export const {
  isControlNetEnabledToggled,
  controlNetAdded,
  controlNetAddedFromImage,
  controlNetRemoved,
  controlNetImageChanged,
  isControlNetImageProcessedToggled,
  controlNetProcessedImageChanged,
  controlNetToggled,
  controlNetModelChanged,
  controlNetWeightChanged,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
  shouldAutoProcessToggled,
} = controlNetSlice.actions;

export default controlNetSlice.reducer;

export const controlNetSelector = (state: RootState) => state.controlNet;
