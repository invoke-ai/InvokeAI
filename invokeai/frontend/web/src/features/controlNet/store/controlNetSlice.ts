import { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { ImageDTO } from 'services/api/types';
import {
  ControlNetProcessorType,
  RequiredCannyImageProcessorInvocation,
  RequiredControlNetProcessorNode,
} from './types';
import {
  CONTROLNET_MODELS,
  CONTROLNET_PROCESSORS,
  ControlNetModelName,
} from './constants';
import { controlNetImageProcessed } from './actions';
import { imageDeleted, imageUrlsReceived } from 'services/api/thunks/image';
import { forEach } from 'lodash-es';
import { isAnySessionRejected } from 'services/api/thunks/session';
import { appSocketInvocationError } from 'services/events/actions';

export const initialControlNet: Omit<ControlNetConfig, 'controlNetId'> = {
  isEnabled: true,
  model: CONTROLNET_MODELS['lllyasviel/control_v11p_sd15_canny'].type,
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
  controlImage: null,
  processedControlImage: null,
  processorType: 'canny_image_processor',
  processorNode: CONTROLNET_PROCESSORS.canny_image_processor
    .default as RequiredCannyImageProcessorInvocation,
  shouldAutoConfig: true,
};

export type ControlNetConfig = {
  controlNetId: string;
  isEnabled: boolean;
  model: ControlNetModelName;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  controlImage: string | null;
  processedControlImage: string | null;
  processorType: ControlNetProcessorType;
  processorNode: RequiredControlNetProcessorNode;
  shouldAutoConfig: boolean;
};

export type ControlNetState = {
  controlNets: Record<string, ControlNetConfig>;
  isEnabled: boolean;
  pendingControlImages: string[];
};

export const initialControlNetState: ControlNetState = {
  controlNets: {},
  isEnabled: false,
  pendingControlImages: [],
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
      action: PayloadAction<{
        controlNetId: string;
        controlNet?: ControlNetConfig;
      }>
    ) => {
      const { controlNetId, controlNet } = action.payload;
      state.controlNets[controlNetId] = {
        ...(controlNet ?? initialControlNet),
        controlNetId,
      };
    },
    controlNetAddedFromImage: (
      state,
      action: PayloadAction<{ controlNetId: string; controlImage: string }>
    ) => {
      const { controlNetId, controlImage } = action.payload;
      state.controlNets[controlNetId] = {
        ...initialControlNet,
        controlNetId,
        controlImage,
      };
    },
    controlNetRemoved: (
      state,
      action: PayloadAction<{ controlNetId: string }>
    ) => {
      const { controlNetId } = action.payload;
      delete state.controlNets[controlNetId];
    },
    controlNetToggled: (
      state,
      action: PayloadAction<{ controlNetId: string }>
    ) => {
      const { controlNetId } = action.payload;
      state.controlNets[controlNetId].isEnabled =
        !state.controlNets[controlNetId].isEnabled;
    },
    controlNetImageChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        controlImage: string | null;
      }>
    ) => {
      const { controlNetId, controlImage } = action.payload;
      state.controlNets[controlNetId].controlImage = controlImage;
      state.controlNets[controlNetId].processedControlImage = null;
      if (
        controlImage !== null &&
        state.controlNets[controlNetId].processorType !== 'none'
      ) {
        state.pendingControlImages.push(controlNetId);
      }
    },
    controlNetProcessedImageChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        processedControlImage: string | null;
      }>
    ) => {
      const { controlNetId, processedControlImage } = action.payload;
      state.controlNets[controlNetId].processedControlImage =
        processedControlImage;
      state.pendingControlImages = state.pendingControlImages.filter(
        (id) => id !== controlNetId
      );
    },
    controlNetModelChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        model: ControlNetModelName;
      }>
    ) => {
      const { controlNetId, model } = action.payload;
      state.controlNets[controlNetId].model = model;
      state.controlNets[controlNetId].processedControlImage = null;

      if (state.controlNets[controlNetId].shouldAutoConfig) {
        const processorType = CONTROLNET_MODELS[model].defaultProcessor;
        if (processorType) {
          state.controlNets[controlNetId].processorType = processorType;
          state.controlNets[controlNetId].processorNode = CONTROLNET_PROCESSORS[
            processorType
          ].default as RequiredControlNetProcessorNode;
        } else {
          state.controlNets[controlNetId].processorType = 'none';
          state.controlNets[controlNetId].processorNode = CONTROLNET_PROCESSORS
            .none.default as RequiredControlNetProcessorNode;
        }
      }
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
      state.controlNets[controlNetId].shouldAutoConfig = false;
    },
    controlNetProcessorTypeChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        processorType: ControlNetProcessorType;
      }>
    ) => {
      const { controlNetId, processorType } = action.payload;
      state.controlNets[controlNetId].processedControlImage = null;
      state.controlNets[controlNetId].processorType = processorType;
      state.controlNets[controlNetId].processorNode = CONTROLNET_PROCESSORS[
        processorType
      ].default as RequiredControlNetProcessorNode;
      state.controlNets[controlNetId].shouldAutoConfig = false;
    },
    controlNetAutoConfigToggled: (
      state,
      action: PayloadAction<{
        controlNetId: string;
      }>
    ) => {
      const { controlNetId } = action.payload;
      const newShouldAutoConfig =
        !state.controlNets[controlNetId].shouldAutoConfig;

      if (newShouldAutoConfig) {
        // manage the processor for the user
        const processorType =
          CONTROLNET_MODELS[state.controlNets[controlNetId].model]
            .defaultProcessor;
        if (processorType) {
          state.controlNets[controlNetId].processorType = processorType;
          state.controlNets[controlNetId].processorNode = CONTROLNET_PROCESSORS[
            processorType
          ].default as RequiredControlNetProcessorNode;
        } else {
          state.controlNets[controlNetId].processorType = 'none';
          state.controlNets[controlNetId].processorNode = CONTROLNET_PROCESSORS
            .none.default as RequiredControlNetProcessorNode;
        }
      }

      state.controlNets[controlNetId].shouldAutoConfig = newShouldAutoConfig;
    },
    controlNetReset: () => {
      return { ...initialControlNetState };
    },
  },
  extraReducers: (builder) => {
    builder.addCase(controlNetImageProcessed, (state, action) => {
      if (
        state.controlNets[action.payload.controlNetId].controlImage !== null
      ) {
        state.pendingControlImages.push(action.payload.controlNetId);
      }
    });

    builder.addCase(imageDeleted.pending, (state, action) => {
      // Preemptively remove the image from the gallery
      const { image_name } = action.meta.arg;
      forEach(state.controlNets, (c) => {
        if (c.controlImage === image_name) {
          c.controlImage = null;
          c.processedControlImage = null;
        }
        if (c.processedControlImage === image_name) {
          c.processedControlImage = null;
        }
      });
    });

    // builder.addCase(imageUrlsReceived.fulfilled, (state, action) => {
    //   const { image_name, image_url, thumbnail_url } = action.payload;

    //   forEach(state.controlNets, (c) => {
    //     if (c.controlImage?.image_name === image_name) {
    //       c.controlImage.image_url = image_url;
    //       c.controlImage.thumbnail_url = thumbnail_url;
    //     }
    //     if (c.processedControlImage?.image_name === image_name) {
    //       c.processedControlImage.image_url = image_url;
    //       c.processedControlImage.thumbnail_url = thumbnail_url;
    //     }
    //   });
    // });

    builder.addCase(appSocketInvocationError, (state, action) => {
      state.pendingControlImages = [];
    });

    builder.addMatcher(isAnySessionRejected, (state, action) => {
      state.pendingControlImages = [];
    });
  },
});

export const {
  isControlNetEnabledToggled,
  controlNetAdded,
  controlNetAddedFromImage,
  controlNetRemoved,
  controlNetImageChanged,
  controlNetProcessedImageChanged,
  controlNetToggled,
  controlNetModelChanged,
  controlNetWeightChanged,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
  controlNetReset,
  controlNetAutoConfigToggled,
} = controlNetSlice.actions;

export default controlNetSlice.reducer;

export const controlNetSelector = (state: RootState) => state.controlNet;
