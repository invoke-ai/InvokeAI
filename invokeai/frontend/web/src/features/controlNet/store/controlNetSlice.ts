import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { ControlNetModelParam } from 'features/parameters/types/parameterSchemas';
import { cloneDeep, forEach } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import { isAnySessionRejected } from 'services/api/thunks/session';
import { appSocketInvocationError } from 'services/events/actions';
import { controlNetImageProcessed } from './actions';
import {
  CONTROLNET_MODEL_DEFAULT_PROCESSORS,
  CONTROLNET_PROCESSORS,
} from './constants';
import {
  ControlNetProcessorType,
  RequiredCannyImageProcessorInvocation,
  RequiredControlNetProcessorNode,
} from './types';

export type ControlModes =
  | 'balanced'
  | 'more_prompt'
  | 'more_control'
  | 'unbalanced';

export const initialControlNet: Omit<ControlNetConfig, 'controlNetId'> = {
  isEnabled: true,
  model: null,
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
  controlMode: 'balanced',
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
  model: ControlNetModelParam | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  controlMode: ControlModes;
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
    controlNetDuplicated: (
      state,
      action: PayloadAction<{
        sourceControlNetId: string;
        newControlNetId: string;
      }>
    ) => {
      const { sourceControlNetId, newControlNetId } = action.payload;

      const newControlnet = cloneDeep(state.controlNets[sourceControlNetId]);
      newControlnet.controlNetId = newControlNetId;
      state.controlNets[newControlNetId] = newControlnet;
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
        model: ControlNetModelParam;
      }>
    ) => {
      const { controlNetId, model } = action.payload;
      state.controlNets[controlNetId].model = model;
      state.controlNets[controlNetId].processedControlImage = null;

      if (state.controlNets[controlNetId].shouldAutoConfig) {
        let processorType: ControlNetProcessorType | undefined = undefined;

        for (const modelSubstring in CONTROLNET_MODEL_DEFAULT_PROCESSORS) {
          if (model.model_name.includes(modelSubstring)) {
            processorType = CONTROLNET_MODEL_DEFAULT_PROCESSORS[modelSubstring];
            break;
          }
        }

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
    controlNetControlModeChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; controlMode: ControlModes }>
    ) => {
      const { controlNetId, controlMode } = action.payload;
      state.controlNets[controlNetId].controlMode = controlMode;
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
        let processorType: ControlNetProcessorType | undefined = undefined;

        for (const modelSubstring in CONTROLNET_MODEL_DEFAULT_PROCESSORS) {
          if (
            state.controlNets[controlNetId].model?.model_name.includes(
              modelSubstring
            )
          ) {
            processorType = CONTROLNET_MODEL_DEFAULT_PROCESSORS[modelSubstring];
            break;
          }
        }

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

    builder.addCase(appSocketInvocationError, (state, action) => {
      state.pendingControlImages = [];
    });

    builder.addMatcher(isAnySessionRejected, (state, action) => {
      state.pendingControlImages = [];
    });

    builder.addMatcher(
      imagesApi.endpoints.deleteImage.matchFulfilled,
      (state, action) => {
        // Preemptively remove the image from all controlnets
        // TODO: doesn't the imageusage stuff do this for us?
        const { image_name } = action.meta.arg.originalArgs;
        forEach(state.controlNets, (c) => {
          if (c.controlImage === image_name) {
            c.controlImage = null;
            c.processedControlImage = null;
          }
          if (c.processedControlImage === image_name) {
            c.processedControlImage = null;
          }
        });
      }
    );
  },
});

export const {
  isControlNetEnabledToggled,
  controlNetAdded,
  controlNetDuplicated,
  controlNetAddedFromImage,
  controlNetRemoved,
  controlNetImageChanged,
  controlNetProcessedImageChanged,
  controlNetToggled,
  controlNetModelChanged,
  controlNetWeightChanged,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
  controlNetControlModeChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
  controlNetReset,
  controlNetAutoConfigToggled,
} = controlNetSlice.actions;

export default controlNetSlice.reducer;

export const controlNetSelector = (state: RootState) => state.controlNet;
