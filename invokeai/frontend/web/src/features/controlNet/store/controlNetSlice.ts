import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import {
  ControlNetModelParam,
  IPAdapterModelParam,
} from 'features/parameters/types/parameterSchemas';
import { cloneDeep, forEach } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import { components } from 'services/api/schema';
import { ImageDTO } from 'services/api/types';
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

export type ControlModes = NonNullable<
  components['schemas']['ControlNetInvocation']['control_mode']
>;

export type ResizeModes = NonNullable<
  components['schemas']['ControlNetInvocation']['resize_mode']
>;

export const initialControlNet: Omit<ControlNetConfig, 'controlNetId'> = {
  isEnabled: true,
  model: null,
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
  controlMode: 'balanced',
  resizeMode: 'just_resize',
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
  resizeMode: ResizeModes;
  controlImage: string | null;
  processedControlImage: string | null;
  processorType: ControlNetProcessorType;
  processorNode: RequiredControlNetProcessorNode;
  shouldAutoConfig: boolean;
};

export type IPAdapterConfig = {
  adapterImage: ImageDTO | null;
  model: IPAdapterModelParam | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
};

export type ControlNetState = {
  controlNets: Record<string, ControlNetConfig>;
  isEnabled: boolean;
  pendingControlImages: string[];
  isIPAdapterEnabled: boolean;
  ipAdapterInfo: IPAdapterConfig;
};

export const initialIPAdapterState: IPAdapterConfig = {
  adapterImage: null,
  model: null,
  weight: 1,
  beginStepPct: 0,
  endStepPct: 1,
};

export const initialControlNetState: ControlNetState = {
  controlNets: {},
  isEnabled: false,
  pendingControlImages: [],
  isIPAdapterEnabled: false,
  ipAdapterInfo: { ...initialIPAdapterState },
};

export const controlNetSlice = createSlice({
  name: 'controlNet',
  initialState: initialControlNetState,
  reducers: {
    isControlNetEnabledToggled: (state) => {
      state.isEnabled = !state.isEnabled;
    },
    controlNetEnabled: (state) => {
      state.isEnabled = true;
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
    controlNetRecalled: (state, action: PayloadAction<ControlNetConfig>) => {
      const controlNet = action.payload;
      state.controlNets[controlNet.controlNetId] = {
        ...controlNet,
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
      const oldControlNet = state.controlNets[sourceControlNetId];
      if (!oldControlNet) {
        return;
      }
      const newControlnet = cloneDeep(oldControlNet);
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
    controlNetIsEnabledChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; isEnabled: boolean }>
    ) => {
      const { controlNetId, isEnabled } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }
      cn.isEnabled = isEnabled;
    },
    controlNetImageChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        controlImage: string | null;
      }>
    ) => {
      const { controlNetId, controlImage } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.controlImage = controlImage;
      cn.processedControlImage = null;
      if (controlImage !== null && cn.processorType !== 'none') {
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
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.processedControlImage = processedControlImage;
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
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.model = model;
      cn.processedControlImage = null;

      if (cn.shouldAutoConfig) {
        let processorType: ControlNetProcessorType | undefined = undefined;

        for (const modelSubstring in CONTROLNET_MODEL_DEFAULT_PROCESSORS) {
          if (model.model_name.includes(modelSubstring)) {
            processorType = CONTROLNET_MODEL_DEFAULT_PROCESSORS[modelSubstring];
            break;
          }
        }

        if (processorType) {
          cn.processorType = processorType;
          cn.processorNode = CONTROLNET_PROCESSORS[processorType]
            .default as RequiredControlNetProcessorNode;
        } else {
          cn.processorType = 'none';
          cn.processorNode = CONTROLNET_PROCESSORS.none
            .default as RequiredControlNetProcessorNode;
        }
      }
    },
    controlNetWeightChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; weight: number }>
    ) => {
      const { controlNetId, weight } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.weight = weight;
    },
    controlNetBeginStepPctChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; beginStepPct: number }>
    ) => {
      const { controlNetId, beginStepPct } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.beginStepPct = beginStepPct;
    },
    controlNetEndStepPctChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; endStepPct: number }>
    ) => {
      const { controlNetId, endStepPct } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.endStepPct = endStepPct;
    },
    controlNetControlModeChanged: (
      state,
      action: PayloadAction<{ controlNetId: string; controlMode: ControlModes }>
    ) => {
      const { controlNetId, controlMode } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.controlMode = controlMode;
    },
    controlNetResizeModeChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        resizeMode: ResizeModes;
      }>
    ) => {
      const { controlNetId, resizeMode } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.resizeMode = resizeMode;
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
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      const processorNode = cn.processorNode;
      cn.processorNode = {
        ...processorNode,
        ...changes,
      };
      cn.shouldAutoConfig = false;
    },
    controlNetProcessorTypeChanged: (
      state,
      action: PayloadAction<{
        controlNetId: string;
        processorType: ControlNetProcessorType;
      }>
    ) => {
      const { controlNetId, processorType } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      cn.processedControlImage = null;
      cn.processorType = processorType;
      cn.processorNode = CONTROLNET_PROCESSORS[processorType]
        .default as RequiredControlNetProcessorNode;
      cn.shouldAutoConfig = false;
    },
    controlNetAutoConfigToggled: (
      state,
      action: PayloadAction<{
        controlNetId: string;
      }>
    ) => {
      const { controlNetId } = action.payload;
      const cn = state.controlNets[controlNetId];
      if (!cn) {
        return;
      }

      const newShouldAutoConfig = !cn.shouldAutoConfig;

      if (newShouldAutoConfig) {
        // manage the processor for the user
        let processorType: ControlNetProcessorType | undefined = undefined;

        for (const modelSubstring in CONTROLNET_MODEL_DEFAULT_PROCESSORS) {
          if (cn.model?.model_name.includes(modelSubstring)) {
            processorType = CONTROLNET_MODEL_DEFAULT_PROCESSORS[modelSubstring];
            break;
          }
        }

        if (processorType) {
          cn.processorType = processorType;
          cn.processorNode = CONTROLNET_PROCESSORS[processorType]
            .default as RequiredControlNetProcessorNode;
        } else {
          cn.processorType = 'none';
          cn.processorNode = CONTROLNET_PROCESSORS.none
            .default as RequiredControlNetProcessorNode;
        }
      }

      cn.shouldAutoConfig = newShouldAutoConfig;
    },
    controlNetReset: () => {
      return { ...initialControlNetState };
    },
    isIPAdapterEnabledChanged: (state, action: PayloadAction<boolean>) => {
      state.isIPAdapterEnabled = action.payload;
    },
    ipAdapterRecalled: (state, action: PayloadAction<IPAdapterConfig>) => {
      state.ipAdapterInfo = action.payload;
    },
    ipAdapterImageChanged: (state, action: PayloadAction<ImageDTO | null>) => {
      state.ipAdapterInfo.adapterImage = action.payload;
    },
    ipAdapterWeightChanged: (state, action: PayloadAction<number>) => {
      state.ipAdapterInfo.weight = action.payload;
    },
    ipAdapterModelChanged: (
      state,
      action: PayloadAction<IPAdapterModelParam | null>
    ) => {
      state.ipAdapterInfo.model = action.payload;
    },
    ipAdapterBeginStepPctChanged: (state, action: PayloadAction<number>) => {
      state.ipAdapterInfo.beginStepPct = action.payload;
    },
    ipAdapterEndStepPctChanged: (state, action: PayloadAction<number>) => {
      state.ipAdapterInfo.endStepPct = action.payload;
    },
    ipAdapterStateReset: (state) => {
      state.isIPAdapterEnabled = false;
      state.ipAdapterInfo = { ...initialIPAdapterState };
    },
    clearPendingControlImages: (state) => {
      state.pendingControlImages = [];
    },
  },
  extraReducers: (builder) => {
    builder.addCase(controlNetImageProcessed, (state, action) => {
      const cn = state.controlNets[action.payload.controlNetId];
      if (!cn) {
        return;
      }
      if (cn.controlImage !== null) {
        state.pendingControlImages.push(action.payload.controlNetId);
      }
    });

    builder.addCase(appSocketInvocationError, (state) => {
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
  controlNetEnabled,
  controlNetAdded,
  controlNetRecalled,
  controlNetDuplicated,
  controlNetAddedFromImage,
  controlNetRemoved,
  controlNetImageChanged,
  controlNetProcessedImageChanged,
  controlNetIsEnabledChanged,
  controlNetModelChanged,
  controlNetWeightChanged,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
  controlNetControlModeChanged,
  controlNetResizeModeChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
  controlNetReset,
  controlNetAutoConfigToggled,
  isIPAdapterEnabledChanged,
  ipAdapterRecalled,
  ipAdapterImageChanged,
  ipAdapterWeightChanged,
  ipAdapterModelChanged,
  ipAdapterBeginStepPctChanged,
  ipAdapterEndStepPctChanged,
  ipAdapterStateReset,
  clearPendingControlImages,
} = controlNetSlice.actions;

export default controlNetSlice.reducer;
