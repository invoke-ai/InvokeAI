import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { CONTROLNET_MODEL_DEFAULT_PROCESSORS } from 'features/controlAdapters/store/constants';
import {
  controlAdapterAutoConfigToggled,
  controlAdapterModelChanged,
  controlAdapterProcessortTypeChanged,
  selectControlAdapterById,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterProcessorType } from 'features/controlAdapters/store/types';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { modelsApi } from 'services/api/endpoints/models';

export const addControlAdapterAutoProcessorUpdateListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(controlAdapterModelChanged, controlAdapterAutoConfigToggled),
    effect: async (action, { getState, dispatch }) => {
      let id;
      let model;

      const state = getState();
      const { controlAdapters: controlAdaptersState } = state;

      if (controlAdapterModelChanged.match(action)) {
        model = action.payload.model;
        id = action.payload.id;

        const cn = selectControlAdapterById(controlAdaptersState, id);
        if (!cn) {
          return;
        }

        if (!isControlNetOrT2IAdapter(cn)) {
          return;
        }
      }

      if (controlAdapterAutoConfigToggled.match(action)) {
        id = action.payload.id;

        const cn = selectControlAdapterById(controlAdaptersState, id);
        if (!cn) {
          return;
        }

        if (!isControlNetOrT2IAdapter(cn)) {
          return;
        }

        // if they turned off autoconfig, return
        if (!cn.shouldAutoConfig) {
          return;
        }

        model = cn.model;
      }

      if (!model || !id) {
        return;
      }

      let processorType: ControlAdapterProcessorType | undefined = undefined;
      const { data: modelConfig } = modelsApi.endpoints.getModelConfig.select(model.key)(state);

      for (const modelSubstring in CONTROLNET_MODEL_DEFAULT_PROCESSORS) {
        if (modelConfig?.name.includes(modelSubstring)) {
          processorType = CONTROLNET_MODEL_DEFAULT_PROCESSORS[modelSubstring];
          break;
        }
      }

      dispatch(
        controlAdapterProcessortTypeChanged({ id, processorType: processorType || 'none', shouldAutoConfig: true })
      );
    },
  });
};
