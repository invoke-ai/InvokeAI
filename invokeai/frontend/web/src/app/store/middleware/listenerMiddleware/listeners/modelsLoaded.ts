import { logger } from 'app/logging/logger';
import {
  controlAdapterModelCleared,
  selectAllControlNets,
  selectAllIPAdapters,
  selectAllT2IAdapters,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { loraRemoved } from 'features/lora/store/loraSlice';
import { modelChanged, vaeSelected } from 'features/parameters/store/generationSlice';
import { zParameterModel, zParameterVAEModel } from 'features/parameters/types/parameterSchemas';
import { refinerModelChanged } from 'features/sdxl/store/sdxlSlice';
import { forEach, some } from 'lodash-es';
import { mainModelsAdapterSelectors, modelsApi, vaeModelsAdapterSelectors } from 'services/api/endpoints/models';
import type { TypeGuardFor } from 'services/api/types';

import { startAppListening } from '..';

export const addModelsLoadedListener = () => {
  startAppListening({
    predicate: (action): action is TypeGuardFor<typeof modelsApi.endpoints.getMainModels.matchFulfilled> =>
      modelsApi.endpoints.getMainModels.matchFulfilled(action) &&
      !action.meta.arg.originalArgs.includes('sdxl-refiner'),
    effect: async (action, { getState, dispatch }) => {
      // models loaded, we need to ensure the selected model is available and if not, select the first one
      const log = logger('models');
      log.info({ models: action.payload.entities }, `Main models loaded (${action.payload.ids.length})`);

      const currentModel = getState().generation.model;
      const models = mainModelsAdapterSelectors.selectAll(action.payload);

      if (models.length === 0) {
        // No models loaded at all
        dispatch(modelChanged(null));
        return;
      }

      const isCurrentModelAvailable = currentModel
        ? models.some(
            (m) =>
              m.model_name === currentModel.model_name &&
              m.base_model === currentModel.base_model &&
              m.model_type === currentModel.model_type
          )
        : false;

      if (isCurrentModelAvailable) {
        return;
      }

      const result = zParameterModel.safeParse(models[0]);

      if (!result.success) {
        log.error({ error: result.error.format() }, 'Failed to parse main model');
        return;
      }

      dispatch(modelChanged(result.data, currentModel));
    },
  });
  startAppListening({
    predicate: (action): action is TypeGuardFor<typeof modelsApi.endpoints.getMainModels.matchFulfilled> =>
      modelsApi.endpoints.getMainModels.matchFulfilled(action) && action.meta.arg.originalArgs.includes('sdxl-refiner'),
    effect: async (action, { getState, dispatch }) => {
      // models loaded, we need to ensure the selected model is available and if not, select the first one
      const log = logger('models');
      log.info({ models: action.payload.entities }, `SDXL Refiner models loaded (${action.payload.ids.length})`);

      const currentModel = getState().sdxl.refinerModel;
      const models = mainModelsAdapterSelectors.selectAll(action.payload);

      if (models.length === 0) {
        // No models loaded at all
        dispatch(refinerModelChanged(null));
        return;
      }

      const isCurrentModelAvailable = currentModel
        ? models.some(
            (m) =>
              m.model_name === currentModel.model_name &&
              m.base_model === currentModel.base_model &&
              m.model_type === currentModel.model_type
          )
        : false;

      if (!isCurrentModelAvailable) {
        dispatch(refinerModelChanged(null));
        return;
      }
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getVaeModels.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // VAEs loaded, need to reset the VAE is it's no longer available
      const log = logger('models');
      log.info({ models: action.payload.entities }, `VAEs loaded (${action.payload.ids.length})`);

      const currentVae = getState().generation.vae;

      if (currentVae === null) {
        // null is a valid VAE! it means "use the default with the main model"
        return;
      }

      const isCurrentVAEAvailable = some(
        action.payload.entities,
        (m) => m?.model_name === currentVae?.model_name && m?.base_model === currentVae?.base_model
      );

      if (isCurrentVAEAvailable) {
        return;
      }

      const firstModel = vaeModelsAdapterSelectors.selectAll(action.payload)[0];

      if (!firstModel) {
        // No custom VAEs loaded at all; use the default
        dispatch(vaeSelected(null));
        return;
      }

      const result = zParameterVAEModel.safeParse(firstModel);

      if (!result.success) {
        log.error({ error: result.error.format() }, 'Failed to parse VAE model');
        return;
      }

      dispatch(vaeSelected(result.data));
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getLoRAModels.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // LoRA models loaded - need to remove missing LoRAs from state
      const log = logger('models');
      log.info({ models: action.payload.entities }, `LoRAs loaded (${action.payload.ids.length})`);

      const loras = getState().lora.loras;

      forEach(loras, (lora, id) => {
        const isLoRAAvailable = some(
          action.payload.entities,
          (m) => m?.model_name === lora?.model_name && m?.base_model === lora?.base_model
        );

        if (isLoRAAvailable) {
          return;
        }

        dispatch(loraRemoved(id));
      });
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getControlNetModels.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // ControlNet models loaded - need to remove missing ControlNets from state
      const log = logger('models');
      log.info({ models: action.payload.entities }, `ControlNet models loaded (${action.payload.ids.length})`);

      selectAllControlNets(getState().controlAdapters).forEach((ca) => {
        const isModelAvailable = some(
          action.payload.entities,
          (m) => m?.model_name === ca?.model?.model_name && m?.base_model === ca?.model?.base_model
        );

        if (isModelAvailable) {
          return;
        }

        dispatch(controlAdapterModelCleared({ id: ca.id }));
      });
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getT2IAdapterModels.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // ControlNet models loaded - need to remove missing ControlNets from state
      const log = logger('models');
      log.info({ models: action.payload.entities }, `T2I Adapter models loaded (${action.payload.ids.length})`);

      selectAllT2IAdapters(getState().controlAdapters).forEach((ca) => {
        const isModelAvailable = some(
          action.payload.entities,
          (m) => m?.model_name === ca?.model?.model_name && m?.base_model === ca?.model?.base_model
        );

        if (isModelAvailable) {
          return;
        }

        dispatch(controlAdapterModelCleared({ id: ca.id }));
      });
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getIPAdapterModels.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // ControlNet models loaded - need to remove missing ControlNets from state
      const log = logger('models');
      log.info({ models: action.payload.entities }, `IP Adapter models loaded (${action.payload.ids.length})`);

      selectAllIPAdapters(getState().controlAdapters).forEach((ca) => {
        const isModelAvailable = some(
          action.payload.entities,
          (m) => m?.model_name === ca?.model?.model_name && m?.base_model === ca?.model?.base_model
        );

        if (isModelAvailable) {
          return;
        }

        dispatch(controlAdapterModelCleared({ id: ca.id }));
      });
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getTextualInversionModels.matchFulfilled,
    effect: async (action) => {
      const log = logger('models');
      log.info({ models: action.payload.entities }, `Embeddings loaded (${action.payload.ids.length})`);
    },
  });
};
