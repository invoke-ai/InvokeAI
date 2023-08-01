import { logger } from 'app/logging/logger';
import { controlNetRemoved } from 'features/controlNet/store/controlNetSlice';
import { loraRemoved } from 'features/lora/store/loraSlice';
import {
  modelChanged,
  vaeSelected,
} from 'features/parameters/store/generationSlice';
import {
  zMainOrOnnxModel,
  zSDXLRefinerModel,
  zVaeModel,
} from 'features/parameters/types/parameterSchemas';
import {
  refinerModelChanged,
  setShouldUseSDXLRefiner,
} from 'features/sdxl/store/sdxlSlice';
import { forEach, some } from 'lodash-es';
import { modelsApi } from 'services/api/endpoints/models';
import { startAppListening } from '..';

export const addModelsLoadedListener = () => {
  startAppListening({
    predicate: (state, action) =>
      modelsApi.endpoints.getMainModels.matchFulfilled(action) &&
      !action.meta.arg.originalArgs.includes('sdxl-refiner'),
    effect: async (action, { getState, dispatch }) => {
      // models loaded, we need to ensure the selected model is available and if not, select the first one
      const log = logger('models');
      log.info(
        { models: action.payload.entities },
        `Main models loaded (${action.payload.ids.length})`
      );

      const currentModel = getState().generation.model;

      const isCurrentModelAvailable = some(
        action.payload.entities,
        (m) =>
          m?.model_name === currentModel?.model_name &&
          m?.base_model === currentModel?.base_model &&
          m?.model_type === currentModel?.model_type
      );

      if (isCurrentModelAvailable) {
        return;
      }

      const firstModelId = action.payload.ids[0];
      const firstModel = action.payload.entities[firstModelId];

      if (!firstModel) {
        // No models loaded at all
        dispatch(modelChanged(null));
        return;
      }

      const result = zMainOrOnnxModel.safeParse(firstModel);

      if (!result.success) {
        log.error(
          { error: result.error.format() },
          'Failed to parse main model'
        );
        return;
      }

      dispatch(modelChanged(result.data));
    },
  });
  startAppListening({
    predicate: (state, action) =>
      modelsApi.endpoints.getMainModels.matchFulfilled(action) &&
      action.meta.arg.originalArgs.includes('sdxl-refiner'),
    effect: async (action, { getState, dispatch }) => {
      // models loaded, we need to ensure the selected model is available and if not, select the first one
      const log = logger('models');
      log.info(
        { models: action.payload.entities },
        `SDXL Refiner models loaded (${action.payload.ids.length})`
      );

      const currentModel = getState().sdxl.refinerModel;

      const isCurrentModelAvailable = some(
        action.payload.entities,
        (m) =>
          m?.model_name === currentModel?.model_name &&
          m?.base_model === currentModel?.base_model &&
          m?.model_type === currentModel?.model_type
      );

      if (isCurrentModelAvailable) {
        return;
      }

      const firstModelId = action.payload.ids[0];
      const firstModel = action.payload.entities[firstModelId];

      if (!firstModel) {
        // No models loaded at all
        dispatch(refinerModelChanged(null));
        dispatch(setShouldUseSDXLRefiner(false));
        return;
      }

      const result = zSDXLRefinerModel.safeParse(firstModel);

      if (!result.success) {
        log.error(
          { error: result.error.format() },
          'Failed to parse SDXL Refiner Model'
        );
        return;
      }

      dispatch(refinerModelChanged(result.data));
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getVaeModels.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // VAEs loaded, need to reset the VAE is it's no longer available
      const log = logger('models');
      log.info(
        { models: action.payload.entities },
        `VAEs loaded (${action.payload.ids.length})`
      );

      const currentVae = getState().generation.vae;

      if (currentVae === null) {
        // null is a valid VAE! it means "use the default with the main model"
        return;
      }

      const isCurrentVAEAvailable = some(
        action.payload.entities,
        (m) =>
          m?.model_name === currentVae?.model_name &&
          m?.base_model === currentVae?.base_model
      );

      if (isCurrentVAEAvailable) {
        return;
      }

      const firstModelId = action.payload.ids[0];
      const firstModel = action.payload.entities[firstModelId];

      if (!firstModel) {
        // No custom VAEs loaded at all; use the default
        dispatch(modelChanged(null));
        return;
      }

      const result = zVaeModel.safeParse(firstModel);

      if (!result.success) {
        log.error(
          { error: result.error.format() },
          'Failed to parse VAE model'
        );
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
      log.info(
        { models: action.payload.entities },
        `LoRAs loaded (${action.payload.ids.length})`
      );

      const loras = getState().lora.loras;

      forEach(loras, (lora, id) => {
        const isLoRAAvailable = some(
          action.payload.entities,
          (m) =>
            m?.model_name === lora?.model_name &&
            m?.base_model === lora?.base_model
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
      log.info(
        { models: action.payload.entities },
        `ControlNet models loaded (${action.payload.ids.length})`
      );

      const controlNets = getState().controlNet.controlNets;

      forEach(controlNets, (controlNet, controlNetId) => {
        const isControlNetAvailable = some(
          action.payload.entities,
          (m) =>
            m?.model_name === controlNet?.model?.model_name &&
            m?.base_model === controlNet?.model?.base_model
        );

        if (isControlNetAvailable) {
          return;
        }

        dispatch(controlNetRemoved({ controlNetId }));
      });
    },
  });
  startAppListening({
    matcher: modelsApi.endpoints.getTextualInversionModels.matchFulfilled,
    effect: async (action) => {
      const log = logger('models');
      log.info(
        { models: action.payload.entities },
        `Embeddings loaded (${action.payload.ids.length})`
      );
    },
  });
};
