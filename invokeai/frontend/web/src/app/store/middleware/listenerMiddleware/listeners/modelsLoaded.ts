import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { AppDispatch, RootState } from 'app/store/store';
import type { JSONObject } from 'common/types';
import {
  controlAdapterModelCleared,
  selectControlAdapterAll,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { loraRemoved } from 'features/lora/store/loraSlice';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { heightChanged, modelChanged, vaeSelected, widthChanged } from 'features/parameters/store/generationSlice';
import { zParameterModel, zParameterVAEModel } from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { refinerModelChanged } from 'features/sdxl/store/sdxlSlice';
import { forEach } from 'lodash-es';
import type { Logger } from 'roarr';
import { modelConfigsAdapterSelectors, modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import { isNonRefinerMainModelConfig, isRefinerMainModelModelConfig, isVAEModelConfig } from 'services/api/types';

export const addModelsLoadedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: modelsApi.endpoints.getModelConfigs.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // models loaded, we need to ensure the selected model is available and if not, select the first one
      const log = logger('models');
      log.info({ models: action.payload.entities }, `Models loaded (${action.payload.ids.length})`);

      const state = getState();

      const models = modelConfigsAdapterSelectors.selectAll(action.payload);

      handleMainModels(models, state, dispatch, log);
      handleRefinerModels(models, state, dispatch, log);
      handleVAEModels(models, state, dispatch, log);
      handleLoRAModels(models, state, dispatch, log);
      handleControlAdapterModels(models, state, dispatch, log);
    },
  });
};

type ModelHandler = (
  models: AnyModelConfig[],
  state: RootState,
  dispatch: AppDispatch,
  log: Logger<JSONObject>
) => undefined;

const handleMainModels: ModelHandler = (models, state, dispatch, log) => {
  const currentModel = state.generation.model;
  const mainModels = models.filter(isNonRefinerMainModelConfig);
  if (mainModels.length === 0) {
    // No models loaded at all
    dispatch(modelChanged(null));
    return;
  }

  const isCurrentMainModelAvailable = currentModel ? mainModels.some((m) => m.key === currentModel.key) : false;
  if (isCurrentMainModelAvailable) {
    return;
  }

  const defaultModel = state.config.sd.defaultModel;
  const defaultModelInList = defaultModel ? mainModels.find((m) => m.key === defaultModel) : false;

  if (defaultModelInList) {
    const result = zParameterModel.safeParse(defaultModelInList);
    if (result.success) {
      dispatch(modelChanged(defaultModelInList, currentModel));

      const optimalDimension = getOptimalDimension(defaultModelInList);
      if (getIsSizeOptimal(state.generation.width, state.generation.height, optimalDimension)) {
        return;
      }
      const { width, height } = calculateNewSize(
        state.generation.aspectRatio.value,
        optimalDimension * optimalDimension
      );

      dispatch(widthChanged(width));
      dispatch(heightChanged(height));
      return;
    }
  }

  const result = zParameterModel.safeParse(mainModels[0]);

  if (!result.success) {
    log.error({ error: result.error.format() }, 'Failed to parse main model');
    return;
  }

  dispatch(modelChanged(result.data, currentModel));
};

const handleRefinerModels: ModelHandler = (models, state, dispatch, _log) => {
  const currentRefinerModel = state.sdxl.refinerModel;
  const refinerModels = models.filter(isRefinerMainModelModelConfig);
  if (models.length === 0) {
    // No models loaded at all
    dispatch(refinerModelChanged(null));
    return;
  }

  const isCurrentRefinerModelAvailable = currentRefinerModel
    ? refinerModels.some((m) => m.key === currentRefinerModel.key)
    : false;

  if (!isCurrentRefinerModelAvailable) {
    dispatch(refinerModelChanged(null));
    return;
  }
};

const handleVAEModels: ModelHandler = (models, state, dispatch, log) => {
  const currentVae = state.generation.vae;

  if (currentVae === null) {
    // null is a valid VAE! it means "use the default with the main model"
    return;
  }
  const vaeModels = models.filter(isVAEModelConfig);

  const isCurrentVAEAvailable = vaeModels.some((m) => m.key === currentVae.key);

  if (isCurrentVAEAvailable) {
    return;
  }

  const firstModel = vaeModels[0];

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
};

const handleLoRAModels: ModelHandler = (models, state, dispatch, _log) => {
  const loras = state.lora.loras;

  forEach(loras, (lora, id) => {
    const isLoRAAvailable = models.some((m) => m.key === lora.model.key);

    if (isLoRAAvailable) {
      return;
    }

    dispatch(loraRemoved(id));
  });
};

const handleControlAdapterModels: ModelHandler = (models, state, dispatch, _log) => {
  selectControlAdapterAll(state.controlAdapters).forEach((ca) => {
    const isModelAvailable = models.some((m) => m.key === ca.model?.key);

    if (isModelAvailable) {
      return;
    }

    dispatch(controlAdapterModelCleared({ id: ca.id }));
  });
};
