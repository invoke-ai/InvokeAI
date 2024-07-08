import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { AppDispatch, RootState } from 'app/store/store';
import type { JSONObject } from 'common/types';
import {
  caModelChanged,
  documentHeightChanged,
  documentWidthChanged,
  ipaModelChanged,
  loraDeleted,
  modelChanged,
  refinerModelChanged,
  rgIPAdapterModelChanged,
  vaeSelected,
} from 'features/controlLayers/store/canvasV2Slice';
import { calculateNewSize } from 'features/parameters/components/DocumentSize/calculateNewSize';
import { postProcessingModelChanged, upscaleModelChanged } from 'features/parameters/store/upscaleSlice';
import { zParameterModel, zParameterVAEModel } from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { Logger } from 'roarr';
import { modelConfigsAdapterSelectors, modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import {
  isControlNetOrT2IAdapterModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isRefinerMainModelModelConfig,
  isSpandrelImageToImageModelConfig,
  isVAEModelConfig,
} from 'services/api/types';

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
      handleSpandrelImageToImageModels(models, state, dispatch, log);
      handleIPAdapterModels(models, state, dispatch, log);
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
  const currentModel = state.canvasV2.params.model;
  const mainModels = models.filter(isNonRefinerMainModelConfig);
  if (mainModels.length === 0) {
    // No models loaded at all
    dispatch(modelChanged({ model: null }));
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
      dispatch(modelChanged({ model: defaultModelInList, previousModel: currentModel }));

      const optimalDimension = getOptimalDimension(defaultModelInList);
      if (getIsSizeOptimal(state.canvasV2.document.rect.width, state.canvasV2.document.rect.height, optimalDimension)) {
        return;
      }
      const { width, height } = calculateNewSize(
        state.canvasV2.document.aspectRatio.value,
        optimalDimension * optimalDimension
      );

      dispatch(documentWidthChanged({ width }));
      dispatch(documentHeightChanged({ height }));
      return;
    }
  }

  const result = zParameterModel.safeParse(mainModels[0]);

  if (!result.success) {
    log.error({ error: result.error.format() }, 'Failed to parse main model');
    return;
  }

  dispatch(modelChanged({ model: result.data, previousModel: currentModel }));
};

const handleRefinerModels: ModelHandler = (models, state, dispatch, _log) => {
  const currentRefinerModel = state.canvasV2.params.refinerModel;
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
  const currentVae = state.canvasV2.params.vae;

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
  const loraModels = models.filter(isLoRAModelConfig);
  state.canvasV2.loras.forEach((lora) => {
    const isLoRAAvailable = loraModels.some((m) => m.key === lora.model.key);
    if (isLoRAAvailable) {
      return;
    }
    dispatch(loraDeleted({ id: lora.id }));
  });
};

const handleControlAdapterModels: ModelHandler = (models, state, dispatch, _log) => {
  const caModels = models.filter(isControlNetOrT2IAdapterModelConfig);
  state.canvasV2.controlAdapters.entities.forEach((ca) => {
    const isModelAvailable = caModels.some((m) => m.key === ca.model?.key);
    if (isModelAvailable) {
      return;
    }
    dispatch(caModelChanged({ id: ca.id, modelConfig: null }));
  });
};

const handleIPAdapterModels: ModelHandler = (models, state, dispatch, _log) => {
  const ipaModels = models.filter(isIPAdapterModelConfig);
  state.canvasV2.ipAdapters.entities.forEach(({ id, model }) => {
    const isModelAvailable = ipaModels.some((m) => m.key === model?.key);
    if (isModelAvailable) {
      return;
    }
    dispatch(ipaModelChanged({ id, modelConfig: null }));
  });

  state.canvasV2.regions.entities.forEach(({ id, ipAdapters }) => {
    ipAdapters.forEach(({ id: ipAdapterId, model }) => {
      const isModelAvailable = ipaModels.some((m) => m.key === model?.key);
      if (isModelAvailable) {
        return;
      }
      dispatch(rgIPAdapterModelChanged({ id, ipAdapterId, modelConfig: null }));
    });
  });
};

const handleSpandrelImageToImageModels: ModelHandler = (models, state, dispatch, _log) => {
  const { upscaleModel: currentUpscaleModel, postProcessingModel: currentPostProcessingModel } = state.upscale;
  const upscaleModels = models.filter(isSpandrelImageToImageModelConfig);
  const firstModel = upscaleModels[0] || null;

  const isCurrentUpscaleModelAvailable = currentUpscaleModel
    ? upscaleModels.some((m) => m.key === currentUpscaleModel.key)
    : false;

  if (!isCurrentUpscaleModelAvailable) {
    dispatch(upscaleModelChanged(firstModel));
  }

  const isCurrentPostProcessingModelAvailable = currentPostProcessingModel
    ? upscaleModels.some((m) => m.key === currentPostProcessingModel.key)
    : false;

  if (!isCurrentPostProcessingModelAvailable) {
    dispatch(postProcessingModelChanged(firstModel));
  }
};
