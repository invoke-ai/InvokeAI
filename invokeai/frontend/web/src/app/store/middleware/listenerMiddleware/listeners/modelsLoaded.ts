import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { AppDispatch, RootState } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import {
  bboxHeightChanged,
  bboxWidthChanged,
  controlLayerModelChanged,
  referenceImageIPAdapterModelChanged,
  rgIPAdapterModelChanged,
} from 'features/controlLayers/store/canvasSlice';
import { loraDeleted } from 'features/controlLayers/store/lorasSlice';
import {
  clipEmbedModelSelected,
  fluxVAESelected,
  modelChanged,
  refinerModelChanged,
  t5EncoderModelSelected,
  vaeSelected,
} from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { calculateNewSize } from 'features/parameters/components/Bbox/calculateNewSize';
import { postProcessingModelChanged, upscaleModelChanged } from 'features/parameters/store/upscaleSlice';
import { zParameterModel, zParameterVAEModel } from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { Logger } from 'roarr';
import { modelConfigsAdapterSelectors, modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import {
  isCLIPEmbedModelConfig,
  isControlNetOrT2IAdapterModelConfig,
  isFluxVAEModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonFluxVAEModelConfig,
  isNonRefinerMainModelConfig,
  isRefinerMainModelModelConfig,
  isSpandrelImageToImageModelConfig,
  isT5EncoderModelConfig,
} from 'services/api/types';

const log = logger('models');

export const addModelsLoadedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: modelsApi.endpoints.getModelConfigs.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      // models loaded, we need to ensure the selected model is available and if not, select the first one
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
      handleT5EncoderModels(models, state, dispatch, log);
      handleCLIPEmbedModels(models, state, dispatch, log);
      handleFLUXVAEModels(models, state, dispatch, log);
    },
  });
};

type ModelHandler = (
  models: AnyModelConfig[],
  state: RootState,
  dispatch: AppDispatch,
  log: Logger<SerializableObject>
) => undefined;

const handleMainModels: ModelHandler = (models, state, dispatch, log) => {
  const currentModel = state.params.model;
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
      const { bbox } = selectCanvasSlice(state);
      const optimalDimension = getOptimalDimension(defaultModelInList);
      if (getIsSizeOptimal(bbox.rect.width, bbox.rect.height, optimalDimension)) {
        return;
      }
      const { width, height } = calculateNewSize(bbox.aspectRatio.value, optimalDimension * optimalDimension);

      dispatch(bboxWidthChanged({ width }));
      dispatch(bboxHeightChanged({ height }));
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
  const currentRefinerModel = state.params.refinerModel;
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
  const currentVae = state.params.vae;

  if (currentVae === null) {
    // null is a valid VAE! it means "use the default with the main model"
    return;
  }
  const vaeModels = models.filter(isNonFluxVAEModelConfig);

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
  state.loras.loras.forEach((lora) => {
    const isLoRAAvailable = loraModels.some((m) => m.key === lora.model.key);
    if (isLoRAAvailable) {
      return;
    }
    dispatch(loraDeleted({ id: lora.id }));
  });
};

const handleControlAdapterModels: ModelHandler = (models, state, dispatch, _log) => {
  const caModels = models.filter(isControlNetOrT2IAdapterModelConfig);
  selectCanvasSlice(state).controlLayers.entities.forEach((entity) => {
    const isModelAvailable = caModels.some((m) => m.key === entity.controlAdapter.model?.key);
    if (isModelAvailable) {
      return;
    }
    dispatch(controlLayerModelChanged({ entityIdentifier: getEntityIdentifier(entity), modelConfig: null }));
  });
};

const handleIPAdapterModels: ModelHandler = (models, state, dispatch, _log) => {
  const ipaModels = models.filter(isIPAdapterModelConfig);
  selectCanvasSlice(state).referenceImages.entities.forEach((entity) => {
    const isModelAvailable = ipaModels.some((m) => m.key === entity.ipAdapter.model?.key);
    if (isModelAvailable) {
      return;
    }
    dispatch(referenceImageIPAdapterModelChanged({ entityIdentifier: getEntityIdentifier(entity), modelConfig: null }));
  });

  selectCanvasSlice(state).regionalGuidance.entities.forEach((entity) => {
    entity.referenceImages.forEach(({ id: referenceImageId, ipAdapter }) => {
      const isModelAvailable = ipaModels.some((m) => m.key === ipAdapter.model?.key);
      if (isModelAvailable) {
        return;
      }
      dispatch(
        rgIPAdapterModelChanged({ entityIdentifier: getEntityIdentifier(entity), referenceImageId, modelConfig: null })
      );
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

const handleT5EncoderModels: ModelHandler = (models, state, dispatch, _log) => {
  const { t5EncoderModel: currentT5EncoderModel } = state.params;
  const t5EncoderModels = models.filter(isT5EncoderModelConfig);
  const firstModel = t5EncoderModels[0] || null;

  const isCurrentT5EncoderModelAvailable = currentT5EncoderModel
    ? t5EncoderModels.some((m) => m.key === currentT5EncoderModel.key)
    : false;

  if (!isCurrentT5EncoderModelAvailable) {
    dispatch(t5EncoderModelSelected(firstModel));
  }
};

const handleCLIPEmbedModels: ModelHandler = (models, state, dispatch, _log) => {
  const { clipEmbedModel: currentCLIPEmbedModel } = state.params;
  const CLIPEmbedModels = models.filter(isCLIPEmbedModelConfig);
  const firstModel = CLIPEmbedModels[0] || null;

  const isCurrentCLIPEmbedModelAvailable = currentCLIPEmbedModel
    ? CLIPEmbedModels.some((m) => m.key === currentCLIPEmbedModel.key)
    : false;

  if (!isCurrentCLIPEmbedModelAvailable) {
    dispatch(clipEmbedModelSelected(firstModel));
  }
};

const handleFLUXVAEModels: ModelHandler = (models, state, dispatch, _log) => {
  const { fluxVAE: currentFLUXVAEModel } = state.params;
  const fluxVAEModels = models.filter(isFluxVAEModelConfig);
  const firstModel = fluxVAEModels[0] || null;

  const isCurrentFLUXVAEModelAvailable = currentFLUXVAEModel
    ? fluxVAEModels.some((m) => m.key === currentFLUXVAEModel.key)
    : false;

  if (!isCurrentFLUXVAEModelAvailable) {
    dispatch(fluxVAESelected(firstModel));
  }
};
