import { logger } from 'app/logging/logger';
import type { AppDispatch, AppStartListening, RootState } from 'app/store/store';
import { controlLayerModelChanged, rgRefImageModelChanged } from 'features/controlLayers/store/canvasSlice';
import { loraDeleted } from 'features/controlLayers/store/lorasSlice';
import {
  clipEmbedModelSelected,
  fluxVAESelected,
  modelChanged,
  refinerModelChanged,
  t5EncoderModelSelected,
  vaeSelected,
} from 'features/controlLayers/store/paramsSlice';
import { refImageModelChanged, selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import {
  getEntityIdentifier,
  isFLUXReduxConfig,
  isIPAdapterConfig,
  isRegionalGuidanceFLUXReduxConfig,
  isRegionalGuidanceIPAdapterConfig,
} from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { modelSelected } from 'features/parameters/store/actions';
import {
  postProcessingModelChanged,
  tileControlnetModelChanged,
  upscaleModelChanged,
} from 'features/parameters/store/upscaleSlice';
import { videoModelChanged } from 'features/parameters/store/videoSlice';
import {
  zParameterCLIPEmbedModel,
  zParameterSpandrelImageToImageModel,
  zParameterT5EncoderModel,
  zParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import type { Logger } from 'roarr';
import { modelConfigsAdapterSelectors, modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import {
  isCLIPEmbedModelConfigOrSubmodel,
  isControlLayerModelConfig,
  isControlNetModelConfig,
  isFluxReduxModelConfig,
  isFluxVAEModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonFluxVAEModelConfig,
  isNonRefinerMainModelConfig,
  isRefinerMainModelModelConfig,
  isSpandrelImageToImageModelConfig,
  isT5EncoderModelConfigOrSubmodel,
  isVideoModelConfig,
} from 'services/api/types';
import type { JsonObject } from 'type-fest';

const log = logger('models');

/**
 * This listener handles resetting or selecting models as we receive the big list of models from the API.
 *
 * For example, if a selected model is no longer available, it resets that models selection in redux.
 *
 * Or, if the model selection is one that should always be populated if possible, like main models, the listener
 * attempts to populate it.
 *
 * Some models, like VAEs, are optional and can be `null` - this listener will only clear the selection if the model is
 * no longer available, it will not attempt to select a new model.
 */
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
      handlePostProcessingModel(models, state, dispatch, log);
      handleUpscaleModel(models, state, dispatch, log);
      handleTileControlNetModel(models, state, dispatch, log);
      handleIPAdapterModels(models, state, dispatch, log);
      handleT5EncoderModels(models, state, dispatch, log);
      handleCLIPEmbedModels(models, state, dispatch, log);
      handleFLUXVAEModels(models, state, dispatch, log);
      handleFLUXReduxModels(models, state, dispatch, log);
      handleVideoModels(models, state, dispatch, log);
    },
  });
};

type ModelHandler = (
  models: AnyModelConfig[],
  state: RootState,
  dispatch: AppDispatch,
  log: Logger<JsonObject>
) => undefined;

const handleMainModels: ModelHandler = (models, state, dispatch, log) => {
  const selectedMainModel = state.params.model;
  const allMainModels = models.filter(isNonRefinerMainModelConfig).sort((a) => (a.base === 'sdxl' ? -1 : 1));

  const firstModel = allMainModels[0];

  // If we have no models, we may need to clear the selected model
  if (!firstModel) {
    // Only clear the model if we have one currently selected
    if (selectedMainModel !== null) {
      log.debug({ selectedMainModel }, 'No main models available, clearing');
      dispatch(modelChanged({ model: null }));
    }
    return;
  }

  // If the current model is available, we don't need to do anything
  if (allMainModels.some((m) => m.key === selectedMainModel?.key)) {
    return;
  }

  // If we have a default model, try to use it
  if (state.config.sd.defaultModel) {
    const defaultModel = allMainModels.find((m) => m.key === state.config.sd.defaultModel);
    if (defaultModel) {
      log.debug(
        { selectedMainModel, defaultModel },
        'No selected main model or selected main model is not available, selecting default model'
      );
      dispatch(modelSelected(defaultModel));
      return;
    }
  }

  log.debug(
    { selectedMainModel, firstModel },
    'No selected main model or selected main model is not available, selecting first available model'
  );
  dispatch(modelSelected(firstModel));
};

const handleRefinerModels: ModelHandler = (models, state, dispatch, log) => {
  const selectedRefinerModel = state.params.refinerModel;

  // `null` is a valid refiner model - no need to do anything.
  if (selectedRefinerModel === null) {
    return;
  }

  // We have a refiner model selected, need to check if it is available

  // Grab just the refiner models
  const allRefinerModels = models.filter(isRefinerMainModelModelConfig);

  // If the current refiner model is available, we don't need to do anything
  if (allRefinerModels.some((m) => m.key === selectedRefinerModel.key)) {
    return;
  }

  // Else, we need to clear the refiner model
  log.debug({ selectedRefinerModel }, 'Selected refiner model is not available, clearing');
  dispatch(refinerModelChanged(null));
  return;
};

const handleVAEModels: ModelHandler = (models, state, dispatch, log) => {
  const selectedVAEModel = state.params.vae;

  // `null` is a valid VAE - it means "use the VAE baked into the currently-selected main model"
  if (selectedVAEModel === null) {
    return;
  }

  // We have a VAE selected, need to check if it is available

  // Grab just the VAE models
  const vaeModels = models.filter((m) => isNonFluxVAEModelConfig(m));

  // If the current VAE model is available, we don't need to do anything
  if (vaeModels.some((m) => m.key === selectedVAEModel.key)) {
    return;
  }

  // Else, we need to clear the VAE model
  log.debug({ selectedVAEModel }, 'Selected VAE model is not available, clearing');
  dispatch(vaeSelected(null));
  return;
};

const handleLoRAModels: ModelHandler = (models, state, dispatch, log) => {
  const loraModels = models.filter(isLoRAModelConfig);
  state.loras.loras.forEach((lora) => {
    const isLoRAAvailable = loraModels.some((m) => m.key === lora.model.key);
    if (isLoRAAvailable) {
      return;
    }
    log.debug({ model: lora.model }, 'LoRA model is not available, clearing');
    dispatch(loraDeleted({ id: lora.id }));
  });
};

const handleVideoModels: ModelHandler = (models, state, dispatch, log) => {
  const videoModels = models.filter(isVideoModelConfig);
  const selectedVideoModel = state.video.videoModel;

  if (selectedVideoModel && videoModels.some((m) => m.key === selectedVideoModel.key)) {
    return;
  }

  const firstModel = videoModels[0] || null;
  if (firstModel) {
    log.debug({ firstModel }, 'No video model selected, selecting first available video model');
    dispatch(videoModelChanged({ videoModel: zModelIdentifierField.parse(firstModel) }));
    return;
  }
};

const handleControlAdapterModels: ModelHandler = (models, state, dispatch, log) => {
  const caModels = models.filter(isControlLayerModelConfig);
  selectCanvasSlice(state).controlLayers.entities.forEach((entity) => {
    const selectedControlAdapterModel = entity.controlAdapter.model;
    // `null` is a valid control adapter model - no need to do anything.
    if (!selectedControlAdapterModel) {
      return;
    }
    const isModelAvailable = caModels.some((m) => m.key === selectedControlAdapterModel.key);
    if (isModelAvailable) {
      return;
    }
    log.debug({ selectedControlAdapterModel }, 'Selected control adapter model is not available, clearing');
    dispatch(controlLayerModelChanged({ entityIdentifier: getEntityIdentifier(entity), modelConfig: null }));
  });
};

const handleIPAdapterModels: ModelHandler = (models, state, dispatch, log) => {
  const ipaModels = models.filter(isIPAdapterModelConfig);
  selectRefImagesSlice(state).entities.forEach((entity) => {
    if (!isIPAdapterConfig(entity.config)) {
      return;
    }

    const selectedIPAdapterModel = entity.config.model;
    // `null` is a valid IP adapter model - no need to do anything.
    if (!selectedIPAdapterModel) {
      return;
    }
    const isModelAvailable = ipaModels.some((m) => m.key === selectedIPAdapterModel.key);
    if (isModelAvailable) {
      return;
    }
    log.debug({ selectedIPAdapterModel }, 'Selected IP adapter model is not available, clearing');
    dispatch(refImageModelChanged({ id: entity.id, modelConfig: null }));
  });

  selectCanvasSlice(state).regionalGuidance.entities.forEach((entity) => {
    entity.referenceImages.forEach(({ id: referenceImageId, config }) => {
      if (!isRegionalGuidanceIPAdapterConfig(config)) {
        return;
      }

      const selectedIPAdapterModel = config.model;
      // `null` is a valid IP adapter model - no need to do anything.
      if (!selectedIPAdapterModel) {
        return;
      }
      const isModelAvailable = ipaModels.some((m) => m.key === selectedIPAdapterModel.key);
      if (isModelAvailable) {
        return;
      }
      log.debug({ selectedIPAdapterModel }, 'Selected IP adapter model is not available, clearing');
      dispatch(
        rgRefImageModelChanged({ entityIdentifier: getEntityIdentifier(entity), referenceImageId, modelConfig: null })
      );
    });
  });
};

const handleFLUXReduxModels: ModelHandler = (models, state, dispatch, log) => {
  const fluxReduxModels = models.filter(isFluxReduxModelConfig);

  selectRefImagesSlice(state).entities.forEach((entity) => {
    if (!isFLUXReduxConfig(entity.config)) {
      return;
    }
    const selectedFLUXReduxModel = entity.config.model;
    // `null` is a valid FLUX Redux model - no need to do anything.
    if (!selectedFLUXReduxModel) {
      return;
    }
    const isModelAvailable = fluxReduxModels.some((m) => m.key === selectedFLUXReduxModel.key);
    if (isModelAvailable) {
      return;
    }
    log.debug({ selectedFLUXReduxModel }, 'Selected FLUX Redux model is not available, clearing');
    dispatch(refImageModelChanged({ id: entity.id, modelConfig: null }));
  });

  selectCanvasSlice(state).regionalGuidance.entities.forEach((entity) => {
    entity.referenceImages.forEach(({ id: referenceImageId, config }) => {
      if (!isRegionalGuidanceFLUXReduxConfig(config)) {
        return;
      }

      const selectedFLUXReduxModel = config.model;
      // `null` is a valid FLUX Redux model - no need to do anything.
      if (!selectedFLUXReduxModel) {
        return;
      }
      const isModelAvailable = fluxReduxModels.some((m) => m.key === selectedFLUXReduxModel.key);
      if (isModelAvailable) {
        return;
      }
      log.debug({ selectedFLUXReduxModel }, 'Selected FLUX Redux model is not available, clearing');
      dispatch(
        rgRefImageModelChanged({ entityIdentifier: getEntityIdentifier(entity), referenceImageId, modelConfig: null })
      );
    });
  });
};

const handlePostProcessingModel: ModelHandler = (models, state, dispatch, log) => {
  const selectedPostProcessingModel = state.upscale.postProcessingModel;
  const allSpandrelModels = models.filter(isSpandrelImageToImageModelConfig);

  // If the currently selected model is available, we don't need to do anything
  if (selectedPostProcessingModel && allSpandrelModels.some((m) => m.key === selectedPostProcessingModel.key)) {
    return;
  }

  // Else we should select the first available model
  const firstModel = allSpandrelModels[0] || null;
  if (firstModel) {
    log.debug(
      { selectedPostProcessingModel, firstModel },
      'No selected post-processing model or selected post-processing model is not available, selecting first available model'
    );
    dispatch(postProcessingModelChanged(zParameterSpandrelImageToImageModel.parse(firstModel)));
    return;
  }

  // No available models, we should clear the selected model - but only if we have one selected
  if (selectedPostProcessingModel) {
    log.debug({ selectedPostProcessingModel }, 'Selected post-processing model is not available, clearing');
    dispatch(postProcessingModelChanged(null));
  }
};

const handleUpscaleModel: ModelHandler = (models, state, dispatch, log) => {
  const selectedUpscaleModel = state.upscale.upscaleModel;
  const allSpandrelModels = models.filter(isSpandrelImageToImageModelConfig);

  // If the currently selected model is available, we don't need to do anything
  if (selectedUpscaleModel && allSpandrelModels.some((m) => m.key === selectedUpscaleModel.key)) {
    return;
  }

  // Else we should select the first available model
  const firstModel = allSpandrelModels[0] || null;
  if (firstModel) {
    log.debug(
      { selectedUpscaleModel, firstModel },
      'No selected upscale model or selected upscale model is not available, selecting first available model'
    );
    dispatch(upscaleModelChanged(zParameterSpandrelImageToImageModel.parse(firstModel)));
    return;
  }

  // No available models, we should clear the selected model - but only if we have one selected
  if (selectedUpscaleModel) {
    log.debug({ selectedUpscaleModel }, 'Selected upscale model is not available, clearing');
    dispatch(upscaleModelChanged(null));
  }
};

const handleTileControlNetModel: ModelHandler = (models, state, dispatch, log) => {
  const selectedTileControlNetModel = state.upscale.tileControlnetModel;
  const controlNetModels = models.filter(isControlNetModelConfig);

  // If the currently selected model is available, we don't need to do anything
  if (selectedTileControlNetModel && controlNetModels.some((m) => m.key === selectedTileControlNetModel.key)) {
    return;
  }

  // The only way we have to identify a model as a tile model is by its name containing 'tile' :)
  const tileModel = controlNetModels.find((m) => m.name.toLowerCase().includes('tile'));

  // If we have a tile model, select it
  if (tileModel) {
    log.debug(
      { selectedTileControlNetModel, tileModel },
      'No selected tile ControlNet model or selected model is not available, selecting tile model'
    );
    dispatch(tileControlnetModelChanged(tileModel));
    return;
  }

  // Otherwise, select the first available ControlNet model
  const firstModel = controlNetModels[0] || null;
  if (firstModel) {
    log.debug(
      { selectedTileControlNetModel, firstModel },
      'No tile ControlNet model found, selecting first available ControlNet model'
    );
    dispatch(tileControlnetModelChanged(firstModel));
    return;
  }

  // No available models, we should clear the selected model - but only if we have one selected
  if (selectedTileControlNetModel) {
    log.debug({ selectedTileControlNetModel }, 'Selected tile ControlNet model is not available, clearing');
    dispatch(tileControlnetModelChanged(null));
  }
};

const handleT5EncoderModels: ModelHandler = (models, state, dispatch, log) => {
  const selectedT5EncoderModel = state.params.t5EncoderModel;
  const t5EncoderModels = models.filter((m) => isT5EncoderModelConfigOrSubmodel(m));

  // If the currently selected model is available, we don't need to do anything
  if (selectedT5EncoderModel && t5EncoderModels.some((m) => m.key === selectedT5EncoderModel.key)) {
    return;
  }

  // Else we should select the first available model
  const firstModel = t5EncoderModels[0] || null;
  if (firstModel) {
    log.debug(
      { selectedT5EncoderModel, firstModel },
      'No selected T5 encoder model or selected T5 encoder model is not available, selecting first available model'
    );
    dispatch(t5EncoderModelSelected(zParameterT5EncoderModel.parse(firstModel)));
    return;
  }

  // No available models, we should clear the selected model - but only if we have one selected
  if (selectedT5EncoderModel) {
    log.debug({ selectedT5EncoderModel }, 'Selected T5 encoder model is not available, clearing');
    dispatch(t5EncoderModelSelected(null));
    return;
  }
};

const handleCLIPEmbedModels: ModelHandler = (models, state, dispatch, log) => {
  const selectedCLIPEmbedModel = state.params.clipEmbedModel;
  const CLIPEmbedModels = models.filter((m) => isCLIPEmbedModelConfigOrSubmodel(m));

  // If the currently selected model is available, we don't need to do anything
  if (selectedCLIPEmbedModel && CLIPEmbedModels.some((m) => m.key === selectedCLIPEmbedModel.key)) {
    return;
  }

  // Else we should select the first available model
  const firstModel = CLIPEmbedModels[0] || null;
  if (firstModel) {
    log.debug(
      { selectedCLIPEmbedModel, firstModel },
      'No selected CLIP embed model or selected CLIP embed model is not available, selecting first available model'
    );
    dispatch(clipEmbedModelSelected(zParameterCLIPEmbedModel.parse(firstModel)));
    return;
  }

  // No available models, we should clear the selected model - but only if we have one selected
  if (selectedCLIPEmbedModel) {
    log.debug({ selectedCLIPEmbedModel }, 'Selected CLIP embed model is not available, clearing');
    dispatch(clipEmbedModelSelected(null));
    return;
  }
};

const handleFLUXVAEModels: ModelHandler = (models, state, dispatch, log) => {
  const selectedFLUXVAEModel = state.params.fluxVAE;
  const fluxVAEModels = models.filter((m) => isFluxVAEModelConfig(m));

  // If the currently selected model is available, we don't need to do anything
  if (selectedFLUXVAEModel && fluxVAEModels.some((m) => m.key === selectedFLUXVAEModel.key)) {
    return;
  }

  // Else we should select the first available model
  const firstModel = fluxVAEModels[0] || null;
  if (firstModel) {
    log.debug(
      { selectedFLUXVAEModel, firstModel },
      'No selected FLUX VAE model or selected FLUX VAE model is not available, selecting first available model'
    );
    dispatch(fluxVAESelected(zParameterVAEModel.parse(firstModel)));
    return;
  }

  // No available models, we should clear the selected model - but only if we have one selected
  if (selectedFLUXVAEModel) {
    log.debug({ selectedFLUXVAEModel }, 'Selected FLUX VAE model is not available, clearing');
    dispatch(fluxVAESelected(null));
    return;
  }
};
