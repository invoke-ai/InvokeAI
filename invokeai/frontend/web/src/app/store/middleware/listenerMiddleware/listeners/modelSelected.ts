import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { AppDispatch, RootState } from 'app/store/store';
import { bboxSyncedToOptimalDimension, rgRefImageModelChanged } from 'features/controlLayers/store/canvasSlice';
import type { AppDispatch, RootState } from 'app/store/store';
import { bboxSyncedToOptimalDimension, rgRefImageModelChanged } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { loraDeleted } from 'features/controlLayers/store/lorasSlice';
import { modelChanged, syncedToOptimalDimension, vaeSelected } from 'features/controlLayers/store/paramsSlice';
import { refImageModelChanged, selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectBboxModelBase, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { refImageModelChanged, selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectBboxModelBase, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { modelSelected } from 'features/parameters/store/actions';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { selectIPAdapterModels } from 'services/api/hooks/modelsByType';
import { modelConfigsAdapterSelectors, modelsApi } from 'services/api/endpoints/models';
import type { ApiModelConfig, FLUXReduxModelConfig, IPAdapterModelConfig, MainModelConfig } from 'services/api/types';
import {
  isChatGPT4oModelConfig,
  isFluxKontextModelConfig,
  isFluxReduxModelConfig,
  isImagen3ModelConfig,
  isImagen4ModelConfig,
  isIPAdapterModelConfig,
} from 'services/api/types';

const log = logger('models');

export const addModelSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: modelSelected,
    effect: (action, { getState, dispatch }) => {
      const state = getState();
      const result = zParameterModel.safeParse(action.payload);

      if (!result.success) {
        log.error({ error: result.error.format() }, 'Failed to parse main model');
        return;
      }

      const newModel = result.data;

      const newBaseModel = newModel.base;
      const didBaseModelChange = state.params.model?.base !== newBaseModel;

      if (didBaseModelChange) {
        let modelsCleared = 0;

        state.loras.loras.forEach((lora) => {
          if (lora.model.base !== newBaseModel) {
            dispatch(loraDeleted({ id: lora.id }));
            modelsCleared += 1;
          }
        });

        const { vae } = state.params;
        if (vae && vae.base !== newBaseModel) {
          dispatch(vaeSelected(null));
          modelsCleared += 1;
        }

        // handle incompatible controlnets
        // state.canvas.present.controlAdapters.entities.forEach((ca) => {
        //   if (ca.model?.base !== newBaseModel) {
        //     modelsCleared += 1;
        //     if (ca.isEnabled) {
        //       dispatch(entityIsEnabledToggled({ entityIdentifier: { id: ca.id, type: 'control_adapter' } }));
        //     }
        //   }
        // });

        // Handle reference image model switching
        handleReferenceImageModelSwitching(state, dispatch, newModel, log);

        if (modelsCleared > 0) {
          toast({
            id: 'BASE_MODEL_CHANGED',
            title: t('toast.baseModelChanged'),
            description: t('toast.baseModelChangedCleared', {
              count: modelsCleared,
            }),
            status: 'warning',
          });
        }
      }

      dispatch(modelChanged({ model: newModel, previousModel: state.params.model }));

      const modelBase = selectBboxModelBase(state);

      if (modelBase !== state.params.model?.base) {
        // Sync generate tab settings whenever the model base changes
        dispatch(syncedToOptimalDimension());
        if (!selectIsStaging(state)) {
          // Canvas tab only syncs if not staging
          dispatch(bboxSyncedToOptimalDimension());
        }
      }
    },
  });
};

const handleReferenceImageModelSwitching = (
  state: RootState,
  dispatch: AppDispatch,
  newModel: MainModelConfig,
  log: typeof logger
) => {
  // Get all available models from the models query
  const modelsQueryResult = modelsApi.endpoints.getModelConfigs.select()(state);
  if (!modelsQueryResult.data) {
    return;
  }

  const allModels = modelConfigsAdapterSelectors.selectAll(modelsQueryResult.data);

  // Filter models compatible with the new main model
  const compatibleModels = allModels.filter((model) => {
    return (
      (isIPAdapterModelConfig(model) ||
        isFluxReduxModelConfig(model) ||
        isChatGPT4oModelConfig(model) ||
        isImagen3ModelConfig(model) ||
        isImagen4ModelConfig(model) ||
        isFluxKontextModelConfig(model)) &&
      model.base === newModel.base
    );
  });

  const isNewModelAPI =
    isChatGPT4oModelConfig(newModel) ||
    isImagen3ModelConfig(newModel) ||
    isImagen4ModelConfig(newModel) ||
    isFluxKontextModelConfig(newModel);

  // Function to get the best compatible model
  const getBestCompatibleModel = (
    _currentModel: IPAdapterModelConfig | FLUXReduxModelConfig | ApiModelConfig | null
  ): IPAdapterModelConfig | FLUXReduxModelConfig | ApiModelConfig | null => {
    if (isNewModelAPI) {
      // For API models, try to find an API model with the same name
      const matchingApiModel = compatibleModels.find(
        (model) =>
          (isChatGPT4oModelConfig(model) ||
            isImagen3ModelConfig(model) ||
            isImagen4ModelConfig(model) ||
            isFluxKontextModelConfig(model)) &&
          model.name === newModel.name
      );
      if (
        matchingApiModel &&
        (isChatGPT4oModelConfig(matchingApiModel) ||
          isImagen3ModelConfig(matchingApiModel) ||
          isImagen4ModelConfig(matchingApiModel) ||
          isFluxKontextModelConfig(matchingApiModel))
      ) {
        return matchingApiModel as ApiModelConfig;
      }
    }

    // Otherwise, return the first compatible model for this architecture
    const firstCompatible = compatibleModels[0];
    if (firstCompatible) {
      if (isIPAdapterModelConfig(firstCompatible)) {
        return firstCompatible as IPAdapterModelConfig;
      } else if (isFluxReduxModelConfig(firstCompatible)) {
        return firstCompatible as FLUXReduxModelConfig;
      } else if (
        isChatGPT4oModelConfig(firstCompatible) ||
        isImagen3ModelConfig(firstCompatible) ||
        isImagen4ModelConfig(firstCompatible) ||
        isFluxKontextModelConfig(firstCompatible)
      ) {
        return firstCompatible as ApiModelConfig;
      }
    }

    return null;
  };

  // Handle global reference images
  selectRefImagesSlice(state).entities.forEach((entity) => {
    const currentModel = entity.config.model;

    // If current model is incompatible or null, try to set a compatible one
    if (!currentModel || currentModel.base !== newModel.base) {
      const bestModel = getBestCompatibleModel(currentModel);
      if (bestModel) {
        log.debug(
          { previousModel: currentModel, newModel: bestModel },
          'Switching global reference image model to compatible model'
        );
        dispatch(refImageModelChanged({ id: entity.id, modelConfig: bestModel }));
      } else if (currentModel) {
        // Clear the model if no compatible model is found
        log.debug({ previousModel: currentModel }, 'Clearing incompatible global reference image model');
        dispatch(refImageModelChanged({ id: entity.id, modelConfig: null }));
      }
    }
  });

  // Handle regional guidance reference images
  selectCanvasSlice(state).regionalGuidance.entities.forEach((entity) => {
    entity.referenceImages.forEach(({ id: referenceImageId, config }) => {
      const currentModel = config.model;

      // If current model is incompatible or null, try to set a compatible one
      if (!currentModel || currentModel.base !== newModel.base) {
        const bestModel = getBestCompatibleModel(currentModel);
        if (bestModel) {
          log.debug(
            { previousModel: currentModel, newModel: bestModel },
            'Switching regional guidance reference image model to compatible model'
          );
          dispatch(
            rgRefImageModelChanged({
              entityIdentifier: getEntityIdentifier(entity),
              referenceImageId,
              modelConfig: bestModel,
            })
          );
        } else if (currentModel) {
          // Clear the model if no compatible model is found
          log.debug({ previousModel: currentModel }, 'Clearing incompatible regional guidance reference image model');
          dispatch(
            rgRefImageModelChanged({
              entityIdentifier: getEntityIdentifier(entity),
              referenceImageId,
              modelConfig: null,
            })
          );
        }
      }
    });
  });
};

const handleReferenceImageModelSwitching = (
  state: RootState,
  dispatch: AppDispatch,
  newModel: ParameterModel,
  log: ReturnType<typeof logger>
) => {
  const allIPAdapterModels = selectIPAdapterModels(state);
  const newBase = newModel.base;
  const compatibleIPAdapterModels = allIPAdapterModels.filter((model) => model.base === newBase);
  const firstCompatibleModel = compatibleIPAdapterModels[0] ?? null;

  selectRefImagesSlice(state).entities.forEach((entity) => {
    const currentModel = entity.config.model;

    if (!currentModel || currentModel.base !== newBase) {
      if (firstCompatibleModel) {
        log.debug(
          { previousModel: currentModel, newModel: firstCompatibleModel },
          'Switching global reference image model to compatible model'
        );
        dispatch(refImageModelChanged({ id: entity.id, modelConfig: firstCompatibleModel }));
      } else if (currentModel) {
        log.debug({ previousModel: currentModel }, 'Clearing incompatible global reference image model');
        dispatch(refImageModelChanged({ id: entity.id, modelConfig: null }));
      }
    }
  });

  selectCanvasSlice(state).regionalGuidance.entities.forEach((entity) => {
    entity.referenceImages.forEach(({ id: referenceImageId, config }) => {
      const currentModel = config.model;

      if (!currentModel || currentModel.base !== newBase) {
        if (firstCompatibleModel) {
          log.debug(
            { previousModel: currentModel, newModel: firstCompatibleModel },
            'Switching regional guidance reference image model to compatible model'
          );
          dispatch(
            rgRefImageModelChanged({
              entityIdentifier: getEntityIdentifier(entity),
              referenceImageId,
              modelConfig: firstCompatibleModel,
            })
          );
        } else if (currentModel) {
          log.debug({ previousModel: currentModel }, 'Clearing incompatible regional guidance reference image model');
          dispatch(
            rgRefImageModelChanged({
              entityIdentifier: getEntityIdentifier(entity),
              referenceImageId,
              modelConfig: null,
            })
          );
        }
      }
    });
  });
};
