import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/store';
import { bboxSyncedToOptimalDimension, rgRefImageModelChanged } from 'features/controlLayers/store/canvasSlice';
import { buildSelectIsStaging, selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { loraIsEnabledChanged } from 'features/controlLayers/store/lorasSlice';
import {
  modelChanged,
  syncedToOptimalDimension,
  vaeSelected,
} from 'features/controlLayers/store/paramsSlice';
import { refImageModelChanged, selectReferenceImageEntities } from 'features/controlLayers/store/refImagesSlice';
import {
  selectAllEntitiesOfType,
  selectBboxModelBase,
  selectCanvasSlice,
} from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { modelSelected } from 'features/parameters/store/actions';
import { SUPPORTS_REF_IMAGES_BASE_MODELS } from 'features/parameters/types/constants';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { selectGlobalRefImageModels, selectRegionalRefImageModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';
import {
  isChatGPT4oModelConfig,
  isFluxKontextApiModelConfig,
  isFluxKontextModelConfig,
  isFluxReduxModelConfig,
  isGemini2_5ModelConfig,
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
      const newBase = newModel.base;
      const didBaseModelChange = state.params.model?.base !== newBase;

      if (didBaseModelChange) {
        // we may need to reset some incompatible submodels
        let modelsUpdatedDisabledOrCleared = 0;

        // handle incompatible loras
        state.loras.loras.forEach((lora) => {
          if (lora.model.base !== newBase) {
            dispatch(loraIsEnabledChanged({ id: lora.id, isEnabled: false }));
            modelsUpdatedDisabledOrCleared += 1;
          }
        });

        // handle incompatible vae
        const { vae } = state.params;
        if (vae && vae.base !== newBase) {
          dispatch(vaeSelected(null));
          modelsUpdatedDisabledOrCleared += 1;
        }

        if (SUPPORTS_REF_IMAGES_BASE_MODELS.includes(newModel.base)) {
          // Handle incompatible reference image models - switch to first compatible model, with some smart logic
          // to choose the best available model based on the new main model.
          const allRefImageModels = selectGlobalRefImageModels(state).filter(({ base }) => base === newBase);

          let newGlobalRefImageModel = null;

          // Certain models require the ref image model to be the same as the main model - others just need a matching
          // base. Helper to grab the first exact match or the first available model if no exact match is found.
          const exactMatchOrFirst = <T extends AnyModelConfig>(candidates: T[]): T | null =>
            candidates.find(({ key }) => key === newModel.key) ?? candidates[0] ?? null;

          // The only way we can differentiate between FLUX and FLUX Kontext is to check for "kontext" in the name
          if (newModel.base === 'flux' && newModel.name.toLowerCase().includes('kontext')) {
            const fluxKontextDevModels = allRefImageModels.filter(isFluxKontextModelConfig);
            newGlobalRefImageModel = exactMatchOrFirst(fluxKontextDevModels);
          } else if (newModel.base === 'chatgpt-4o') {
            const chatGPT4oModels = allRefImageModels.filter(isChatGPT4oModelConfig);
            newGlobalRefImageModel = exactMatchOrFirst(chatGPT4oModels);
          } else if (newModel.base === 'gemini-2.5') {
            const gemini2_5Models = allRefImageModels.filter(isGemini2_5ModelConfig);
            newGlobalRefImageModel = exactMatchOrFirst(gemini2_5Models);
          } else if (newModel.base === 'flux-kontext') {
            const fluxKontextApiModels = allRefImageModels.filter(isFluxKontextApiModelConfig);
            newGlobalRefImageModel = exactMatchOrFirst(fluxKontextApiModels);
          } else if (newModel.base === 'flux') {
            const fluxReduxModels = allRefImageModels.filter(isFluxReduxModelConfig);
            newGlobalRefImageModel = fluxReduxModels[0] ?? null;
          } else {
            newGlobalRefImageModel = allRefImageModels[0] ?? null;
          }

          // All ref image entities are updated to use the same new model
          const refImageEntities = selectReferenceImageEntities(state);
          for (const entity of refImageEntities) {
            const shouldUpdateModel =
              (entity.config.model && entity.config.model.base !== newBase) ||
              (!entity.config.model && newGlobalRefImageModel);

            if (shouldUpdateModel) {
              dispatch(
                refImageModelChanged({
                  id: entity.id,
                  modelConfig: newGlobalRefImageModel,
                })
              );
              modelsUpdatedDisabledOrCleared += 1;
            }
          }
        }

        // For regional guidance, there is no smart logic - we just pick the first available model.
        const newRegionalRefImageModel = selectRegionalRefImageModels(state)[0] ?? null;

        // All regional guidance entities are updated to use the same new model.
        const canvasState = selectCanvasSlice(state);
        const canvasRegionalGuidanceEntities = selectAllEntitiesOfType(canvasState, 'regional_guidance');
        for (const entity of canvasRegionalGuidanceEntities) {
          for (const refImage of entity.referenceImages) {
            // Only change the model if the current one is not compatible with the new base model.
            const shouldUpdateModel =
              (refImage.config.model && refImage.config.model.base !== newBase) ||
              (!refImage.config.model && newRegionalRefImageModel);

            if (shouldUpdateModel) {
              dispatch(
                rgRefImageModelChanged({
                  entityIdentifier: getEntityIdentifier(entity),
                  referenceImageId: refImage.id,
                  modelConfig: newRegionalRefImageModel,
                })
              );
              modelsUpdatedDisabledOrCleared += 1;
            }
          }
        }

        if (modelsUpdatedDisabledOrCleared > 0) {
          toast({
            id: 'BASE_MODEL_CHANGED',
            title: t('toast.baseModelChanged'),
            description: t('toast.baseModelChangedCleared', {
              count: modelsUpdatedDisabledOrCleared,
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
        const isStaging = buildSelectIsStaging(selectCanvasSessionId(state))(state);
        if (!isStaging) {
          // Canvas tab only syncs if not staging
          dispatch(bboxSyncedToOptimalDimension());
        }
      }
    },
  });
};
