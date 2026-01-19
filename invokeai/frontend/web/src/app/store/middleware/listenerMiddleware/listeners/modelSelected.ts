import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/store';
import { bboxSyncedToOptimalDimension, rgRefImageModelChanged } from 'features/controlLayers/store/canvasSlice';
import { buildSelectIsStaging, selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { loraIsEnabledChanged } from 'features/controlLayers/store/lorasSlice';
import {
  kleinQwen3EncoderModelSelected,
  kleinVaeModelSelected,
  modelChanged,
  syncedToOptimalDimension,
  vaeSelected,
  zImageQwen3EncoderModelSelected,
  zImageQwen3SourceModelSelected,
  zImageVaeModelSelected,
} from 'features/controlLayers/store/paramsSlice';
import { refImageModelChanged, selectReferenceImageEntities } from 'features/controlLayers/store/refImagesSlice';
import {
  selectAllEntitiesOfType,
  selectBboxModelBase,
  selectCanvasSlice,
} from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { SUPPORTS_REF_IMAGES_BASE_MODELS } from 'features/modelManagerV2/models';
import { modelSelected } from 'features/parameters/store/actions';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import {
  selectFluxVAEModels,
  selectGlobalRefImageModels,
  selectQwen3EncoderModels,
  selectRegionalRefImageModels,
  selectZImageDiffusersModels,
} from 'services/api/hooks/modelsByType';
import type { FLUXKontextModelConfig, FLUXReduxModelConfig, IPAdapterModelConfig } from 'services/api/types';
import { isFluxKontextModelConfig, isFluxReduxModelConfig } from 'services/api/types';

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

        // handle incompatible Z-Image models - clear if switching away from z-image
        const { zImageVaeModel, zImageQwen3EncoderModel, zImageQwen3SourceModel } = state.params;
        if (newBase !== 'z-image') {
          if (zImageVaeModel) {
            dispatch(zImageVaeModelSelected(null));
            modelsUpdatedDisabledOrCleared += 1;
          }
          if (zImageQwen3EncoderModel) {
            dispatch(zImageQwen3EncoderModelSelected(null));
            modelsUpdatedDisabledOrCleared += 1;
          }
          if (zImageQwen3SourceModel) {
            dispatch(zImageQwen3SourceModelSelected(null));
            modelsUpdatedDisabledOrCleared += 1;
          }
        } else {
          // Switching to Z-Image - set defaults if no valid configuration exists
          const hasValidConfig = zImageQwen3SourceModel || (zImageVaeModel && zImageQwen3EncoderModel);

          if (!hasValidConfig) {
            // Prefer Qwen3 Source (Diffusers model) if available
            const availableZImageDiffusers = selectZImageDiffusersModels(state);

            if (availableZImageDiffusers.length > 0) {
              const diffusersModel = availableZImageDiffusers[0];
              if (diffusersModel) {
                dispatch(
                  zImageQwen3SourceModelSelected({
                    key: diffusersModel.key,
                    hash: diffusersModel.hash,
                    name: diffusersModel.name,
                    base: diffusersModel.base,
                    type: diffusersModel.type,
                  })
                );
              }
            } else {
              // Fallback: try to set Qwen3 Encoder + VAE
              const availableQwen3Encoders = selectQwen3EncoderModels(state);
              const availableFluxVAEs = selectFluxVAEModels(state);

              if (availableQwen3Encoders.length > 0 && availableFluxVAEs.length > 0) {
                const qwen3Encoder = availableQwen3Encoders[0];
                const fluxVAE = availableFluxVAEs[0];

                if (qwen3Encoder) {
                  dispatch(
                    zImageQwen3EncoderModelSelected({
                      key: qwen3Encoder.key,
                      name: qwen3Encoder.name,
                      base: qwen3Encoder.base,
                    })
                  );
                }
                if (fluxVAE) {
                  dispatch(
                    zImageVaeModelSelected({
                      key: fluxVAE.key,
                      hash: fluxVAE.hash,
                      name: fluxVAE.name,
                      base: fluxVAE.base,
                      type: fluxVAE.type,
                    })
                  );
                }
              }
            }
          }
        }

        // handle incompatible FLUX.2 Klein models - clear if switching away from flux2
        const { kleinVaeModel, kleinQwen3EncoderModel } = state.params;
        if (newBase !== 'flux2') {
          if (kleinVaeModel) {
            dispatch(kleinVaeModelSelected(null));
            modelsUpdatedDisabledOrCleared += 1;
          }
          if (kleinQwen3EncoderModel) {
            dispatch(kleinQwen3EncoderModelSelected(null));
            modelsUpdatedDisabledOrCleared += 1;
          }
        }

        if (SUPPORTS_REF_IMAGES_BASE_MODELS.includes(newModel.base)) {
          // Handle incompatible reference image models - switch to first compatible model, with some smart logic
          // to choose the best available model based on the new main model.
          const allRefImageModels = selectGlobalRefImageModels(state).filter(({ base }) => base === newBase);

          let newGlobalRefImageModel: IPAdapterModelConfig | FLUXKontextModelConfig | FLUXReduxModelConfig | null =
            null;

          // Certain models require the ref image model to be the same as the main model - others just need a matching
          // base. Helper to grab the first exact match or the first available model if no exact match is found.
          const exactMatchOrFirst = <T extends IPAdapterModelConfig | FLUXKontextModelConfig | FLUXReduxModelConfig>(
            candidates: T[]
          ): T | null => candidates.find(({ key }) => key === newModel.key) ?? candidates[0] ?? null;

          // The only way we can differentiate between FLUX and FLUX Kontext is to check for "kontext" in the name
          if (newModel.base === 'flux' && newModel.name.toLowerCase().includes('kontext')) {
            const fluxKontextDevModels = allRefImageModels.filter(isFluxKontextModelConfig);
            newGlobalRefImageModel = exactMatchOrFirst(fluxKontextDevModels);
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

      // Handle FLUX.2 Klein model changes within the same base (different variants need different encoders)
      // Clear the Qwen3 encoder when switching between different Klein models, as variants require matching encoders
      // (e.g., klein_4b needs qwen3_4b, klein_9b needs qwen3_8b)
      if (newBase === 'flux2' && state.params.model?.base === 'flux2' && newModel.key !== state.params.model?.key) {
        const { kleinQwen3EncoderModel } = state.params;
        if (kleinQwen3EncoderModel) {
          dispatch(kleinQwen3EncoderModelSelected(null));
          toast({
            id: 'KLEIN_ENCODER_CLEARED',
            title: t('toast.kleinEncoderCleared'),
            description: t('toast.kleinEncoderClearedDescription'),
            status: 'info',
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
