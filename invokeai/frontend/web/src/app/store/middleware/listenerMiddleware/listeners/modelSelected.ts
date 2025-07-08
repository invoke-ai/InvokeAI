import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { bboxSyncedToOptimalDimension } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { loraDeleted } from 'features/controlLayers/store/lorasSlice';
import { modelChanged, syncedToOptimalDimension, vaeSelected } from 'features/controlLayers/store/paramsSlice';
import { 
  refImageModelChanged,
  selectReferenceImageEntities 
} from 'features/controlLayers/store/refImagesSlice';
import { selectBboxModelBase, selectAllEntities } from 'features/controlLayers/store/selectors';
import { 
  rgRefImageModelChanged
} from 'features/controlLayers/store/canvasSlice';
import { 
  getEntityIdentifier,
  isRegionalGuidanceEntityIdentifier
} from 'features/controlLayers/store/types';
import type { 
  CanvasEntityState,
  RefImageState 
} from 'features/controlLayers/store/types';
import { modelSelected } from 'features/parameters/store/actions';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { toast } from 'features/toast/toast';
import { 
  selectIPAdapterModels
} from 'services/api/hooks/modelsByType';
import type { 
  AnyModelConfig
} from 'services/api/types';
import { 
  isIPAdapterModelConfig,
  isFluxReduxModelConfig,
  isChatGPT4oModelConfig,
  isFluxKontextApiModelConfig,
  isFluxKontextModelConfig
} from 'services/api/types';
import type { RootState } from 'app/store/store';
import { t } from 'i18next';

const log = logger('models');

// Selector for global reference image models
const selectGlobalReferenceImageModels = (state: RootState): AnyModelConfig[] => {
  const allModels = selectIPAdapterModels(state);
  // Add other model types that can be used as reference images
  return allModels.filter((model: AnyModelConfig) => 
    isIPAdapterModelConfig(model) ||
    isFluxReduxModelConfig(model) ||
    isChatGPT4oModelConfig(model) ||
    isFluxKontextApiModelConfig(model) ||
    isFluxKontextModelConfig(model)
  );
};

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
        // we may need to reset some incompatible submodels
        let modelsCleared = 0;

        // handle incompatible loras
        state.loras.loras.forEach((lora) => {
          if (lora.model.base !== newBaseModel) {
            dispatch(loraDeleted({ id: lora.id }));
            modelsCleared += 1;
          }
        });

        // handle incompatible vae
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

        // Handle incompatible reference image models - switch to first compatible model
        const availableRefImageModels = selectGlobalReferenceImageModels(state).filter((model: AnyModelConfig) => model.base === newBaseModel);
        const firstCompatibleModel = availableRefImageModels[0] || null;

        // Handle global reference images
        const refImageEntities = selectReferenceImageEntities(state);
        refImageEntities.forEach((entity: RefImageState) => {
          if (entity.config.model && entity.config.model.base !== newBaseModel) {
            dispatch(refImageModelChanged({ 
              id: entity.id, 
              modelConfig: firstCompatibleModel 
            }));
            if (firstCompatibleModel) {
              log.debug(
                { oldModel: entity.config.model, newModel: firstCompatibleModel },
                'Switched global reference image model to compatible model'
              );
            } else {
              log.debug(
                { oldModel: entity.config.model },
                'Cleared global reference image model - no compatible models available'
              );
              modelsCleared += 1;
            }
          }
        });

        // Handle regional guidance reference images
        const canvasEntities = selectAllEntities(state.canvas.present);
        canvasEntities.forEach((entity: CanvasEntityState) => {
          if (isRegionalGuidanceEntityIdentifier(getEntityIdentifier(entity))) {
            entity.referenceImages.forEach((refImage: any) => {
              if (refImage.config.model && refImage.config.model.base !== newBaseModel) {
                dispatch(rgRefImageModelChanged({
                  entityIdentifier: getEntityIdentifier(entity),
                  referenceImageId: refImage.id,
                  modelConfig: firstCompatibleModel
                }));
                if (firstCompatibleModel) {
                  log.debug(
                    { oldModel: refImage.config.model, newModel: firstCompatibleModel },
                    'Switched regional guidance reference image model to compatible model'
                  );
                } else {
                  log.debug(
                    { oldModel: refImage.config.model },
                    'Cleared regional guidance reference image model - no compatible models available'
                  );
                  modelsCleared += 1;
                }
              }
            });
          }
        });

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
