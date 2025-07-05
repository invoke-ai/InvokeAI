import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { AppDispatch, RootState } from 'app/store/store';
import { bboxSyncedToOptimalDimension, rgRefImageModelChanged } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { loraDeleted } from 'features/controlLayers/store/lorasSlice';
import { modelChanged, vaeSelected } from 'features/controlLayers/store/paramsSlice';
import { refImageModelChanged, selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectBboxModelBase, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { modelSelected } from 'features/parameters/store/actions';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { selectIPAdapterModels } from 'services/api/hooks/modelsByType';

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
      if (!selectIsStaging(state) && modelBase !== state.params.model?.base) {
        dispatch(bboxSyncedToOptimalDimension());
      }
    },
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
