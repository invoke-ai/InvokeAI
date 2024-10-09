import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { bboxSyncedToOptimalDimension } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { loraDeleted } from 'features/controlLayers/store/lorasSlice';
import { modelChanged, vaeSelected } from 'features/controlLayers/store/paramsSlice';
import { selectBboxModelBase } from 'features/controlLayers/store/selectors';
import { modelSelected } from 'features/parameters/store/actions';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';

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
