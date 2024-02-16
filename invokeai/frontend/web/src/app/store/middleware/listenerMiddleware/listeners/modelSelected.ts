import { logger } from 'app/logging/logger';
import {
  controlAdapterIsEnabledChanged,
  selectControlAdapterAll,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { loraRemoved } from 'features/lora/store/loraSlice';
import { modelSelected } from 'features/parameters/store/actions';
import { modelChanged, vaeSelected } from 'features/parameters/store/generationSlice';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { forEach } from 'lodash-es';

import { startAppListening } from '..';

export const addModelSelectedListener = () => {
  startAppListening({
    actionCreator: modelSelected,
    effect: (action, { getState, dispatch }) => {
      const log = logger('models');

      const state = getState();
      const result = zParameterModel.safeParse(action.payload);

      if (!result.success) {
        log.error({ error: result.error.format() }, 'Failed to parse main model');
        return;
      }

      const newModel = result.data;

      const newBaseModel = newModel.base;
      const didBaseModelChange = state.generation.model?.base !== newBaseModel;

      if (didBaseModelChange) {
        // we may need to reset some incompatible submodels
        let modelsCleared = 0;

        // handle incompatible loras
        forEach(state.lora.loras, (lora, id) => {
          if (lora.base !== newBaseModel) {
            dispatch(loraRemoved(id));
            modelsCleared += 1;
          }
        });

        // handle incompatible vae
        const { vae } = state.generation;
        if (vae && vae.base !== newBaseModel) {
          dispatch(vaeSelected(null));
          modelsCleared += 1;
        }

        // handle incompatible controlnets
        selectControlAdapterAll(state.controlAdapters).forEach((ca) => {
          if (ca.model?.base !== newBaseModel) {
            dispatch(controlAdapterIsEnabledChanged({ id: ca.id, isEnabled: false }));
            modelsCleared += 1;
          }
        });

        if (modelsCleared > 0) {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.baseModelChangedCleared', {
                  count: modelsCleared,
                }),
                status: 'warning',
              })
            )
          );
        }
      }

      dispatch(modelChanged(newModel, state.generation.model));
    },
  });
};
