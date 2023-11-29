import { logger } from 'app/logging/logger';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import {
  controlAdapterIsEnabledChanged,
  selectControlAdapterAll,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { loraRemoved } from 'features/lora/store/loraSlice';
import { modelSelected } from 'features/parameters/store/actions';
import {
  modelChanged,
  setHeight,
  setWidth,
  vaeSelected,
} from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { forEach } from 'lodash-es';
import { startAppListening } from '..';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';

export const addModelSelectedListener = () => {
  startAppListening({
    actionCreator: modelSelected,
    effect: (action, { getState, dispatch }) => {
      const log = logger('models');

      const state = getState();
      const result = zParameterModel.safeParse(action.payload);

      if (!result.success) {
        log.error(
          { error: result.error.format() },
          'Failed to parse main model'
        );
        return;
      }

      const newModel = result.data;

      const { base_model } = newModel;

      if (state.generation.model?.base_model !== base_model) {
        // we may need to reset some incompatible submodels
        let modelsCleared = 0;

        // handle incompatible loras
        forEach(state.lora.loras, (lora, id) => {
          if (lora.base_model !== base_model) {
            dispatch(loraRemoved(id));
            modelsCleared += 1;
          }
        });

        // handle incompatible vae
        const { vae } = state.generation;
        if (vae && vae.base_model !== base_model) {
          dispatch(vaeSelected(null));
          modelsCleared += 1;
        }

        // handle incompatible controlnets
        selectControlAdapterAll(state.controlAdapters).forEach((ca) => {
          if (ca.model?.base_model !== base_model) {
            dispatch(
              controlAdapterIsEnabledChanged({ id: ca.id, isEnabled: false })
            );
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

      // Update Width / Height / Bounding Box Dimensions on Model Change
      if (
        state.generation.model?.base_model !== newModel.base_model &&
        state.ui.shouldAutoChangeDimensions
      ) {
        if (['sdxl', 'sdxl-refiner'].includes(newModel.base_model)) {
          dispatch(setWidth(1024));
          dispatch(setHeight(1024));
          dispatch(setBoundingBoxDimensions({ width: 1024, height: 1024 }));
        } else {
          dispatch(setWidth(512));
          dispatch(setHeight(512));
          dispatch(setBoundingBoxDimensions({ width: 512, height: 512 }));
        }
      }

      dispatch(modelChanged(newModel));
    },
  });
};
