import { makeToast } from 'app/components/Toaster';
import { log } from 'app/logging/useLogger';
import { loraRemoved } from 'features/lora/store/loraSlice';
import { modelSelected } from 'features/parameters/store/actions';
import {
  modelChanged,
  vaeSelected,
} from 'features/parameters/store/generationSlice';
import { zMainModel } from 'features/parameters/types/parameterSchemas';
import { addToast } from 'features/system/store/systemSlice';
import { forEach } from 'lodash-es';
import { startAppListening } from '..';

const moduleLog = log.child({ module: 'models' });

export const addModelSelectedListener = () => {
  startAppListening({
    actionCreator: modelSelected,
    effect: (action, { getState, dispatch }) => {
      const state = getState();
      const result = zMainModel.safeParse(action.payload);

      if (!result.success) {
        moduleLog.error(
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

        // TODO: handle incompatible controlnet; pending model manager support
        if (modelsCleared > 0) {
          dispatch(
            addToast(
              makeToast({
                title: `Base model changed, cleared ${modelsCleared} incompatible submodel${
                  modelsCleared === 1 ? '' : 's'
                }`,
                status: 'warning',
              })
            )
          );
        }
      }

      dispatch(modelChanged(newModel));
    },
  });
};
