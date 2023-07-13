import { makeToast } from 'app/components/Toaster';
import { modelSelected } from 'features/parameters/store/actions';
import {
  modelChanged,
  vaeSelected,
} from 'features/parameters/store/generationSlice';
import { zMainModel } from 'features/parameters/store/parameterZodSchemas';
import { addToast } from 'features/system/store/systemSlice';
import { startAppListening } from '..';
import { lorasCleared } from '../../../../../features/lora/store/loraSlice';

export const addModelSelectedListener = () => {
  startAppListening({
    actionCreator: modelSelected,
    effect: (action, { getState, dispatch }) => {
      const state = getState();
      const { base_model, model_name } = action.payload;

      if (state.generation.model?.base_model !== base_model) {
        dispatch(
          addToast(
            makeToast({
              title: 'Base model changed, clearing submodels',
              status: 'warning',
            })
          )
        );
        dispatch(vaeSelected(null));
        dispatch(lorasCleared());
        // TODO: controlnet cleared
      }

      const newModel = zMainModel.parse(action.payload);

      dispatch(modelChanged(newModel));
    },
  });
};
