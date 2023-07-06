import {
  modelChanged,
  vaeSelected,
} from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { startAppListening } from '..';
import { modelSelected } from 'features/parameters/store/actions';
import { makeToast } from 'app/components/Toaster';
import { lorasCleared } from '../../../../../features/lora/store/loraSlice';

export const addModelSelectedListener = () => {
  startAppListening({
    actionCreator: modelSelected,
    effect: (action, { getState, dispatch }) => {
      const state = getState();
      const [base_model, type, name] = action.payload.split('/');

      if (state.generation.model?.base_model !== base_model) {
        dispatch(
          addToast(
            makeToast({
              title: 'Base model changed, clearing submodels',
              status: 'warning',
            })
          )
        );
        dispatch(vaeSelected('auto'));
        dispatch(lorasCleared());
        // TODO: controlnet cleared
      }

      dispatch(modelChanged({ id: action.payload, base_model, name, type }));
    },
  });
};
