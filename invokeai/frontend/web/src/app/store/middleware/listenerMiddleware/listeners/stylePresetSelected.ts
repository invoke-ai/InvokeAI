import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { activeStylePresetIdChanged } from 'features/stylePresets/store/stylePresetSlice';
import { stylePresetsApi } from 'services/api/endpoints/stylePresets';

export const addStylePresetSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: activeStylePresetIdChanged,
    effect: async (action, { dispatch }) => {
      if (!action.payload) {
        return;
      }
      dispatch(stylePresetsApi.endpoints.selectStylePreset.initiate(action.payload));
    },
  });
};
