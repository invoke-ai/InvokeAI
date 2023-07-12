import { modelChanged } from 'features/parameters/store/generationSlice';
import { some } from 'lodash-es';
import { modelsApi } from 'services/api/endpoints/models';
import { startAppListening } from '..';

export const addModelsLoadedListener = () => {
  startAppListening({
    matcher: modelsApi.endpoints.getMainModels.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      // models loaded, we need to ensure the selected model is available and if not, select the first one

      const currentModel = getState().generation.model;

      const isCurrentModelAvailable = some(
        action.payload.entities,
        (m) =>
          m?.model_name === currentModel?.model_name &&
          m?.base_model === currentModel?.base_model
      );

      if (isCurrentModelAvailable) {
        return;
      }

      const firstModelId = action.payload.ids[0];
      const firstModel = action.payload.entities[firstModelId];

      if (!firstModel) {
        // No models loaded at all
        dispatch(modelChanged(null));
        return;
      }

      dispatch(
        modelChanged({
          base_model: firstModel.base_model,
          model_name: firstModel.model_name,
        })
      );
    },
  });
};
