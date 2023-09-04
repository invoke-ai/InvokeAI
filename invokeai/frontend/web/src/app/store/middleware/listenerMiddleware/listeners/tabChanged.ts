import { modelChanged } from 'features/parameters/store/generationSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { NON_REFINER_BASE_MODELS } from 'services/api/constants';
import { mainModelsAdapter, modelsApi } from 'services/api/endpoints/models';
import { startAppListening } from '..';

export const addTabChangedListener = () => {
  startAppListening({
    actionCreator: setActiveTab,
    effect: async (action, { getState, dispatch }) => {
      const activeTabName = action.payload;
      if (activeTabName === 'unifiedCanvas') {
        const currentBaseModel = getState().generation.model?.base_model;

        if (
          currentBaseModel &&
          ['sd-1', 'sd-2', 'sdxl'].includes(currentBaseModel)
        ) {
          // if we're already on a valid model, no change needed
          return;
        }

        try {
          // just grab fresh models
          const modelsRequest = dispatch(
            modelsApi.endpoints.getMainModels.initiate(NON_REFINER_BASE_MODELS)
          );
          const models = await modelsRequest.unwrap();
          // cancel this cache subscription
          modelsRequest.unsubscribe();

          if (!models.ids.length) {
            // no valid canvas models
            dispatch(modelChanged(null));
            return;
          }

          // need to filter out all the invalid canvas models (currently sdxl & refiner)
          const validCanvasModels = mainModelsAdapter
            .getSelectors()
            .selectAll(models)
            .filter((model) =>
              ['sd-1', 'sd-2', 'sxdl'].includes(model.base_model)
            );

          const firstValidCanvasModel = validCanvasModels[0];

          if (!firstValidCanvasModel) {
            // no valid canvas models
            dispatch(modelChanged(null));
            return;
          }

          const { base_model, model_name, model_type } = firstValidCanvasModel;

          dispatch(modelChanged({ base_model, model_name, model_type }));
        } catch {
          // network request failed, bail
          dispatch(modelChanged(null));
        }
      }
    },
  });
};
