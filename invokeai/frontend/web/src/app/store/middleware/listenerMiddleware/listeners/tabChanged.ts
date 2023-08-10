import { modelChanged } from 'features/parameters/store/generationSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { find, forEach } from 'lodash-es';
import { NON_REFINER_BASE_MODELS } from 'services/api/constants';
import {
  MainModelConfigEntity,
  modelsApi,
} from 'services/api/endpoints/models';
import { startAppListening } from '..';

export const addTabChangedListener = () => {
  startAppListening({
    actionCreator: setActiveTab,
    effect: (action, { getState, dispatch }) => {
      const activeTabName = action.payload;
      if (activeTabName === 'unifiedCanvas') {
        // grab the models from RTK Query cache
        const { data } = modelsApi.endpoints.getMainModels.select(
          NON_REFINER_BASE_MODELS
        )(getState());

        if (!data) {
          // no models yet, so we can't do anything
          dispatch(modelChanged(null));
          return;
        }

        // need to filter out all the invalid canvas models (currently, this is just sdxl)
        const validCanvasModels: MainModelConfigEntity[] = [];

        forEach(data.entities, (entity) => {
          if (!entity) {
            return;
          }
          if (['sd-1', 'sd-2'].includes(entity.base_model)) {
            validCanvasModels.push(entity);
          }
        });

        // Check if current selected model exists on Canvas Models
        const currentModel = find(
          validCanvasModels,
          (model) =>
            model.model_name === getState().generation.model?.model_name
        );

        // If it does, set it or else set the first in list
        const validModel = currentModel ? currentModel : validCanvasModels[0];

        // Make sure the valid model exists. Can be undefined in some cases.
        if (!validModel) {
          // uh oh, we have no models that are valid for canvas
          dispatch(modelChanged(null));
          return;
        }

        // only store the model name and base model in redux
        const { base_model, model_name, model_type } = validModel;

        dispatch(modelChanged({ base_model, model_name, model_type }));
      }
    },
  });
};
