import { createAppAsyncThunk } from 'app/storeUtils';
import { Model } from 'features/system/store/modelSlice';
import { reduce } from 'lodash';
import { ModelsService } from 'services/api';

export const IMAGES_PER_PAGE = 20;

export const receivedModels = createAppAsyncThunk(
  'models/receivedModels',
  async (_arg) => {
    const response = await ModelsService.listModels();
    const deserializedModels = reduce(
      response.models,
      (modelsAccumulator, model, modelName) => {
        modelsAccumulator[modelName] = { ...model, name: modelName };

        return modelsAccumulator;
      },
      {} as Record<string, Model>
    );

    return deserializedModels;
  }
);
