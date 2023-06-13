import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { Model } from 'features/system/store/modelSlice';
import { reduce, size } from 'lodash-es';
import { ModelsService } from 'services/api';

const models = log.child({ namespace: 'model' });

export const IMAGES_PER_PAGE = 20;

export const receivedModels = createAppAsyncThunk(
  'models/receivedModels',
  async (_) => {
    const response = await ModelsService.listModels();

    const deserializedModels = reduce(
      response.models['sd-1']['pipeline'],
      (modelsAccumulator, model, modelName) => {
        modelsAccumulator[modelName] = { ...model, name: modelName };

        return modelsAccumulator;
      },
      {} as Record<string, Model>
    );

    models.info(
      { response },
      `Received ${size(response.models['sd-1']['pipeline'])} models`
    );

    return deserializedModels;
  }
);
