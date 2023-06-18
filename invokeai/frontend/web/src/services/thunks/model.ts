import { log } from 'app/logging/useLogger';
import { createAppAsyncThunk } from 'app/store/storeUtils';
import { SD1PipelineModel } from 'features/system/store/models/sd1PipelineModelSlice';
import { SD2PipelineModel } from 'features/system/store/models/sd2PipelineModelSlice';
import { reduce, size } from 'lodash-es';
import { BaseModelType, ModelType, ModelsService } from 'services/api';

const models = log.child({ namespace: 'model' });

export const IMAGES_PER_PAGE = 20;

type receivedModelsArg = {
  baseModel: BaseModelType | undefined;
  modelType: ModelType | undefined;
};

export const receivedModels = createAppAsyncThunk(
  'models/receivedModels',
  async (arg: receivedModelsArg) => {
    const response = await ModelsService.listModels(arg);

    let deserializedModels = {};

    if (arg.baseModel === undefined) return response.models;
    if (arg.modelType === undefined) return response.models;

    if (arg.baseModel === 'sd-1') {
      deserializedModels = reduce(
        response.models[arg.baseModel][arg.modelType],
        (modelsAccumulator, model, modelName) => {
          modelsAccumulator[modelName] = { ...model, name: modelName };
          return modelsAccumulator;
        },
        {} as Record<string, SD1PipelineModel>
      );
    }

    if (arg.baseModel === 'sd-2') {
      deserializedModels = reduce(
        response.models[arg.baseModel][arg.modelType],
        (modelsAccumulator, model, modelName) => {
          modelsAccumulator[modelName] = { ...model, name: modelName };
          return modelsAccumulator;
        },
        {} as Record<string, SD2PipelineModel>
      );
    }

    models.info(
      { response },
      `Received ${size(response.models[arg.baseModel][arg.modelType])} ${[
        arg.baseModel,
      ]} models`
    );

    return deserializedModels;
  }
);
