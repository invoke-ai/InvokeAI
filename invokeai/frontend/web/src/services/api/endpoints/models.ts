import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { cloneDeep } from 'lodash-es';
import {
  AnyModelConfig,
  BaseModelType,
  CheckpointModelConfig,
  ControlNetModelConfig,
  DiffusersModelConfig,
  ImportModelConfig,
  LoRAModelConfig,
  MainModelConfig,
  OnnxModelConfig,
  MergeModelConfig,
  TextualInversionModelConfig,
  VaeModelConfig,
  ModelType,
} from 'services/api/types';

import queryString from 'query-string';
import { ApiFullTagDescription, LIST_TAG, api } from '..';
import { operations, paths } from '../schema';

export type DiffusersModelConfigEntity = DiffusersModelConfig & { id: string };
export type CheckpointModelConfigEntity = CheckpointModelConfig & {
  id: string;
};
export type MainModelConfigEntity =
  | DiffusersModelConfigEntity
  | CheckpointModelConfigEntity;

export type OnnxModelConfigEntity = OnnxModelConfig & { id: string };

export type LoRAModelConfigEntity = LoRAModelConfig & { id: string };

export type ControlNetModelConfigEntity = ControlNetModelConfig & {
  id: string;
};

export type TextualInversionModelConfigEntity = TextualInversionModelConfig & {
  id: string;
};

export type VaeModelConfigEntity = VaeModelConfig & { id: string };

type AnyModelConfigEntity =
  | MainModelConfigEntity
  | OnnxModelConfigEntity
  | LoRAModelConfigEntity
  | ControlNetModelConfigEntity
  | TextualInversionModelConfigEntity
  | VaeModelConfigEntity;

type UpdateMainModelArg = {
  base_model: BaseModelType;
  model_name: string;
  body: MainModelConfig;
};

type UpdateLoRAModelArg = {
  base_model: BaseModelType;
  model_name: string;
  body: LoRAModelConfig;
};

type UpdateMainModelResponse =
  paths['/api/v1/models/{base_model}/{model_type}/{model_name}']['patch']['responses']['200']['content']['application/json'];

type UpdateLoRAModelResponse = UpdateMainModelResponse;

type DeleteMainModelArg = {
  base_model: BaseModelType;
  model_name: string;
  model_type: ModelType;
};

type DeleteMainModelResponse = void;

type DeleteLoRAModelArg = DeleteMainModelArg;

type DeleteLoRAModelResponse = void;

type ConvertMainModelArg = {
  base_model: BaseModelType;
  model_name: string;
  convert_dest_directory?: string;
};

type ConvertMainModelResponse =
  paths['/api/v1/models/convert/{base_model}/{model_type}/{model_name}']['put']['responses']['200']['content']['application/json'];

type MergeMainModelArg = {
  base_model: BaseModelType;
  body: MergeModelConfig;
};

type MergeMainModelResponse =
  paths['/api/v1/models/merge/{base_model}']['put']['responses']['200']['content']['application/json'];

type ImportMainModelArg = {
  body: ImportModelConfig;
};

type ImportMainModelResponse =
  paths['/api/v1/models/import']['post']['responses']['201']['content']['application/json'];

type AddMainModelArg = {
  body: MainModelConfig;
};

type AddMainModelResponse =
  paths['/api/v1/models/add']['post']['responses']['201']['content']['application/json'];

type SyncModelsResponse =
  paths['/api/v1/models/sync']['post']['responses']['201']['content']['application/json'];

export type SearchFolderResponse =
  paths['/api/v1/models/search']['get']['responses']['200']['content']['application/json'];

type CheckpointConfigsResponse =
  paths['/api/v1/models/ckpt_confs']['get']['responses']['200']['content']['application/json'];

type SearchFolderArg = operations['search_for_models']['parameters']['query'];

export const mainModelsAdapter = createEntityAdapter<MainModelConfigEntity>({
  sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
});

const onnxModelsAdapter = createEntityAdapter<OnnxModelConfigEntity>({
  sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
});
const loraModelsAdapter = createEntityAdapter<LoRAModelConfigEntity>({
  sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
});
export const controlNetModelsAdapter =
  createEntityAdapter<ControlNetModelConfigEntity>({
    sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
  });
export const textualInversionModelsAdapter =
  createEntityAdapter<TextualInversionModelConfigEntity>({
    sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
  });
export const vaeModelsAdapter = createEntityAdapter<VaeModelConfigEntity>({
  sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
});

export const getModelId = ({
  base_model,
  model_type,
  model_name,
}: AnyModelConfig) => `${base_model}/${model_type}/${model_name}`;

const createModelEntities = <T extends AnyModelConfigEntity>(
  models: AnyModelConfig[]
): T[] => {
  const entityArray: T[] = [];
  models.forEach((model) => {
    const entity = {
      ...cloneDeep(model),
      id: getModelId(model),
    } as T;
    entityArray.push(entity);
  });
  return entityArray;
};

export const modelsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getOnnxModels: build.query<
      EntityState<OnnxModelConfigEntity>,
      BaseModelType[]
    >({
      query: (base_models) => {
        const params = {
          model_type: 'onnx',
          base_models,
        };

        const query = queryString.stringify(params, { arrayFormat: 'none' });
        return `models/?${query}`;
      },
      providesTags: (result) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'OnnxModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'OnnxModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (response: { models: OnnxModelConfig[] }) => {
        const entities = createModelEntities<OnnxModelConfigEntity>(
          response.models
        );
        return onnxModelsAdapter.setAll(
          onnxModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getMainModels: build.query<
      EntityState<MainModelConfigEntity>,
      BaseModelType[]
    >({
      query: (base_models) => {
        const params = {
          model_type: 'main',
          base_models,
        };

        const query = queryString.stringify(params, { arrayFormat: 'none' });
        return `models/?${query}`;
      },
      providesTags: (result) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'MainModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'MainModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (response: { models: MainModelConfig[] }) => {
        const entities = createModelEntities<MainModelConfigEntity>(
          response.models
        );
        return mainModelsAdapter.setAll(
          mainModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    updateMainModels: build.mutation<
      UpdateMainModelResponse,
      UpdateMainModelArg
    >({
      query: ({ base_model, model_name, body }) => {
        return {
          url: `models/${base_model}/main/${model_name}`,
          method: 'PATCH',
          body: body,
        };
      },
      invalidatesTags: [
        { type: 'MainModel', id: LIST_TAG },
        { type: 'SDXLRefinerModel', id: LIST_TAG },
        { type: 'OnnxModel', id: LIST_TAG },
      ],
    }),
    importMainModels: build.mutation<
      ImportMainModelResponse,
      ImportMainModelArg
    >({
      query: ({ body }) => {
        return {
          url: `models/import`,
          method: 'POST',
          body: body,
        };
      },
      invalidatesTags: [
        { type: 'MainModel', id: LIST_TAG },
        { type: 'SDXLRefinerModel', id: LIST_TAG },
        { type: 'OnnxModel', id: LIST_TAG },
      ],
    }),
    addMainModels: build.mutation<AddMainModelResponse, AddMainModelArg>({
      query: ({ body }) => {
        return {
          url: `models/add`,
          method: 'POST',
          body: body,
        };
      },
      invalidatesTags: [
        { type: 'MainModel', id: LIST_TAG },
        { type: 'SDXLRefinerModel', id: LIST_TAG },
        { type: 'OnnxModel', id: LIST_TAG },
      ],
    }),
    deleteMainModels: build.mutation<
      DeleteMainModelResponse,
      DeleteMainModelArg
    >({
      query: ({ base_model, model_name, model_type }) => {
        return {
          url: `models/${base_model}/${model_type}/${model_name}`,
          method: 'DELETE',
        };
      },
      invalidatesTags: [
        { type: 'MainModel', id: LIST_TAG },
        { type: 'SDXLRefinerModel', id: LIST_TAG },
        { type: 'OnnxModel', id: LIST_TAG },
      ],
    }),
    convertMainModels: build.mutation<
      ConvertMainModelResponse,
      ConvertMainModelArg
    >({
      query: ({ base_model, model_name, convert_dest_directory }) => {
        return {
          url: `models/convert/${base_model}/main/${model_name}`,
          method: 'PUT',
          params: { convert_dest_directory },
        };
      },
      invalidatesTags: [
        { type: 'MainModel', id: LIST_TAG },
        { type: 'SDXLRefinerModel', id: LIST_TAG },
        { type: 'OnnxModel', id: LIST_TAG },
      ],
    }),
    mergeMainModels: build.mutation<MergeMainModelResponse, MergeMainModelArg>({
      query: ({ base_model, body }) => {
        return {
          url: `models/merge/${base_model}`,
          method: 'PUT',
          body: body,
        };
      },
      invalidatesTags: [
        { type: 'MainModel', id: LIST_TAG },
        { type: 'SDXLRefinerModel', id: LIST_TAG },
        { type: 'OnnxModel', id: LIST_TAG },
      ],
    }),
    syncModels: build.mutation<SyncModelsResponse, void>({
      query: () => {
        return {
          url: `models/sync`,
          method: 'POST',
        };
      },
      invalidatesTags: [
        { type: 'MainModel', id: LIST_TAG },
        { type: 'SDXLRefinerModel', id: LIST_TAG },
        { type: 'OnnxModel', id: LIST_TAG },
      ],
    }),
    getLoRAModels: build.query<EntityState<LoRAModelConfigEntity>, void>({
      query: () => ({ url: 'models/', params: { model_type: 'lora' } }),
      providesTags: (result) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'LoRAModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'LoRAModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (response: { models: LoRAModelConfig[] }) => {
        const entities = createModelEntities<LoRAModelConfigEntity>(
          response.models
        );
        return loraModelsAdapter.setAll(
          loraModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    updateLoRAModels: build.mutation<
      UpdateLoRAModelResponse,
      UpdateLoRAModelArg
    >({
      query: ({ base_model, model_name, body }) => {
        return {
          url: `models/${base_model}/lora/${model_name}`,
          method: 'PATCH',
          body: body,
        };
      },
      invalidatesTags: [{ type: 'LoRAModel', id: LIST_TAG }],
    }),
    deleteLoRAModels: build.mutation<
      DeleteLoRAModelResponse,
      DeleteLoRAModelArg
    >({
      query: ({ base_model, model_name }) => {
        return {
          url: `models/${base_model}/lora/${model_name}`,
          method: 'DELETE',
        };
      },
      invalidatesTags: [{ type: 'LoRAModel', id: LIST_TAG }],
    }),
    getControlNetModels: build.query<
      EntityState<ControlNetModelConfigEntity>,
      void
    >({
      query: () => ({ url: 'models/', params: { model_type: 'controlnet' } }),
      providesTags: (result) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'ControlNetModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'ControlNetModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (response: { models: ControlNetModelConfig[] }) => {
        const entities = createModelEntities<ControlNetModelConfigEntity>(
          response.models
        );
        return controlNetModelsAdapter.setAll(
          controlNetModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getVaeModels: build.query<EntityState<VaeModelConfigEntity>, void>({
      query: () => ({ url: 'models/', params: { model_type: 'vae' } }),
      providesTags: (result) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'VaeModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'VaeModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (response: { models: VaeModelConfig[] }) => {
        const entities = createModelEntities<VaeModelConfigEntity>(
          response.models
        );
        return vaeModelsAdapter.setAll(
          vaeModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getTextualInversionModels: build.query<
      EntityState<TextualInversionModelConfigEntity>,
      void
    >({
      query: () => ({ url: 'models/', params: { model_type: 'embedding' } }),
      providesTags: (result) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'TextualInversionModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'TextualInversionModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (response: {
        models: TextualInversionModelConfig[];
      }) => {
        const entities = createModelEntities<TextualInversionModelConfigEntity>(
          response.models
        );
        return textualInversionModelsAdapter.setAll(
          textualInversionModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getModelsInFolder: build.query<SearchFolderResponse, SearchFolderArg>({
      query: (arg) => {
        const folderQueryStr = queryString.stringify(arg, {});
        return {
          url: `/models/search?${folderQueryStr}`,
        };
      },
      providesTags: (result) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'ScannedModels', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.map((id) => ({
              type: 'ScannedModels' as const,
              id,
            }))
          );
        }
        return tags;
      },
    }),
    getCheckpointConfigs: build.query<CheckpointConfigsResponse, void>({
      query: () => {
        return {
          url: `/models/ckpt_confs`,
        };
      },
    }),
  }),
});

export const {
  useGetMainModelsQuery,
  useGetOnnxModelsQuery,
  useGetControlNetModelsQuery,
  useGetLoRAModelsQuery,
  useGetTextualInversionModelsQuery,
  useGetVaeModelsQuery,
  useUpdateMainModelsMutation,
  useDeleteMainModelsMutation,
  useImportMainModelsMutation,
  useAddMainModelsMutation,
  useConvertMainModelsMutation,
  useMergeMainModelsMutation,
  useDeleteLoRAModelsMutation,
  useUpdateLoRAModelsMutation,
  useSyncModelsMutation,
  useGetModelsInFolderQuery,
  useGetCheckpointConfigsQuery,
} = modelsApi;
