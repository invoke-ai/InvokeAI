import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { cloneDeep } from 'lodash-es';
import {
  AnyModelConfig,
  BaseModelType,
  CheckpointModelConfig,
  ControlNetModelConfig,
  ConvertModelConfig,
  DiffusersModelConfig,
  LoRAModelConfig,
  MainModelConfig,
  MergeModelConfig,
  TextualInversionModelConfig,
  VaeModelConfig,
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
  | LoRAModelConfigEntity
  | ControlNetModelConfigEntity
  | TextualInversionModelConfigEntity
  | VaeModelConfigEntity;

type UpdateMainModelArg = {
  base_model: BaseModelType;
  model_name: string;
  body: MainModelConfig;
};

type UpdateMainModelResponse =
  paths['/api/v1/models/{base_model}/{model_type}/{model_name}']['patch']['responses']['200']['content']['application/json'];

type DeleteMainModelArg = {
  base_model: BaseModelType;
  model_name: string;
};

type DeleteMainModelResponse = void;

type ConvertMainModelArg = {
  base_model: BaseModelType;
  model_name: string;
  body: ConvertModelConfig;
};

type ConvertMainModelResponse =
  paths['/api/v1/models/convert/{base_model}/{model_type}/{model_name}']['put']['responses']['200']['content']['application/json'];

type MergeMainModelArg = {
  base_model: BaseModelType;
  body: MergeModelConfig;
};

type MergeMainModelResponse =
  paths['/api/v1/models/merge/{base_model}']['put']['responses']['200']['content']['application/json'];

type SearchFolderResponse =
  paths['/api/v1/models/search']['get']['responses']['200']['content']['application/json'];

type SearchFolderArg = operations['search_for_models']['parameters']['query'];

const mainModelsAdapter = createEntityAdapter<MainModelConfigEntity>({
  sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
});
const loraModelsAdapter = createEntityAdapter<LoRAModelConfigEntity>({
  sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
});
const controlNetModelsAdapter =
  createEntityAdapter<ControlNetModelConfigEntity>({
    sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
  });
const textualInversionModelsAdapter =
  createEntityAdapter<TextualInversionModelConfigEntity>({
    sortComparer: (a, b) => a.model_name.localeCompare(b.model_name),
  });
const vaeModelsAdapter = createEntityAdapter<VaeModelConfigEntity>({
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
    getMainModels: build.query<EntityState<MainModelConfigEntity>, void>({
      query: () => ({ url: 'models/', params: { model_type: 'main' } }),
      providesTags: (result, error, arg) => {
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
      transformResponse: (
        response: { models: MainModelConfig[] },
        meta,
        arg
      ) => {
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
      invalidatesTags: [{ type: 'MainModel', id: LIST_TAG }],
    }),
    deleteMainModels: build.mutation<
      DeleteMainModelResponse,
      DeleteMainModelArg
    >({
      query: ({ base_model, model_name }) => {
        return {
          url: `models/${base_model}/main/${model_name}`,
          method: 'DELETE',
        };
      },
      invalidatesTags: [{ type: 'MainModel', id: LIST_TAG }],
    }),
    convertMainModels: build.mutation<
      ConvertMainModelResponse,
      ConvertMainModelArg
    >({
      query: ({ base_model, model_name, body }) => {
        return {
          url: `models/convert/${base_model}/main/${model_name}`,
          method: 'PUT',
          body: body,
        };
      },
      invalidatesTags: [{ type: 'MainModel', id: LIST_TAG }],
    }),
    mergeMainModels: build.mutation<MergeMainModelResponse, MergeMainModelArg>({
      query: ({ base_model, body }) => {
        return {
          url: `models/merge/${base_model}`,
          method: 'PUT',
          body: body,
        };
      },
      invalidatesTags: [{ type: 'MainModel', id: LIST_TAG }],
    }),
    getLoRAModels: build.query<EntityState<LoRAModelConfigEntity>, void>({
      query: () => ({ url: 'models/', params: { model_type: 'lora' } }),
      providesTags: (result, error, arg) => {
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
      transformResponse: (
        response: { models: LoRAModelConfig[] },
        meta,
        arg
      ) => {
        const entities = createModelEntities<LoRAModelConfigEntity>(
          response.models
        );
        return loraModelsAdapter.setAll(
          loraModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getControlNetModels: build.query<
      EntityState<ControlNetModelConfigEntity>,
      void
    >({
      query: () => ({ url: 'models/', params: { model_type: 'controlnet' } }),
      providesTags: (result, error, arg) => {
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
      transformResponse: (
        response: { models: ControlNetModelConfig[] },
        meta,
        arg
      ) => {
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
      providesTags: (result, error, arg) => {
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
      transformResponse: (
        response: { models: VaeModelConfig[] },
        meta,
        arg
      ) => {
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
      providesTags: (result, error, arg) => {
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
      transformResponse: (
        response: { models: TextualInversionModelConfig[] },
        meta,
        arg
      ) => {
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
    }),
  }),
});

export const {
  useGetMainModelsQuery,
  useGetControlNetModelsQuery,
  useGetLoRAModelsQuery,
  useGetTextualInversionModelsQuery,
  useGetVaeModelsQuery,
  useUpdateMainModelsMutation,
  useDeleteMainModelsMutation,
  useConvertMainModelsMutation,
  useMergeMainModelsMutation,
  useGetModelsInFolderQuery,
} = modelsApi;
