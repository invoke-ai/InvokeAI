import type { EntityAdapter, EntityState } from '@reduxjs/toolkit';
import { createEntityAdapter } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import queryString from 'query-string';
import type { operations, paths } from 'services/api/schema';
import type {
  AnyModelConfig,
  BaseModelType,
  ControlNetConfig,
  ImportModelConfig,
  IPAdapterConfig,
  LoRAConfig,
  MainModelConfig,
  MergeModelConfig,
  ModelType,
  T2IAdapterConfig,
  TextualInversionConfig,
  VAEConfig,
} from 'services/api/types';

import type { ApiTagDescription, tagTypes } from '..';
import { api, buildV2Url, LIST_TAG } from '..';

/* eslint-disable @typescript-eslint/no-explicit-any */
export const getModelId = (input: any): any => input;

type UpdateMainModelArg = {
  base_model: BaseModelType;
  model_name: string;
  body: MainModelConfig;
};

type UpdateLoRAModelArg = {
  base_model: BaseModelType;
  model_name: string;
  body: LoRAConfig;
};

type UpdateMainModelResponse =
  paths['/api/v2/models/i/{key}']['patch']['responses']['200']['content']['application/json'];

type ListModelsArg = NonNullable<paths['/api/models_v2/']['get']['parameters']['query']>;

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

type AddMainModelResponse = paths['/api/v1/models/add']['post']['responses']['201']['content']['application/json'];

type SyncModelsResponse = paths['/api/v1/models/sync']['post']['responses']['201']['content']['application/json'];

export type SearchFolderResponse =
  paths['/api/v1/models/search']['get']['responses']['200']['content']['application/json'];

type CheckpointConfigsResponse =
  paths['/api/v1/models/ckpt_confs']['get']['responses']['200']['content']['application/json'];

type SearchFolderArg = operations['search_for_models']['parameters']['query'];

export const mainModelsAdapter = createEntityAdapter<MainModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const mainModelsAdapterSelectors = mainModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const loraModelsAdapter = createEntityAdapter<LoRAConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const loraModelsAdapterSelectors = loraModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const controlNetModelsAdapter = createEntityAdapter<ControlNetConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const controlNetModelsAdapterSelectors = controlNetModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const ipAdapterModelsAdapter = createEntityAdapter<IPAdapterConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const ipAdapterModelsAdapterSelectors = ipAdapterModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const t2iAdapterModelsAdapter = createEntityAdapter<T2IAdapterConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const t2iAdapterModelsAdapterSelectors = t2iAdapterModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const textualInversionModelsAdapter = createEntityAdapter<TextualInversionConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const textualInversionModelsAdapterSelectors = textualInversionModelsAdapter.getSelectors(
  undefined,
  getSelectorsOptions
);
export const vaeModelsAdapter = createEntityAdapter<VAEConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const vaeModelsAdapterSelectors = vaeModelsAdapter.getSelectors(undefined, getSelectorsOptions);

const buildProvidesTags =
  <TEntity extends AnyModelConfig>(tagType: (typeof tagTypes)[number]) =>
  (result: EntityState<TEntity, string> | undefined) => {
    const tags: ApiTagDescription[] = [{ type: tagType, id: LIST_TAG }, 'Model'];

    if (result) {
      tags.push(
        ...result.ids.map((id) => ({
          type: tagType,
          id,
        }))
      );
    }

    return tags;
  };

const buildTransformResponse =
  <T extends AnyModelConfig>(adapter: EntityAdapter<T, string>) =>
  (response: { models: T[] }) => {
    return adapter.setAll(adapter.getInitialState(), response.models);
  };

/**
 * Builds an endpoint URL for the models router
 * @example
 * buildModelsUrl('some-path')
 * // '/api/v1/models/some-path'
 */
const buildModelsUrl = (path: string = '') => buildV2Url(`models/${path}`);

export const modelsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getMainModels: build.query<EntityState<MainModelConfig, string>, BaseModelType[]>({
      query: (base_models) => {
        const params: ListModelsArg = {
          model_type: 'main',
          base_models,
        };

        const query = queryString.stringify(params, { arrayFormat: 'none' });
        return buildModelsUrl(`?${query}`);
      },
      providesTags: buildProvidesTags<MainModelConfig>('MainModel'),
      transformResponse: buildTransformResponse<MainModelConfig>(mainModelsAdapter),
    }),
    updateMainModels: build.mutation<UpdateMainModelResponse, UpdateMainModelArg>({
      query: ({ base_model, model_name, body }) => {
        return {
          url: buildModelsUrl(`${base_model}/main/${model_name}`),
          method: 'PATCH',
          body: body,
        };
      },
      invalidatesTags: ['Model'],
    }),
    importMainModels: build.mutation<ImportMainModelResponse, ImportMainModelArg>({
      query: ({ body }) => {
        return {
          url: buildModelsUrl('import'),
          method: 'POST',
          body: body,
        };
      },
      invalidatesTags: ['Model'],
    }),
    addMainModels: build.mutation<AddMainModelResponse, AddMainModelArg>({
      query: ({ body }) => {
        return {
          url: buildModelsUrl('add'),
          method: 'POST',
          body: body,
        };
      },
      invalidatesTags: ['Model'],
    }),
    deleteMainModels: build.mutation<DeleteMainModelResponse, DeleteMainModelArg>({
      query: ({ base_model, model_name, model_type }) => {
        return {
          url: buildModelsUrl(`${base_model}/${model_type}/${model_name}`),
          method: 'DELETE',
        };
      },
      invalidatesTags: ['Model'],
    }),
    convertMainModels: build.mutation<ConvertMainModelResponse, ConvertMainModelArg>({
      query: ({ base_model, model_name, convert_dest_directory }) => {
        return {
          url: buildModelsUrl(`convert/${base_model}/main/${model_name}`),
          method: 'PUT',
          params: { convert_dest_directory },
        };
      },
      invalidatesTags: ['Model'],
    }),
    mergeMainModels: build.mutation<MergeMainModelResponse, MergeMainModelArg>({
      query: ({ base_model, body }) => {
        return {
          url: buildModelsUrl(`merge/${base_model}`),
          method: 'PUT',
          body: body,
        };
      },
      invalidatesTags: ['Model'],
    }),
    syncModels: build.mutation<SyncModelsResponse, void>({
      query: () => {
        return {
          url: buildModelsUrl('sync'),
          method: 'POST',
        };
      },
      invalidatesTags: ['Model'],
    }),
    getLoRAModels: build.query<EntityState<LoRAConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'lora' } }),
      providesTags: buildProvidesTags<LoRAConfig>('LoRAModel'),
      transformResponse: buildTransformResponse<LoRAConfig>(loraModelsAdapter),
    }),
    updateLoRAModels: build.mutation<UpdateLoRAModelResponse, UpdateLoRAModelArg>({
      query: ({ base_model, model_name, body }) => {
        return {
          url: buildModelsUrl(`${base_model}/lora/${model_name}`),
          method: 'PATCH',
          body: body,
        };
      },
      invalidatesTags: [{ type: 'LoRAModel', id: LIST_TAG }],
    }),
    deleteLoRAModels: build.mutation<DeleteLoRAModelResponse, DeleteLoRAModelArg>({
      query: ({ base_model, model_name }) => {
        return {
          url: buildModelsUrl(`${base_model}/lora/${model_name}`),
          method: 'DELETE',
        };
      },
      invalidatesTags: [{ type: 'LoRAModel', id: LIST_TAG }],
    }),
    getControlNetModels: build.query<EntityState<ControlNetConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'controlnet' } }),
      providesTags: buildProvidesTags<ControlNetConfig>('ControlNetModel'),
      transformResponse: buildTransformResponse<ControlNetConfig>(controlNetModelsAdapter),
    }),
    getIPAdapterModels: build.query<EntityState<IPAdapterConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'ip_adapter' } }),
      providesTags: buildProvidesTags<IPAdapterConfig>('IPAdapterModel'),
      transformResponse: buildTransformResponse<IPAdapterConfig>(ipAdapterModelsAdapter),
    }),
    getT2IAdapterModels: build.query<EntityState<T2IAdapterConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 't2i_adapter' } }),
      providesTags: buildProvidesTags<T2IAdapterConfig>('T2IAdapterModel'),
      transformResponse: buildTransformResponse<T2IAdapterConfig>(t2iAdapterModelsAdapter),
    }),
    getVaeModels: build.query<EntityState<VAEConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'vae' } }),
      providesTags: buildProvidesTags<VAEConfig>('VaeModel'),
      transformResponse: buildTransformResponse<VAEConfig>(vaeModelsAdapter),
    }),
    getTextualInversionModels: build.query<EntityState<TextualInversionConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'embedding' } }),
      providesTags: buildProvidesTags<TextualInversionConfig>('TextualInversionModel'),
      transformResponse: buildTransformResponse<TextualInversionConfig>(textualInversionModelsAdapter),
    }),
    getModelsInFolder: build.query<SearchFolderResponse, SearchFolderArg>({
      query: (arg) => {
        const folderQueryStr = queryString.stringify(arg, {});
        return {
          url: buildModelsUrl(`search?${folderQueryStr}`),
        };
      },
    }),
    getCheckpointConfigs: build.query<CheckpointConfigsResponse, void>({
      query: () => {
        return {
          url: buildModelsUrl(`ckpt_confs`),
        };
      },
    }),
  }),
});

export const {
  useGetMainModelsQuery,
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetT2IAdapterModelsQuery,
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
