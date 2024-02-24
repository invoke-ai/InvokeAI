import type { EntityAdapter, EntityState } from '@reduxjs/toolkit';
import { createEntityAdapter } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import queryString from 'query-string';
import type { operations, paths } from 'services/api/schema';
import type {
  AnyModelConfig,
  BaseModelType,
  ControlNetModelConfig,
  IPAdapterModelConfig,
  LoRAModelConfig,
  MainModelConfig,
  T2IAdapterModelConfig,
  TextualInversionModelConfig,
  VAEModelConfig,
} from 'services/api/types';

import type { ApiTagDescription, tagTypes } from '..';
import { api, buildV2Url, LIST_TAG } from '..';

type UpdateModelArg = {
  key: NonNullable<operations['update_model_record']['parameters']['path']['key']>;
  body: NonNullable<operations['update_model_record']['requestBody']['content']['application/json']>;
};

type UpdateModelResponse = paths['/api/v2/models/i/{key}']['patch']['responses']['200']['content']['application/json'];
type GetModelConfigResponse = paths['/api/v2/models/i/{key}']['get']['responses']['200']['content']['application/json'];

type GetModelMetadataResponse =
  paths['/api/v2/models/i/{key}/metadata']['get']['responses']['200']['content']['application/json'];

type ListModelsArg = NonNullable<paths['/api/v2/models/']['get']['parameters']['query']>;

type DeleteMainModelArg = {
  key: string;
};

type DeleteMainModelResponse = void;

type ConvertMainModelResponse =
  paths['/api/v2/models/convert/{key}']['put']['responses']['200']['content']['application/json'];

type ImportMainModelArg = {
  source: NonNullable<operations['heuristic_install_model']['parameters']['query']['source']>;
  access_token?: operations['heuristic_install_model']['parameters']['query']['access_token'];
  config: NonNullable<operations['heuristic_install_model']['requestBody']['content']['application/json']>;
};

type ImportMainModelResponse =
  paths['/api/v2/models/heuristic_install']['post']['responses']['201']['content']['application/json'];

type ListImportModelsResponse =
  paths['/api/v2/models/import']['get']['responses']['200']['content']['application/json'];

type DeleteImportModelsResponse =
  paths['/api/v2/models/import/{id}']['delete']['responses']['201']['content']['application/json'];

type PruneModelImportsResponse =
  paths['/api/v2/models/import']['patch']['responses']['200']['content']['application/json'];

type ImportAdvancedModelArg = {
  source: NonNullable<operations['import_model']['requestBody']['content']['application/json']['source']>;
  config: NonNullable<operations['import_model']['requestBody']['content']['application/json']['config']>;
};

type ImportAdvancedModelResponse =
  paths['/api/v2/models/import']['post']['responses']['201']['content']['application/json'];

export type ScanFolderResponse =
  paths['/api/v2/models/scan_folder']['get']['responses']['200']['content']['application/json'];
type ScanFolderArg = operations['scan_for_models']['parameters']['query'];

export const mainModelsAdapter = createEntityAdapter<MainModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const mainModelsAdapterSelectors = mainModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const loraModelsAdapter = createEntityAdapter<LoRAModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const loraModelsAdapterSelectors = loraModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const controlNetModelsAdapter = createEntityAdapter<ControlNetModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const controlNetModelsAdapterSelectors = controlNetModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const ipAdapterModelsAdapter = createEntityAdapter<IPAdapterModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const ipAdapterModelsAdapterSelectors = ipAdapterModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const t2iAdapterModelsAdapter = createEntityAdapter<T2IAdapterModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const t2iAdapterModelsAdapterSelectors = t2iAdapterModelsAdapter.getSelectors(undefined, getSelectorsOptions);
export const textualInversionModelsAdapter = createEntityAdapter<TextualInversionModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const textualInversionModelsAdapterSelectors = textualInversionModelsAdapter.getSelectors(
  undefined,
  getSelectorsOptions
);
export const vaeModelsAdapter = createEntityAdapter<VAEModelConfig, string>({
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

// TODO(psyche): Ideally we can share the cache between the `getXYZModels` queries and `getModelConfig` query

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
    getModelMetadata: build.query<GetModelMetadataResponse, string>({
      query: (key) => {
        return buildModelsUrl(`i/${key}/metadata`);
      },
      providesTags: ['Model'],
    }),
    updateModels: build.mutation<UpdateModelResponse, UpdateModelArg>({
      query: ({ key, body }) => {
        return {
          url: buildModelsUrl(`i/${key}`),
          method: 'PATCH',
          body: body,
        };
      },
      invalidatesTags: ['Model'],
    }),
    importMainModels: build.mutation<ImportMainModelResponse, ImportMainModelArg>({
      query: ({ source, config, access_token }) => {
        return {
          url: buildModelsUrl('heuristic_install'),
          params: { source, access_token },
          method: 'POST',
          body: config,
        };
      },
      invalidatesTags: ['Model', 'ModelImports'],
    }),
    importAdvancedModel: build.mutation<ImportAdvancedModelResponse, ImportAdvancedModelArg>({
      query: ({ source, config }) => {
        return {
          url: buildModelsUrl('install'),
          method: 'POST',
          body: { source, config },
        };
      },
      invalidatesTags: ['Model', 'ModelImports'],
    }),
    deleteModels: build.mutation<DeleteMainModelResponse, DeleteMainModelArg>({
      query: ({ key }) => {
        return {
          url: buildModelsUrl(`i/${key}`),
          method: 'DELETE',
        };
      },
      invalidatesTags: ['Model'],
    }),
    convertMainModels: build.mutation<ConvertMainModelResponse, string>({
      query: (key) => {
        return {
          url: buildModelsUrl(`convert/${key}`),
          method: 'PUT',
        };
      },
      invalidatesTags: ['ModelConfig'],
    }),
    getModelConfig: build.query<GetModelConfigResponse, string>({
      query: (key) => buildModelsUrl(`i/${key}`),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['Model'];

        if (result) {
          tags.push({ type: 'ModelConfig', id: result.key });
        }

        return tags;
      },
    }),
    syncModels: build.mutation<void, void>({
      query: () => {
        return {
          url: buildModelsUrl('sync'),
          method: 'PATCH',
        };
      },
      invalidatesTags: ['Model'],
    }),
    getLoRAModels: build.query<EntityState<LoRAModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'lora' } }),
      providesTags: buildProvidesTags<LoRAModelConfig>('LoRAModel'),
      transformResponse: buildTransformResponse<LoRAModelConfig>(loraModelsAdapter),
    }),
    getControlNetModels: build.query<EntityState<ControlNetModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'controlnet' } }),
      providesTags: buildProvidesTags<ControlNetModelConfig>('ControlNetModel'),
      transformResponse: buildTransformResponse<ControlNetModelConfig>(controlNetModelsAdapter),
    }),
    getIPAdapterModels: build.query<EntityState<IPAdapterModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'ip_adapter' } }),
      providesTags: buildProvidesTags<IPAdapterModelConfig>('IPAdapterModel'),
      transformResponse: buildTransformResponse<IPAdapterModelConfig>(ipAdapterModelsAdapter),
    }),
    getT2IAdapterModels: build.query<EntityState<T2IAdapterModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 't2i_adapter' } }),
      providesTags: buildProvidesTags<T2IAdapterModelConfig>('T2IAdapterModel'),
      transformResponse: buildTransformResponse<T2IAdapterModelConfig>(t2iAdapterModelsAdapter),
    }),
    getVaeModels: build.query<EntityState<VAEModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'vae' } }),
      providesTags: buildProvidesTags<VAEModelConfig>('VaeModel'),
      transformResponse: buildTransformResponse<VAEModelConfig>(vaeModelsAdapter),
    }),
    getTextualInversionModels: build.query<EntityState<TextualInversionModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'embedding' } }),
      providesTags: buildProvidesTags<TextualInversionModelConfig>('TextualInversionModel'),
      transformResponse: buildTransformResponse<TextualInversionModelConfig>(textualInversionModelsAdapter),
    }),
    scanModels: build.query<ScanFolderResponse, ScanFolderArg>({
      query: (arg) => {
        const folderQueryStr = arg ? queryString.stringify(arg, {}) : '';
        return {
          url: buildModelsUrl(`scan_folder?${folderQueryStr}`),
        };
      },
    }),
    getModelImports: build.query<ListImportModelsResponse, void>({
      query: () => {
        return {
          url: buildModelsUrl(`import`),
        };
      },
      providesTags: ['ModelImports'],
    }),
    deleteModelImport: build.mutation<DeleteImportModelsResponse, number>({
      query: (id) => {
        return {
          url: buildModelsUrl(`import/${id}`),
          method: 'DELETE',
        };
      },
      invalidatesTags: ['ModelImports'],
    }),
    pruneModelImports: build.mutation<PruneModelImportsResponse, void>({
      query: () => {
        return {
          url: buildModelsUrl('import'),
          method: 'PATCH',
        };
      },
      invalidatesTags: ['ModelImports'],
    }),
  }),
});

export const {
  useGetModelConfigQuery,
  useGetMainModelsQuery,
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetT2IAdapterModelsQuery,
  useGetLoRAModelsQuery,
  useGetTextualInversionModelsQuery,
  useGetVaeModelsQuery,
  useDeleteModelsMutation,
  useUpdateModelsMutation,
  useImportMainModelsMutation,
  useImportAdvancedModelMutation,
  useConvertMainModelsMutation,
  useSyncModelsMutation,
  useScanModelsQuery,
  useLazyScanModelsQuery,
  useGetModelImportsQuery,
  useGetModelMetadataQuery,
  useDeleteModelImportMutation,
  usePruneModelImportsMutation,
} = modelsApi;
