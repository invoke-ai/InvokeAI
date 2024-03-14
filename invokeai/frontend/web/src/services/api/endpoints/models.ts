import type { EntityState } from '@reduxjs/toolkit';
import { createEntityAdapter } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import queryString from 'query-string';
import type { operations, paths } from 'services/api/schema';
import type { AnyModelConfig } from 'services/api/types';

import type { ApiTagDescription } from '..';
import { api, buildV2Url, LIST_TAG } from '..';

export type UpdateModelArg = {
  key: paths['/api/v2/models/i/{key}']['patch']['parameters']['path']['key'];
  body: paths['/api/v2/models/i/{key}']['patch']['requestBody']['content']['application/json'];
};

type UpdateModelImageArg = {
  key: string;
  image: Blob;
};

type UpdateModelResponse = paths['/api/v2/models/i/{key}']['patch']['responses']['200']['content']['application/json'];
type UpdateModelImageResponse =
  paths['/api/v2/models/i/{key}/image']['patch']['responses']['200']['content']['application/json'];

type GetModelConfigResponse = paths['/api/v2/models/i/{key}']['get']['responses']['200']['content']['application/json'];
type GetModelConfigsResponse = NonNullable<
  paths['/api/v2/models/']['get']['responses']['200']['content']['application/json']
>;

type DeleteModelArg = {
  key: string;
};
type DeleteModelResponse = void;
type DeleteModelImageResponse = void;

type ConvertMainModelResponse =
  paths['/api/v2/models/convert/{key}']['put']['responses']['200']['content']['application/json'];

type InstallModelArg = {
  source: paths['/api/v2/models/install']['post']['parameters']['query']['source'];
  inplace?: paths['/api/v2/models/install']['post']['parameters']['query']['inplace'];
};
type InstallModelResponse = paths['/api/v2/models/install']['post']['responses']['201']['content']['application/json'];

type ListModelInstallsResponse =
  paths['/api/v2/models/install']['get']['responses']['200']['content']['application/json'];

type CancelModelInstallResponse =
  paths['/api/v2/models/install/{id}']['delete']['responses']['201']['content']['application/json'];

type PruneCompletedModelInstallsResponse =
  paths['/api/v2/models/install']['delete']['responses']['200']['content']['application/json'];

export type ScanFolderResponse =
  paths['/api/v2/models/scan_folder']['get']['responses']['200']['content']['application/json'];
type ScanFolderArg = operations['scan_for_models']['parameters']['query'];

type GetHuggingFaceModelsResponse =
  paths['/api/v2/models/hugging_face']['get']['responses']['200']['content']['application/json'];

type GetByAttrsArg = operations['get_model_records_by_attrs']['parameters']['query'];

const modelConfigsAdapter = createEntityAdapter<AnyModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const modelConfigsAdapterSelectors = modelConfigsAdapter.getSelectors(undefined, getSelectorsOptions);

/**
 * Builds an endpoint URL for the models router
 * @example
 * buildModelsUrl('some-path')
 * // '/api/v1/models/some-path'
 */
const buildModelsUrl = (path: string = '') => buildV2Url(`models/${path}`);

export const modelsApi = api.injectEndpoints({
  endpoints: (build) => ({
    updateModel: build.mutation<UpdateModelResponse, UpdateModelArg>({
      query: ({ key, body }) => {
        return {
          url: buildModelsUrl(`i/${key}`),
          method: 'PATCH',
          body: body,
        };
      },
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        try {
          const { data } = await queryFulfilled;

          // Update the individual model query caches
          dispatch(modelsApi.util.upsertQueryData('getModelConfig', data.key, data));

          const { base, name, type } = data;
          dispatch(modelsApi.util.upsertQueryData('getModelConfigByAttrs', { base, name, type }, data));

          // Update the list query cache
          dispatch(
            modelsApi.util.updateQueryData('getModelConfigs', undefined, (draft) => {
              modelConfigsAdapter.updateOne(draft, {
                id: data.key,
                changes: data,
              });
            })
          );
        } catch {
          // no-op
        }
      },
    }),
    updateModelImage: build.mutation<UpdateModelImageResponse, UpdateModelImageArg>({
      query: ({ key, image }) => {
        const formData = new FormData();
        formData.append('image', image);
        return {
          url: buildModelsUrl(`i/${key}/image`),
          method: 'PATCH',
          body: formData,
        };
      },
      invalidatesTags: ['Model'],
    }),
    installModel: build.mutation<InstallModelResponse, InstallModelArg>({
      query: ({ source, inplace = true }) => {
        return {
          url: buildModelsUrl('install'),
          params: { source, inplace },
          method: 'POST',
        };
      },
      invalidatesTags: ['Model', 'ModelInstalls'],
    }),
    deleteModels: build.mutation<DeleteModelResponse, DeleteModelArg>({
      query: ({ key }) => {
        return {
          url: buildModelsUrl(`i/${key}`),
          method: 'DELETE',
        };
      },
      invalidatesTags: ['Model'],
    }),
    deleteModelImage: build.mutation<DeleteModelImageResponse, string>({
      query: (key) => {
        return {
          url: buildModelsUrl(`i/${key}/image`),
          method: 'DELETE',
        };
      },
      invalidatesTags: ['Model'],
    }),
    getModelImage: build.query<string, string>({
      query: (key) => buildModelsUrl(`i/${key}/image`),
    }),
    convertModel: build.mutation<ConvertMainModelResponse, string>({
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
    getModelConfigByAttrs: build.query<AnyModelConfig, GetByAttrsArg>({
      query: (arg) => buildModelsUrl(`get_by_attrs?${queryString.stringify(arg)}`),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['Model'];

        if (result) {
          tags.push({ type: 'ModelConfig', id: result.key });
        }

        return tags;
      },
      serializeQueryArgs: ({ queryArgs }) => `${queryArgs.name}.${queryArgs.base}.${queryArgs.type}`,
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
    scanFolder: build.query<ScanFolderResponse, ScanFolderArg>({
      query: (arg) => {
        const folderQueryStr = arg ? queryString.stringify(arg, {}) : '';
        return {
          url: buildModelsUrl(`scan_folder?${folderQueryStr}`),
        };
      },
    }),
    getHuggingFaceModels: build.query<GetHuggingFaceModelsResponse, string>({
      query: (hugging_face_repo) => {
        return {
          url: buildModelsUrl(`hugging_face?hugging_face_repo=${hugging_face_repo}`),
        };
      },
    }),
    listModelInstalls: build.query<ListModelInstallsResponse, void>({
      query: () => {
        return {
          url: buildModelsUrl('install'),
        };
      },
      providesTags: ['ModelInstalls'],
    }),
    cancelModelInstall: build.mutation<CancelModelInstallResponse, number>({
      query: (id) => {
        return {
          url: buildModelsUrl(`install/${id}`),
          method: 'DELETE',
        };
      },
      invalidatesTags: ['ModelInstalls'],
    }),
    pruneCompletedModelInstalls: build.mutation<PruneCompletedModelInstallsResponse, void>({
      query: () => {
        return {
          url: buildModelsUrl('install'),
          method: 'DELETE',
        };
      },
      invalidatesTags: ['ModelInstalls'],
    }),
    getModelConfigs: build.query<EntityState<AnyModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl() }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = [{ type: 'ModelConfig', id: LIST_TAG }];
        if (result) {
          const modelTags = result.ids.map((id) => ({ type: 'ModelConfig', id }) as const);
          tags.push(...modelTags);
        }
        return tags;
      },
      keepUnusedDataFor: 60 * 60 * 1000 * 24, // 1 day (infinite)
      transformResponse: (response: GetModelConfigsResponse) => {
        return modelConfigsAdapter.setAll(modelConfigsAdapter.getInitialState(), response.models);
      },
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          modelConfigsAdapterSelectors.selectAll(data).forEach((modelConfig) => {
            dispatch(modelsApi.util.upsertQueryData('getModelConfig', modelConfig.key, modelConfig));
            const { base, name, type } = modelConfig;
            dispatch(modelsApi.util.upsertQueryData('getModelConfigByAttrs', { base, name, type }, modelConfig));
          });
        });
      },
    }),
  }),
});

export const {
  useGetModelConfigsQuery,
  useGetModelConfigQuery,
  useDeleteModelsMutation,
  useDeleteModelImageMutation,
  useUpdateModelMutation,
  useUpdateModelImageMutation,
  useInstallModelMutation,
  useConvertModelMutation,
  useSyncModelsMutation,
  useLazyScanFolderQuery,
  useLazyGetHuggingFaceModelsQuery,
  useListModelInstallsQuery,
  useCancelModelInstallMutation,
  usePruneCompletedModelInstallsMutation,
} = modelsApi;
