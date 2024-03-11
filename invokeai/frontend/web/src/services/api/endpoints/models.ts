import type { EntityAdapter, EntityState, ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import { createEntityAdapter } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import queryString from 'query-string';
import {
  ALL_BASE_MODELS,
  NON_REFINER_BASE_MODELS,
  NON_SDXL_MAIN_MODELS,
  REFINER_BASE_MODELS,
  SDXL_MAIN_MODELS,
} from 'services/api/constants';
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

type ListModelsArg = NonNullable<paths['/api/v2/models/']['get']['parameters']['query']>;

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

const mainModelsAdapter = createEntityAdapter<MainModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const mainModelsAdapterSelectors = mainModelsAdapter.getSelectors(undefined, getSelectorsOptions);
const loraModelsAdapter = createEntityAdapter<LoRAModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const loraModelsAdapterSelectors = loraModelsAdapter.getSelectors(undefined, getSelectorsOptions);
const controlNetModelsAdapter = createEntityAdapter<ControlNetModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const controlNetModelsAdapterSelectors = controlNetModelsAdapter.getSelectors(undefined, getSelectorsOptions);
const ipAdapterModelsAdapter = createEntityAdapter<IPAdapterModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const ipAdapterModelsAdapterSelectors = ipAdapterModelsAdapter.getSelectors(undefined, getSelectorsOptions);
const t2iAdapterModelsAdapter = createEntityAdapter<T2IAdapterModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const t2iAdapterModelsAdapterSelectors = t2iAdapterModelsAdapter.getSelectors(undefined, getSelectorsOptions);
const textualInversionModelsAdapter = createEntityAdapter<TextualInversionModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const textualInversionModelsAdapterSelectors = textualInversionModelsAdapter.getSelectors(
  undefined,
  getSelectorsOptions
);
const vaeModelsAdapter = createEntityAdapter<VAEModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
export const vaeModelsAdapterSelectors = vaeModelsAdapter.getSelectors(undefined, getSelectorsOptions);

const anyModelConfigAdapter = createEntityAdapter<AnyModelConfig, string>({
  selectId: (entity) => entity.key,
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
const anyModelConfigAdapterSelectors = anyModelConfigAdapter.getSelectors(undefined, getSelectorsOptions);

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
    updateModel: build.mutation<UpdateModelResponse, UpdateModelArg>({
      query: ({ key, body }) => {
        return {
          url: buildModelsUrl(`i/${key}`),
          method: 'PATCH',
          body: body,
        };
      },
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertSingleModelConfig(data, dispatch);
        });
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
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertModelConfigs(data, dispatch);
        });
      },
    }),
    getLoRAModels: build.query<EntityState<LoRAModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'lora' } }),
      providesTags: buildProvidesTags<LoRAModelConfig>('LoRAModel'),
      transformResponse: buildTransformResponse<LoRAModelConfig>(loraModelsAdapter),
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertModelConfigs(data, dispatch);
        });
      },
    }),
    getControlNetModels: build.query<EntityState<ControlNetModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'controlnet' } }),
      providesTags: buildProvidesTags<ControlNetModelConfig>('ControlNetModel'),
      transformResponse: buildTransformResponse<ControlNetModelConfig>(controlNetModelsAdapter),
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertModelConfigs(data, dispatch);
        });
      },
    }),
    getIPAdapterModels: build.query<EntityState<IPAdapterModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'ip_adapter' } }),
      providesTags: buildProvidesTags<IPAdapterModelConfig>('IPAdapterModel'),
      transformResponse: buildTransformResponse<IPAdapterModelConfig>(ipAdapterModelsAdapter),
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertModelConfigs(data, dispatch);
        });
      },
    }),
    getT2IAdapterModels: build.query<EntityState<T2IAdapterModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 't2i_adapter' } }),
      providesTags: buildProvidesTags<T2IAdapterModelConfig>('T2IAdapterModel'),
      transformResponse: buildTransformResponse<T2IAdapterModelConfig>(t2iAdapterModelsAdapter),
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertModelConfigs(data, dispatch);
        });
      },
    }),
    getVaeModels: build.query<EntityState<VAEModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'vae' } }),
      providesTags: buildProvidesTags<VAEModelConfig>('VaeModel'),
      transformResponse: buildTransformResponse<VAEModelConfig>(vaeModelsAdapter),
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertModelConfigs(data, dispatch);
        });
      },
    }),
    getTextualInversionModels: build.query<EntityState<TextualInversionModelConfig, string>, void>({
      query: () => ({ url: buildModelsUrl(), params: { model_type: 'embedding' } }),
      providesTags: buildProvidesTags<TextualInversionModelConfig>('TextualInversionModel'),
      transformResponse: buildTransformResponse<TextualInversionModelConfig>(textualInversionModelsAdapter),
      onQueryStarted: async (_, { dispatch, queryFulfilled }) => {
        queryFulfilled.then(({ data }) => {
          upsertModelConfigs(data, dispatch);
        });
      },
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

const upsertModelConfigs = (
  modelConfigs: EntityState<AnyModelConfig, string>,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dispatch: ThunkDispatch<any, any, UnknownAction>
) => {
  /**
   * Once a list of models of a specific type is received, fetching any of those models individually is a waste of a
   * network request. This function takes the received list of models and upserts them into the individual query caches
   * for each model type.
   */

  // Iterate over all the models and upsert them into the individual query caches for each model type.
  anyModelConfigAdapterSelectors.selectAll(modelConfigs).forEach((modelConfig) => {
    dispatch(modelsApi.util.upsertQueryData('getModelConfig', modelConfig.key, modelConfig));
    const { base, name, type } = modelConfig;
    dispatch(modelsApi.util.upsertQueryData('getModelConfigByAttrs', { base, name, type }, modelConfig));
  });
};

const upsertSingleModelConfig = (
  modelConfig: AnyModelConfig,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dispatch: ThunkDispatch<any, any, UnknownAction>
) => {
  /**
   * When a model is updated, the individual query caches for each model type need to be updated, as well as the list
   * query caches of models of that type.
   */

  // Update the individual model query caches.
  dispatch(modelsApi.util.upsertQueryData('getModelConfig', modelConfig.key, modelConfig));
  const { base, name, type } = modelConfig;
  dispatch(modelsApi.util.upsertQueryData('getModelConfigByAttrs', { base, name, type }, modelConfig));

  // Update the list query caches for each model type.
  if (modelConfig.type === 'main') {
    [ALL_BASE_MODELS, NON_REFINER_BASE_MODELS, SDXL_MAIN_MODELS, NON_SDXL_MAIN_MODELS, REFINER_BASE_MODELS].forEach(
      (queryArg) => {
        dispatch(
          modelsApi.util.updateQueryData('getMainModels', queryArg, (draft) => {
            mainModelsAdapter.updateOne(draft, {
              id: modelConfig.key,
              changes: modelConfig,
            });
          })
        );
      }
    );
    return;
  }

  if (modelConfig.type === 'controlnet') {
    dispatch(
      modelsApi.util.updateQueryData('getControlNetModels', undefined, (draft) => {
        controlNetModelsAdapter.updateOne(draft, {
          id: modelConfig.key,
          changes: modelConfig,
        });
      })
    );
    return;
  }

  if (modelConfig.type === 'embedding') {
    dispatch(
      modelsApi.util.updateQueryData('getTextualInversionModels', undefined, (draft) => {
        textualInversionModelsAdapter.updateOne(draft, {
          id: modelConfig.key,
          changes: modelConfig,
        });
      })
    );
    return;
  }

  if (modelConfig.type === 'ip_adapter') {
    dispatch(
      modelsApi.util.updateQueryData('getIPAdapterModels', undefined, (draft) => {
        ipAdapterModelsAdapter.updateOne(draft, {
          id: modelConfig.key,
          changes: modelConfig,
        });
      })
    );
    return;
  }

  if (modelConfig.type === 'lora') {
    dispatch(
      modelsApi.util.updateQueryData('getLoRAModels', undefined, (draft) => {
        loraModelsAdapter.updateOne(draft, {
          id: modelConfig.key,
          changes: modelConfig,
        });
      })
    );
    return;
  }

  if (modelConfig.type === 't2i_adapter') {
    dispatch(
      modelsApi.util.updateQueryData('getT2IAdapterModels', undefined, (draft) => {
        t2iAdapterModelsAdapter.updateOne(draft, {
          id: modelConfig.key,
          changes: modelConfig,
        });
      })
    );
    return;
  }

  if (modelConfig.type === 'vae') {
    dispatch(
      modelsApi.util.updateQueryData('getVaeModels', undefined, (draft) => {
        vaeModelsAdapter.updateOne(draft, {
          id: modelConfig.key,
          changes: modelConfig,
        });
      })
    );
    return;
  }
};
