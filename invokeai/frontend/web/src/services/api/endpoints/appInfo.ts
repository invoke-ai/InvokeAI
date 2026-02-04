import type { OpenAPIV3_1 } from 'openapi-types';
import type { stringify } from 'querystring';
import type { paths } from 'services/api/schema';
import type { AppVersion, ExternalProviderConfig, ExternalProviderStatus } from 'services/api/types';

import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the app router
 * @example
 * buildAppInfoUrl('some-path')
 * // '/api/v1/app/some-path'
 */
const buildAppInfoUrl = (path: string = '', query?: Parameters<typeof stringify>[0]) =>
  buildV1Url(`app/${path}`, query);

export const appInfoApi = api.injectEndpoints({
  endpoints: (build) => ({
    getAppVersion: build.query<AppVersion, void>({
      query: () => ({
        url: buildAppInfoUrl('version'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    getAppDeps: build.query<
      paths['/api/v1/app/app_deps']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildAppInfoUrl('app_deps'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    getPatchmatchStatus: build.query<
      paths['/api/v1/app/patchmatch_status']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildAppInfoUrl('patchmatch_status'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    getRuntimeConfig: build.query<
      paths['/api/v1/app/runtime_config']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildAppInfoUrl('runtime_config'),
        method: 'GET',
      }),
    }),
    getExternalProviderStatuses: build.query<ExternalProviderStatus[], void>({
      query: () => ({
        url: buildAppInfoUrl('external_providers/status'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    getExternalProviderConfigs: build.query<ExternalProviderConfig[], void>({
      query: () => ({
        url: buildAppInfoUrl('external_providers/config'),
        method: 'GET',
      }),
      providesTags: ['AppConfig', 'FetchOnReconnect'],
    }),
    setExternalProviderConfig: build.mutation<ExternalProviderConfig, SetExternalProviderConfigArg>({
      query: ({ provider_id, ...body }) => ({
        url: buildAppInfoUrl(`external_providers/config/${provider_id}`),
        method: 'POST',
        body,
      }),
      invalidatesTags: ['AppConfig', 'FetchOnReconnect'],
    }),
    resetExternalProviderConfig: build.mutation<ExternalProviderConfig, string>({
      query: (provider_id) => ({
        url: buildAppInfoUrl(`external_providers/config/${provider_id}`),
        method: 'DELETE',
      }),
      invalidatesTags: ['AppConfig', 'FetchOnReconnect'],
    }),
    getInvocationCacheStatus: build.query<
      paths['/api/v1/app/invocation_cache/status']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildAppInfoUrl('invocation_cache/status'),
        method: 'GET',
      }),
      providesTags: ['InvocationCacheStatus', 'FetchOnReconnect'],
    }),
    clearInvocationCache: build.mutation<void, void>({
      query: () => ({
        url: buildAppInfoUrl('invocation_cache'),
        method: 'DELETE',
      }),
      invalidatesTags: ['InvocationCacheStatus'],
    }),
    enableInvocationCache: build.mutation<void, void>({
      query: () => ({
        url: buildAppInfoUrl('invocation_cache/enable'),
        method: 'PUT',
      }),
      invalidatesTags: ['InvocationCacheStatus'],
    }),
    disableInvocationCache: build.mutation<void, void>({
      query: () => ({
        url: buildAppInfoUrl('invocation_cache/disable'),
        method: 'PUT',
      }),
      invalidatesTags: ['InvocationCacheStatus'],
    }),
    getOpenAPISchema: build.query<OpenAPIV3_1.Document, void>({
      query: () => `${window.location.href.replace(/\/$/, '')}/openapi.json`,
      providesTags: ['Schema'],
    }),
  }),
});

export const {
  useGetAppVersionQuery,
  useGetAppDepsQuery,
  useGetPatchmatchStatusQuery,
  useGetRuntimeConfigQuery,
  useGetExternalProviderStatusesQuery,
  useGetExternalProviderConfigsQuery,
  useSetExternalProviderConfigMutation,
  useResetExternalProviderConfigMutation,
  useClearInvocationCacheMutation,
  useDisableInvocationCacheMutation,
  useEnableInvocationCacheMutation,
  useGetInvocationCacheStatusQuery,
  useGetOpenAPISchemaQuery,
  useLazyGetOpenAPISchemaQuery,
} = appInfoApi;

type SetExternalProviderConfigArg =
  paths['/api/v1/app/external_providers/config/{provider_id}']['post']['requestBody']['content']['application/json'] & {
    provider_id: paths['/api/v1/app/external_providers/config/{provider_id}']['post']['parameters']['path']['provider_id'];
  };
