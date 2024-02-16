import { $openAPISchemaUrl } from 'app/store/nanostores/openAPISchemaUrl';
import type { OpenAPIV3_1 } from 'openapi-types';
import type { paths } from 'services/api/schema';
import type { AppConfig, AppDependencyVersions, AppVersion } from 'services/api/types';

import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the app router
 * @example
 * buildAppInfoUrl('some-path')
 * // '/api/v1/app/some-path'
 */
const buildAppInfoUrl = (path: string = '') => buildV1Url(`app/${path}`);

export const appInfoApi = api.injectEndpoints({
  endpoints: (build) => ({
    getAppVersion: build.query<AppVersion, void>({
      query: () => ({
        url: buildAppInfoUrl('version'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    getAppDeps: build.query<AppDependencyVersions, void>({
      query: () => ({
        url: buildAppInfoUrl('app_deps'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    getAppConfig: build.query<AppConfig, void>({
      query: () => ({
        url: buildAppInfoUrl('config'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
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
      query: () => {
        const openAPISchemaUrl = $openAPISchemaUrl.get();
        const url = openAPISchemaUrl ? openAPISchemaUrl : `${window.location.href.replace(/\/$/, '')}/openapi.json`;
        return url;
      },
      providesTags: ['Schema'],
    }),
  }),
});

export const {
  useGetAppVersionQuery,
  useGetAppDepsQuery,
  useGetAppConfigQuery,
  useClearInvocationCacheMutation,
  useDisableInvocationCacheMutation,
  useEnableInvocationCacheMutation,
  useGetInvocationCacheStatusQuery,
  useGetOpenAPISchemaQuery,
  useLazyGetOpenAPISchemaQuery,
} = appInfoApi;
