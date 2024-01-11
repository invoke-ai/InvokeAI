import type { paths } from 'services/api/schema';
import type { AppConfig, AppVersion } from 'services/api/types';

import { api } from '..';

export const appInfoApi = api.injectEndpoints({
  endpoints: (build) => ({
    getAppVersion: build.query<AppVersion, void>({
      query: () => ({
        url: `app/version`,
        method: 'GET',
      }),
      providesTags: ['AppVersion'],
      keepUnusedDataFor: 86400000, // 1 day
    }),
    getAppConfig: build.query<AppConfig, void>({
      query: () => ({
        url: `app/config`,
        method: 'GET',
      }),
      providesTags: ['AppConfig'],
      keepUnusedDataFor: 86400000, // 1 day
    }),
    getInvocationCacheStatus: build.query<
      paths['/api/v1/app/invocation_cache/status']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: `app/invocation_cache/status`,
        method: 'GET',
      }),
      providesTags: ['InvocationCacheStatus', 'FetchOnReconnect'],
    }),
    clearInvocationCache: build.mutation<void, void>({
      query: () => ({
        url: `app/invocation_cache`,
        method: 'DELETE',
      }),
      invalidatesTags: ['InvocationCacheStatus'],
    }),
    enableInvocationCache: build.mutation<void, void>({
      query: () => ({
        url: `app/invocation_cache/enable`,
        method: 'PUT',
      }),
      invalidatesTags: ['InvocationCacheStatus'],
    }),
    disableInvocationCache: build.mutation<void, void>({
      query: () => ({
        url: `app/invocation_cache/disable`,
        method: 'PUT',
      }),
      invalidatesTags: ['InvocationCacheStatus'],
    }),
  }),
});

export const {
  useGetAppVersionQuery,
  useGetAppConfigQuery,
  useClearInvocationCacheMutation,
  useDisableInvocationCacheMutation,
  useEnableInvocationCacheMutation,
  useGetInvocationCacheStatusQuery,
} = appInfoApi;
