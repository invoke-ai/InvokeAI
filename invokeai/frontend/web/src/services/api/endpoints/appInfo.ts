import { api } from '..';
import { AppConfig, AppVersion } from '../types';

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
  }),
});

export const { useGetAppVersionQuery, useGetAppConfigQuery } = appInfoApi;
