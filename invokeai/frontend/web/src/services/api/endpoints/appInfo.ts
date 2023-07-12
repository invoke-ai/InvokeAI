import { api } from '..';
import { AppVersion, AppConfig } from '../types';

export const appInfoApi = api.injectEndpoints({
  endpoints: (build) => ({
    getAppVersion: build.query<AppVersion, void>({
      query: () => ({
        url: `app/version`,
        method: 'GET',
      }),
    }),
    getAppConfig: build.query<AppConfig, void>({
      query: () => ({
        url: `app/config`,
        method: 'GET',
      }),
    }),
  }),
});

export const { useGetAppVersionQuery, useGetAppConfigQuery } = appInfoApi;
