import { api } from '..';
import { AppVersion } from '../types';

export const appInfoApi = api.injectEndpoints({
  endpoints: (build) => ({
    getAppVersion: build.query<AppVersion, void>({
      query: () => ({
        url: `app/version`,
        method: 'GET',
      }),
    }),
  }),
});

export const { useGetAppVersionQuery } = appInfoApi;
