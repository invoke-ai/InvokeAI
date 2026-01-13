import { api } from 'services/api';
import type { components } from 'services/api/schema';

type LoginRequest = {
  email: string;
  password: string;
  remember_me?: boolean;
};

type LoginResponse = {
  token: string;
  user: components['schemas']['UserDTO'];
  expires_in: number;
};

type SetupRequest = {
  email: string;
  display_name: string;
  password: string;
};

type SetupResponse = {
  success: boolean;
  user: components['schemas']['UserDTO'];
};

type MeResponse = components['schemas']['UserDTO'];

type LogoutResponse = {
  success: boolean;
};

type SetupStatusResponse = {
  setup_required: boolean;
};

export const authApi = api.injectEndpoints({
  endpoints: (build) => ({
    login: build.mutation<LoginResponse, LoginRequest>({
      query: (credentials) => ({
        url: 'api/v1/auth/login',
        method: 'POST',
        body: credentials,
      }),
      // Invalidate boards and images cache on successful login to refresh data for new user
      invalidatesTags: ['Board', 'Image', 'ImageList', 'ImageNameList', 'ImageCollection', 'ImageMetadata'],
    }),
    logout: build.mutation<LogoutResponse, void>({
      query: () => ({
        url: 'api/v1/auth/logout',
        method: 'POST',
      }),
      // Invalidate boards and images cache on logout to clear stale data
      invalidatesTags: ['Board', 'Image', 'ImageList', 'ImageNameList', 'ImageCollection', 'ImageMetadata'],
    }),
    getCurrentUser: build.query<MeResponse, void>({
      query: () => 'api/v1/auth/me',
    }),
    setup: build.mutation<SetupResponse, SetupRequest>({
      query: (setupData) => ({
        url: 'api/v1/auth/setup',
        method: 'POST',
        body: setupData,
      }),
    }),
    getSetupStatus: build.query<SetupStatusResponse, void>({
      query: () => 'api/v1/auth/status',
    }),
  }),
});

export const { useLoginMutation, useLogoutMutation, useGetCurrentUserQuery, useSetupMutation, useGetSetupStatusQuery } =
  authApi;
