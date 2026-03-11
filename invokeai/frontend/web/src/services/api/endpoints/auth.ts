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
  multiuser_enabled: boolean;
  strict_password_checking: boolean;
};

export type UserDTO = components['schemas']['UserDTO'];

type AdminUserCreateRequest = {
  email: string;
  display_name?: string | null;
  password: string;
  is_admin?: boolean;
};

type AdminUserUpdateRequest = {
  display_name?: string | null;
  password?: string | null;
  is_admin?: boolean | null;
  is_active?: boolean | null;
};

type UserProfileUpdateRequest = {
  display_name?: string | null;
  current_password?: string | null;
  new_password?: string | null;
};

type GeneratePasswordResponse = {
  password: string;
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
    listUsers: build.query<UserDTO[], void>({
      query: () => 'api/v1/auth/users',
      providesTags: ['UserList'],
    }),
    createUser: build.mutation<UserDTO, AdminUserCreateRequest>({
      query: (data) => ({
        url: 'api/v1/auth/users',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['UserList'],
    }),
    getUser: build.query<UserDTO, string>({
      query: (userId) => `api/v1/auth/users/${userId}`,
      providesTags: (_result, _error, userId) => [{ type: 'UserList', id: userId }],
    }),
    updateUser: build.mutation<UserDTO, { userId: string; data: AdminUserUpdateRequest }>({
      query: ({ userId, data }) => ({
        url: `api/v1/auth/users/${userId}`,
        method: 'PATCH',
        body: data,
      }),
      invalidatesTags: ['UserList'],
    }),
    deleteUser: build.mutation<void, string>({
      query: (userId) => ({
        url: `api/v1/auth/users/${userId}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['UserList'],
    }),
    updateCurrentUser: build.mutation<UserDTO, UserProfileUpdateRequest>({
      query: (data) => ({
        url: 'api/v1/auth/me',
        method: 'PATCH',
        body: data,
      }),
    }),
    generatePassword: build.query<GeneratePasswordResponse, void>({
      query: () => 'api/v1/auth/generate-password',
    }),
  }),
});

export const {
  useLoginMutation,
  useLogoutMutation,
  useGetCurrentUserQuery,
  useSetupMutation,
  useGetSetupStatusQuery,
  useListUsersQuery,
  useCreateUserMutation,
  useGetUserQuery,
  useUpdateUserMutation,
  useDeleteUserMutation,
  useUpdateCurrentUserMutation,
  useLazyGeneratePasswordQuery,
} = authApi;
