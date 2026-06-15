import { apiFetch, apiFetchJson } from '@workbench/backend/http';

/**
 * REST surface for the backend's multi-user endpoints (`/api/v1/auth`). Field
 * names mirror the backend DTOs verbatim; the session store and components own
 * any reshaping.
 */

const AUTH_BASE = '/api/v1/auth';

export interface UserDTO {
  user_id: string;
  email: string;
  display_name: string | null;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_login_at: string | null;
}

export interface AuthStatus {
  setup_required: boolean;
  multiuser_enabled: boolean;
  strict_password_checking: boolean;
  admin_email: string | null;
}

export interface LoginRequest {
  email: string;
  password: string;
  remember_me: boolean;
}

export interface LoginResult {
  token: string;
  user: UserDTO;
  expires_in: number;
}

export interface SetupRequest {
  email: string;
  display_name: string | null;
  password: string;
}

export interface ProfileUpdateRequest {
  display_name?: string;
  current_password?: string;
  new_password?: string;
}

export interface UserCreateRequest {
  email: string;
  display_name: string | null;
  password: string;
  is_admin: boolean;
}

export interface UserUpdateRequest {
  display_name?: string;
  password?: string;
  is_admin?: boolean;
  is_active?: boolean;
}

export const getAuthStatus = (): Promise<AuthStatus> => apiFetchJson<AuthStatus>(`${AUTH_BASE}/status`);

export const login = (request: LoginRequest): Promise<LoginResult> =>
  apiFetchJson<LoginResult>(`${AUTH_BASE}/login`, { body: JSON.stringify(request), method: 'POST' });

export const logout = (): Promise<{ success: boolean }> =>
  apiFetchJson<{ success: boolean }>(`${AUTH_BASE}/logout`, { method: 'POST' });

export const getCurrentUser = (): Promise<UserDTO> => apiFetchJson<UserDTO>(`${AUTH_BASE}/me`);

export const setupAdmin = (request: SetupRequest): Promise<{ success: boolean; user: UserDTO }> =>
  apiFetchJson<{ success: boolean; user: UserDTO }>(`${AUTH_BASE}/setup`, {
    body: JSON.stringify(request),
    method: 'POST',
  });

export const updateCurrentUser = (request: ProfileUpdateRequest): Promise<UserDTO> =>
  apiFetchJson<UserDTO>(`${AUTH_BASE}/me`, { body: JSON.stringify(request), method: 'PATCH' });

export const listUsers = (): Promise<UserDTO[]> => apiFetchJson<UserDTO[]>(`${AUTH_BASE}/users`);

export const createUser = (request: UserCreateRequest): Promise<UserDTO> =>
  apiFetchJson<UserDTO>(`${AUTH_BASE}/users`, { body: JSON.stringify(request), method: 'POST' });

export const updateUser = (userId: string, changes: UserUpdateRequest): Promise<UserDTO> =>
  apiFetchJson<UserDTO>(`${AUTH_BASE}/users/${encodeURIComponent(userId)}`, {
    body: JSON.stringify(changes),
    method: 'PATCH',
  });

export const deleteUser = async (userId: string): Promise<void> => {
  await apiFetch(`${AUTH_BASE}/users/${encodeURIComponent(userId)}`, { method: 'DELETE' });
};

export const generatePassword = async (): Promise<string> => {
  const body = await apiFetchJson<{ password: string }>(`${AUTH_BASE}/generate-password`);

  return body.password;
};
