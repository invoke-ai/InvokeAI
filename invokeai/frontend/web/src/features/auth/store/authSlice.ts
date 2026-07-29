import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { SliceConfig } from 'app/store/types';
import { z } from 'zod';

const zUser = z.object({
  user_id: z.string(),
  email: z.string(),
  display_name: z.string().nullable(),
  is_admin: z.boolean(),
  is_active: z.boolean(),
});

const zAuthState = z.object({
  isAuthenticated: z.boolean(),
  token: z.string().nullable(),
  user: zUser.nullable(),
  isLoading: z.boolean(),
  sessionExpired: z.boolean(),
});

type User = z.infer<typeof zUser>;
type AuthState = z.infer<typeof zAuthState>;

const getTokenUserId = (token: string | null): string | null => {
  if (!token) {
    return null;
  }
  try {
    const encodedPayload = token.split('.')[1];
    if (!encodedPayload) {
      return null;
    }
    const normalizedPayload = encodedPayload.replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(normalizedPayload.padEnd(Math.ceil(normalizedPayload.length / 4) * 4, '=')));
    return typeof payload.user_id === 'string' ? payload.user_id : null;
  } catch {
    return null;
  }
};

export const tokensBelongToSameUser = (first: string | null, second: string | null): boolean => {
  const firstUserId = getTokenUserId(first);
  return firstUserId !== null && firstUserId === getTokenUserId(second);
};

// Helper to safely access localStorage (not available in test environment)
const getStoredAuthToken = (): string | null => {
  if (typeof window !== 'undefined' && window.localStorage) {
    return localStorage.getItem('auth_token');
  }
  return null;
};

const initialState: AuthState = {
  isAuthenticated: !!getStoredAuthToken(),
  token: getStoredAuthToken(),
  user: null,
  isLoading: false,
  sessionExpired: false,
};

const getInitialAuthState = (): AuthState => initialState;

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials: (state, action: PayloadAction<{ token: string; user: User }>) => {
      state.token = action.payload.token;
      state.user = action.payload.user;
      state.isAuthenticated = true;
      state.sessionExpired = false;
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem('auth_token', action.payload.token);
      }
    },
    tokenRefreshed: (state, action: PayloadAction<string>) => {
      state.token = action.payload;
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem('auth_token', action.payload);
      }
    },
    externalTokenAdopted: (state, action: PayloadAction<string>) => {
      if (!tokensBelongToSameUser(state.token, action.payload)) {
        state.user = null;
      }
      state.token = action.payload;
      state.isAuthenticated = true;
      state.sessionExpired = false;
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem('auth_token', action.payload);
      }
    },
    currentUserUpdated: (state, action: PayloadAction<User>) => {
      state.user = action.payload;
    },
    logout: (state) => {
      state.token = null;
      state.user = null;
      state.isAuthenticated = false;
      state.sessionExpired = false;
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.removeItem('auth_token');
      }
    },
    sessionExpiredLogout: (state) => {
      state.token = null;
      state.user = null;
      state.isAuthenticated = false;
      state.sessionExpired = true;
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.removeItem('auth_token');
      }
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
  },
});

export const {
  setCredentials,
  tokenRefreshed,
  externalTokenAdopted,
  currentUserUpdated,
  logout,
  sessionExpiredLogout,
  setLoading,
} = authSlice.actions;

export const authSliceConfig: SliceConfig<typeof authSlice> = {
  slice: authSlice,
  schema: zAuthState,
  getInitialState: getInitialAuthState,
  persistConfig: {
    migrate: () => getInitialAuthState(),
    // Don't persist auth state - token is stored in localStorage
    persistDenylist: ['isAuthenticated', 'token', 'user', 'isLoading', 'sessionExpired'],
  },
};

export const selectIsAuthenticated = (state: { auth: AuthState }) => state.auth.isAuthenticated;
export const selectCurrentUser = (state: { auth: AuthState }) => state.auth.user;
export const selectAuthToken = (state: { auth: AuthState }) => state.auth.token;
export const selectIsAuthLoading = (state: { auth: AuthState }) => state.auth.isLoading;
export const selectSessionExpired = (state: { auth: AuthState }) => state.auth.sessionExpired;
