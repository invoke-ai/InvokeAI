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
});

type User = z.infer<typeof zUser>;
type AuthState = z.infer<typeof zAuthState>;

const initialState: AuthState = {
  isAuthenticated: !!localStorage.getItem('auth_token'),
  token: localStorage.getItem('auth_token'),
  user: null,
  isLoading: false,
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
      localStorage.setItem('auth_token', action.payload.token);
    },
    logout: (state) => {
      state.token = null;
      state.user = null;
      state.isAuthenticated = false;
      localStorage.removeItem('auth_token');
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
  },
});

export const { setCredentials, logout, setLoading } = authSlice.actions;

export const authSliceConfig: SliceConfig<typeof authSlice> = {
  slice: authSlice,
  schema: zAuthState,
  getInitialState: getInitialAuthState,
  persistConfig: {
    migrate: () => getInitialAuthState(),
    // Don't persist auth state - token is stored in localStorage
    persistDenylist: ['isAuthenticated', 'token', 'user', 'isLoading'],
  },
};

export const selectIsAuthenticated = (state: { auth: AuthState }) => state.auth.isAuthenticated;
export const selectCurrentUser = (state: { auth: AuthState }) => state.auth.user;
export const selectAuthToken = (state: { auth: AuthState }) => state.auth.token;
export const selectIsAuthLoading = (state: { auth: AuthState }) => state.auth.isLoading;
