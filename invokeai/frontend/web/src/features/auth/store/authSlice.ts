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

/** The token's claims, or null when it is absent or not a readable JWT. */
const decodeTokenPayload = (token: string | null): Record<string, unknown> | null => {
  if (!token) {
    return null;
  }
  try {
    const encodedPayload = token.split('.')[1];
    if (!encodedPayload) {
      return null;
    }
    const normalizedPayload = encodedPayload.replace(/-/g, '+').replace(/_/g, '/');
    return JSON.parse(atob(normalizedPayload.padEnd(Math.ceil(normalizedPayload.length / 4) * 4, '=')));
  } catch {
    return null;
  }
};

const getTokenUserId = (token: string | null): string | null => {
  const payload = decodeTokenPayload(token);
  return typeof payload?.user_id === 'string' ? payload.user_id : null;
};

/**
 * The session a token belongs to, as opposed to the bytes it happens to be made of.
 *
 * The sliding-window middleware mints a replacement token on every mutating request, so a live
 * session's token changes bytes on a fixed cadence (throttled to once a minute by
 * `acceptRefreshedToken`) while the login behind it never changes. Anything that must be rebuilt
 * when the *session* changes — the socket, most notably — keys on this rather than on the token,
 * so a routine refresh does not tear down work that belongs to the same user.
 *
 * The revocation epoch is part of the identity, not incidental to it. A password change bumps
 * `token_epoch` on the user record, and the server force-disconnects every socket that
 * authenticated under the superseded epoch (`sockets.py`, `_handle_user_access_changed`). A
 * server-initiated disconnect is terminal for socket.io — the client sets `skipReconnect` and
 * never retries (`socket.io-client`, `Socket.ondisconnect` -> `Manager._close`) — so the only way
 * back is to build a new socket, and the replacement token the server hands out carries the new
 * epoch. Keying on `user_id` alone would leave that socket dead until a full page reload, with
 * `$isConnected` stuck false and Invoke disabled with it.
 *
 * A token carrying no user id falls back to its own bytes: with no identity to compare, byte
 * equality is the only safe answer, and consumers keep their pre-existing behaviour.
 */
export const getTokenSessionKey = (token: string | null): string | null => {
  if (!token) {
    return null;
  }
  const payload = decodeTokenPayload(token);
  if (typeof payload?.user_id !== 'string') {
    return token;
  }
  // Absent on tokens minted before the claim existed; the server reads those as epoch 0 too.
  const epoch = typeof payload.token_epoch === 'number' ? payload.token_epoch : 0;
  return `${payload.user_id}:${epoch}`;
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
    /**
     * Discards leftover credentials without the account-change semantics of `logout`. The one
     * caller is ProtectedRoute's multiuser-disabled branch: the server has switched to
     * single-user mode and a token from the multiuser era is still lying around. That is a mode
     * switch, not a hand-off to another person — the same human keeps the machine — so the
     * workspace slices, which reset on `logout` to keep one account's canvas and workflow away
     * from the next, must not fire. They would not merely flash empty: in single-user mode the
     * unauthenticated persist is accepted, so the wipe would overwrite the stored workspace for
     * good. The store listener still clears the api cache on this action, since what is cached
     * was fetched under multiuser visibility scoping.
     */
    staleCredentialsDiscarded: (state) => {
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
  staleCredentialsDiscarded,
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
export const selectAuthSessionKey = (state: { auth: AuthState }) => getTokenSessionKey(state.auth.token);
export const selectIsAuthLoading = (state: { auth: AuthState }) => state.auth.isLoading;
export const selectSessionExpired = (state: { auth: AuthState }) => state.auth.sessionExpired;
