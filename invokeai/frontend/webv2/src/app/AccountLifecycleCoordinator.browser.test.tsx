import type * as identityModule from '@features/identity';

import { RouterProvider } from '@tanstack/react-router';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const identityErrors = vi.hoisted(() => ({
  AuthSessionUnavailableError: class AuthSessionUnavailableError extends Error {},
}));

const unavailableScreen = vi.hoisted(() => {
  let retry: (() => Promise<void> | void) | null = null;

  return {
    getRetry: () => retry,
    reset: () => {
      retry = null;
    },
    setRetry: (nextRetry: () => Promise<void> | void) => {
      retry = nextRetry;
    },
  };
});

type SessionSnapshot = {
  accountEpoch: number;
  multiuserEnabled: boolean;
  phase: 'unavailable' | 'ready';
  sessionExpired: boolean;
  setupRequired: boolean;
  strictPasswordChecking: boolean;
  user: { user_id: string } | null;
};

const sessionStore = vi.hoisted(() => {
  type Snapshot = {
    accountEpoch: number;
    multiuserEnabled: boolean;
    phase: 'unavailable' | 'ready';
    sessionExpired: boolean;
    setupRequired: boolean;
    strictPasswordChecking: boolean;
    user: { user_id: string } | null;
  };

  let snapshot: Snapshot = {
    accountEpoch: 0,
    multiuserEnabled: true,
    phase: 'ready',
    sessionExpired: false,
    setupRequired: false,
    strictPasswordChecking: true,
    user: { user_id: 'user-a' },
  };
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => snapshot,
    reset: () => {
      listeners.clear();
    },
    setSnapshot: (nextSnapshot: Snapshot) => {
      snapshot = nextSnapshot;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
});

vi.mock('@features/identity', async () => {
  const React = await import('react');
  // Modules outside this test's mock map (reached via other import specifiers)
  // still bind real identity exports — spread them so the mock only overrides
  // what the test drives instead of rotting on every new identity export.
  const actual = await vi.importActual<typeof identityModule>('@features/identity');

  return {
    ...actual,
    AuthSessionUnavailableError: identityErrors.AuthSessionUnavailableError,
    AuthUnavailableScreen: ({ onRetry }: { onRetry: () => Promise<void> | void }) => {
      unavailableScreen.setRetry(onRetry);

      return React.createElement('div', { 'data-testid': 'auth-unavailable-screen' }, 'Auth unavailable');
    },
    ensureReadyAuthSession: () => {
      const snapshot = sessionStore.getSnapshot();

      return snapshot.phase === 'ready'
        ? Promise.resolve(snapshot)
        : Promise.reject(new identityErrors.AuthSessionUnavailableError());
    },
    getCapabilities: () => ({
      canManageModels: true,
      canManageNodes: true,
      canManageUsers: true,
    }),
    LoginScreen: () => React.createElement('div', { 'data-testid': 'login-screen' }, 'Login'),
    SetupScreen: () => React.createElement('div', null, 'Setup'),
    useAuthSession: () =>
      React.useSyncExternalStore(sessionStore.subscribe, sessionStore.getSnapshot, sessionStore.getSnapshot),
  };
});

vi.mock('@workbench/components/WorkbenchSplashScreen', () => ({
  WorkbenchSplashScreen: () => null,
}));

vi.mock('@workbench/launchpad/Launchpad', () => ({
  Launchpad: () => null,
}));

vi.mock('@workbench/projects/session', () => ({
  peekOpenProjectIds: () => Promise.resolve(null),
}));

vi.mock('@workbench/settings/store', () => ({
  loadWorkbenchSettings: () => Promise.resolve(undefined),
}));

vi.mock('./SocketHubRuntime', async () => {
  const React = await import('react');

  return {
    SocketHubRuntime: () =>
      React.createElement('div', { 'data-testid': 'authenticated-runtime' }, 'Authenticated runtime'),
  };
});

vi.mock('./WorkbenchApp', () => ({
  WorkbenchApp: () => null,
}));

const AUTHENTICATED_PATHS = ['/', '/projects', '/models', '/nodes', '/users', '/app'] as const;

const signedInSession = (accountEpoch: number): SessionSnapshot => ({
  accountEpoch,
  multiuserEnabled: true,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: true,
  user: { user_id: 'user-a' },
});

const signedOutSession = (accountEpoch: number): SessionSnapshot => ({
  ...signedInSession(accountEpoch),
  sessionExpired: true,
  user: null,
});

const singleUserSession = (accountEpoch: number): SessionSnapshot => ({
  ...signedInSession(accountEpoch),
  multiuserEnabled: false,
  user: null,
});

const unavailableSession = (accountEpoch: number): SessionSnapshot => ({
  ...signedOutSession(accountEpoch),
  multiuserEnabled: false,
  phase: 'unavailable',
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  sessionStore.reset();
  unavailableScreen.reset();
});

describe('authenticated route account lifecycle', () => {
  it('never mounts authenticated runtime while auth mode is unavailable', async () => {
    sessionStore.setSnapshot(unavailableSession(1));
    const { router } = await import('./router');

    await router.load();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(<RouterProvider router={router} />);
    });

    expect(host.querySelector('[data-testid="authenticated-runtime"]')).toBeNull();
    expect(host.querySelector('[data-testid="login-screen"]')).toBeNull();
    expect(host.querySelector('[data-testid="auth-unavailable-screen"]')).not.toBeNull();
    const retry = unavailableScreen.getRetry();
    expect(retry).not.toBeNull();

    await act(async () => {
      sessionStore.setSnapshot(signedInSession(2));
      await retry?.();
    });

    await expect.poll(() => host?.querySelector('[data-testid="authenticated-runtime"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="auth-unavailable-screen"]')).toBeNull();
  });

  it('unmounts every authenticated route on multiuser sign-out and keeps every route mounted in single-user mode', async () => {
    sessionStore.setSnapshot(signedInSession(1));
    const { router } = await import('./router');

    await router.load();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(<RouterProvider router={router} />);
    });

    const initialPath = router.state.location.pathname;
    const initialRuntime = host.querySelector<HTMLElement>('[data-testid="authenticated-runtime"]');
    expect(initialRuntime).not.toBeNull();

    await act(() => {
      sessionStore.setSnapshot(signedInSession(2));
    });

    const remountedRuntime = host.querySelector<HTMLElement>('[data-testid="authenticated-runtime"]');
    expect(router.state.location.pathname).toBe(initialPath);
    expect(initialRuntime?.isConnected).toBe(false);
    expect(remountedRuntime).not.toBe(initialRuntime);
    expect(remountedRuntime).not.toBeNull();

    let accountEpoch = 2;

    for (const path of AUTHENTICATED_PATHS) {
      accountEpoch += 1;
      await act(async () => {
        sessionStore.setSnapshot(signedInSession(accountEpoch));
        await router.navigate({ replace: true, to: path });
      });
      await expect.poll(() => router.state.location.pathname).toBe(path);
      const authenticatedRuntime = host.querySelector<HTMLElement>('[data-testid="authenticated-runtime"]');
      expect(authenticatedRuntime).not.toBeNull();

      accountEpoch += 1;
      await act(() => {
        sessionStore.setSnapshot(signedOutSession(accountEpoch));
      });

      await expect.poll(() => router.state.location.pathname).toBe('/login');
      expect(authenticatedRuntime?.isConnected).toBe(false);
      expect(host.querySelector('[data-testid="authenticated-runtime"]')).toBeNull();
      expect(host.querySelector('[data-testid="login-screen"]')).not.toBeNull();
    }

    for (const path of AUTHENTICATED_PATHS) {
      accountEpoch += 1;
      const snapshot = singleUserSession(accountEpoch);
      await act(async () => {
        sessionStore.setSnapshot(snapshot);
        await router.navigate({ replace: true, to: path });
      });
      await expect.poll(() => router.state.location.pathname).toBe(path);
      const authenticatedRuntime = host.querySelector<HTMLElement>('[data-testid="authenticated-runtime"]');
      expect(authenticatedRuntime).not.toBeNull();

      await act(() => {
        sessionStore.setSnapshot({ ...snapshot });
      });

      expect(router.state.location.pathname).toBe(path);
      expect(host.querySelector('[data-testid="authenticated-runtime"]')).toBe(authenticatedRuntime);
      expect(host.querySelector('[data-testid="login-screen"]')).toBeNull();
    }
  });
});
