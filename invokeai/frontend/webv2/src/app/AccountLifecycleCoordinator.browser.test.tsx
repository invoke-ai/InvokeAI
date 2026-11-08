import type * as identityModule from '@features/identity';
import type * as settingsStoreModule from '@workbench/settings/store';

import { ChakraProvider } from '@chakra-ui/react';
import { RouterProvider } from '@tanstack/react-router';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const identityErrors = vi.hoisted(() => ({
  AuthSessionUnavailableError: class AuthSessionUnavailableError extends Error {},
}));

const capabilities = vi.hoisted(() => ({ canManageModels: true, canManageNodes: true, canManageUsers: true }));

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
  switchingAccount?: boolean;
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
    switchingAccount?: boolean;
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
  // Preserve real identity exports; override only those driven by this test.
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
    getCapabilities: () => ({ ...capabilities }),
    LoginScreen: () => React.createElement('div', { 'data-testid': 'login-screen' }, 'Login'),
    SessionExpiryNotice: () => null,
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

vi.mock('@workbench/projects/session', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  peekOpenProjectIds: () => Promise.resolve(null),
}));

vi.mock('@workbench/settings/store', async (importOriginal) => {
  const settingsStore = await importOriginal<typeof settingsStoreModule>();

  return { ...settingsStore, loadWorkbenchSettings: () => Promise.resolve(settingsStore.DEFAULT_PREFERENCES) };
});

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
  capabilities.canManageModels = true;
  vi.restoreAllMocks();
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
      root?.render(
        <ChakraProvider value={system}>
          <RouterProvider router={router} />
        </ChakraProvider>
      );
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

    // Let RouterProvider load: a reused router's retained transition hook would wait forever for a pre-render
    // commit.
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <RouterProvider router={router} />
        </ChakraProvider>
      );
    });

    await expect.poll(() => host?.querySelector('[data-testid="authenticated-runtime"]')).not.toBeNull();
    const initialPath = router.state.location.pathname;
    const initialRuntime = host.querySelector<HTMLElement>('[data-testid="authenticated-runtime"]');

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
      // An expired session keeps the way back through sign-in.
      expect(router.state.location.search).toEqual({ returnTo: path });
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

  it('shows the switch, not the sign-in screen, while following another account, and a sign-out keeps no way back', async () => {
    sessionStore.setSnapshot(signedInSession(1));
    const { router } = await import('./router');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <RouterProvider router={router} />
        </ChakraProvider>
      );
    });
    await act(() => router.navigate({ replace: true, to: '/projects' }));
    await expect.poll(() => host?.querySelector('[data-testid="authenticated-runtime"]')).not.toBeNull();

    await act(() => {
      sessionStore.setSnapshot({ ...signedOutSession(2), sessionExpired: false, switchingAccount: true });
    });
    expect(host.querySelector('[role="status"]')?.textContent).toBe('auth.switchingAccount');
    expect(router.state.location.pathname).toBe('/projects');

    await act(() => {
      sessionStore.setSnapshot({ ...signedOutSession(3), sessionExpired: false });
    });
    await expect.poll(() => router.state.location.pathname).toBe('/login');
    expect(router.state.location.search).toEqual({});
  });

  it('opens Home for an unknown address and Projects for a page the account may not use, saying so once', async () => {
    const { toaster } = await import('@platform/ui/toaster');
    const create = vi.spyOn(toaster, 'create');
    sessionStore.setSnapshot(signedInSession(1));
    const { router } = await import('./router');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <RouterProvider router={router} />
        </ChakraProvider>
      );
    });
    await act(() => router.navigate({ replace: true, to: '/projects' }));

    // A hover preload decides nothing and says nothing.
    await act(() => router.preloadRoute({ to: '/no-such-page' as '/' }));
    capabilities.canManageModels = false;
    await act(() => router.preloadRoute({ to: '/models' }));
    expect(create).not.toHaveBeenCalled();

    await act(() => router.navigate({ to: '/no-such-page' as '/' }));
    await expect.poll(() => router.state.location.pathname).toBe('/');
    expect(create).toHaveBeenCalledWith(expect.objectContaining({ id: 'launchpad-not-found' }));

    await act(() => router.navigate({ to: '/models' }));
    await expect.poll(() => router.state.location.pathname).toBe('/projects');
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'launchpad-no-access:launchpad.sections.models' })
    );
    expect(create).toHaveBeenCalledTimes(2);
  });
});
