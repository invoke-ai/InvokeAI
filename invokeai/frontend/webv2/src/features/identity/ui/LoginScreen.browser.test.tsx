/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { AuthSession } from '@features/identity/session';

import { ChakraProvider } from '@chakra-ui/react';
import { createExternalStore } from '@platform/state/externalStore';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { LoginScreen, readLoginReturnPath } from './LoginScreen';

const identity = vi.hoisted(() => ({ loginWithCredentials: vi.fn() }));
const sessionStore = createExternalStore<AuthSession>({
  accountEpoch: 1,
  expiredUserId: null,
  multiuserEnabled: true,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  switchingAccount: false,
  user: null,
});

vi.mock('@features/identity/session', () => ({
  isLoginAttemptSupersededError: () => false,
  loginWithCredentials: identity.loginWithCredentials,
  useAuthSession: () => sessionStore.useSnapshot(),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const userFor = (userId: string) => ({
  created_at: '2026-01-01T00:00:00Z',
  display_name: null,
  email: `${userId}@example.com`,
  is_active: true,
  is_admin: false,
  last_login_at: null,
  updated_at: '2026-01-01T00:00:00Z',
  user_id: userId,
});

let host: HTMLDivElement;
let root: Root;

const render = async (path: string) => {
  const rootRoute = createRootRoute({ component: Outlet });
  const router = createRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
    routeTree: rootRoute.addChildren([
      createRoute({ getParentRoute: () => rootRoute, path: '/' }),
      createRoute({ getParentRoute: () => rootRoute, path: 'preferences/$section' }),
      createRoute({
        component: LoginScreen,
        getParentRoute: () => rootRoute,
        path: 'login',
        validateSearch: (search: Record<string, unknown>) => ({ returnTo: readLoginReturnPath(search.returnTo) }),
      }),
    ]),
  });
  await act(async () => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <RouterProvider router={router} />
        </I18nextProvider>
      </ChakraProvider>
    );
    await router.load();
  });
  return router;
};

const signInAs = async (userId: string) => {
  identity.loginWithCredentials.mockImplementationOnce(() => {
    sessionStore.patchSnapshot({ expiredUserId: null, sessionExpired: false, user: userFor(userId) });
    return Promise.resolve();
  });
  await userEvent.fill(page.getByRole('textbox', { name: i18n.t('users.email') }), `${userId}@example.com`);
  await userEvent.fill(page.getByLabelText(i18n.t('auth.password'), { exact: true }), 'password');
  await userEvent.click(page.getByRole('button', { exact: true, name: i18n.t('auth.signIn') }));
};

beforeEach(() => {
  identity.loginWithCredentials.mockReset();
  sessionStore.patchSnapshot({ expiredUserId: 'user-a', sessionExpired: true, user: null });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('login screen', () => {
  it('returns the account whose session expired to where it was, and says who resets passwords', async () => {
    const router = await render('/login?returnTo=%2Fpreferences%2Fbehavior');

    await expect.element(page.getByText(i18n.t('auth.sessionExpired'))).toBeVisible();
    await expect.element(page.getByText(i18n.t('auth.forgotPasswordHint'))).toBeVisible();
    await signInAs('user-a');

    await expect.poll(() => router.state.location.pathname).toBe('/preferences/behavior');
  });

  it('starts another account at Home, whatever the way back says', async () => {
    const router = await render('/login?returnTo=%2Fpreferences%2Fbehavior');

    await signInAs('user-b');

    await expect.poll(() => router.state.location.pathname).toBe('/');
  });

  it('returns there too when another tab signs the same account back in', async () => {
    const router = await render('/login?returnTo=%2Fpreferences%2Fbehavior');

    // This tab follows the other tab's sign-in; nothing was submitted here.
    await act(() =>
      sessionStore.patchSnapshot({ expiredUserId: null, sessionExpired: false, user: userFor('user-a') })
    );

    await expect.poll(() => router.state.location.pathname).toBe('/preferences/behavior');
    expect(identity.loginWithCredentials).not.toHaveBeenCalled();
  });

  it('accepts only in-app ways back', () => {
    expect(readLoginReturnPath('/app?project=p1')).toBe('/app?project=p1');
    for (const value of [
      '//evil.example',
      '/\\evil.example',
      '/\tevil.example',
      '/ evil.example',
      'https://evil.example',
      'javascript:alert(1)',
      '/login',
      '/setup',
      3,
    ]) {
      expect(readLoginReturnPath(value)).toBeUndefined();
    }
  });
});
