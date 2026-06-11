import { Center, Spinner } from '@chakra-ui/react';
import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Navigate,
  Outlet,
  redirect,
} from '@tanstack/react-router';

import { ensureAuthSession } from './workbench/auth/session';
import { LoginScreen } from './workbench/auth/components/LoginScreen';
import { SetupScreen } from './workbench/auth/components/SetupScreen';
import { WorkbenchApp } from './WorkbenchApp';

/**
 * Code-based route tree. Every guard starts from `ensureAuthSession()`, which
 * resolves the backend's multi-user status once per app load:
 *
 * - multi-user off → the workbench renders directly and /login + /setup bounce home
 * - setup required → everything funnels to /setup
 * - signed out     → the workbench redirects to /login
 *
 * Hash history is used because the bundle is served with a relative base
 * (`./`), so the app cannot rely on server-side fallback for deep paths.
 */

const rootRoute = createRootRoute({ component: Outlet });

const workbenchRoute = createRoute({
  beforeLoad: async () => {
    const session = await ensureAuthSession();

    if (!session.multiuserEnabled) {
      return;
    }

    if (session.setupRequired) {
      throw redirect({ to: '/setup' });
    }

    if (session.user === null) {
      throw redirect({ to: '/login' });
    }
  },
  component: WorkbenchApp,
  getParentRoute: () => rootRoute,
  path: '/',
});

const loginRoute = createRoute({
  beforeLoad: async () => {
    const session = await ensureAuthSession();

    if (session.multiuserEnabled && session.setupRequired) {
      throw redirect({ to: '/setup' });
    }

    if (!session.multiuserEnabled || session.user !== null) {
      throw redirect({ to: '/' });
    }
  },
  component: LoginScreen,
  getParentRoute: () => rootRoute,
  path: '/login',
});

const setupRoute = createRoute({
  beforeLoad: async () => {
    const session = await ensureAuthSession();

    if (!session.multiuserEnabled || !session.setupRequired) {
      throw redirect({ to: '/' });
    }
  },
  component: SetupScreen,
  getParentRoute: () => rootRoute,
  path: '/setup',
});

const RouterPending = () => (
  <Center bg="bg.shell" minH="100dvh">
    <Spinner color="fg.muted" size="sm" />
  </Center>
);

export const router = createRouter({
  defaultNotFoundComponent: () => <Navigate to="/" />,
  defaultPendingComponent: RouterPending,
  history: createHashHistory(),
  routeTree: rootRoute.addChildren([workbenchRoute, loginRoute, setupRoute]),
});

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
