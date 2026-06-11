import { Center, Spinner } from '@chakra-ui/react';
import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  Navigate,
  Outlet,
  redirect,
} from '@tanstack/react-router';

import { ensureAuthSession } from './workbench/auth/session';
import { LoginScreen } from './workbench/auth/components/LoginScreen';
import { SetupScreen } from './workbench/auth/components/SetupScreen';
import { HomeScreen } from './workbench/home/HomeScreen';
import { peekOpenProjectIds, type WorkbenchSearch } from './workbench/projects/session';

/**
 * Code-based route tree. The authenticated layout route owns the auth guard,
 * which resolves the backend's multi-user status once per app load:
 *
 * - multi-user off → authenticated routes render directly; /login + /setup bounce home
 * - setup required → everything funnels to /setup
 * - signed out     → authenticated routes redirect to /login
 *
 * Under it, `/` is the Home surface (project library, profile, resources) and
 * `/app` is the editor. The editor bundle — canvas, workflow, widgets — is
 * code-split behind `lazyRouteComponent`, so Home stays light; `defaultPreload:
 * 'intent'` starts fetching that chunk the moment a project card is hovered.
 *
 * Hash history is used because the bundle is served with a relative base
 * (`./`), so the app cannot rely on server-side fallback for deep paths.
 */

const rootRoute = createRootRoute({ component: Outlet });

const authenticatedRoute = createRoute({
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
  component: Outlet,
  getParentRoute: () => rootRoute,
  id: 'authenticated',
});

const homeRoute = createRoute({
  component: HomeScreen,
  getParentRoute: () => authenticatedRoute,
  path: '/',
});

const workbenchRoute = createRoute({
  beforeLoad: async ({ cause, search }) => {
    // Photoshop semantics: the editor without documents is Home. A definite
    // empty session redirects unless the URL explicitly asks for a project or
    // a fresh draft; an unknowable session (first run, legacy blob, backend
    // unreachable) falls through and the editor boots from what it can find.
    //
    // Only actual entries are checked: search-only changes ('stay', e.g. the
    // session controller stripping ?new before the draft has autosaved) must
    // not re-evaluate a blob that is still catching up, and hover preloads
    // should not hit the session endpoint at all.
    if (cause !== 'enter' || search.project || search.new) {
      return;
    }

    const openProjectIds = await peekOpenProjectIds();

    if (openProjectIds !== null && openProjectIds.length === 0) {
      throw redirect({ to: '/' });
    }
  },
  component: lazyRouteComponent(() => import('./WorkbenchApp'), 'WorkbenchApp'),
  getParentRoute: () => authenticatedRoute,
  path: '/app',
  validateSearch: (search: Record<string, unknown>): WorkbenchSearch => ({
    new: search.new === true || search.new === 'true' || search.new === 1 ? true : undefined,
    project: typeof search.project === 'string' && search.project.length > 0 ? search.project : undefined,
  }),
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
  defaultPreload: 'intent',
  history: createHashHistory(),
  routeTree: rootRoute.addChildren([
    authenticatedRoute.addChildren([homeRoute, workbenchRoute]),
    loginRoute,
    setupRoute,
  ]),
});

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
