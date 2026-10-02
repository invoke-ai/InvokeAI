import { FontsRuntimeProvider } from '@features/fonts/react';
import {
  AuthSessionUnavailableError,
  AuthUnavailableScreen,
  ensureReadyAuthSession,
  getCapabilities,
  LoginScreen,
  SetupScreen,
  useAuthSession,
  type Capabilities,
} from '@features/identity';
import { ModelInstallRuntime } from '@features/models';
import { createLogger } from '@platform/logging/logger';
import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  ErrorComponent,
  type ErrorComponentProps,
  lazyRouteComponent,
  Navigate,
  Outlet,
  redirect,
  useRouter,
} from '@tanstack/react-router';
import { WorkbenchSplashScreen } from '@workbench/components/WorkbenchSplashScreen';
import { isLaunchpadIntentId } from '@workbench/launchpad/intents';
import { Launchpad } from '@workbench/launchpad/Launchpad';
import { ProjectFileOptionsProvider } from '@workbench/projects/components/ProjectFileOptionsProvider';
import { peekOpenProjectIds, type WorkbenchSearch } from '@workbench/projects/session';
import { loadWorkbenchSettings } from '@workbench/settings/store';
import { Fragment } from 'react';

import { SocketHubRuntime } from './SocketHubRuntime';

/**
 * The authenticated layout owns setup/login guards; lazy /app keeps editor code out of Launchpad. Hash history
 * supports the relative base without server deep-path fallback.
 */

/** Route render and loader failures reach the router's catch boundary; record them once there. */
const routerLogger = createLogger({ area: 'router', namespace: 'app' });

const RouterError = ({ error }: ErrorComponentProps) => {
  'use no memo';
  // Skip compiler memoization on this cold path; retry tests and build budgets guard it.
  const router = useRouter();

  return error instanceof AuthSessionUnavailableError ? (
    <AuthUnavailableScreen onRetry={router.invalidate} />
  ) : (
    <ErrorComponent error={error} />
  );
};

const rootRoute = createRootRoute({ component: Outlet, errorComponent: RouterError });

/**
 * Keep socket and install progress alive across authenticated routes; mount heavier feature listeners only where
 * needed.
 */
const AuthenticatedLayout = () => {
  'use no memo';
  // Account transitions are infrequent; provider composition does not need a memo cache.
  const session = useAuthSession();

  if (session.phase !== 'ready') {
    return null;
  }

  if (session.multiuserEnabled && session.user === null) {
    return <Navigate replace to="/login" />;
  }

  return (
    <Fragment key={session.accountEpoch}>
      <SocketHubRuntime />
      <ModelInstallRuntime />
      <FontsRuntimeProvider>
        <ProjectFileOptionsProvider>
          <Outlet />
        </ProjectFileOptionsProvider>
      </FontsRuntimeProvider>
    </Fragment>
  );
};

const authenticatedRoute = createRoute({
  beforeLoad: async () => {
    const session = await ensureReadyAuthSession();

    if (session.multiuserEnabled) {
      if (session.setupRequired) {
        throw redirect({ to: '/setup' });
      }

      if (session.user === null) {
        throw redirect({ to: '/login' });
      }
    }

    await loadWorkbenchSettings();
  },
  component: AuthenticatedLayout,
  getParentRoute: () => rootRoute,
  id: 'authenticated',
});

const requireLaunchpadCapability = async (capability: keyof Capabilities): Promise<void> => {
  const session = await ensureReadyAuthSession();

  if (!getCapabilities(session)[capability]) {
    throw redirect({ to: '/projects' });
  }
};

const launchpadRouteOptions = {
  component: Launchpad,
  getParentRoute: () => authenticatedRoute,
};

const homeRoute = createRoute({
  ...launchpadRouteOptions,
  path: '/',
});

const projectsHomeRoute = createRoute({
  ...launchpadRouteOptions,
  path: 'projects',
});

const modelsHomeRoute = createRoute({
  beforeLoad: () => requireLaunchpadCapability('canManageModels'),
  ...launchpadRouteOptions,
  path: 'models',
  validateSearch: (search: Record<string, unknown>): { project?: string } => ({
    project: typeof search.project === 'string' && search.project.length > 0 ? search.project : undefined,
  }),
});

const nodesHomeRoute = createRoute({
  beforeLoad: () => requireLaunchpadCapability('canManageNodes'),
  ...launchpadRouteOptions,
  path: 'nodes',
});

const usersHomeRoute = createRoute({
  beforeLoad: () => requireLaunchpadCapability('canManageUsers'),
  ...launchpadRouteOptions,
  path: 'users',
});

const fontsHomeRoute = createRoute({
  ...launchpadRouteOptions,
  path: 'fonts',
});

const validatePreferencesSearch = (search: Record<string, unknown>): { setting?: string } => ({
  setting: typeof search.setting === 'string' && search.setting.length > 0 ? search.setting : undefined,
});

const preferencesHomeRoute = createRoute({
  ...launchpadRouteOptions,
  path: 'preferences',
  validateSearch: validatePreferencesSearch,
});

const preferencesSectionRoute = createRoute({
  ...launchpadRouteOptions,
  path: 'preferences/$section',
  validateSearch: validatePreferencesSearch,
});

const workbenchRoute = createRoute({
  beforeLoad: async ({ cause, search }) => {
    // Preload the editor during the session peek; the module loader deduplicates the lazy-route import.
    void import('./WorkbenchApp');

    // Redirect a definitively empty session to Home unless ?project/?new is supplied. Unknown sessions fall
    // through. Check only entries: search changes can precede autosave, and hover preloads must not fetch the
    // session.
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
    intent: isLaunchpadIntentId(search.intent) ? search.intent : undefined,
    new: search.new === true || search.new === 'true' || search.new === 1 ? true : undefined,
    project: typeof search.project === 'string' && search.project.length > 0 ? search.project : undefined,
  }),
});

const loginRoute = createRoute({
  beforeLoad: async () => {
    const session = await ensureReadyAuthSession();

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
    const session = await ensureReadyAuthSession();

    if (!session.multiuserEnabled || !session.setupRequired) {
      throw redirect({ to: '/' });
    }
  },
  component: SetupScreen,
  getParentRoute: () => rootRoute,
  path: '/setup',
});

const RouterPending = () => <WorkbenchSplashScreen messageKey="splash.loadingApplication" />;

export const router = createRouter({
  defaultNotFoundComponent: () => <Navigate to="/" />,
  defaultOnCatch: (error, errorInfo) =>
    routerLogger.error({
      context: errorInfo?.componentStack ? { componentStack: errorInfo.componentStack } : undefined,
      error,
      message: 'Route rendering failed',
      name: 'app.route-failed',
    }),
  defaultPendingComponent: RouterPending,
  defaultPreload: 'intent',
  history: createHashHistory(),
  routeTree: rootRoute.addChildren([
    authenticatedRoute.addChildren([
      homeRoute,
      projectsHomeRoute,
      modelsHomeRoute,
      nodesHomeRoute,
      usersHomeRoute,
      fontsHomeRoute,
      preferencesHomeRoute,
      preferencesSectionRoute,
      workbenchRoute,
    ]),
    loginRoute,
    setupRoute,
  ]),
});

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
