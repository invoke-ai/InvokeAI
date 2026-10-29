import { Flex, Spinner, Text } from '@chakra-ui/react';
import { FontsRuntimeProvider } from '@features/fonts/react';
import {
  AuthSessionUnavailableError,
  AuthUnavailableScreen,
  ensureReadyAuthSession,
  getAuthSession,
  getCapabilities,
  LoginScreen,
  readLoginReturnPath,
  SessionExpiryNotice,
  SetupScreen,
  subscribeAuthSession,
  useAuthSession,
  useCapabilities,
  type Capabilities,
} from '@features/identity';
import {
  ModelInstallRuntime,
  ModelsUiProvider,
  type ModelsUiAdapter,
  type SchedulerPolicySource,
  type SchedulerPolicyState,
} from '@features/models';
import { createLogger } from '@platform/logging/logger';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { createListenerChannel } from '@platform/state/externalStoreCore';
import { toaster } from '@platform/ui/toaster';
import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  type ErrorComponentProps,
  lazyRouteComponent,
  Navigate,
  Outlet,
  redirect,
  useRouter,
  useRouterState,
} from '@tanstack/react-router';
import { WorkbenchSplashScreen } from '@workbench/components/WorkbenchSplashScreen';
import { isLaunchpadIntentId } from '@workbench/launchpad/intents';
import { Launchpad } from '@workbench/launchpad/Launchpad';
import { StayOnLaunchpadButton } from '@workbench/launchpad/StayOnLaunchpadButton';
import { listProjects } from '@workbench/projects/api';
import { ProjectFileOptionsProvider } from '@workbench/projects/components/ProjectFileOptionsProvider';
import { getProjectLibrary, seedProjectLibrary } from '@workbench/projects/library';
import { peekOpenProjectIds, type WorkbenchSearch } from '@workbench/projects/session';
import {
  loadWorkbenchSettings,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from '@workbench/settings/store';
import { loadAppVersion } from '@workbench/shell/whatsNewStore';
import i18next from 'i18next';
import { Fragment, useCallback, useMemo, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type * as SchedulerRules from './modelSchedulerPolicyReader';

import { RouterErrorScreen } from './RouterErrorScreen';
import { SocketHubRuntime } from './SocketHubRuntime';
import { createStartupRedirects } from './startupRedirect';

/**
 * The authenticated layout owns setup/login guards; lazy /app keeps editor code out of Launchpad. Hash history
 * supports the relative base without server deep-path fallback.
 */

/** Route render and loader failures reach the router's catch boundary; record them once there. */
const routerLogger = createLogger({ area: 'router', namespace: 'app' });

const RouterError = ({ error, reset }: ErrorComponentProps) => {
  'use no memo';
  // Skip compiler memoization on this cold path; retry tests and build budgets guard it.
  const router = useRouter();
  // Home has its own loaders, so leaving the failed route and clearing the boundary is a real way back.
  const goHome = useCallback(() => void router.navigate({ to: '/' }).then(reset), [reset, router]);

  return error instanceof AuthSessionUnavailableError ? (
    <AuthUnavailableScreen onRetry={router.invalidate} />
  ) : (
    <RouterErrorScreen error={error} onGoHome={goHome} />
  );
};

const rootRoute = createRootRoute({ component: Outlet, errorComponent: RouterError });

const SCHEDULER_RULES_LOADING: SchedulerPolicyState = { status: 'loading' };
const SCHEDULER_RULES_UNAVAILABLE: SchedulerPolicyState = { status: 'unavailable' };
let schedulerPolicyState: SchedulerPolicyState = SCHEDULER_RULES_LOADING;
let schedulerRules: Promise<typeof SchedulerRules> | null = null;
const schedulerPolicyChannel = createListenerChannel();

const publishSchedulerPolicy = (next: SchedulerPolicyState): void => {
  schedulerPolicyState = next;
  schedulerPolicyChannel.notify();
};

/**
 * Generate's scheduler rules for the Model Manager's defaults. They and the architecture table they read live in
 * Generation, which the Launchpad does not load at startup, so they arrive with the first Default settings section.
 */
const schedulerPolicySource: SchedulerPolicySource = {
  getSnapshot: () => schedulerPolicyState,
  load: () => {
    if (!schedulerRules) {
      if (schedulerPolicyState.status === 'unavailable') {
        publishSchedulerPolicy(SCHEDULER_RULES_LOADING);
      }
      schedulerRules = import('./modelSchedulerPolicyReader').then((rules) => {
        rules.startSchedulerPolicyReader(publishSchedulerPolicy);
        return rules;
      });
    }
    // Asked on every load: it retries a failed table, and an account change clears the table.
    schedulerRules
      .then((rules) => rules.ensureSchedulerRulesLoaded())
      .catch(() => {
        // The rules' chunk failed to load; the next load (Retry, or another section) imports it again.
        schedulerRules = null;
        publishSchedulerPolicy(SCHEDULER_RULES_UNAVAILABLE);
      });
  },
  subscribe: schedulerPolicyChannel.subscribe,
};

const setInstallInPlace = (installInPlace: boolean) => void patchWorkbenchPreferences({ installInPlace });
const NO_ACTIVE_PROJECT = () => false;

/**
 * The account's Models port for every authenticated route: the Model Manager renders on the Launchpad, and the editor
 * narrows this with its active project. Composed here because a new module would enter the Launchpad's startup graph.
 */
export const AccountModelsUiProvider = ({ children }: { children: ReactNode }) => {
  const enableModelDescriptions = useWorkbenchPreferenceSelector((value) => value.enableModelDescriptions);
  const installInPlace = useWorkbenchPreferenceSelector((value) => value.installInPlace);
  const { canManageModels } = useCapabilities();
  const adapter = useMemo<ModelsUiAdapter>(
    () => ({
      canManageModels,
      enableModelDescriptions,
      installInPlace,
      isProjectActive: NO_ACTIVE_PROJECT,
      managerProjectId: null,
      schedulerPolicy: schedulerPolicySource,
      setInstallInPlace,
    }),
    [canManageModels, enableModelDescriptions, installInPlace]
  );

  return <ModelsUiProvider adapter={adapter}>{children}</ModelsUiProvider>;
};

const NO_SEARCH = {};
const returnToSearch = (returnTo: string) => ({ returnTo });
const isLoginHref = (href: string): boolean => href === '/login' || href.startsWith('/login?');

/** While the session resolves or follows another tab to a different account: say so, never a blank page. */
const AccountPending = ({ messageKey }: { messageKey: string }) => {
  const { t } = useTranslation();

  return (
    <Flex align="center" bg="bg" color="fg.muted" gap="2" justify="center" minH="100dvh">
      <Spinner />
      <Text fontSize="md" role="status">
        {t(messageKey)}
      </Text>
    </Flex>
  );
};

/**
 * Sends a signed-out session to the login screen; an expired one carries the way back (`returnTo`). Only this child
 * follows the location, so navigation never re-renders the layout's runtime providers.
 */
const SignedOutRedirect = ({ sessionExpired }: { sessionExpired: boolean }) => {
  const href = useRouterState({ select: (state) => state.location.href });
  const loginReturnTo = useRouterState({
    select: (state) =>
      isLoginHref(state.location.href) ? (state.location.search as { returnTo?: string }).returnTo : undefined,
  });
  // It can render again on the way to the login screen; it keeps the way back it already chose.
  const returnTo = isLoginHref(href) ? loginReturnTo : sessionExpired ? href : undefined;

  return <Navigate replace search={returnTo === undefined ? NO_SEARCH : returnToSearch(returnTo)} to="/login" />;
};

/**
 * Keep socket and install progress alive across authenticated routes; mount heavier feature listeners only where
 * needed. A sign-in by the expired account, here or followed from another tab, returns to `returnTo`.
 */
const AuthenticatedLayout = () => {
  'use no memo';
  // Account transitions are infrequent; provider composition does not need a memo cache.
  const session = useAuthSession();

  if (session.phase !== 'ready') {
    return <AccountPending messageKey="splash.loadingApplication" />;
  }

  if (session.switchingAccount) {
    return <AccountPending messageKey="auth.switchingAccount" />;
  }

  if (session.multiuserEnabled && session.user === null) {
    return <SignedOutRedirect sessionExpired={session.sessionExpired} />;
  }

  return (
    <>
      {session.multiuserEnabled ? <SessionExpiryNotice /> : null}
      <Fragment key={session.accountEpoch}>
        <SocketHubRuntime />
        <ModelInstallRuntime />
        <FontsRuntimeProvider>
          <ProjectFileOptionsProvider>
            <AccountModelsUiProvider>
              <Outlet />
            </AccountModelsUiProvider>
          </ProjectFileOptionsProvider>
        </FontsRuntimeProvider>
      </Fragment>
    </>
  );
};

/**
 * Captures the initial load before the router's history starts. Only accounts that chose the most recent project pay
 * for the decision: a session peek and, for an empty session, a project listing without waiting for covers.
 */
const startupRedirects = createStartupRedirects(
  {
    hash: window.location.hash,
    navigationType: (performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined)?.type,
  },
  {
    loadProjectLibrary: async () => {
      const owner = captureAccountScope();

      try {
        seedProjectLibrary(await listProjects(owner.signal), owner);
      } catch {
        // The library stays unloaded, so the launch stays on the Launchpad, whose Home lists projects again.
      }

      return getProjectLibrary();
    },
    peekOpenProjectIds,
    preloadEditor: () => void import('./WorkbenchApp'),
    reportSkippedLaunch: () =>
      toaster.create({
        description: i18next.t('splash.launchSkippedDescription'),
        title: i18next.t('splash.launchSkippedTitle'),
        type: 'info',
      }),
  }
);

const authenticatedRoute = createRoute({
  beforeLoad: async ({ location, preload }) => {
    const session = await ensureReadyAuthSession();

    if (session.multiuserEnabled) {
      if (session.setupRequired) {
        throw redirect({ to: '/setup' });
      }

      if (session.user === null) {
        throw redirect({ search: session.sessionExpired ? { returnTo: location.href } : {}, to: '/login' });
      }
    }

    // What's New compares this with the account's last-seen version; it never holds up the route.
    void loadAppVersion();
    const preferences = await loadWorkbenchSettings();

    const startup = await startupRedirects.take(
      { pathname: location.pathname, preload },
      preferences.startupDestination
    );

    if (startup) {
      throw redirect(startup);
    }
  },
  component: AuthenticatedLayout,
  getParentRoute: () => rootRoute,
  id: 'authenticated',
});

/** A page the account may not use opens Projects instead, and says why; a hover preload says nothing. */
const requireLaunchpadCapability = async (
  capability: keyof Capabilities,
  sectionKey: string,
  preload: boolean
): Promise<void> => {
  const session = await ensureReadyAuthSession();

  if (!getCapabilities(session)[capability]) {
    if (!preload) {
      toaster.create({
        description: i18next.t('launchpad.redirects.noAccessDescription'),
        id: `launchpad-no-access:${sectionKey}`,
        title: i18next.t('launchpad.redirects.noAccess', { section: i18next.t(sectionKey) }),
        type: 'info',
      });
    }
    throw redirect({ replace: true, to: '/projects' });
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
  beforeLoad: ({ preload }) => requireLaunchpadCapability('canManageModels', 'launchpad.sections.models', preload),
  ...launchpadRouteOptions,
  path: 'models',
  validateSearch: (search: Record<string, unknown>): { project?: string } => ({
    project: typeof search.project === 'string' && search.project.length > 0 ? search.project : undefined,
  }),
});

const nodesHomeRoute = createRoute({
  beforeLoad: ({ preload }) => requireLaunchpadCapability('canManageNodes', 'launchpad.sections.nodes', preload),
  ...launchpadRouteOptions,
  path: 'nodes',
});

const usersHomeRoute = createRoute({
  beforeLoad: ({ preload }) => requireLaunchpadCapability('canManageUsers', 'launchpad.sections.users', preload),
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

/** An address no page answers opens Home, and says so once; a hover preload says nothing. */
const notFoundRoute = createRoute({
  beforeLoad: ({ preload }) => {
    if (!preload) {
      toaster.create({
        description: i18next.t('launchpad.redirects.notFoundDescription'),
        id: 'launchpad-not-found',
        title: i18next.t('launchpad.redirects.notFound'),
        type: 'info',
      });
    }
    throw redirect({ replace: true, to: '/' });
  },
  getParentRoute: () => authenticatedRoute,
  path: '$',
});

const workbenchRoute = createRoute({
  beforeLoad: async ({ cause, search }) => {
    // Preload the editor during the session peek; the module loader deduplicates the lazy-route import.
    void import('./WorkbenchApp');

    // Redirect a definitively empty session to Home unless ?project/?new is supplied. Unknown sessions fall
    // through. Check only entries: search changes can precede autosave, and hover preloads must not fetch the
    // session. A launch redirect that just found open projects hands over its proof instead of a second peek.
    if (cause !== 'enter' || search.project || search.new || startupRedirects.takeOpenSessionProof()) {
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
  validateSearch: (search: Record<string, unknown>): { returnTo?: string } => ({
    returnTo: readLoginReturnPath(search.returnTo),
  }),
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

const STAY_ON_LAUNCHPAD = <StayOnLaunchpadButton />;
const RouterPending = () => <WorkbenchSplashScreen action={STAY_ON_LAUNCHPAD} messageKey="splash.loadingApplication" />;

export const router = createRouter({
  // Unknown addresses are the splat route's (with its notice); this answers only a `notFound()` a route throws.
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
      notFoundRoute,
    ]),
    loginRoute,
    setupRoute,
  ]),
});

/**
 * Following another tab's sign-in to a different account lands on Home, as signing in here would: the page this tab
 * showed belonged to the previous account (its project, or a manager the new account may not use).
 */
let wasSwitchingAccount = false;
const unsubscribeAccountSwitchLanding = subscribeAuthSession(() => {
  const { switchingAccount, user } = getAuthSession();

  if (switchingAccount) {
    wasSwitchingAccount = true;
    return;
  }

  if (wasSwitchingAccount) {
    wasSwitchingAccount = false;
    if (user !== null) {
      void router.navigate({ replace: true, to: '/' });
    }
  }
});
// A hot update of this module subscribes again; the replaced module's listener must go with it.
import.meta.hot?.dispose(unsubscribeAccountSwitchLanding);

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
