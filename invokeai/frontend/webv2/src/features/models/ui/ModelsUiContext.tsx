import type { ModelConfig } from '@features/models/core/types';
import type { ReactNode } from 'react';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useNavigate, useRouter } from '@tanstack/react-router';
import { createContext, use, useCallback } from 'react';

export interface SchedulerChoice {
  label: string;
  value: string;
}

/** Generate's scheduler rules for one main model, so its defaults offer exactly what Generate runs. */
export interface ModelSchedulerPolicy {
  /** False where the base's graph ignores the scheduler; Generate hides the control there. */
  appliesToGraph: boolean;
  /** The scheduler Generate starts this base on. */
  defaultValue: string;
  /** Generate's choices for the base, led by `currentValue` when it is outside them. */
  options: readonly SchedulerChoice[];
}

/** The main model's architecture; Generate's scheduler rules key on these alone. */
export type SchedulerPolicyModel = Pick<ModelConfig, 'base' | 'format' | 'variant'>;

export type SchedulerPolicyReader = (model: SchedulerPolicyModel, currentValue: string | null) => ModelSchedulerPolicy;

/**
 * Generate's scheduler rules: `loading` until the rules and the architecture table they read arrive, `unavailable`
 * when either failed, and `ready` with a reader that is replaced whenever the table changes.
 */
export type SchedulerPolicyState =
  | { status: 'loading' }
  | { status: 'unavailable' }
  | { read: SchedulerPolicyReader; status: 'ready' };

/** Generate's scheduler rules, loaded on first use so the Model Manager carries no Generation code at startup. */
export interface SchedulerPolicySource {
  getSnapshot: () => SchedulerPolicyState;
  /** Starts loading the rules and the table, or retries after a failure; joins a load already running. */
  load: () => void;
  subscribe: (listener: () => void) => () => void;
}

const SCHEDULER_POLICY_UNAVAILABLE: SchedulerPolicyState = { status: 'unavailable' };
/** For surfaces composed without Generation, such as isolated tests: the rules never arrive. */
export const NO_SCHEDULER_POLICY: SchedulerPolicySource = {
  getSnapshot: () => SCHEDULER_POLICY_UNAVAILABLE,
  load: () => undefined,
  subscribe: () => () => undefined,
};

/** This UI port preserves dependency direction: Models cannot import Workbench or Generation. */
export interface ModelsUiAdapter {
  /** Whether this session may open the Model Manager; its route redirects everyone else away. */
  canManageModels: boolean;
  enableModelDescriptions: boolean;
  /** Add Models' "Install in place" choice, an account preference. */
  installInPlace: boolean;
  isProjectActive: (projectId: string) => boolean;
  managerProjectId: string | null;
  schedulerPolicy: SchedulerPolicySource;
  setInstallInPlace: (installInPlace: boolean) => void;
}

const DEFAULT_MODELS_UI_ADAPTER: ModelsUiAdapter = {
  canManageModels: false,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => false,
  managerProjectId: null,
  schedulerPolicy: NO_SCHEDULER_POLICY,
  setInstallInPlace: () => undefined,
};

const ModelsUiContext = createContext<ModelsUiAdapter>(DEFAULT_MODELS_UI_ADAPTER);

export const ModelsUiProvider = ({ adapter, children }: { adapter: ModelsUiAdapter; children: ReactNode }) => (
  <ModelsUiContext value={adapter}>{children}</ModelsUiContext>
);

export const useModelsUi = (): ModelsUiAdapter => use(ModelsUiContext);

/**
 * Opens the Model Manager on a model's details, returning to the manager's project like its other entry points; null
 * when this session may not manage models. The manager's UI store loads on first use so editor boot does not carry it.
 */
export const useOpenModelInManager = (): ((modelKey: string) => void) | null => {
  const { canManageModels, isProjectActive, managerProjectId } = useModelsUi();
  const navigate = useNavigate();
  const router = useRouter();
  const open = useCallback(
    (modelKey: string) => {
      const owner = captureAccountScope();
      const location = router.state.location;
      void import('./uiStore').then(({ openModelDetail }) => {
        if (
          !isAccountScopeCurrent(owner) ||
          router.state.location !== location ||
          (managerProjectId !== null && !isProjectActive(managerProjectId))
        ) {
          return;
        }
        openModelDetail(modelKey);
        void navigate({ search: { project: managerProjectId ?? undefined }, to: '/models' });
      });
    },
    [isProjectActive, managerProjectId, navigate, router]
  );

  return canManageModels ? open : null;
};

/** Opens Add Models searching the starter catalog for `query`, so the user reviews it before installing. */
export const useOpenAddModelsSearch = (): ((query: string) => void) | null => {
  const { canManageModels, isProjectActive, managerProjectId } = useModelsUi();
  const navigate = useNavigate();
  const router = useRouter();
  const open = useCallback(
    (query: string) => {
      const owner = captureAccountScope();
      const location = router.state.location;
      void import('./uiStore').then(({ requestAddModelsSearch }) => {
        if (
          !isAccountScopeCurrent(owner) ||
          router.state.location !== location ||
          (managerProjectId !== null && !isProjectActive(managerProjectId))
        ) {
          return;
        }
        requestAddModelsSearch(query);
        void navigate({ search: { project: managerProjectId ?? undefined }, to: '/models' });
      });
    },
    [isProjectActive, managerProjectId, navigate, router]
  );

  return canManageModels ? open : null;
};
