import type { ReactNode } from 'react';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useNavigate, useRouter } from '@tanstack/react-router';
import { createContext, use, useCallback } from 'react';

/** This UI port preserves dependency direction: Models cannot import Workbench. */
export interface ModelsUiAdapter {
  /** Whether this session may open the Model Manager; its route redirects everyone else away. */
  canManageModels: boolean;
  enableModelDescriptions: boolean;
  isProjectActive: (projectId: string) => boolean;
  managerProjectId: string | null;
}

const DEFAULT_MODELS_UI_ADAPTER: ModelsUiAdapter = {
  canManageModels: false,
  enableModelDescriptions: true,
  isProjectActive: () => false,
  managerProjectId: null,
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
