import type { ReactNode } from 'react';

import { useNavigate } from '@tanstack/react-router';
import { createContext, use, useCallback } from 'react';

/** This UI port preserves dependency direction: Models cannot import Workbench. */
export interface ModelsUiAdapter {
  /** Whether this session may open the Model Manager; its route redirects everyone else away. */
  canManageModels: boolean;
  enableModelDescriptions: boolean;
  managerProjectId: string | null;
}

const DEFAULT_MODELS_UI_ADAPTER: ModelsUiAdapter = {
  canManageModels: false,
  enableModelDescriptions: true,
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
  const { canManageModels, managerProjectId } = useModelsUi();
  const navigate = useNavigate();
  const open = useCallback(
    (modelKey: string) => {
      void import('./uiStore').then(({ openModelDetail }) => {
        openModelDetail(modelKey);
        void navigate({ search: { project: managerProjectId ?? undefined }, to: '/models' });
      });
    },
    [managerProjectId, navigate]
  );

  return canManageModels ? open : null;
};

/** Opens Add Models searching the starter catalog for `query`, so the user reviews it before installing. */
export const useOpenAddModelsSearch = (): ((query: string) => void) | null => {
  const { canManageModels, managerProjectId } = useModelsUi();
  const navigate = useNavigate();
  const open = useCallback(
    (query: string) => {
      void import('./uiStore').then(({ requestAddModelsSearch }) => {
        requestAddModelsSearch(query);
        void navigate({ search: { project: managerProjectId ?? undefined }, to: '/models' });
      });
    },
    [managerProjectId, navigate]
  );

  return canManageModels ? open : null;
};
