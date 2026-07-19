import type { ReactNode } from 'react';

import { createContext, use } from 'react';

/**
 * Models' UI port. The context is a dependency-direction port (the feature
 * may not import workbench), not a test seam; no second adapter is expected.
 */
export interface ModelsUiAdapter {
  enableModelDescriptions: boolean;
  managerProjectId: string | null;
}

const DEFAULT_MODELS_UI_ADAPTER: ModelsUiAdapter = {
  enableModelDescriptions: true,
  managerProjectId: null,
};

const ModelsUiContext = createContext<ModelsUiAdapter>(DEFAULT_MODELS_UI_ADAPTER);

export const ModelsUiProvider = ({ adapter, children }: { adapter: ModelsUiAdapter; children: ReactNode }) => (
  <ModelsUiContext value={adapter}>{children}</ModelsUiContext>
);

export const useModelsUi = (): ModelsUiAdapter => use(ModelsUiContext);
