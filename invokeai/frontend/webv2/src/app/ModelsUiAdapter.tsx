import type { ModelsUiAdapter } from '@features/models';
import type { ReactNode } from 'react';

import { ModelsUiProvider } from '@features/models';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

/**
 * Production binding of Models' UI port: supplies Workbench preferences and
 * the active project. No second adapter is expected.
 */
export const ModelsUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const enableModelDescriptions = useWorkbenchPreferenceSelector((value) => value.enableModelDescriptions);
  const managerProjectId = useActiveProjectId();
  const adapter = useMemo<ModelsUiAdapter>(
    () => ({ enableModelDescriptions, managerProjectId }),
    [enableModelDescriptions, managerProjectId]
  );

  return <ModelsUiProvider adapter={adapter}>{children}</ModelsUiProvider>;
};
