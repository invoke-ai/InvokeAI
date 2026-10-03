import type { ModelsUiAdapter } from '@features/models';
import type { ReactNode } from 'react';

import { useCapabilities } from '@features/identity';
import { ModelsUiProvider } from '@features/models';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useActiveProjectId, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

export const ModelsUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const enableModelDescriptions = useWorkbenchPreferenceSelector((value) => value.enableModelDescriptions);
  const managerProjectId = useActiveProjectId();
  const { isActiveProject } = useWorkbenchQueries();
  const { canManageModels } = useCapabilities();
  const adapter = useMemo<ModelsUiAdapter>(
    () => ({ canManageModels, enableModelDescriptions, isProjectActive: isActiveProject, managerProjectId }),
    [canManageModels, enableModelDescriptions, isActiveProject, managerProjectId]
  );

  return <ModelsUiProvider adapter={adapter}>{children}</ModelsUiProvider>;
};
