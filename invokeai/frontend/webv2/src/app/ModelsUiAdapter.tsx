import type { ModelsUiAdapter } from '@features/models';
import type { ReactNode } from 'react';

import { ModelsUiProvider, useModelsUi } from '@features/models';
import { useActiveProjectId, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

/** Narrows the account's Models port (see the authenticated layout) so pickers open the manager for this project. */
export const ModelsUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const account = useModelsUi();
  const managerProjectId = useActiveProjectId();
  const { isActiveProject } = useWorkbenchQueries();
  const adapter = useMemo<ModelsUiAdapter>(
    () => ({ ...account, isProjectActive: isActiveProject, managerProjectId }),
    [account, isActiveProject, managerProjectId]
  );

  return <ModelsUiProvider adapter={adapter}>{children}</ModelsUiProvider>;
};
