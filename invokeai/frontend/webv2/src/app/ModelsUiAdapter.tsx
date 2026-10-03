import type { ModelsUiAdapter } from '@features/models';
import type { ReactNode } from 'react';

import { useCapabilities } from '@features/identity';
import { ModelsUiProvider } from '@features/models';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

export const ModelsUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const enableModelDescriptions = useWorkbenchPreferenceSelector((value) => value.enableModelDescriptions);
  const managerProjectId = useActiveProjectId();
  const { canManageModels } = useCapabilities();
  const adapter = useMemo<ModelsUiAdapter>(
    () => ({ canManageModels, enableModelDescriptions, managerProjectId }),
    [canManageModels, enableModelDescriptions, managerProjectId]
  );

  return <ModelsUiProvider adapter={adapter}>{children}</ModelsUiProvider>;
};
