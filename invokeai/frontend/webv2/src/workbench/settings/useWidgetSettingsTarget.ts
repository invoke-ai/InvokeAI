import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useState } from 'react';

/** Selects only one field and fences writes against removed instances and project/account changes. */
export const useWidgetSettingsTarget = <Value extends boolean | string | number | null>(
  widgetId: WidgetTypeId,
  target: SettingsTarget | undefined,
  read: (values: Record<string, unknown>) => Value
) => {
  const [accountScope] = useState(captureAccountScope);
  const queries = useWorkbenchQueries();
  const { widgets } = useWorkbenchCommands();
  const instanceId = useActiveProjectSelector((project) => {
    if (target && target.projectId !== project.id) {
      return undefined;
    }
    const instance = target?.instanceId
      ? project.widgetInstances[target.instanceId]
      : getProjectWidgetInstance(project, widgetId);
    return instance?.typeId === widgetId ? instance.id : undefined;
  });
  const projectId = useActiveProjectSelector((project) => project.id);
  const value = useActiveProjectSelector((project) =>
    read(project.widgetInstances[instanceId ?? '']?.state.values ?? {})
  );
  const patch = useCallback(
    (values: Record<string, unknown>) => {
      if (!instanceId || !isAccountScopeCurrent(accountScope) || !queries.isActiveProject(projectId)) {
        return;
      }
      const project = queries.getProject(projectId);
      if (project?.widgetInstances[instanceId]?.typeId === widgetId) {
        widgets.patchInstanceValues(instanceId, values, projectId);
      }
    },
    [accountScope, instanceId, projectId, queries, widgetId, widgets]
  );
  return { disabled: !instanceId, patch, value };
};
