import { useWorkbenchFocus } from '@workbench/focusRegions';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCallback, useMemo } from 'react';

import type { OpenInWorkspace, WorkspaceNavigationPorts } from './workspaceNavigation';

import { openInWorkspace } from './workspaceNavigation';

/** This editor's store, widget registry and focus, as workspace navigation reads them. */
export const useWorkspaceNavigationPorts = (): WorkspaceNavigationPorts => {
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();

  return useMemo(
    () => ({ commands, focus, getWidgetsForRegion, queries }),
    [commands, focus, getWidgetsForRegion, queries]
  );
};

/** `openInWorkspace` bound to this editor, for controls the user acts on. */
export const useWorkspaceOpen = (): OpenInWorkspace => {
  const ports = useWorkspaceNavigationPorts();

  return useCallback<OpenInWorkspace>((intents, options) => openInWorkspace(ports, intents, options), [ports]);
};
