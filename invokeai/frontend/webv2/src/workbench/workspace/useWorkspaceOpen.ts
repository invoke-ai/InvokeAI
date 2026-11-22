import { useWorkbenchFocus } from '@workbench/focusRegions';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCallback } from 'react';

import type { OpenInWorkspace } from './workspaceNavigation';

import { openInWorkspace } from './workspaceNavigation';

/** `openInWorkspace` bound to this editor's store, widget registry and focus, for controls the user acts on. */
export const useWorkspaceOpen = (): OpenInWorkspace => {
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();

  return useCallback<OpenInWorkspace>(
    (intents, options) => openInWorkspace({ commands, focus, getWidgetsForRegion, queries }, intents, options),
    [commands, focus, getWidgetsForRegion, queries]
  );
};
