import type { WidgetRegion } from '@workbench/layoutContracts';
import type { OpenWorkbenchWidgetOptions, RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';

import { useCallback } from 'react';

import type { WorkbenchWidgetCommands } from './workbenchStore';

import { openWidgetPlacement } from './widgetPlacementCommands';
import { useOptionalWorkbenchCommands, useWorkbenchCommands } from './WorkbenchContext';
import { useOptionalWorkbenchWidgetRegistry, useWorkbenchWidgetRegistry } from './WorkbenchWidgetRegistryContext';

export type OpenWorkbenchWidgetResult = { ok: true; region: WidgetRegion } | { ok: false; reason: 'unavailable' };

export const openWorkbenchWidget = (
  widgets: WorkbenchWidgetCommands,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[],
  widgetId: WidgetTypeId,
  options?: OpenWorkbenchWidgetOptions
): OpenWorkbenchWidgetResult => {
  const result = openWidgetPlacement({ getWidgetsForRegion, options, typeId: widgetId, widgets });

  return result.ok ? { ok: true, region: result.region as WidgetRegion } : { ok: false, reason: 'unavailable' };
};

export const useOpenWorkbenchWidget = () => {
  const { widgets } = useWorkbenchCommands();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();

  return useCallback(
    (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult =>
      openWorkbenchWidget(widgets, getWidgetsForRegion, widgetId, options),
    [getWidgetsForRegion, widgets]
  );
};

export const useOptionalOpenWorkbenchWidget = () => {
  const commands = useOptionalWorkbenchCommands();
  const registry = useOptionalWorkbenchWidgetRegistry();

  return useCallback(
    (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult => {
      if (!commands || !registry) {
        return { ok: false, reason: 'unavailable' };
      }

      return openWorkbenchWidget(commands.widgets, registry.getWidgetsForRegion, widgetId, options);
    },
    [commands, registry]
  );
};
