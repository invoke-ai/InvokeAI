import type { WidgetRegion } from '@workbench/layoutContracts';
import type { OpenWorkbenchWidgetOptions, RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';

import { useCallback } from 'react';

import type { WorkbenchWidgetCommands } from './workbenchStore';

import { focusOpenedWidget } from './focusRegions';
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

/** For controls the user acts on: the widget they open also takes focus (see `focusOpenedWidget`). */
const openAndFocus = (
  widgets: WorkbenchWidgetCommands,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[],
  widgetId: WidgetTypeId,
  options?: OpenWorkbenchWidgetOptions
): OpenWorkbenchWidgetResult => {
  const result = openWorkbenchWidget(widgets, getWidgetsForRegion, widgetId, options);

  if (result.ok) {
    focusOpenedWidget(result.region, widgetId);
  }

  return result;
};

export const useOpenWorkbenchWidget = () => {
  const { widgets } = useWorkbenchCommands();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();

  return useCallback(
    (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult =>
      openAndFocus(widgets, getWidgetsForRegion, widgetId, options),
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

      return openAndFocus(commands.widgets, registry.getWidgetsForRegion, widgetId, options);
    },
    [commands, registry]
  );
};
