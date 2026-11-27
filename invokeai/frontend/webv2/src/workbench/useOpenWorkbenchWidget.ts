import type { WidgetRegion } from '@workbench/layoutContracts';
import type { OpenWorkbenchWidgetOptions, RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';

import { useCallback } from 'react';

import type { WorkbenchFocusController } from './focusRegions';
import type { WorkbenchQueries, WorkbenchWidgetCommands } from './workbenchStore';

import { useWorkbenchFocus } from './focusRegions';
import { openWidgetPlacement, revealPlacedWidget } from './widgetPlacementCommands';
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

type FocusRegion = ReturnType<typeof useWorkbenchFocus>['focusRegion'];

/** For controls the user acts on: the widget they open also takes focus (see `focusRegion`). */
const openAndFocus = (
  widgets: WorkbenchWidgetCommands,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[],
  focusRegion: FocusRegion,
  widgetId: WidgetTypeId,
  options?: OpenWorkbenchWidgetOptions
): OpenWorkbenchWidgetResult => {
  const result = openWorkbenchWidget(widgets, getWidgetsForRegion, widgetId, options);

  if (result.ok) {
    focusRegion(result.region, widgetId);
  }

  return result;
};

/**
 * For "show me" controls: where the layout already has the widget, bring that one forward (raising its window) rather
 * than open another; otherwise open it. Either way it takes focus.
 */
export const revealOrOpenWorkbenchWidget = (
  {
    focus,
    getWidgetsForRegion,
    queries,
    widgets,
  }: {
    focus: Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion'>;
    getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
    queries: Pick<WorkbenchQueries, 'getSnapshot'>;
    widgets: WorkbenchWidgetCommands;
  },
  widgetId: WidgetTypeId
): void => {
  const revealed = revealPlacedWidget({ project: queries.getSnapshot().activeProject, typeId: widgetId, widgets });

  if (revealed?.kind === 'floating') {
    focus.focusFloating(revealed.instanceId);
  } else if (revealed) {
    focus.focusRegion(revealed.region, widgetId);
  } else {
    openAndFocus(widgets, getWidgetsForRegion, focus.focusRegion, widgetId);
  }
};

export const useOpenWorkbenchWidget = () => {
  const { widgets } = useWorkbenchCommands();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const { focusRegion } = useWorkbenchFocus();

  return useCallback(
    (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult =>
      openAndFocus(widgets, getWidgetsForRegion, focusRegion, widgetId, options),
    [focusRegion, getWidgetsForRegion, widgets]
  );
};

export const useOptionalOpenWorkbenchWidget = () => {
  const commands = useOptionalWorkbenchCommands();
  const registry = useOptionalWorkbenchWidgetRegistry();
  const { focusRegion } = useWorkbenchFocus();

  return useCallback(
    (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult => {
      if (!commands || !registry) {
        return { ok: false, reason: 'unavailable' };
      }

      return openAndFocus(commands.widgets, registry.getWidgetsForRegion, focusRegion, widgetId, options);
    },
    [commands, focusRegion, registry]
  );
};
