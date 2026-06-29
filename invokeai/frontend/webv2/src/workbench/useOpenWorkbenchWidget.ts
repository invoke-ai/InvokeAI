import { useCallback, type Dispatch } from 'react';

import type { OpenWorkbenchWidgetOptions, RegisteredWidget, WidgetRegion, WidgetTypeId } from './types';
import type { WorkbenchAction } from './workbenchState';

import { openWidgetPlacement } from './widgetPlacementCommands';
import { useOptionalWorkbenchDispatch, useWorkbenchDispatch } from './WorkbenchContext';
import { useOptionalWorkbenchWidgetRegistry, useWorkbenchWidgetRegistry } from './WorkbenchWidgetRegistryContext';

export type OpenWorkbenchWidgetResult = { ok: true; region: WidgetRegion } | { ok: false; reason: 'unavailable' };

export const openWorkbenchWidget = (
  dispatch: Dispatch<WorkbenchAction>,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[],
  widgetId: WidgetTypeId,
  options?: OpenWorkbenchWidgetOptions
): OpenWorkbenchWidgetResult => {
  const result = openWidgetPlacement({ dispatch, getWidgetsForRegion, options, typeId: widgetId });

  return result.ok ? { ok: true, region: result.region as WidgetRegion } : { ok: false, reason: 'unavailable' };
};

export const useOpenWorkbenchWidget = () => {
  const dispatch = useWorkbenchDispatch();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();

  return useCallback(
    (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult =>
      openWorkbenchWidget(dispatch, getWidgetsForRegion, widgetId, options),
    [dispatch, getWidgetsForRegion]
  );
};

export const useOptionalOpenWorkbenchWidget = () => {
  const dispatch = useOptionalWorkbenchDispatch();
  const registry = useOptionalWorkbenchWidgetRegistry();

  return useCallback(
    (widgetId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult => {
      if (!dispatch || !registry) {
        return { ok: false, reason: 'unavailable' };
      }

      return openWorkbenchWidget(dispatch, registry.getWidgetsForRegion, widgetId, options);
    },
    [dispatch, registry]
  );
};
