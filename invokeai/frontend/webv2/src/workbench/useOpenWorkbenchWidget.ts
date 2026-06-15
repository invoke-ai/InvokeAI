import { useCallback } from 'react';

import type { RegisteredWidget, WidgetId, WidgetRegion } from './types';

import { useOptionalWorkbenchDispatch, useWorkbenchDispatch } from './WorkbenchContext';
import { useOptionalWorkbenchWidgetRegistry, useWorkbenchWidgetRegistry } from './WorkbenchWidgetRegistryContext';

const DEFAULT_OPEN_REGIONS: ReadonlyArray<WidgetRegion> = ['center', 'right', 'left', 'bottom'];

export interface OpenWorkbenchWidgetOptions {
  preferredRegions?: ReadonlyArray<WidgetRegion>;
  requireCenterView?: boolean;
}

export type OpenWorkbenchWidgetResult = { ok: true; region: WidgetRegion } | { ok: false; reason: 'unavailable' };

const canOpenWidgetInRegion = (widget: RegisteredWidget, region: WidgetRegion): boolean => {
  if (widget.status !== 'enabled' || !widget.manifest.view) {
    return false;
  }

  if (region === 'center') {
    return widget.manifest.centerPlacement !== 'toolbar';
  }

  if (region === 'bottom') {
    return widget.manifest.bottomPanel !== 'tooltip';
  }

  return true;
};

const getOpenableRegionFromRegistry = (
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[],
  widgetId: WidgetId,
  options: OpenWorkbenchWidgetOptions = {}
): WidgetRegion | null => {
  const preferredRegions = options.preferredRegions ?? DEFAULT_OPEN_REGIONS;
  const seenRegions = new Set<WidgetRegion>();

  for (const region of preferredRegions) {
    if (seenRegions.has(region) || (options.requireCenterView && region !== 'center')) {
      continue;
    }

    seenRegions.add(region);

    const widget = getWidgetsForRegion(region).find((candidate) => candidate.manifest.id === widgetId);

    if (widget && canOpenWidgetInRegion(widget, region)) {
      return region;
    }
  }

  return null;
};

type OpenWorkbenchWidgetDispatch = (action: {
  region: WidgetRegion;
  type: 'openRegionWidget';
  widgetId: WidgetId;
}) => void;

export const openWorkbenchWidget = (
  dispatch: OpenWorkbenchWidgetDispatch,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[],
  widgetId: WidgetId,
  options?: OpenWorkbenchWidgetOptions
): OpenWorkbenchWidgetResult => {
  const region = getOpenableRegionFromRegistry(getWidgetsForRegion, widgetId, options);

  if (!region) {
    return { ok: false, reason: 'unavailable' };
  }

  dispatch({ region, type: 'openRegionWidget', widgetId });

  return { ok: true, region };
};

export const useOpenWorkbenchWidget = () => {
  const dispatch = useWorkbenchDispatch();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();

  return useCallback(
    (widgetId: WidgetId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult =>
      openWorkbenchWidget(dispatch, getWidgetsForRegion, widgetId, options),
    [dispatch, getWidgetsForRegion]
  );
};

export const useOptionalOpenWorkbenchWidget = () => {
  const dispatch = useOptionalWorkbenchDispatch();
  const registry = useOptionalWorkbenchWidgetRegistry();

  return useCallback(
    (widgetId: WidgetId, options?: OpenWorkbenchWidgetOptions): OpenWorkbenchWidgetResult => {
      if (!dispatch || !registry) {
        return { ok: false, reason: 'unavailable' };
      }

      return openWorkbenchWidget(dispatch, registry.getWidgetsForRegion, widgetId, options);
    },
    [dispatch, registry]
  );
};
