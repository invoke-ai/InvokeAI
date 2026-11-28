import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkspacePlacement } from '@workbench/workspace/workspaceDerivations';

import { autoSwitchDestinations } from './autoRoutePolicy';
import { getSourceLabel } from './invocation';
import { findWidgetTypeIdForSourceId, invocationSources } from './invocationSources';

export const getNaturalDestination = (sourceId: InvocationSourceId): ResultDestination =>
  autoSwitchDestinations[sourceId];

export interface GraphWidgetSource {
  sourceId: InvocationSourceId;
  typeId: WidgetTypeId;
  label: string;
}

let graphWidgetSources: GraphWidgetSource[] | undefined;

/** Every available source a widget runs, with that widget type, in picker order. */
export const getGraphWidgetSources = (): readonly GraphWidgetSource[] =>
  (graphWidgetSources ??= invocationSources.flatMap(({ available, id }) => {
    const typeId = available ? findWidgetTypeIdForSourceId(id) : null;

    return typeId ? [{ label: getSourceLabel(id), sourceId: id, typeId }] : [];
  }));

/** Floating graph widgets are both visible and placed, so include them in invocation-source discovery. */
const addFloatingWidgetTypeIds = (layout: WorkspacePlacement, typeIds: Set<WidgetTypeId>): void => {
  for (const instanceId of Object.keys(layout.floatingWidgets ?? {})) {
    const typeId = layout.widgetInstances[instanceId]?.typeId;

    if (typeId) {
      typeIds.add(typeId);
    }
  }
};

// Collapsed regions still count as visible because disclosure does not change routing.
export const getVisibleWidgetTypeIds = (layout: WorkspacePlacement): Set<WidgetTypeId> => {
  const typeIds = new Set<WidgetTypeId>();

  for (const region of Object.values(layout.widgetRegions)) {
    const typeId = layout.widgetInstances[region.activeInstanceId]?.typeId;

    if (typeId) {
      typeIds.add(typeId);
    }
  }

  // Shaded and maximized windows count too, for the same reason.
  addFloatingWidgetTypeIds(layout, typeIds);

  return typeIds;
};

export const getPlacedWidgetTypeIds = (layout: WorkspacePlacement): Set<WidgetTypeId> => {
  const typeIds = new Set<WidgetTypeId>();

  for (const region of Object.values(layout.widgetRegions)) {
    for (const instanceId of region.instanceIds) {
      const typeId = layout.widgetInstances[instanceId]?.typeId;

      if (typeId) {
        typeIds.add(typeId);
      }
    }
  }

  addFloatingWidgetTypeIds(layout, typeIds);

  return typeIds;
};
