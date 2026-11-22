import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkspacePlacement } from '@workbench/workspace/workspaceDerivations';

import { autoSwitchDestinations } from './autoRoutePolicy';
import { getSourceLabel } from './invocation';
import {
  graphWidgetTypeIds,
  isInvocationSourceAvailable,
  sourceIdsByWidgetTypeId,
  type GraphWidgetTypeId,
} from './invocationSources';

export const getNaturalDestination = (sourceId: InvocationSourceId): ResultDestination =>
  autoSwitchDestinations[sourceId];

export interface GraphWidgetSource {
  sourceId: InvocationSourceId;
  typeId: GraphWidgetTypeId;
  label: string;
}

export const graphWidgetSources: GraphWidgetSource[] = graphWidgetTypeIds
  .map((typeId) => ({
    label: getSourceLabel(sourceIdsByWidgetTypeId[typeId]),
    sourceId: sourceIdsByWidgetTypeId[typeId],
    typeId,
  }))
  .filter((source) => isInvocationSourceAvailable(source.sourceId));

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
