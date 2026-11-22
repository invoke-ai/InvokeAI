import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

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
const addFloatingWidgetTypeIds = (project: Project, typeIds: Set<WidgetTypeId>): void => {
  for (const instanceId of Object.keys(project.floatingWidgets ?? {})) {
    const typeId = project.widgetInstances[instanceId]?.typeId;

    if (typeId) {
      typeIds.add(typeId);
    }
  }
};

// Collapsed regions still count as visible because disclosure does not change routing.
export const getVisibleWidgetTypeIds = (project: Project): Set<WidgetTypeId> => {
  const typeIds = new Set<WidgetTypeId>();

  for (const region of Object.values(project.widgetRegions)) {
    const typeId = project.widgetInstances[region.activeInstanceId]?.typeId;

    if (typeId) {
      typeIds.add(typeId);
    }
  }

  // Shaded and maximized windows count too, for the same reason.
  addFloatingWidgetTypeIds(project, typeIds);

  return typeIds;
};

export const getPlacedWidgetTypeIds = (project: Project): Set<WidgetTypeId> => {
  const typeIds = new Set<WidgetTypeId>();

  for (const region of Object.values(project.widgetRegions)) {
    for (const instanceId of region.instanceIds) {
      const typeId = project.widgetInstances[instanceId]?.typeId;

      if (typeId) {
        typeIds.add(typeId);
      }
    }
  }

  addFloatingWidgetTypeIds(project, typeIds);

  return typeIds;
};
