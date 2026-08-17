import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { autoSwitchDestinations } from './autoRoutePolicy';
import { getDestinationLabel, getSourceLabel, isInvocationSourceAvailable } from './invocation';

const graphWidgetTypeIds = ['generate', 'canvas', 'upscale', 'workflow'] as const;

export type GraphWidgetTypeId = (typeof graphWidgetTypeIds)[number];

const sourceIdsByWidgetTypeId: Record<GraphWidgetTypeId, InvocationSourceId> = {
  canvas: 'canvas',
  generate: 'generate',
  upscale: 'upscale',
  workflow: 'workflow',
};

const widgetTypeIdsBySourceId: Record<InvocationSourceId, GraphWidgetTypeId> = {
  canvas: 'canvas',
  generate: 'generate',
  upscale: 'upscale',
  workflow: 'workflow',
};

export const isGraphWidgetTypeId = (typeId: WidgetTypeId): typeId is GraphWidgetTypeId =>
  (graphWidgetTypeIds as readonly string[]).includes(typeId);

export const getSourceIdForWidgetTypeId = (typeId: WidgetTypeId): InvocationSourceId | null =>
  isGraphWidgetTypeId(typeId) ? sourceIdsByWidgetTypeId[typeId] : null;

export const getWidgetTypeIdForSourceId = (sourceId: InvocationSourceId): GraphWidgetTypeId =>
  widgetTypeIdsBySourceId[sourceId];

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

/**
 * Floated instances belong in both sets below: a widget in a window is on
 * screen and placed, just not in a rail. Reading only `widgetRegions` would
 * drop the first graph-bearing widget that opts into floating out of the
 * invoke-source list while it sits in plain view.
 */
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

export const describeRoute = ({
  destination,
  destinationLocked,
  hasSource,
  sourceId,
  sourceLocked,
}: {
  destination: ResultDestination;
  destinationLocked: boolean;
  hasSource: boolean;
  sourceId: InvocationSourceId;
  sourceLocked: boolean;
}): string => {
  const from = hasSource ? `Invoke from ${getSourceLabel(sourceId).toLowerCase()}` : 'No source widget open';
  const to = `output to ${getDestinationLabel(destination).toLowerCase()}`;
  const mode = sourceLocked ? 'source locked' : 'following edits';
  const destinationMode = destinationLocked ? ', destination locked' : '';

  return `${from}, ${to}, ${mode}${destinationMode}`;
};
