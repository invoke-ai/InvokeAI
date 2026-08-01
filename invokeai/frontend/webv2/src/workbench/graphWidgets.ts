import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { autoSwitchDestinations } from './autoRoutePolicy';
import { getDestinationLabel, getSourceLabel, isInvocationSourceAvailable } from './invocation';

/**
 * The invocation source model behind the topbar's routing control.
 *
 * A *graph widget* is a widget that can expose a generation graph and can
 * therefore act as a source. Which ones are open is a property of the live
 * layout, not of the layout preset — a preset routinely places two or three of
 * them at once, which is exactly why a preset can never determine the source.
 */

/** Widget type ids that carry a graph, in the order the routing menu lists them. */
const graphWidgetTypeIds = ['generate', 'canvas', 'upscale', 'workflow'] as const;

export type GraphWidgetTypeId = (typeof graphWidgetTypeIds)[number];

/**
 * Source id and widget type id are the same string for every graph widget today.
 * They are still distinct concepts — one names a route, the other names a
 * dockable surface — so the mapping is written out rather than assumed.
 */
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

/** Where a source's output naturally lands when the destination is not locked. */
export const getNaturalDestination = (sourceId: InvocationSourceId): ResultDestination =>
  autoSwitchDestinations[sourceId];

export interface GraphWidgetSource {
  sourceId: InvocationSourceId;
  typeId: GraphWidgetTypeId;
  label: string;
}

/** Every graph widget, in the order the routing menu lists them. */
export const graphWidgetSources: GraphWidgetSource[] = graphWidgetTypeIds
  .map((typeId) => ({
    label: getSourceLabel(sourceIdsByWidgetTypeId[typeId]),
    sourceId: sourceIdsByWidgetTypeId[typeId],
    typeId,
  }))
  .filter((source) => isInvocationSourceAvailable(source.sourceId));

/**
 * The widget type each region is currently showing.
 *
 * This is what "visible" means in the routing menu — a surface the current
 * layout is showing. It is deliberately *not* what the menu lists:
 * every source stays selectable, because being unable to pick Workflow just
 * because the centre happens to show Canvas is a dead end, not a safeguard.
 *
 * A collapsed region still counts. Collapsing a panel is a viewing choice, not
 * a statement about what should run.
 */
export const getVisibleWidgetTypeIds = (project: Project): Set<WidgetTypeId> => {
  const typeIds = new Set<WidgetTypeId>();

  for (const region of Object.values(project.widgetRegions)) {
    const typeId = project.widgetInstances[region.activeInstanceId]?.typeId;

    if (typeId) {
      typeIds.add(typeId);
    }
  }

  return typeIds;
};

/** Widget types placed anywhere in the project, whether or not they are showing. */
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

  return typeIds;
};

/**
 * The routing indicator is icon-only, so it carries the whole route in its
 * accessible name — including whether the route is pinned or follows edits.
 */
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
