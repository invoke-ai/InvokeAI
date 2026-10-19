import type { InvocationRoute, InvocationSourceId } from './invocationContracts';
import type { LayoutPreset, LayoutPresetRoute, WidgetRegion } from './layoutContracts';

import { getNaturalDestination, getSourceIdForWidgetTypeId, graphWidgetSources } from './graphWidgets';
import { isInvocationSourceAvailable } from './invocation';

const activeSourceRegionPriority: WidgetRegion[] = ['center', 'left', 'right', 'bottom'];

const getPresetSourceIds = (preset: LayoutPreset): Set<InvocationSourceId> => {
  const sourceIds = new Set<InvocationSourceId>();

  for (const region of Object.values(preset.snapshot.widgetRegions)) {
    for (const instanceId of region.instanceIds) {
      const typeId = preset.snapshot.widgetInstances[instanceId]?.typeId;
      const sourceId = typeId ? getSourceIdForWidgetTypeId(typeId) : null;

      if (sourceId && isInvocationSourceAvailable(sourceId)) {
        sourceIds.add(sourceId);
      }
    }
  }

  return sourceIds;
};

/** Graph sources that the saved arrangement can actually mount, in routing-menu order. */
export const getLayoutPresetSourceOptions = (preset: LayoutPreset) => {
  const sourceIds = getPresetSourceIds(preset);

  return graphWidgetSources.filter((source) => sourceIds.has(source.sourceId));
};

const getActivePresetSource = (preset: LayoutPreset): InvocationSourceId | null => {
  for (const regionId of activeSourceRegionPriority) {
    const region = preset.snapshot.widgetRegions[regionId];
    const typeId = preset.snapshot.widgetInstances[region.activeInstanceId]?.typeId;
    const sourceId = typeId ? getSourceIdForWidgetTypeId(typeId) : null;

    if (sourceId && isInvocationSourceAvailable(sourceId)) {
      return sourceId;
    }
  }

  return null;
};

const resolvePresetRouteCandidate = (invocation: InvocationRoute, preset: LayoutPreset): LayoutPresetRoute => {
  const sourceIds = getPresetSourceIds(preset);

  if (preset.defaultRoute && sourceIds.has(preset.defaultRoute.sourceId)) {
    return preset.defaultRoute;
  }

  if (sourceIds.has(invocation.sourceId)) {
    return { destination: invocation.destination, sourceId: invocation.sourceId };
  }

  const activeSourceId = getActivePresetSource(preset);

  return activeSourceId
    ? { destination: getNaturalDestination(activeSourceId), sourceId: activeSourceId }
    : { destination: invocation.destination, sourceId: invocation.sourceId };
};

/** Resolves a safe preset route, then applies each unlocked field independently. */
export const getInvocationAfterLayoutPreset = (invocation: InvocationRoute, preset: LayoutPreset): InvocationRoute => {
  const candidate = resolvePresetRouteCandidate(invocation, preset);

  return {
    ...invocation,
    destination: invocation.destinationLocked ? invocation.destination : candidate.destination,
    sourceId: invocation.sourceLocked ? invocation.sourceId : candidate.sourceId,
  };
};
