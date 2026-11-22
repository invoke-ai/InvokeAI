import type { InvocationRoute } from './invocationContracts';
import type { LayoutPreset, LayoutPresetRoute } from './layoutContracts';

import { getNaturalDestination, graphWidgetSources } from './graphWidgets';
import { getAvailableSourceIdForWidgetTypeId as getAvailableSourceId } from './invocationSources';
import { getFrontSource, getPlacedSources } from './workspace/workspaceDerivations';

/** Graph sources that the saved arrangement can actually mount, in routing-menu order. */
export const getLayoutPresetSourceOptions = (preset: LayoutPreset) => {
  const sourceIds = getPlacedSources(preset.snapshot, getAvailableSourceId);

  return graphWidgetSources.filter((source) => sourceIds.has(source.sourceId));
};

const resolvePresetRouteCandidate = (invocation: InvocationRoute, preset: LayoutPreset): LayoutPresetRoute => {
  const sourceIds = getPlacedSources(preset.snapshot, getAvailableSourceId);

  if (preset.defaultRoute && sourceIds.has(preset.defaultRoute.sourceId)) {
    return preset.defaultRoute;
  }

  if (sourceIds.has(invocation.sourceId)) {
    return { destination: invocation.destination, sourceId: invocation.sourceId };
  }

  const activeSourceId = getFrontSource(preset.snapshot, getAvailableSourceId);

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
