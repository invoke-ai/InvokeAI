import type { GraphBearingSurfaceContract, WidgetManifest, WorkbenchRegion } from './types';

import { isInvocationSourceAvailable } from './invocation';

export const createGraphBearingSurface = (
  manifest: WidgetManifest,
  region: WorkbenchRegion,
  label: string
): GraphBearingSurfaceContract | null => {
  const graphBearing = manifest.graphBearing;

  if (!graphBearing || !graphBearing.surfaces.includes(region)) {
    return null;
  }

  return {
    canPreviewGraph: true,
    canSetSource: isInvocationSourceAvailable(graphBearing.sourceId),
    graphId: graphBearing.defaultGraphId,
    label,
    region,
    sourceId: graphBearing.sourceId,
    surfaceId: `${manifest.id}:${region}`,
    widgetId: manifest.id,
  };
};
