import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { LayoutPresetRoute } from '@workbench/layoutContracts';

import { getNaturalDestination } from '@workbench/graphWidgets';

export const getInitialLayoutPresetRoute = (
  defaultRoute: LayoutPresetRoute | undefined,
  sourceOptions: readonly GraphWidgetSource[]
): LayoutPresetRoute | undefined => {
  if (defaultRoute && sourceOptions.some((source) => source.sourceId === defaultRoute.sourceId)) {
    return { ...defaultRoute };
  }

  const sourceId = sourceOptions[0]?.sourceId;

  return sourceId ? { destination: getNaturalDestination(sourceId), sourceId } : undefined;
};
