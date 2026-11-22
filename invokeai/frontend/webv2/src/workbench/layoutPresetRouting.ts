import type { LayoutPreset } from './layoutContracts';

import { graphWidgetSources } from './graphWidgets';
import { getAvailableSourceIdForWidgetTypeId as getAvailableSourceId } from './invocationSources';
import { getPlacedSources } from './workspace/workspaceDerivations';

/** Graph sources that the saved arrangement can actually mount, in routing-menu order. */
export const getLayoutPresetSourceOptions = (preset: LayoutPreset) => {
  const sourceIds = getPlacedSources(preset.snapshot, getAvailableSourceId);

  return graphWidgetSources.filter((source) => sourceIds.has(source.sourceId));
};
