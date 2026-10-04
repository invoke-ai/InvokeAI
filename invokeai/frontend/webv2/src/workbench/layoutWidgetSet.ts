import type { WidgetTypeId } from '@workbench/widgetContracts';

/**
 * The subset of a project or preset snapshot that determines which widgets a
 * layout renders. Both `Project` and `LayoutPresetSnapshot` satisfy it
 * structurally.
 */
export interface LayoutWidgetSource {
  widgetInstances: Record<string, { typeId: WidgetTypeId }>;
  widgetRegions: Record<string, { activeInstanceId: string; instanceIds: string[] }>;
}

/** Order regions by boot reveal priority. */
const REGION_ORDER = ['center', 'left', 'right', 'bottom'];

/**
 * Include every active region widget and all bottom widgets, since the status bar renders them all. Share this set
 * between preload and activation gating.
 */
export const getLayoutWidgetTypeIds = (layout: LayoutWidgetSource): WidgetTypeId[] => {
  const typeIds = new Set<WidgetTypeId>();
  const regions = Object.entries(layout.widgetRegions).sort(
    (left, right) => REGION_ORDER.indexOf(left[0]) - REGION_ORDER.indexOf(right[0])
  );

  const add = (instanceId: string): void => {
    const typeId = layout.widgetInstances[instanceId]?.typeId;

    if (typeId) {
      typeIds.add(typeId);
    }
  };

  for (const [, state] of regions) {
    add(state.activeInstanceId);
  }

  for (const [region, state] of regions) {
    if (region === 'bottom') {
      state.instanceIds.forEach(add);
    }
  }

  return [...typeIds];
};
