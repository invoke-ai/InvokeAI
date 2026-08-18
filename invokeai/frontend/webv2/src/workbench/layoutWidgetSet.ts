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

/**
 * Regions in the order a boot reveals them, so callers that care about
 * priority get it for free and callers that do not are unaffected.
 */
const REGION_ORDER = ['center', 'left', 'right', 'bottom'];

/**
 * Every widget type a layout renders: each region's active instance, plus every
 * placed bottom instance — the status bar mounts all of its items as compact
 * widgets, not just the active one.
 *
 * Single definition on purpose. The boot preloader and the preset activation
 * gate previously disagreed: activation only awaited region actives, so a
 * switch never gated on four of the five bottom instances every built-in preset
 * places, and the atomic reveal the deadline buys was incomplete.
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
