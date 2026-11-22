import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { LayoutPresetRoute } from '@workbench/layoutContracts';

import { getNaturalDestination } from '@workbench/graphWidgets';

import { DEFAULT_LAYOUT_PRESET_ICON_ID, isLayoutPresetIconId } from './layoutPresetIcons';

export const getInitialLayoutPresetIconId = (iconId: unknown): string =>
  isLayoutPresetIconId(iconId) ? iconId : DEFAULT_LAYOUT_PRESET_ICON_ID;

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

/**
 * The route a workspace shows in Customize and Done saves: Auto (`null`) where the workspace may go without one, else
 * an explicit route through a source it places. A built-in always has its shipped route to fall back on, so it has no
 * Auto; nor does one whose sources are all gone.
 */
export const resolveLayoutPresetRoute = (
  defaultRoute: LayoutPresetRoute | null | undefined,
  sourceOptions: readonly GraphWidgetSource[],
  allowsAuto: boolean
): LayoutPresetRoute | null =>
  (defaultRoute ?? null) === null && allowsAuto
    ? null
    : (getInitialLayoutPresetRoute(defaultRoute ?? undefined, sourceOptions) ?? null);
