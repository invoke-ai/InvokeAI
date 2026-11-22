import type { LayoutPreset, LayoutPresetId, WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type { AccountState } from '@workbench/projectContracts';

import {
  builtInLayoutPresetDescriptors,
  getLayoutPreset,
  isBuiltInLayoutPresetId,
  resolveLayoutPresetId,
} from '@workbench/layoutPresets';

/** Palette labels follow account-owned preset names without making command registration reactive. */
export const getLayoutPresetCommandTitleOverrides = (
  account: AccountState,
  formatTitle: (presetName: string) => string
): Readonly<Record<string, string>> =>
  Object.fromEntries(
    builtInLayoutPresetDescriptors.flatMap(({ hotkeyId, preset }) => {
      const saved = resolveSavedLayoutPreset(account, preset.id);

      // Only a rename: a preset still on its shipped name keeps the catalog title.
      return saved.labelKey !== undefined || saved.label === preset.label
        ? []
        : [[`app.${hotkeyId}`, formatTitle(saved.label)]];
    })
  );

/** A workspace as this account has it: a built-in with the account's overrides, or a custom one. */
export const resolveSavedLayoutPreset = (account: AccountState, presetId: LayoutPresetId): LayoutPreset => {
  const resolvedPresetId = resolveLayoutPresetId(presetId);
  const customPreset = isBuiltInLayoutPresetId(resolvedPresetId)
    ? undefined
    : account.customLayoutPresets?.find((preset) => preset.id === presetId);

  if (customPreset) {
    return customPreset;
  }

  const builtInPreset = getLayoutPreset(resolvedPresetId);
  const metadata = account.layoutPresetMetadataOverrides?.[builtInPreset.id];
  const override = account.layoutPresetOverrides?.[builtInPreset.id];
  const defaultRoute = account.layoutPresetRouteOverrides?.[builtInPreset.id] ?? builtInPreset.defaultRoute;

  if (!metadata && !override && defaultRoute === builtInPreset.defaultRoute) {
    return builtInPreset;
  }

  const resolved: LayoutPreset = {
    ...builtInPreset,
    ...metadata,
    defaultRoute,
    snapshot: override ?? builtInPreset.snapshot,
  };

  // A renamed preset is shown by the name the account gave it, in every language.
  if (metadata?.label !== undefined) {
    delete resolved.labelKey;
  }

  return resolved;
};

/** Built-in presets always exist; a custom one exists while the account still has it. */
export const doesLayoutPresetExist = (account: AccountState, presetId: LayoutPresetId): boolean =>
  isBuiltInLayoutPresetId(presetId) || (account.customLayoutPresets ?? []).some((preset) => preset.id === presetId);

const cloneWidgetRegionState = (region: WidgetRegionState): WidgetRegionState => ({
  ...region,
  instanceIds: [...region.instanceIds],
  ...(region.alignEndInstanceIds ? { alignEndInstanceIds: [...region.alignEndInstanceIds] } : {}),
});

export const cloneLayoutPresetWidgetRegions = (
  widgetRegionState: Record<WidgetRegion, WidgetRegionState>
): Record<WidgetRegion, WidgetRegionState> => ({
  bottom: cloneWidgetRegionState(widgetRegionState.bottom),
  center: cloneWidgetRegionState(widgetRegionState.center),
  left: cloneWidgetRegionState(widgetRegionState.left),
  right: cloneWidgetRegionState(widgetRegionState.right),
});
