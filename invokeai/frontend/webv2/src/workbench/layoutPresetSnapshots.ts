import type {
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetSnapshot,
  LayoutPresetWidgetInstanceSnapshot,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { AccountState, Project } from '@workbench/projectContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { getLayoutPreset } from '@workbench/layoutPresets';

const widgetRegions: WidgetRegion[] = ['left', 'right', 'bottom', 'center'];

/**
 * The preset as saved *for this account*: a custom preset, or a built-in with
 * the account's saved edits layered over it. Everything that answers "what does
 * this preset look like" — applying it, reverting to it, and the drift
 * comparison behind the strip's dot — reads through here, or `Save changes`
 * would appear to do nothing.
 */
export const resolveSavedLayoutPreset = (account: AccountState, presetId: LayoutPresetId): LayoutPreset => {
  const customPreset = account.customLayoutPresets?.find((preset) => preset.id === presetId);

  if (customPreset) {
    return customPreset;
  }

  const builtInPreset = getLayoutPreset(presetId);
  const override = account.layoutPresetOverrides?.[builtInPreset.id];
  const defaultRoute = account.layoutPresetRouteOverrides?.[builtInPreset.id] ?? builtInPreset.defaultRoute;

  return override || defaultRoute !== builtInPreset.defaultRoute
    ? { ...builtInPreset, defaultRoute, snapshot: override ?? builtInPreset.snapshot }
    : builtInPreset;
};

export const cloneLayoutPresetWidgetRegions = (
  widgetRegionState: Record<WidgetRegion, WidgetRegionState>
): Record<WidgetRegion, WidgetRegionState> => ({
  bottom: { ...widgetRegionState.bottom, instanceIds: [...widgetRegionState.bottom.instanceIds] },
  center: { ...widgetRegionState.center, instanceIds: [...widgetRegionState.center.instanceIds] },
  left: { ...widgetRegionState.left, instanceIds: [...widgetRegionState.left.instanceIds] },
  right: { ...widgetRegionState.right, instanceIds: [...widgetRegionState.right.instanceIds] },
});

export const createLayoutPresetSnapshot = (project: Project): LayoutPresetSnapshot => {
  const referencedInstanceIds = new Set<WidgetInstanceId>();

  for (const region of widgetRegions) {
    referencedInstanceIds.add(project.widgetRegions[region].activeInstanceId);

    for (const instanceId of project.widgetRegions[region].instanceIds) {
      referencedInstanceIds.add(instanceId);
    }
  }

  const widgetInstances: Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot> = {};

  for (const instanceId of referencedInstanceIds) {
    const instance = project.widgetInstances[instanceId];

    if (instance) {
      widgetInstances[instanceId] = { id: instance.id, title: instance.title, typeId: instance.typeId };
    }
  }

  return {
    layout: { ...project.layout, panels: { ...project.layout.panels } },
    widgetInstances,
    widgetRegions: cloneLayoutPresetWidgetRegions(project.widgetRegions),
  };
};

const areArraysEqual = (left: string[], right: string[]): boolean =>
  left.length === right.length && left.every((value, index) => value === right[index]);

const areWidgetRegionsEqual = (left: WidgetRegionState, right: WidgetRegionState): boolean =>
  left.activeInstanceId === right.activeInstanceId &&
  left.isCollapsed === right.isCollapsed &&
  left.sizePx === right.sizePx &&
  areArraysEqual(left.instanceIds, right.instanceIds);

const areWidgetInstanceSnapshotsEqual = (
  left: LayoutPresetSnapshot['widgetInstances'],
  right: LayoutPresetSnapshot['widgetInstances']
): boolean => {
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();

  return (
    areArraysEqual(leftKeys, rightKeys) &&
    leftKeys.every((key) => left[key]?.typeId === right[key]?.typeId && left[key]?.title === right[key]?.title)
  );
};

export const areLayoutPresetSnapshotsEqual = (left: LayoutPresetSnapshot, right: LayoutPresetSnapshot): boolean =>
  left.layout.centerViewId === right.layout.centerViewId &&
  left.layout.panels.isBottomOpen === right.layout.panels.isBottomOpen &&
  left.layout.panels.isLeftOpen === right.layout.panels.isLeftOpen &&
  left.layout.panels.isRightOpen === right.layout.panels.isRightOpen &&
  widgetRegions.every((region) => areWidgetRegionsEqual(left.widgetRegions[region], right.widgetRegions[region])) &&
  areWidgetInstanceSnapshotsEqual(left.widgetInstances, right.widgetInstances);

export const doesProjectMatchLayoutPreset = (project: Project, preset: LayoutPreset): boolean =>
  areLayoutPresetSnapshotsEqual(createLayoutPresetSnapshot(project), preset.snapshot);
