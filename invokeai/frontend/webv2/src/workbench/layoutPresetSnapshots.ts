import type {
  FloatingWidgetState,
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetSnapshot,
  LayoutPresetWidgetInstanceSnapshot,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { AccountState, Project } from '@workbench/projectContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

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
      const savedLabel = resolveSavedLayoutPreset(account, preset.id).label;

      return savedLabel === preset.label ? [] : [[`app.${hotkeyId}`, formatTitle(savedLabel)]];
    })
  );

const widgetRegions: WidgetRegion[] = ['left', 'right', 'bottom', 'center'];

/** Apply, revert, and drift checks all resolve account overrides here so Save changes has consistent meaning. */
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

  return metadata || override || defaultRoute !== builtInPreset.defaultRoute
    ? { ...builtInPreset, ...metadata, defaultRoute, snapshot: override ?? builtInPreset.snapshot }
    : builtInPreset;
};

export const cloneFloatingWidgets = (
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState>
): Record<WidgetInstanceId, FloatingWidgetState> =>
  Object.fromEntries(Object.entries(floatingWidgets).map(([instanceId, state]) => [instanceId, { ...state }]));

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

export const createLayoutPresetSnapshot = (project: Project): LayoutPresetSnapshot => {
  const referencedInstanceIds = new Set<WidgetInstanceId>();

  for (const region of widgetRegions) {
    referencedInstanceIds.add(project.widgetRegions[region].activeInstanceId);

    for (const instanceId of project.widgetRegions[region].instanceIds) {
      referencedInstanceIds.add(instanceId);
    }
  }

  // A floated instance sits in no region, so it has to be named here too or
  // applying the preset would recreate neither the window nor a docked copy.
  for (const instanceId of Object.keys(project.floatingWidgets ?? {})) {
    referencedInstanceIds.add(instanceId);
  }

  const widgetInstances: Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot> = {};

  for (const instanceId of referencedInstanceIds) {
    const instance = project.widgetInstances[instanceId];

    if (instance) {
      widgetInstances[instanceId] = { id: instance.id, title: instance.title, typeId: instance.typeId };
    }
  }

  return {
    ...(project.floatingWidgets ? { floatingWidgets: cloneFloatingWidgets(project.floatingWidgets) } : {}),
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
  areArraysEqual(left.instanceIds, right.instanceIds) &&
  areArraysEqual(left.alignEndInstanceIds ?? [], right.alignEndInstanceIds ?? []);

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

/** Geometry counts as saved-layout drift; focus-driven stack ordering does not. */
const areFloatingWidgetsEqual = (
  left: LayoutPresetSnapshot['floatingWidgets'],
  right: LayoutPresetSnapshot['floatingWidgets']
): boolean => {
  const leftKeys = Object.keys(left ?? {}).sort();
  const rightKeys = Object.keys(right ?? {}).sort();

  return (
    areArraysEqual(leftKeys, rightKeys) &&
    leftKeys.every((key) => {
      const leftState = left?.[key];
      const rightState = right?.[key];

      return (
        leftState?.x === rightState?.x &&
        leftState?.y === rightState?.y &&
        leftState?.widthPx === rightState?.widthPx &&
        leftState?.heightPx === rightState?.heightPx &&
        leftState?.mode === rightState?.mode &&
        leftState?.returnRegion === rightState?.returnRegion
      );
    })
  );
};

export const areLayoutPresetSnapshotsEqual = (left: LayoutPresetSnapshot, right: LayoutPresetSnapshot): boolean =>
  left.layout.centerViewId === right.layout.centerViewId &&
  left.layout.panels.isBottomOpen === right.layout.panels.isBottomOpen &&
  left.layout.panels.isLeftOpen === right.layout.panels.isLeftOpen &&
  left.layout.panels.isRightOpen === right.layout.panels.isRightOpen &&
  widgetRegions.every((region) => areWidgetRegionsEqual(left.widgetRegions[region], right.widgetRegions[region])) &&
  areFloatingWidgetsEqual(left.floatingWidgets, right.floatingWidgets) &&
  areWidgetInstanceSnapshotsEqual(left.widgetInstances, right.widgetInstances);

export const doesProjectMatchLayoutPreset = (project: Project, preset: LayoutPreset): boolean =>
  areLayoutPresetSnapshotsEqual(createLayoutPresetSnapshot(project), preset.snapshot);
