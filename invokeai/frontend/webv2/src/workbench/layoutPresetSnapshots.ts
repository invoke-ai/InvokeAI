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

import { normalizeFloatingPlacement } from '@workbench/floatingWindows';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';
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

/** Built-in presets always exist; a custom one exists while the account still has it. */
export const doesLayoutPresetExist = (account: AccountState, presetId: LayoutPresetId): boolean =>
  isBuiltInLayoutPresetId(presetId) || (account.customLayoutPresets ?? []).some((preset) => preset.id === presetId);

export const findLayoutPresetWorkingCopy = (
  presetWorkingLayouts: Project['presetWorkingLayouts'],
  presetId: LayoutPresetId
): LayoutPresetSnapshot | undefined => presetWorkingLayouts?.find((copy) => copy.presetId === presetId)?.snapshot;

/** What switching to a preset lays out in this project: its working copy here, else its saved arrangement. */
export const getLayoutPresetArrangement = (project: Project, preset: LayoutPreset): LayoutPresetSnapshot =>
  (project.layout.presetId === preset.id
    ? undefined
    : findLayoutPresetWorkingCopy(project.presetWorkingLayouts, preset.id)) ?? preset.snapshot;

/**
 * Presets this project holds an unsaved arrangement of besides the active one, whose drift is the live layout's.
 * A copy that matches its saved preset (saved since, here or in another project) is not unsaved.
 */
export const getUnsavedInactiveLayoutPresetIds = (
  presetWorkingLayouts: Project['presetWorkingLayouts'],
  activePresetId: LayoutPresetId,
  account: AccountState
): LayoutPresetId[] =>
  (presetWorkingLayouts ?? []).flatMap(({ presetId, snapshot }) =>
    presetId !== activePresetId &&
    doesLayoutPresetExist(account, presetId) &&
    !areLayoutPresetSnapshotsEqual(snapshot, resolveSavedLayoutPreset(account, presetId).snapshot)
      ? [presetId]
      : []
  );

const cloneFloatingWidgets = (
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

  for (const region of WIDGET_REGIONS) {
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

/**
 * Read a snapshot's placements the way a hydrated project's are. Stored presets, applied presets, and drift
 * comparison share this, so a freshly applied preset always matches the snapshot it came from.
 */
export const normalizeLayoutPresetSnapshot = (snapshot: LayoutPresetSnapshot): LayoutPresetSnapshot => {
  const { floatingWidgets, widgetRegions } = normalizeFloatingPlacement(
    snapshot.widgetRegions,
    snapshot.floatingWidgets,
    (instanceId) => Object.hasOwn(snapshot.widgetInstances, instanceId)
  );
  // A window normalization dropped leaves an instance nothing places; applying the preset would not place it
  // either, so the snapshot stops naming it.
  const orphanedIds = Object.keys(snapshot.floatingWidgets ?? {}).filter(
    (instanceId) =>
      !floatingWidgets?.[instanceId] &&
      WIDGET_REGIONS.every(
        (region) =>
          widgetRegions[region].activeInstanceId !== instanceId &&
          !widgetRegions[region].instanceIds.includes(instanceId)
      )
  );

  return {
    ...(floatingWidgets ? { floatingWidgets } : {}),
    layout: snapshot.layout,
    widgetInstances:
      orphanedIds.length > 0
        ? Object.fromEntries(Object.entries(snapshot.widgetInstances).filter(([id]) => !orphanedIds.includes(id)))
        : snapshot.widgetInstances,
    widgetRegions,
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

/** Geometry and the return slot count as saved-layout drift; focus-driven stack ordering does not. */
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
        leftState?.returnRegion === rightState?.returnRegion &&
        leftState?.returnIndex === rightState?.returnIndex
      );
    })
  );
};

export const areLayoutPresetSnapshotsEqual = (
  leftSnapshot: LayoutPresetSnapshot,
  rightSnapshot: LayoutPresetSnapshot
): boolean => {
  const left = normalizeLayoutPresetSnapshot(leftSnapshot);
  const right = normalizeLayoutPresetSnapshot(rightSnapshot);

  return (
    left.layout.centerViewId === right.layout.centerViewId &&
    left.layout.panels.isBottomOpen === right.layout.panels.isBottomOpen &&
    left.layout.panels.isLeftOpen === right.layout.panels.isLeftOpen &&
    left.layout.panels.isRightOpen === right.layout.panels.isRightOpen &&
    WIDGET_REGIONS.every((region) => areWidgetRegionsEqual(left.widgetRegions[region], right.widgetRegions[region])) &&
    areFloatingWidgetsEqual(left.floatingWidgets, right.floatingWidgets) &&
    areWidgetInstanceSnapshotsEqual(left.widgetInstances, right.widgetInstances)
  );
};

export const doesProjectMatchLayoutPreset = (project: Project, preset: LayoutPreset): boolean =>
  areLayoutPresetSnapshotsEqual(createLayoutPresetSnapshot(project), preset.snapshot);
