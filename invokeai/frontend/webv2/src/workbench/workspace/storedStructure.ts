import type {
  CenterViewId,
  LayoutPresetSnapshot,
  LayoutPresetSnapshotLayout,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { cloneLayoutPresetWidgetRegions } from '@workbench/layoutPresetSnapshots';

import type { WorkspaceStructure } from './structure';

import { CENTER_VIEW_TYPE_IDS, normalizeStructure } from './structure';

/** Retired widget types; anything persisted about them drops on load. */
export const RETIRED_WIDGET_TYPE_IDS: ReadonlySet<string> = new Set(['properties', 'transform', 'project']);

/** Widget types a region no longer hosts; stored copies drop on load. */
export const RETIRED_REGION_WIDGET_TYPE_IDS: Readonly<Partial<Record<WidgetRegion, ReadonlySet<string>>>> = {
  bottom: new Set(['diagnostics']),
};

const withoutRetiredInstances = (
  region: WidgetRegionState,
  retired: ReadonlySet<WidgetInstanceId>
): WidgetRegionState => {
  if (!region.instanceIds.some((instanceId) => retired.has(instanceId))) {
    return region;
  }
  const instanceIds = region.instanceIds.filter((instanceId) => !retired.has(instanceId));
  return {
    ...region,
    activeInstanceId: retired.has(region.activeInstanceId) ? (instanceIds[0] ?? '') : region.activeInstanceId,
    ...(region.alignEndInstanceIds
      ? { alignEndInstanceIds: region.alignEndInstanceIds.filter((instanceId) => !retired.has(instanceId)) }
      : {}),
    instanceIds,
  };
};

const isInstancePlaced = (
  widgetRegions: Readonly<Record<WidgetRegion, WidgetRegionState>>,
  floatingWidgets: Readonly<Record<WidgetInstanceId, unknown>> | undefined,
  instanceId: WidgetInstanceId
): boolean =>
  (floatingWidgets !== undefined && Object.hasOwn(floatingWidgets, instanceId)) ||
  WIDGET_REGIONS.some(
    (regionId) =>
      widgetRegions[regionId].instanceIds.includes(instanceId) ||
      widgetRegions[regionId].activeInstanceId === instanceId
  );

/** A docked id without an instance is read by its default type: its id, or the part before `:`. */
export const stripRetiredWidgets = <Instance extends { typeId: string }>(
  widgetRegions: Readonly<Record<WidgetRegion, WidgetRegionState>>,
  widgetInstances: Readonly<Record<WidgetInstanceId, Instance>>,
  storedFloatingWidgets: unknown
): {
  widgetInstances: Record<WidgetInstanceId, Instance>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
} => {
  const typeOf = (instanceId: WidgetInstanceId): string =>
    widgetInstances[instanceId]?.typeId ?? instanceId.split(':', 1)[0]!;
  const isTypeRetired = (instanceId: WidgetInstanceId) => RETIRED_WIDGET_TYPE_IDS.has(typeOf(instanceId));
  const regions = { ...widgetRegions };
  const displaced = new Set<WidgetInstanceId>();

  for (const regionId of WIDGET_REGIONS) {
    const retiredTypes = RETIRED_REGION_WIDGET_TYPE_IDS[regionId];
    const region = regions[regionId];
    const regionRetired = new Set(
      region.instanceIds.filter((instanceId) => retiredTypes?.has(typeOf(instanceId)) === true)
    );
    const typeRetired = new Set(region.instanceIds.filter(isTypeRetired));

    for (const instanceId of regionRetired) {
      displaced.add(instanceId);
    }
    const stripped = withoutRetiredInstances(region, new Set([...typeRetired, ...regionRetired]));
    const isEmptied = stripped !== region && stripped.instanceIds.length === 0;

    regions[regionId] =
      regionId !== 'center' && (isEmptied || regionRetired.has(region.activeInstanceId))
        ? { ...stripped, isCollapsed: true }
        : stripped;
  }

  const floatingWidgets =
    storedFloatingWidgets && typeof storedFloatingWidgets === 'object'
      ? (storedFloatingWidgets as Record<WidgetInstanceId, unknown>)
      : undefined;
  const isStillPlaced = (instanceId: WidgetInstanceId) => isInstancePlaced(regions, floatingWidgets, instanceId);
  const instances = Object.fromEntries(
    Object.entries(widgetInstances).filter(
      ([instanceId]) => !isTypeRetired(instanceId) && (!displaced.has(instanceId) || isStillPlaced(instanceId))
    )
  );

  return { widgetInstances: instances, widgetRegions: regions };
};

const isStoredWidgetInstance = (instanceId: string, value: unknown): boolean => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as { id?: unknown; title?: unknown; typeId?: unknown };

  return (
    instanceId.length > 0 &&
    record.id === instanceId &&
    typeof record.typeId === 'string' &&
    record.typeId.length > 0 &&
    (record.title === undefined || typeof record.title === 'string')
  );
};

const isWidgetRegionState = (
  value: unknown,
  widgetInstances: Readonly<Record<string, unknown>>
): value is WidgetRegionState => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Partial<WidgetRegionState>;
  const instanceIds = record.instanceIds;

  const isEmpty = Array.isArray(instanceIds) && instanceIds.length === 0 && record.activeInstanceId === '';

  return (
    typeof record.activeInstanceId === 'string' &&
    Array.isArray(instanceIds) &&
    instanceIds.every((instanceId) => typeof instanceId === 'string' && instanceId in widgetInstances) &&
    new Set(instanceIds).size === instanceIds.length &&
    (isEmpty ||
      (record.activeInstanceId.length > 0 &&
        record.activeInstanceId in widgetInstances &&
        (instanceIds.length === 0 || instanceIds.includes(record.activeInstanceId)))) &&
    (record.alignEndInstanceIds === undefined ||
      (Array.isArray(record.alignEndInstanceIds) &&
        record.alignEndInstanceIds.every((instanceId) => typeof instanceId === 'string'))) &&
    typeof record.isCollapsed === 'boolean' &&
    typeof record.sizePx === 'number' &&
    Number.isFinite(record.sizePx) &&
    record.sizePx >= 0
  );
};

/** Whether a stored value has the shape of a workspace structure, as every build since presets wrote one. */
export const isStoredStructure = (value: unknown): value is LayoutPresetSnapshot => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const snapshot = value as Partial<LayoutPresetSnapshot>;
  const layout = snapshot.layout as Partial<LayoutPresetSnapshotLayout> | undefined;
  const widgetInstances = snapshot.widgetInstances as Record<string, unknown> | undefined;

  return (
    !!layout &&
    typeof layout.presetId === 'string' &&
    layout.presetId.length > 0 &&
    typeof layout.centerViewId === 'string' &&
    CENTER_VIEW_TYPE_IDS.has(layout.centerViewId as CenterViewId) &&
    !!layout.panels &&
    typeof layout.panels.isBottomOpen === 'boolean' &&
    typeof layout.panels.isLeftOpen === 'boolean' &&
    typeof layout.panels.isRightOpen === 'boolean' &&
    !!widgetInstances &&
    typeof widgetInstances === 'object' &&
    !Array.isArray(widgetInstances) &&
    Object.keys(widgetInstances).length > 0 &&
    Object.entries(widgetInstances).every(([instanceId, instance]) => isStoredWidgetInstance(instanceId, instance)) &&
    !!snapshot.widgetRegions &&
    typeof snapshot.widgetRegions === 'object' &&
    isWidgetRegionState(snapshot.widgetRegions.left, widgetInstances) &&
    isWidgetRegionState(snapshot.widgetRegions.right, widgetInstances) &&
    isWidgetRegionState(snapshot.widgetRegions.bottom, widgetInstances) &&
    isWidgetRegionState(snapshot.widgetRegions.center, widgetInstances)
  );
};

/** A stored structure as this build reads it: retired widgets stripped, then canonical (`normalizeStructure`). */
export const readStoredStructure = (snapshot: LayoutPresetSnapshot): WorkspaceStructure => {
  const { widgetInstances, widgetRegions } = stripRetiredWidgets(
    cloneLayoutPresetWidgetRegions(snapshot.widgetRegions),
    snapshot.widgetInstances,
    snapshot.floatingWidgets
  );
  return normalizeStructure({
    floatingWidgets: snapshot.floatingWidgets,
    layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
    widgetInstances: Object.fromEntries(
      Object.entries(widgetInstances).map(([instanceId, instance]) => [instanceId, { ...instance }])
    ),
    widgetRegions,
  });
};

/** A project's widget instances without those of retired types; the same record when it holds none. */
export const withoutRetiredTypes = <Instance extends { typeId: string }>(
  widgetInstances: Record<WidgetInstanceId, Instance>
): Record<WidgetInstanceId, Instance> =>
  Object.values(widgetInstances).some((instance) => RETIRED_WIDGET_TYPE_IDS.has(instance.typeId))
    ? Object.fromEntries(
        Object.entries(widgetInstances).filter(([, instance]) => !RETIRED_WIDGET_TYPE_IDS.has(instance.typeId))
      )
    : widgetInstances;
