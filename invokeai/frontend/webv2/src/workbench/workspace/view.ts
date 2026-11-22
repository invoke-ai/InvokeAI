import type {
  FloatingWidgetGeometry,
  FloatingWidgetMode,
  FloatingWidgetState,
  LayoutPresetId,
  WidgetRegion,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { clampSizeToMinimum, nextStackOrder } from '@workbench/floatingWindows';
import { isWidgetRegion } from '@workbench/layoutContracts';

import type { LayoutAvailability } from './resolveLayout';
import type { SideRegion, WorkspaceStructure } from './structure';

import { clampRegionSize, getRegionDefaults, SIDE_REGIONS } from './structure';
import { getRegionFront } from './workspaceDerivations';

/** What a project may change about one of a workspace's windows; where it docks back is structure. */
export type WorkspaceWindowView = Partial<
  Pick<FloatingWidgetState, 'heightPx' | 'mode' | 'stackOrder' | 'widthPx' | 'x' | 'y'>
>;

/**
 * How one project looks at one workspace: sparse overrides of the structure's defaults. An override equal to its
 * default is never stored, except a side region's front of an instance it shares with the center
 * (`isSharedWithCenter`): there it records that the user asked for it in that region, which repair 4 of
 * `resolveLayout` lets win over the center's home front. `undefined` is the empty view.
 */
export interface WorkspaceView {
  fronts?: Partial<Record<WidgetRegion, WidgetInstanceId>>;
  sizes?: Partial<Record<SideRegion, number>>;
  collapsed?: Partial<Record<SideRegion, boolean>>;
  windows?: Record<WidgetInstanceId, WorkspaceWindowView>;
}

/** A project's views by workspace; a workspace it never changed the look of has none. */
export type WorkspaceViews = Partial<Record<LayoutPresetId, WorkspaceView>>;

const WINDOW_FIELDS = ['heightPx', 'mode', 'stackOrder', 'widthPx', 'x', 'y'] as const;
const FLOATING_WIDGET_MODES: ReadonlySet<unknown> = new Set<FloatingWidgetMode>(['windowed', 'maximized', 'shaded']);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const isSideRegion = (value: string): value is SideRegion => (SIDE_REGIONS as readonly string[]).includes(value);

const compact = <Value extends object>(record: Value): Value | undefined =>
  Object.keys(record).length > 0 ? record : undefined;

const normalizeWindow = (value: unknown): WorkspaceWindowView | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const window: WorkspaceWindowView = {};

  for (const field of WINDOW_FIELDS) {
    const fieldValue = value[field];

    if (field === 'mode' ? FLOATING_WIDGET_MODES.has(fieldValue) : isFiniteNumber(fieldValue)) {
      Object.assign(window, { [field]: fieldValue });
    }
  }

  return compact(window);
};

const normalizeEntries = <Value>(
  value: unknown,
  acceptKey: (key: string) => boolean,
  readValue: (entry: unknown) => Value | undefined
): Record<string, Value> | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const entries = Object.entries(value).flatMap(([key, entry]) => {
    const read = acceptKey(key) ? readValue(entry) : undefined;

    return read === undefined ? [] : [[key, read] as const];
  });

  return compact(Object.fromEntries(entries));
};

/** Reads a stored view as untrusted data: unknown regions and ill-typed values are dropped. Never throws. */
export const normalizeView = (value: unknown): WorkspaceView | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const fronts = normalizeEntries(value.fronts, isWidgetRegion, (entry) =>
    typeof entry === 'string' && entry.length > 0 ? entry : undefined
  );
  const sizes = normalizeEntries(value.sizes, isSideRegion, (entry) =>
    isFiniteNumber(entry) && entry >= 0 ? entry : undefined
  );
  const collapsed = normalizeEntries(value.collapsed, isSideRegion, (entry) =>
    typeof entry === 'boolean' ? entry : undefined
  );
  const windows = normalizeEntries(value.windows, (key) => key.length > 0, normalizeWindow);

  return compact({
    ...(collapsed ? { collapsed } : {}),
    ...(fronts ? { fronts } : {}),
    ...(sizes ? { sizes } : {}),
    ...(windows ? { windows } : {}),
  });
};

/** A project's views by workspace, read as untrusted data; views of workspaces the account lacks are dropped. */
export const normalizeViews = (
  value: unknown,
  hasWorkspace: (workspaceId: LayoutPresetId) => boolean
): WorkspaceViews => normalizeEntries(value, hasWorkspace, normalizeView) ?? {};

/** Sets or, with `undefined`, drops (resets) one workspace's view; the same map when nothing changes. */
export const withView = (
  views: WorkspaceViews,
  workspaceId: LayoutPresetId,
  view: WorkspaceView | undefined
): WorkspaceViews => {
  if (views[workspaceId] === view) {
    return views;
  }

  const { [workspaceId]: _previous, ...others } = views;

  return view === undefined ? others : { ...others, [workspaceId]: view };
};

type ViewSection = 'collapsed' | 'fronts' | 'sizes' | 'windows';

/** Writes one override, or removes it when it equals its default; the same view when nothing changes. */
const setOverride = (
  view: WorkspaceView | undefined,
  section: ViewSection,
  key: string,
  value: boolean | number | string | WorkspaceWindowView | undefined
): WorkspaceView | undefined => {
  const current = view?.[section] as Record<string, unknown> | undefined;

  if (current?.[key] === value) {
    return view;
  }

  const { [key]: _previous, ...others } = current ?? {};
  const next = compact(value === undefined ? others : { ...others, [key]: value });
  const { [section]: _section, ...rest } = view ?? {};

  return compact(next ? { ...rest, [section]: next } : rest);
};

/** The registry's answer the shared-instance rule reads: whether a type renders its own surface in a region. */
export type SurfaceAvailability = Pick<LayoutAvailability, 'hasDistinctSurface'>;

/**
 * The shared-instance rule's one predicate: an instance of this type fronted in the center and in side `region` shows
 * once, unless the type renders a distinct surface in each (a workflow's graph and its form).
 */
export const showsOnceWithCenter = (typeId: WidgetTypeId, region: WidgetRegion, surfaces: SurfaceAvailability) =>
  region !== 'center' &&
  !(surfaces.hasDistinctSurface(typeId, 'center') && surfaces.hasDistinctSurface(typeId, region));

/** Whether the structure places the instance in the center and in side `region`, where it shows once. */
export const isSharedWithCenter = (
  structure: WorkspaceStructure,
  region: WidgetRegion,
  instanceId: WidgetInstanceId,
  surfaces: SurfaceAvailability
): boolean => {
  const typeId = structure.widgetInstances[instanceId]?.typeId;

  return (
    typeId !== undefined &&
    structure.widgetRegions.center.instanceIds.includes(instanceId) &&
    structure.widgetRegions[region].instanceIds.includes(instanceId) &&
    showsOnceWithCenter(typeId, region, surfaces)
  );
};

/**
 * Writes a region's front as a stored view holds it, asking nothing of other regions: a home front is no override,
 * unless shared with the center (above).
 */
export const setFrontOverride = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  region: WidgetRegion,
  instanceId: WidgetInstanceId,
  surfaces: SurfaceAvailability
): WorkspaceView | undefined => {
  if (!structure.widgetRegions[region].instanceIds.includes(instanceId)) {
    return view;
  }

  const isDefault =
    getRegionFront(structure.widgetRegions[region]) === instanceId &&
    !isSharedWithCenter(structure, region, instanceId, surfaces);

  return setOverride(view, 'fronts', region, isDefault ? undefined : instanceId);
};

/**
 * Fronts a member of the region; never adds one. For an instance shared with the center the latest ask wins: fronting
 * it in the center drops every side region's explicit front of it, and fronting it in a side region drops the
 * center's.
 */
export const frontRegion = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  region: WidgetRegion,
  instanceId: WidgetInstanceId,
  surfaces: SurfaceAvailability
): WorkspaceView | undefined => {
  if (!structure.widgetRegions[region].instanceIds.includes(instanceId)) {
    return view;
  }

  const released =
    region === 'center'
      ? SIDE_REGIONS.reduce(
          (next, side) =>
            next?.fronts?.[side] === instanceId && isSharedWithCenter(structure, side, instanceId, surfaces)
              ? setOverride(next, 'fronts', side, undefined)
              : next,
          view
        )
      : view?.fronts?.center === instanceId && isSharedWithCenter(structure, region, instanceId, surfaces)
        ? setOverride(view, 'fronts', 'center', undefined)
        : view;

  return setFrontOverride(released, structure, region, instanceId, surfaces);
};

export const setRegionSize = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  region: WidgetRegion,
  sizePx: number
): WorkspaceView | undefined => {
  if (region === 'center' || !Number.isFinite(sizePx)) {
    return view;
  }

  const clamped = clampRegionSize(region, sizePx);

  return setOverride(
    view,
    'sizes',
    region,
    clamped === clampRegionSize(region, getRegionDefaults(structure, region).sizePx) ? undefined : clamped
  );
};

export const setRegionCollapsed = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  region: WidgetRegion,
  collapsed: boolean
): WorkspaceView | undefined =>
  region === 'center'
    ? view
    : setOverride(
        view,
        'collapsed',
        region,
        collapsed === getRegionDefaults(structure, region).collapsed ? undefined : collapsed
      );

/** Overrides fields of one of the structure's windows; fields equal to the window's defaults are dropped. */
export const setWindowView = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId,
  patch: WorkspaceWindowView
): WorkspaceView | undefined => {
  const placed = structure.floatingWidgets?.[instanceId];

  if (!placed) {
    return view;
  }

  const current = view?.windows?.[instanceId];
  const merged = { ...current, ...patch };
  const next: WorkspaceWindowView = {};

  for (const field of WINDOW_FIELDS) {
    const value = merged[field];

    if (value !== undefined && value !== placed[field]) {
      Object.assign(next, { [field]: value });
    }
  }

  const isUnchanged =
    WINDOW_FIELDS.every((field) => next[field] === current?.[field]) &&
    Object.keys(next).length === Object.keys(current ?? {}).length;

  return isUnchanged ? view : setOverride(view, 'windows', instanceId, compact(next));
};

export const setWindowGeometry = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId,
  geometry: FloatingWidgetGeometry
): WorkspaceView | undefined => setWindowView(view, structure, instanceId, clampSizeToMinimum(geometry));

export const setWindowMode = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId,
  mode: FloatingWidgetMode
): WorkspaceView | undefined => setWindowView(view, structure, instanceId, { mode });

/** A window as this view shows it, before the resolver's clamps; undefined when the structure has no such window. */
export const getViewedWindow = (
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  instanceId: WidgetInstanceId
): FloatingWidgetState | undefined => {
  const placed = structure.floatingWidgets?.[instanceId];

  return placed && { ...placed, ...view?.windows?.[instanceId] };
};

/** Brings a window to the top, compacting every window's stack order to 1..N as the reducer does. */
export const raiseWindow = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId
): WorkspaceView | undefined => {
  const windows = Object.fromEntries(
    Object.keys(structure.floatingWidgets ?? {}).map((id) => [id, getViewedWindow(structure, view, id)!])
  );
  const raised = windows[instanceId];

  if (!raised || raised.stackOrder === nextStackOrder(windows) - 1) {
    return view;
  }

  const below = Object.entries(windows)
    .filter(([id]) => id !== instanceId)
    .sort(([, left], [, right]) => left.stackOrder - right.stackOrder);

  return [...below.map(([id]) => id), instanceId].reduce<WorkspaceView | undefined>(
    (next, id, index) => setWindowView(next, structure, id, { stackOrder: index + 1 }),
    view
  );
};

const areSectionsEqual = (left: object | undefined, right: object | undefined): boolean => {
  const leftEntries = Object.entries(left ?? {});

  return (
    leftEntries.length === Object.keys(right ?? {}).length &&
    leftEntries.every(([key, value]) => {
      const other = (right as Record<string, unknown> | undefined)?.[key];

      return typeof value === 'object'
        ? areSectionsEqual(value as object, other as object | undefined)
        : value === other;
    })
  );
};

const areViewsEqual = (left: WorkspaceView | undefined, right: WorkspaceView | undefined): boolean =>
  left === right ||
  (['collapsed', 'fronts', 'sizes', 'windows'] as const).every((section) =>
    areSectionsEqual(left?.[section], right?.[section])
  );

/**
 * Drops what no longer applies to the structure: fronts of non-members, windows it does not place, and overrides
 * equal to its defaults (after Customize changed them). The same view when nothing changes.
 */
export const pruneView = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  surfaces: SurfaceAvailability
): WorkspaceView | undefined => {
  if (!view) {
    return view;
  }

  let next: WorkspaceView | undefined = view;

  for (const [region, instanceId] of Object.entries(view.fronts ?? {})) {
    next = structure.widgetRegions[region as WidgetRegion].instanceIds.includes(instanceId)
      ? setFrontOverride(next, structure, region as WidgetRegion, instanceId, surfaces)
      : setOverride(next, 'fronts', region, undefined);
  }
  for (const [region, sizePx] of Object.entries(view.sizes ?? {})) {
    next = setRegionSize(next, structure, region as SideRegion, sizePx);
  }
  for (const [region, collapsed] of Object.entries(view.collapsed ?? {})) {
    next = setRegionCollapsed(next, structure, region as SideRegion, collapsed);
  }
  for (const [instanceId, window] of Object.entries(view.windows ?? {})) {
    next = structure.floatingWidgets?.[instanceId]
      ? setWindowView(setOverride(next, 'windows', instanceId, undefined), structure, instanceId, window)
      : setOverride(next, 'windows', instanceId, undefined);
  }

  return areViewsEqual(next, view) ? view : next;
};
