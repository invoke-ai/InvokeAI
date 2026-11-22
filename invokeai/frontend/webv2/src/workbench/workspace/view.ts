import type {
  FloatingWidgetGeometry,
  FloatingWidgetMode,
  FloatingWidgetState,
  LayoutPresetId,
  WidgetRegion,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { cascadeDefaultGeometry, clampSizeToMinimum } from '@workbench/floatingWindows';
import { isWidgetRegion } from '@workbench/layoutContracts';

import type { LayoutAvailability } from './resolveLayout';
import type { SideRegion, WorkspaceStructure } from './structure';

import { clampRegionSize, getRegionDefaults, SIDE_REGIONS } from './structure';
import { getRegionFront } from './workspaceDerivations';

/** What a project may change about one of a workspace's windows; where it docks back is structure. */
export type WorkspaceWindowView = Partial<
  Pick<FloatingWidgetState, 'heightPx' | 'mode' | 'stackOrder' | 'widthPx' | 'x' | 'y'>
>;

/** Where a guest shows: fronted in the center (Away), or in a window. */
export type GuestSurface = 'center' | 'window';

/** A widget the workspace does not place, shown by this project; its window's geometry lives in `windows`. */
export interface WorkspaceGuest {
  typeId: WidgetTypeId;
  surface: GuestSurface;
}

/**
 * At most one guest per widget type and one in the center, so a view holds no more guests than there are types; a
 * stored view is cut to this many on load.
 */
export const MAX_GUESTS = 12;

/**
 * How one project looks at one workspace: sparse overrides of the structure's defaults, and its guests. An override
 * equal to its default is never stored, except a side region's front of an instance it shares with the center
 * (`isSharedWithCenter`): there it records that the user asked for it in that region, which repair 4 of
 * `resolveLayout` lets win over the center's home front. A guest window has no default, so its entry in `windows` is
 * whole. `undefined` is the empty view.
 */
export interface WorkspaceView {
  fronts?: Partial<Record<WidgetRegion, WidgetInstanceId>>;
  sizes?: Partial<Record<SideRegion, number>>;
  collapsed?: Partial<Record<SideRegion, boolean>>;
  windows?: Record<WidgetInstanceId, WorkspaceWindowView>;
  guests?: Record<WidgetInstanceId, WorkspaceGuest>;
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

const GUEST_SURFACES: ReadonlySet<unknown> = new Set<GuestSurface>(['center', 'window']);

/** One guest per type, one in the center, at most `MAX_GUESTS`; the first stored wins. */
const normalizeGuests = (value: unknown): Record<WidgetInstanceId, WorkspaceGuest> | undefined => {
  const read = normalizeEntries(
    value,
    (key) => key.length > 0,
    (entry): WorkspaceGuest | undefined =>
      isRecord(entry) &&
      typeof entry.typeId === 'string' &&
      entry.typeId.length > 0 &&
      GUEST_SURFACES.has(entry.surface)
        ? { surface: entry.surface as GuestSurface, typeId: entry.typeId }
        : undefined
  );
  const types = new Set<WidgetTypeId>();
  let hasCenter = false;

  return compact(
    Object.fromEntries(
      Object.entries(read ?? {})
        .filter(([, guest]) => {
          if (types.has(guest.typeId) || (guest.surface === 'center' && hasCenter)) {
            return false;
          }

          types.add(guest.typeId);
          hasCenter ||= guest.surface === 'center';

          return true;
        })
        .slice(0, MAX_GUESTS)
    )
  );
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
  const guests = normalizeGuests(value.guests);

  return compact({
    ...(collapsed ? { collapsed } : {}),
    ...(fronts ? { fronts } : {}),
    ...(guests ? { guests } : {}),
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

type ViewSection = 'collapsed' | 'fronts' | 'guests' | 'sizes' | 'windows';

/** Writes one override, or removes it when it equals its default; the same view when nothing changes. */
const setOverride = (
  view: WorkspaceView | undefined,
  section: ViewSection,
  key: string,
  value: boolean | number | string | WorkspaceGuest | WorkspaceWindowView | undefined
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
 * Fronts a member of the region; never adds one. Fronting the center forgets its guest. For an instance shared with
 * the center the latest ask wins: fronting it in the center drops every side region's explicit front of it, and
 * fronting it in a side region drops the center's.
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
          withoutCenterGuest(view)
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

const isWindowGuest = (view: WorkspaceView | undefined, instanceId: WidgetInstanceId): boolean =>
  view?.guests?.[instanceId]?.surface === 'window';

/** What a guest window is before it was moved: where a first float lands. */
const GUEST_WINDOW: Required<WorkspaceWindowView> = { ...cascadeDefaultGeometry(0), mode: 'windowed', stackOrder: 0 };

/**
 * Overrides fields of one of the structure's windows, fields equal to its defaults dropped, or of a guest window,
 * which has no defaults.
 */
export const setWindowView = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId,
  patch: WorkspaceWindowView
): WorkspaceView | undefined => {
  const placed = structure.floatingWidgets?.[instanceId];

  if (!placed && !isWindowGuest(view, instanceId)) {
    return view;
  }

  const current = view?.windows?.[instanceId];
  const merged = { ...current, ...patch };
  const next: WorkspaceWindowView = {};

  for (const field of WINDOW_FIELDS) {
    const value = merged[field];

    if (value !== undefined && value !== placed?.[field]) {
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

/** A structure window or a guest window as this view shows it, before the resolver's clamps. */
export const getWindowView = (
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  instanceId: WidgetInstanceId
): Required<WorkspaceWindowView> | undefined => {
  const placed = getViewedWindow(structure, view, instanceId);

  if (placed) {
    const { heightPx, mode, stackOrder, widthPx, x, y } = placed;

    return { heightPx, mode, stackOrder, widthPx, x, y };
  }

  return isWindowGuest(view, instanceId) ? { ...GUEST_WINDOW, ...view?.windows?.[instanceId] } : undefined;
};

/** Every window this view can show, guests included, by instance. */
const getWindowViews = (
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined
): Record<WidgetInstanceId, Required<WorkspaceWindowView>> =>
  Object.fromEntries(
    [
      ...Object.keys(structure.floatingWidgets ?? {}),
      ...Object.keys(view?.guests ?? {}).filter((instanceId) => isWindowGuest(view, instanceId)),
    ].map((instanceId) => [instanceId, getWindowView(structure, view, instanceId)!])
  );

/** Brings a window to the top, compacting every window's stack order to 1..N as the reducer does. */
export const raiseWindow = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId
): WorkspaceView | undefined => {
  const windows = getWindowViews(structure, view);
  const raised = windows[instanceId];
  const top = Math.max(0, ...Object.values(windows).map((window) => window.stackOrder));

  if (!raised || raised.stackOrder === top) {
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

/**
 * The side region a guest window of the type stands for, where "Keep in workspace" docks it: the first of left,
 * right and bottom that shows it as a panel. Null when no side region does, so it cannot be a guest window.
 */
export const getGuestWindowRegion = (
  getPresentation: LayoutAvailability['getPresentation'],
  typeId: WidgetTypeId
): SideRegion | null => SIDE_REGIONS.find((region) => getPresentation(typeId, region) === 'view') ?? null;

/** What `isGuestSuperseded` reads from the registry: where a type shows, and where it has a surface of its own. */
export type GuestAvailability = Pick<LayoutAvailability, 'getPresentation' | 'hasDistinctSurface'>;

/** The surface a side region or window standing for it shows a type on: the region's own, else the type's panel. */
const getPanelSurface = (typeId: WidgetTypeId, region: WidgetRegion, availability: GuestAvailability): string =>
  availability.hasDistinctSurface(typeId, region) ? region : 'panel';

/**
 * Whether the structure supersedes a guest, which is then neither shown (repair 5 of `resolveLayout`), kept
 * (`pruneView`) nor opened (`showGuest`): the structure uses its instance id, or shows its type where the guest does. That is the center for a
 * center guest, and for a guest window any side region or window that shows the type's same surface: any at all, but
 * for a type with a surface per region (`graphBearing.surfaces`) only those of the region the window stands for (a
 * workflow's form in a window is not its graph in the status bar's panel).
 */
export const isGuestSuperseded = (
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId,
  { surface, typeId }: WorkspaceGuest,
  availability: GuestAvailability
): boolean => {
  if (Object.hasOwn(structure.widgetInstances, instanceId)) {
    return true;
  }

  const isType = (id: WidgetInstanceId) => structure.widgetInstances[id]?.typeId === typeId;
  // Only what the resolver shows supersedes: a placement repair 1 hides (the type cannot show there) does not.
  const showsIn = (region: WidgetRegion) => availability.getPresentation(typeId, region) !== null;

  if (surface === 'center') {
    return showsIn('center') && structure.widgetRegions.center.instanceIds.some(isType);
  }

  const region = getGuestWindowRegion(availability.getPresentation, typeId);

  if (region === null) {
    return false;
  }

  const guestSurface = getPanelSurface(typeId, region, availability);

  return (
    SIDE_REGIONS.some(
      (side) =>
        showsIn(side) &&
        structure.widgetRegions[side].instanceIds.some(isType) &&
        getPanelSurface(typeId, side, availability) === guestSurface
    ) ||
    Object.entries(structure.floatingWidgets ?? {}).some(
      ([id, window]) =>
        isType(id) &&
        showsIn(window.returnRegion) &&
        getPanelSurface(typeId, window.returnRegion, availability) === guestSurface
    )
  );
};

/**
 * The instance a guest of the type is: one of the project's own instances of the type that this workspace's structure
 * does not use, the one named after the type first (what graph builders and recall read by type), then the others in
 * id order; with none, a new `{type}:guest`.
 */
export const getGuestInstanceId = (
  structure: WorkspaceStructure,
  typeId: WidgetTypeId,
  projectInstanceIds: readonly WidgetInstanceId[]
): WidgetInstanceId => {
  const isFree = (id: WidgetInstanceId) => !Object.hasOwn(structure.widgetInstances, id);
  const owned = [...projectInstanceIds].sort((left, right) =>
    left === typeId ? -1 : right === typeId ? 1 : left < right ? -1 : left > right ? 1 : 0
  );
  const minted = [
    `${typeId}:guest`,
    ...Array.from({ length: MAX_GUESTS }, (_, index) => `${typeId}:guest-${index + 2}`),
  ];

  return owned.find(isFree) ?? minted.find(isFree) ?? `${typeId}:guest`;
};

/** Forgets a guest and its window. */
export const removeGuest = (
  view: WorkspaceView | undefined,
  instanceId: WidgetInstanceId
): WorkspaceView | undefined =>
  view?.guests?.[instanceId]
    ? setOverride(setOverride(view, 'guests', instanceId, undefined), 'windows', instanceId, undefined)
    : view;

const withoutCenterGuest = (view: WorkspaceView | undefined): WorkspaceView | undefined =>
  Object.entries(view?.guests ?? {})
    .filter(([, guest]) => guest.surface === 'center')
    .reduce((next, [instanceId]) => removeGuest(next, instanceId), view);

/**
 * Shows a guest: a center guest replaces the center's, a window guest opens on top at `geometry` (or comes to the top
 * when already open). Another guest of the same type goes, and past `MAX_GUESTS` the lowest window does.
 */
export const showGuest = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  instanceId: WidgetInstanceId,
  guest: WorkspaceGuest,
  geometry: FloatingWidgetGeometry
): WorkspaceView | undefined => {
  const current = view?.guests?.[instanceId];

  if (current?.typeId === guest.typeId && current.surface === guest.surface) {
    return guest.surface === 'window'
      ? setWindowMode(raiseWindow(view, structure, instanceId), structure, instanceId, 'windowed')
      : view;
  }

  let next = Object.entries(view?.guests ?? {})
    .filter(([id, other]) => id === instanceId || other.typeId === guest.typeId)
    .reduce((cleared, [id]) => removeGuest(cleared, id), guest.surface === 'center' ? withoutCenterGuest(view) : view);
  const windows = getWindowViews(structure, next);
  const guestWindows = Object.keys(next?.guests ?? {})
    .filter((id) => isWindowGuest(next, id))
    .sort((left, right) => windows[left]!.stackOrder - windows[right]!.stackOrder);

  if (Object.keys(next?.guests ?? {}).length >= MAX_GUESTS && guestWindows[0] !== undefined) {
    next = removeGuest(next, guestWindows[0]);
  }

  next = setOverride(next, 'guests', instanceId, { surface: guest.surface, typeId: guest.typeId });

  if (guest.surface === 'center') {
    return next;
  }

  const top = Math.max(0, ...Object.values(getWindowViews(structure, next)).map((window) => window.stackOrder));

  return setWindowView(next, structure, instanceId, {
    ...clampSizeToMinimum(geometry),
    mode: 'windowed',
    stackOrder: top + 1,
  });
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
  (['collapsed', 'fronts', 'guests', 'sizes', 'windows'] as const).every((section) =>
    areSectionsEqual(left?.[section], right?.[section])
  );

/**
 * Drops what no longer applies to the structure: fronts of non-members, windows that are neither its own nor a
 * guest's, guests it supersedes (`isGuestSuperseded`), and overrides equal to its defaults
 * (after Customize changed them). The same view when nothing changes.
 */
export const pruneView = (
  view: WorkspaceView | undefined,
  structure: WorkspaceStructure,
  availability: GuestAvailability
): WorkspaceView | undefined => {
  if (!view) {
    return view;
  }

  // A kept guest's window entry stays: it is the placed window's view now.
  let next: WorkspaceView | undefined = Object.entries(view.guests ?? {})
    .filter(([instanceId, guest]) => isGuestSuperseded(structure, instanceId, guest, availability))
    .reduce<WorkspaceView | undefined>(
      (pruned, [instanceId]) => setOverride(pruned, 'guests', instanceId, undefined),
      view
    );

  for (const [region, instanceId] of Object.entries(view.fronts ?? {})) {
    next = structure.widgetRegions[region as WidgetRegion].instanceIds.includes(instanceId)
      ? setFrontOverride(next, structure, region as WidgetRegion, instanceId, availability)
      : setOverride(next, 'fronts', region, undefined);
  }
  for (const [region, sizePx] of Object.entries(view.sizes ?? {})) {
    next = setRegionSize(next, structure, region as SideRegion, sizePx);
  }
  for (const [region, collapsed] of Object.entries(view.collapsed ?? {})) {
    next = setRegionCollapsed(next, structure, region as SideRegion, collapsed);
  }
  for (const [instanceId, window] of Object.entries(next?.windows ?? {})) {
    if (structure.floatingWidgets?.[instanceId]) {
      next = setWindowView(setOverride(next, 'windows', instanceId, undefined), structure, instanceId, window);
    } else if (!isWindowGuest(next, instanceId)) {
      next = setOverride(next, 'windows', instanceId, undefined);
    }
  }

  return areViewsEqual(next, view) ? view : next;
};
