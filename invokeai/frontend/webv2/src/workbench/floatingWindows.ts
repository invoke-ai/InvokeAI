import type {
  FloatingWidgetGeometry,
  FloatingWidgetMode,
  FloatingWidgetState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { isWidgetRegion, WIDGET_REGIONS } from '@workbench/layoutContracts';

/**
 * Pure geometry, stacking, and placement rules for floating widget windows. The reducer owns the state;
 * components clamp against the live viewport (the reducer never reads window dimensions).
 */

export const FLOATING_DEFAULT_WIDTH_PX = 520;
export const FLOATING_DEFAULT_HEIGHT_PX = 440;
export const FLOATING_MIN_WIDTH_PX = 280;
export const FLOATING_MIN_HEIGHT_PX = 200;

/** Minimum sliver of a window that must stay reachable inside the viewport. */
export const FLOATING_VIEWPORT_MARGIN_PX = 48;
const CASCADE_ORIGIN_PX = 96;
const CASCADE_STEP_PX = 32;
const CASCADE_WRAP = 8;

export type FloatingGeometry = FloatingWidgetGeometry;

export const nextStackOrder = (floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined): number =>
  Object.values(floatingWidgets ?? {}).reduce((max, state) => Math.max(max, state.stackOrder), 0) + 1;

/** Default placement for the Nth window: a classic cascading offset. */
export const cascadeDefaultGeometry = (existingCount: number): FloatingGeometry => {
  const step = (existingCount % CASCADE_WRAP) * CASCADE_STEP_PX;

  return {
    heightPx: FLOATING_DEFAULT_HEIGHT_PX,
    widthPx: FLOATING_DEFAULT_WIDTH_PX,
    x: CASCADE_ORIGIN_PX + step,
    y: CASCADE_ORIGIN_PX + step,
  };
};

export const clampSizeToMinimum = (geometry: FloatingGeometry): FloatingGeometry => ({
  ...geometry,
  heightPx: Math.max(FLOATING_MIN_HEIGHT_PX, geometry.heightPx),
  widthPx: Math.max(FLOATING_MIN_WIDTH_PX, geometry.widthPx),
});

/**
 * Keep at least a grabbable corner of the window inside the viewport so a
 * drag (or a shrunk browser window) can never strand it off-screen. The window is never shown wider than the
 * viewport, so that displayed width — not a larger stored one — is what has to stay reachable.
 */
export const clampWindowToViewport = (
  geometry: FloatingGeometry,
  viewport: { width: number; height: number }
): FloatingGeometry => ({
  ...geometry,
  x: Math.min(
    Math.max(geometry.x, FLOATING_VIEWPORT_MARGIN_PX - Math.min(geometry.widthPx, viewport.width)),
    viewport.width - FLOATING_VIEWPORT_MARGIN_PX
  ),
  y: Math.min(Math.max(geometry.y, 0), Math.max(0, viewport.height - FLOATING_VIEWPORT_MARGIN_PX)),
});

/**
 * What a resize commits: on each axis the resize actually changed, the result from the rectangle on screen; on an
 * axis it left alone, what was stored. An untouched axis may be showing the viewport's clamp or cap, which is not
 * the user's choice to persist. Null when the resize changed nothing — a drag against the minimum or the cap.
 */
export const commitResizedAxes = (
  stored: FloatingGeometry,
  start: FloatingGeometry,
  resized: FloatingGeometry
): FloatingGeometry | null => {
  // The rectangle on screen can sit on fractional pixels, and what is committed is whole ones: a difference under
  // half a pixel is arithmetic, not the user.
  const changed = (from: number, to: number) => Math.abs(to - from) >= 0.5;
  const horizontal = changed(start.widthPx, resized.widthPx) || changed(start.x, resized.x);
  const vertical = changed(start.heightPx, resized.heightPx) || changed(start.y, resized.y);

  return horizontal || vertical
    ? {
        heightPx: vertical ? resized.heightPx : stored.heightPx,
        widthPx: horizontal ? resized.widthPx : stored.widthPx,
        x: horizontal ? resized.x : stored.x,
        y: vertical ? resized.y : stored.y,
      }
    : null;
};

/** A window edge or corner, by compass point. */
export type FloatingResizeEdge = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

/**
 * Resize from one edge or corner by a pointer offset. The opposite side stays where it is, the size stays between
 * the minimum and `max` (the viewport, which is as large as a window is ever shown), and the top edge stops at the
 * viewport's top so the title bar stays reachable. A viewport smaller than the minimum lowers the minimum to it:
 * the window is shown at the viewport's size there, and raising it to the minimum would read as a resize.
 */
export const resizeFloatingGeometry = (
  start: FloatingGeometry,
  edge: FloatingResizeEdge,
  deltaX: number,
  deltaY: number,
  max: { heightPx: number; widthPx: number } = { heightPx: Number.POSITIVE_INFINITY, widthPx: Number.POSITIVE_INFINITY }
): FloatingGeometry => {
  const right = start.x + start.widthPx;
  const bottom = start.y + start.heightPx;
  const clampWidth = (widthPx: number) =>
    Math.max(Math.min(FLOATING_MIN_WIDTH_PX, max.widthPx), Math.min(widthPx, max.widthPx));
  const clampHeight = (heightPx: number) =>
    Math.max(Math.min(FLOATING_MIN_HEIGHT_PX, max.heightPx), Math.min(heightPx, max.heightPx));
  const geometry = { ...start };

  if (edge.includes('e')) {
    geometry.widthPx = clampWidth(start.widthPx + deltaX);
  }
  if (edge.includes('w')) {
    geometry.widthPx = clampWidth(start.widthPx - deltaX);
    geometry.x = right - geometry.widthPx;
  }
  if (edge.includes('s')) {
    geometry.heightPx = clampHeight(start.heightPx + deltaY);
  }
  if (edge.includes('n')) {
    geometry.heightPx = clampHeight(Math.min(start.heightPx - deltaY, bottom));
    geometry.y = bottom - geometry.heightPx;
  }

  return geometry;
};

// Placement: where a floating window returns to. A floating instance belongs to no region's `instanceIds`; its
// return region keeps a marker for it instead, and reducers, normalization, and rails all read that marker's
// position here.

/** The placement half of a window: the slot its return region keeps for it. */
export type FloatingWidgetPlacement = Pick<FloatingWidgetState, 'returnIndex' | 'returnRegion'>;

export interface RegionOrderSlot {
  instanceId: WidgetInstanceId;
  isFloating: boolean;
}

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

// Code-unit order, not `localeCompare`: the result must not vary with the browser's locale.
const compareInstanceIds = (left: WidgetInstanceId, right: WidgetInstanceId): number =>
  left < right ? -1 : left > right ? 1 : 0;

/**
 * One region's complete order: its docked instances plus a marker for each window that returns to it. A marker's
 * `returnIndex` is its position in this order, so docking never moves anything the rail shows.
 *
 * Indices persisted before markers were counted, and missing or colliding ones, are read best-effort: ascending,
 * ties broken by instance id, absent ones appended. That is deterministic, but it does not reconstruct the
 * arrangement those windows were floated from.
 */
export const getRegionOrder = (
  region: WidgetRegion,
  instanceIds: readonly WidgetInstanceId[],
  floatingWidgets: Readonly<Record<WidgetInstanceId, FloatingWidgetPlacement>> | undefined
): RegionOrderSlot[] => {
  const slots: RegionOrderSlot[] = instanceIds.map((instanceId) => ({ instanceId, isFloating: false }));

  if (!floatingWidgets) {
    return slots;
  }

  const markers = Object.entries(floatingWidgets)
    .filter(([instanceId, placement]) => placement.returnRegion === region && !instanceIds.includes(instanceId))
    .map(([instanceId, { returnIndex }]) => ({
      index: isFiniteNumber(returnIndex) && returnIndex >= 0 ? Math.floor(returnIndex) : Number.POSITIVE_INFINITY,
      instanceId,
    }))
    .sort((left, right) => left.index - right.index || compareInstanceIds(left.instanceId, right.instanceId));
  let previousPosition = -1;

  for (const { index, instanceId } of markers) {
    // Each marker lands after the one before it, so colliding indices keep their id order.
    const position = Math.max(Math.min(index, slots.length), previousPosition + 1);

    slots.splice(position, 0, { instanceId, isFloating: true });
    previousPosition = position;
  }

  return slots;
};

/**
 * The one shape a window's state is written in. Persistence compares documents as JSON, key order included, so a
 * window built by a reducer has to serialize exactly as the same window read back by a reload.
 */
export const toFloatingWidgetState = ({
  heightPx,
  mode,
  returnIndex,
  returnRegion,
  stackOrder,
  widthPx,
  x,
  y,
}: FloatingWidgetState): FloatingWidgetState => ({
  heightPx,
  widthPx,
  x,
  y,
  mode,
  returnIndex,
  returnRegion,
  stackOrder,
});

/** Write a complete order back as the region's docked members and its markers' return indices. */
export const writeRegionOrder = (
  slots: readonly RegionOrderSlot[],
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined
): { floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined; instanceIds: WidgetInstanceId[] } => {
  let nextFloatingWidgets = floatingWidgets;

  slots.forEach(({ instanceId, isFloating }, index) => {
    const floating = nextFloatingWidgets?.[instanceId];

    if (isFloating && floating && floating.returnIndex !== index) {
      nextFloatingWidgets = {
        ...nextFloatingWidgets,
        [instanceId]: toFloatingWidgetState({ ...floating, returnIndex: index }),
      };
    }
  });

  return {
    floatingWidgets: nextFloatingWidgets,
    instanceIds: slots.filter((slot) => !slot.isFloating).map((slot) => slot.instanceId),
  };
};

/**
 * Whether the center is still waiting for this instance: it emptied when the instance floated and nothing has
 * taken its place since. Docking or removing the window then gives the center its view back. Emptiness is by
 * membership, not by what can render — state has no widget registry to ask.
 */
export const isAwaitedCenterView = (
  center: Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds'>,
  instanceId: WidgetInstanceId
): boolean => center.instanceIds.length === 0 && center.activeInstanceId === instanceId;

const FLOATING_WIDGET_MODES: readonly FloatingWidgetMode[] = ['windowed', 'maximized', 'shaded'];

/** Drop malformed windows so invalid region names or geometry cannot reach reducers or CSS. */
const validateFloatingWidgets = (
  value: unknown,
  hasInstance: (instanceId: WidgetInstanceId) => boolean
): Record<WidgetInstanceId, FloatingWidgetState> => {
  const floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> = {};

  if (!value || typeof value !== 'object') {
    return floatingWidgets;
  }

  for (const [instanceId, entry] of Object.entries(value as Record<string, unknown>)) {
    if (!entry || typeof entry !== 'object' || !hasInstance(instanceId)) {
      continue;
    }

    const state = entry as Partial<FloatingWidgetState>;
    // The right rail's docks folded back into one region; a window floated out of one returns to the rail.
    const rawReturnRegion: unknown = state.returnRegion;
    const returnRegion =
      rawReturnRegion === 'rightTop' || rawReturnRegion === 'rightBottom' ? 'right' : rawReturnRegion;

    if (
      !isFiniteNumber(state.x) ||
      !isFiniteNumber(state.y) ||
      !isFiniteNumber(state.widthPx) ||
      !isFiniteNumber(state.heightPx) ||
      !isFiniteNumber(state.stackOrder) ||
      !FLOATING_WIDGET_MODES.includes(state.mode as FloatingWidgetMode) ||
      !isWidgetRegion(returnRegion)
    ) {
      continue;
    }

    floatingWidgets[instanceId] = toFloatingWidgetState({
      ...clampSizeToMinimum({ heightPx: state.heightPx, widthPx: state.widthPx, x: state.x, y: state.y }),
      mode: state.mode as FloatingWidgetMode,
      returnIndex: state.returnIndex,
      returnRegion,
      stackOrder: state.stackOrder,
    });
  }

  return floatingWidgets;
};

/**
 * The one normalization for stored placements — hydrated projects, stored presets, applied presets, and preset
 * comparison all pass through it, so the same arrangement always reads the same way. Malformed windows are
 * dropped, a floating instance leaves every region (an emptied center keeps naming it, so it can come back), and
 * every marker's return index becomes its position in {@link getRegionOrder}. Running it twice changes nothing.
 */
export const normalizeFloatingPlacement = (
  widgetRegions: Record<WidgetRegion, WidgetRegionState>,
  storedFloatingWidgets: unknown,
  hasInstance: (instanceId: WidgetInstanceId) => boolean
): {
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
} => {
  let floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined = validateFloatingWidgets(
    storedFloatingWidgets,
    hasInstance
  );

  if (Object.keys(floatingWidgets).length === 0) {
    return { floatingWidgets: undefined, widgetRegions };
  }

  const normalizedRegions = { ...widgetRegions };

  for (const regionId of WIDGET_REGIONS) {
    const region = normalizedRegions[regionId];
    const dockedIds = region.instanceIds.filter((instanceId) => !floatingWidgets?.[instanceId]);
    const order = writeRegionOrder(getRegionOrder(regionId, dockedIds, floatingWidgets), floatingWidgets);

    floatingWidgets = order.floatingWidgets;

    if (dockedIds.length !== region.instanceIds.length) {
      normalizedRegions[regionId] = withoutFloatedInstances(regionId, region, dockedIds);
    }
  }

  return { floatingWidgets, widgetRegions: normalizedRegions };
};

/**
 * A region after some of its instances floated away. The center never collapses and, once empty, keeps naming the
 * view it lost; an emptied rail collapses and names nothing.
 */
export const withoutFloatedInstances = (
  regionId: WidgetRegion,
  region: WidgetRegionState,
  instanceIds: WidgetInstanceId[]
): WidgetRegionState => ({
  ...region,
  activeInstanceId: instanceIds.includes(region.activeInstanceId)
    ? region.activeInstanceId
    : (instanceIds[0] ?? (regionId === 'center' ? region.activeInstanceId : '')),
  instanceIds,
  isCollapsed: instanceIds.length === 0 ? regionId !== 'center' : region.isCollapsed,
});

/** The one shape remembered geometry is written in; persistence compares documents as JSON, key order included. */
const toRememberedGeometry = ({ heightPx, widthPx, x, y }: FloatingWidgetGeometry): FloatingWidgetGeometry => ({
  heightPx,
  widthPx,
  x,
  y,
});

/**
 * Keep the memory of closed windows in step with a change to the open ones, so it always reads the way
 * {@link normalizeLastFloatingGeometry} would leave it: a window that was open and no longer is leaves its
 * geometry behind, one that is open now has none (its geometry is its own state), and an instance that no longer
 * exists is forgotten. Returns the same memory when nothing changed, and undefined when nothing is remembered.
 */
export const rememberClosedWindows = (
  remembered: Record<WidgetInstanceId, FloatingWidgetGeometry> | undefined,
  before: Record<WidgetInstanceId, FloatingWidgetState> | undefined,
  after: Record<WidgetInstanceId, FloatingWidgetState> | undefined,
  hasInstance: (instanceId: WidgetInstanceId) => boolean = () => true
): Record<WidgetInstanceId, FloatingWidgetGeometry> | undefined => {
  const closed = Object.entries(before ?? {}).filter(([instanceId]) => !after?.[instanceId] && hasInstance(instanceId));
  const staleIds = Object.keys(remembered ?? {}).filter(
    (instanceId) => after?.[instanceId] !== undefined || !hasInstance(instanceId)
  );

  if (closed.length === 0 && staleIds.length === 0) {
    return remembered;
  }

  const next = { ...remembered };

  for (const instanceId of staleIds) {
    delete next[instanceId];
  }
  for (const [instanceId, geometry] of closed) {
    next[instanceId] = toRememberedGeometry(geometry);
  }

  return Object.keys(next).length > 0 ? next : undefined;
};

/**
 * Bring a remembered rectangle wholly into the viewport. The sliver policy suits a window that was already open
 * when the viewport shrank; a window the user has just asked to float should arrive where they can use it. The
 * size is left alone: CSS caps what is shown, and the stored size returns when there is room.
 */
export const fitWindowIntoViewport = (
  geometry: FloatingGeometry,
  viewport: { width: number; height: number }
): FloatingGeometry => ({
  ...geometry,
  x: Math.max(0, Math.min(geometry.x, viewport.width - Math.min(geometry.widthPx, viewport.width))),
  y: Math.max(0, Math.min(geometry.y, viewport.height - Math.min(geometry.heightPx, viewport.height))),
});

/**
 * Stored memory of where docked or closed windows last floated. Keeps well-formed geometry for instances that
 * exist and are not floating now — a floating window's geometry is its own state — and reports none as undefined.
 */
export const normalizeLastFloatingGeometry = (
  stored: unknown,
  hasInstance: (instanceId: WidgetInstanceId) => boolean,
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined
): Record<WidgetInstanceId, FloatingWidgetGeometry> | undefined => {
  if (!stored || typeof stored !== 'object') {
    return undefined;
  }

  const remembered: Record<WidgetInstanceId, FloatingWidgetGeometry> = {};

  for (const [instanceId, entry] of Object.entries(stored as Record<string, unknown>)) {
    if (!entry || typeof entry !== 'object' || !hasInstance(instanceId) || floatingWidgets?.[instanceId]) {
      continue;
    }

    const { heightPx, widthPx, x, y } = entry as Partial<FloatingWidgetGeometry>;

    if (isFiniteNumber(x) && isFiniteNumber(y) && isFiniteNumber(widthPx) && isFiniteNumber(heightPx)) {
      remembered[instanceId] = toRememberedGeometry(clampSizeToMinimum({ heightPx, widthPx, x, y }));
    }
  }

  return Object.keys(remembered).length > 0 ? remembered : undefined;
};
