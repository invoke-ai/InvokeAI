import type {
  FloatingWidgetGeometry,
  FloatingWidgetMembership,
  FloatingWidgetMode,
  FloatingWidgetState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { isWidgetRegion, WIDGET_REGIONS } from '@workbench/layoutContracts';

/** Pure rules; the reducer never reads window dimensions, components clamp against the live viewport. */

export const FLOATING_DEFAULT_WIDTH_PX = 520;
export const FLOATING_DEFAULT_HEIGHT_PX = 440;
export const FLOATING_MIN_WIDTH_PX = 280;
export const FLOATING_MIN_HEIGHT_PX = 200;

export const FLOATING_VIEWPORT_MARGIN_PX = 48;
const CASCADE_ORIGIN_PX = 96;
const CASCADE_STEP_PX = 32;
const CASCADE_WRAP = 8;

export type FloatingGeometry = FloatingWidgetGeometry;

export const nextStackOrder = (floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined): number =>
  Object.values(floatingWidgets ?? {}).reduce((max, state) => Math.max(max, state.stackOrder), 0) + 1;

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

/** The displayed width, never wider than the viewport, is what must stay reachable. */
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
 * Commits only axes the gesture visibly changed; an untouched axis may show a clamp the user never chose. Null when
 * nothing changed on screen. `viewport` must be in the same fractional pixels CSS holds `start` in.
 */
export const commitChangedAxes = (
  stored: FloatingGeometry,
  start: FloatingGeometry,
  changedTo: FloatingGeometry,
  viewport: { width: number; height: number }
): FloatingGeometry | null => {
  // Sub-half-pixel differences are arithmetic, not the user; changes are read before the clamp.
  const changed = (from: number, to: number) => Math.abs(to - from) >= 0.5;
  const placed = clampWindowToViewport(changedTo, viewport);
  const horizontal =
    (changed(start.widthPx, changedTo.widthPx) || changed(start.x, changedTo.x)) &&
    (changed(start.widthPx, placed.widthPx) || changed(start.x, placed.x));
  const vertical =
    (changed(start.heightPx, changedTo.heightPx) || changed(start.y, changedTo.y)) &&
    (changed(start.heightPx, placed.heightPx) || changed(start.y, placed.y));

  return horizontal || vertical
    ? {
        heightPx: vertical ? placed.heightPx : stored.heightPx,
        widthPx: horizontal ? placed.widthPx : stored.widthPx,
        x: horizontal ? placed.x : stored.x,
        y: vertical ? placed.y : stored.y,
      }
    : null;
};

export type FloatingResizeEdge = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

/** A viewport smaller than the minimum lowers the minimum to it, or showing it there would read as a resize. */
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

// A floating instance belongs to no region's `instanceIds`; its return region keeps a marker for it instead.

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
 * A marker's `returnIndex` is its position in this order, so docking never moves anything the rail shows. Legacy or
 * colliding indices are read deterministically, not reconstructed.
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

type AlsoIn = NonNullable<FloatingWidgetState['alsoIn']>;

/** Regions in a fixed order, the return region and ill-typed entries left out: one shape for reducers and reloads. */
const normalizeAlsoIn = (value: unknown, returnRegion: WidgetRegion): AlsoIn | undefined => {
  if (!value || typeof value !== 'object') {
    return undefined;
  }

  const entries = WIDGET_REGIONS.flatMap((region): [WidgetRegion, FloatingWidgetMembership][] => {
    const entry = (value as Record<string, unknown>)[region] as Partial<FloatingWidgetMembership> | undefined;

    if (region === returnRegion || !entry || typeof entry !== 'object' || !isFiniteNumber(entry.index)) {
      return [];
    }

    const index = Math.max(0, Math.floor(entry.index));

    return [[region, entry.isFront === true ? { index, isFront: true } : { index }]];
  });

  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
};

/** Persistence compares documents as JSON, key order included: reducers and reloads must serialize alike. */
export const toFloatingWidgetState = ({
  alsoIn,
  heightPx,
  mode,
  returnIndex,
  returnRegion,
  stackOrder,
  widthPx,
  x,
  y,
}: FloatingWidgetState): FloatingWidgetState => {
  const memberships = normalizeAlsoIn(alsoIn, returnRegion);

  return {
    heightPx,
    widthPx,
    x,
    y,
    mode,
    returnIndex,
    returnRegion,
    stackOrder,
    ...(memberships ? { alsoIn: memberships } : {}),
  };
};

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

const FLOATING_WIDGET_MODES: readonly FloatingWidgetMode[] = ['windowed', 'maximized', 'shaded'];

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
      alsoIn: state.alsoIn,
      mode: state.mode as FloatingWidgetMode,
      returnIndex: state.returnIndex,
      returnRegion,
      stackOrder: state.stackOrder,
    });
  }

  return floatingWidgets;
};

/** The one normalization for stored placements, so the same arrangement always reads the same; idempotent. */
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

/** Keeps closed-window memory as {@link normalizeLastFloatingGeometry} would leave it; same object when unchanged. */
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

/** Unlike the sliver policy, a newly floated window arrives wholly usable; CSS caps the shown size. */
export const fitWindowIntoViewport = (
  geometry: FloatingGeometry,
  viewport: { width: number; height: number }
): FloatingGeometry => ({
  ...geometry,
  x: Math.max(0, Math.min(geometry.x, viewport.width - Math.min(geometry.widthPx, viewport.width))),
  y: Math.max(0, Math.min(geometry.y, viewport.height - Math.min(geometry.heightPx, viewport.height))),
});

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
