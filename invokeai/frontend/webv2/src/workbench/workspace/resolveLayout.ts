import type {
  FloatingWidgetState,
  LayoutPresetWidgetInstanceSnapshot,
  PanelState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { clampSizeToMinimum, getRegionOrder, toFloatingWidgetState } from '@workbench/floatingWindows';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';

import type { RegionPresentation } from './resolveOpen';
import type { WorkspaceStructure } from './structure';
import type { WorkspaceView } from './view';

import { clampRegionSize, getRegionDefaults, PANEL_OPEN_KEYS, SIDE_REGIONS } from './structure';
import { getViewedWindow, showsOnceWithCenter } from './view';

/** What this account and session can show, from the widget registry; the resolver reads nothing else. */
export interface LayoutAvailability {
  /** Null where the type cannot show: disabled, hidden, admin-only, or not allowed in the region. */
  getPresentation: (typeId: WidgetTypeId, region: WidgetRegion) => RegionPresentation;
  /**
   * Whether the type renders its own surface in the region (its manifest's `graphBearing.surfaces`), so one instance
   * fronted in the center and in that region shows twice by design: a workflow's graph and its form.
   */
  hasDistinctSurface: (typeId: WidgetTypeId, region: WidgetRegion) => boolean;
  /** The default workspace's stage, which a center with no available view shows as a guest. */
  fallbackStage: LayoutPresetWidgetInstanceSnapshot | null;
}

/** What is on screen, in the shapes the shell reads from a project today. */
export interface ResolvedLayout {
  /** Every region shows its front; a side region is shown when not collapsed. */
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined;
  /** Mirrors the side regions' `isCollapsed`: open exactly when not collapsed. */
  panels: PanelState;
  /** Instance identity for everything placed above, guests included. */
  widgetInstances: Readonly<Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot>>;
  /** Shown though the structure does not place them here: the center's fallback stage. */
  guests: readonly WidgetInstanceId[];
}

const areListsEqual = (left: readonly string[] | undefined, right: readonly string[] | undefined): boolean =>
  left === right ||
  (left !== undefined &&
    right !== undefined &&
    left.length === right.length &&
    left.every((value, index) => value === right[index]));

const areRegionsEqual = (left: WidgetRegionState, right: WidgetRegionState | undefined): boolean =>
  right !== undefined &&
  left.activeInstanceId === right.activeInstanceId &&
  left.isCollapsed === right.isCollapsed &&
  left.sizePx === right.sizePx &&
  areListsEqual(left.instanceIds, right.instanceIds) &&
  areListsEqual(left.alignEndInstanceIds, right.alignEndInstanceIds);

const areWindowsEqual = (left: FloatingWidgetState, right: FloatingWidgetState | undefined): boolean =>
  right !== undefined &&
  left.x === right.x &&
  left.y === right.y &&
  left.widthPx === right.widthPx &&
  left.heightPx === right.heightPx &&
  left.mode === right.mode &&
  left.stackOrder === right.stackOrder &&
  left.returnRegion === right.returnRegion &&
  left.returnIndex === right.returnIndex;

/** The first candidate equal to `next`, so an unchanged part keeps the identity its subscribers hold. */
const share = <Value>(
  next: Value,
  isEqual: (left: Value, right: Value | undefined) => boolean,
  ...candidates: (Value | undefined)[]
): Value => candidates.find((candidate) => isEqual(next, candidate)) ?? next;

const shareRecord = <Value>(
  next: Record<string, Value>,
  previous: Record<string, Value> | undefined
): Record<string, Value> =>
  previous !== undefined &&
  Object.keys(previous).length === Object.keys(next).length &&
  Object.entries(next).every(([key, value]) => previous[key] === value)
    ? previous
    : next;

/**
 * What is on screen for a structure seen through a project's view. Pure, total and idempotent; the only repairs:
 * 1. An instance whose type is unavailable in its region is skipped (the structure keeps it).
 * 2. A front that is not an available member falls back to the home front, then the first available member.
 * 3. A center with no available view shows the fallback stage as a guest.
 * 4. An instance is shown once where `showsOnceWithCenter` says so. The center keeps it over a side region's home
 *    front; an open side region the view explicitly fronts it in keeps it instead, unless the view fronts it in the
 *    center too, and the center fronts its next view (with none, both show it). Every other side region holding it
 *    fronts its next member, or collapses.
 * 5. Sizes and window sizes are clamped by the panel and window limits.
 * Parts equal to `previous` (or to the structure's own) are returned as those objects.
 */
export const resolveLayout = (
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  availability: LayoutAvailability,
  previous?: ResolvedLayout
): ResolvedLayout => {
  const storedWindows = structure.floatingWidgets ?? {};
  let widgetInstances = structure.widgetInstances;
  const presentationOf = (instanceId: WidgetInstanceId, region: WidgetRegion): RegionPresentation => {
    const typeId = widgetInstances[instanceId]?.typeId;

    return typeId === undefined ? null : availability.getPresentation(typeId, region);
  };
  const shownWindowIds = new Set(
    Object.entries(storedWindows)
      .filter(([instanceId, window]) => presentationOf(instanceId, window.returnRegion) !== null)
      .map(([instanceId]) => instanceId)
  );
  // A floating instance is never a member; then repair 1.
  const docked = Object.fromEntries(
    WIDGET_REGIONS.map((region) => [
      region,
      [...new Set(structure.widgetRegions[region].instanceIds)].filter(
        (instanceId) => !Object.hasOwn(storedWindows, instanceId)
      ),
    ])
  ) as Record<WidgetRegion, WidgetInstanceId[]>;
  const members = Object.fromEntries(
    WIDGET_REGIONS.map((region) => [
      region,
      docked[region].filter((instanceId) => presentationOf(instanceId, region) !== null),
    ])
  ) as Record<WidgetRegion, WidgetInstanceId[]>;
  const canFront = (region: WidgetRegion, instanceId: WidgetInstanceId | undefined): instanceId is WidgetInstanceId =>
    instanceId !== undefined &&
    members[region].includes(instanceId) &&
    (region !== 'center' || presentationOf(instanceId, 'center') === 'view');
  // Repair 2.
  const fronts = Object.fromEntries(
    WIDGET_REGIONS.map((region) => [
      region,
      [view?.fronts?.[region], structure.widgetRegions[region].activeInstanceId, ...members[region]].find(
        (instanceId) => canFront(region, instanceId)
      ) ?? null,
    ])
  ) as Record<WidgetRegion, WidgetInstanceId | null>;
  const guests: WidgetInstanceId[] = [];
  const stage = availability.fallbackStage;

  // Repair 3: a window already shows the fallback, so it is not shown twice.
  if (
    fronts.center === null &&
    stage !== null &&
    availability.getPresentation(stage.typeId, 'center') === 'view' &&
    !Object.hasOwn(storedWindows, stage.id) &&
    (widgetInstances[stage.id]?.typeId ?? stage.typeId) === stage.typeId
  ) {
    members.center = [...members.center, stage.id];
    fronts.center = stage.id;
    guests.push(stage.id);
    widgetInstances = Object.hasOwn(widgetInstances, stage.id)
      ? widgetInstances
      : { ...widgetInstances, [stage.id]: stage };
  }

  const collapsed = Object.fromEntries(
    WIDGET_REGIONS.map((region) => [
      region,
      region !== 'center' && (view?.collapsed?.[region] ?? getRegionDefaults(structure, region).collapsed),
    ])
  ) as Record<WidgetRegion, boolean>;
  const shared = fronts.center;
  const sharedTypeId = shared === null ? undefined : widgetInstances[shared]?.typeId;

  // Repair 4.
  if (shared !== null && sharedTypeId !== undefined) {
    const sharing = SIDE_REGIONS.filter(
      (region) =>
        fronts[region] === shared &&
        presentationOf(shared, region) === 'view' &&
        showsOnceWithCenter(sharedTypeId, region, availability)
    );
    const holder =
      view?.fronts?.center === shared
        ? undefined
        : sharing.find((region) => view?.fronts?.[region] === shared && !collapsed[region]);

    if (holder !== undefined) {
      const sideFronts = SIDE_REGIONS.map((region) => fronts[region]);
      const alternative = members.center.find(
        (instanceId) => instanceId !== shared && canFront('center', instanceId) && !sideFronts.includes(instanceId)
      );

      if (alternative !== undefined) {
        fronts.center = alternative;
      }
    }

    for (const region of sharing.filter((candidate) => candidate !== holder)) {
      const neighbour = members[region].find((instanceId) => instanceId !== shared);

      if (neighbour === undefined) {
        collapsed[region] = true;
      } else {
        fronts[region] = neighbour;
      }
    }
  }

  const widgetRegions = Object.fromEntries(
    WIDGET_REGIONS.map((region) => {
      const stored = structure.widgetRegions[region];
      const instanceIds = members[region];
      const alignEnd = stored.alignEndInstanceIds;
      const next: WidgetRegionState = {
        // The documented empty center keeps naming what it lost; an empty side region fronts nothing.
        activeInstanceId: fronts[region] ?? instanceIds[0] ?? (region === 'center' ? stored.activeInstanceId : ''),
        ...(alignEnd ? { alignEndInstanceIds: alignEnd.filter((instanceId) => instanceIds.includes(instanceId)) } : {}),
        instanceIds,
        isCollapsed: collapsed[region],
        // Repair 5.
        sizePx: region === 'center' ? stored.sizePx : clampRegionSize(region, view?.sizes?.[region] ?? stored.sizePx),
      };

      return [region, share(next, areRegionsEqual, previous?.widgetRegions[region], stored)];
    })
  ) as Record<WidgetRegion, WidgetRegionState>;
  const floatingWidgets = resolveWindows(structure, view, docked, members, shownWindowIds, previous?.floatingWidgets);
  const panels: PanelState = {
    isBottomOpen: !collapsed.bottom,
    isLeftOpen: !collapsed.left,
    isRightOpen: !collapsed.right,
  };
  const next: ResolvedLayout = {
    floatingWidgets,
    guests: share(guests, areListsEqual, previous?.guests),
    panels: share(
      panels,
      (left, right) =>
        right !== undefined &&
        SIDE_REGIONS.every((region) => left[PANEL_OPEN_KEYS[region]] === right[PANEL_OPEN_KEYS[region]]),
      previous?.panels
    ),
    widgetInstances: shareRecord(widgetInstances, previous?.widgetInstances),
    widgetRegions: shareRecord(widgetRegions, previous?.widgetRegions) as Record<WidgetRegion, WidgetRegionState>,
  };

  return previous !== undefined &&
    (Object.keys(next) as (keyof ResolvedLayout)[]).every((key) => next[key] === previous[key])
    ? previous
    : next;
};

/** Shown windows as the view places them, each return index its position among what its region shows. */
const resolveWindows = (
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  docked: Record<WidgetRegion, WidgetInstanceId[]>,
  members: Record<WidgetRegion, WidgetInstanceId[]>,
  shownWindowIds: ReadonlySet<WidgetInstanceId>,
  previous: Record<WidgetInstanceId, FloatingWidgetState> | undefined
): Record<WidgetInstanceId, FloatingWidgetState> | undefined => {
  if (shownWindowIds.size === 0) {
    return undefined;
  }

  const windows: Record<WidgetInstanceId, FloatingWidgetState> = {};

  for (const region of WIDGET_REGIONS) {
    const regionMembers = members[region];
    const slots = getRegionOrder(region, docked[region], structure.floatingWidgets).filter((slot) =>
      slot.isFloating ? shownWindowIds.has(slot.instanceId) : regionMembers.includes(slot.instanceId)
    );

    slots.forEach((slot, returnIndex) => {
      const viewed = slot.isFloating ? getViewedWindow(structure, view, slot.instanceId) : undefined;

      if (viewed) {
        const next = toFloatingWidgetState({ ...viewed, ...clampSizeToMinimum(viewed), returnIndex });

        windows[slot.instanceId] = share(next, areWindowsEqual, previous?.[slot.instanceId]);
      }
    });
  }

  return shareRecord(windows, previous);
};

/**
 * A resolver whose output keeps the identity of every part that did not change since its last call, and the whole
 * layout for identical inputs, so selectors can call it per render. Holds only its last call; the caller owns it.
 */
export const createLayoutResolver = (): ((
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  availability: LayoutAvailability
) => ResolvedLayout) => {
  let last:
    | {
        availability: LayoutAvailability;
        layout: ResolvedLayout;
        structure: WorkspaceStructure;
        view: WorkspaceView | undefined;
      }
    | undefined;

  return (structure, view, availability) => {
    if (last?.structure !== structure || last.view !== view || last.availability !== availability) {
      last = { availability, layout: resolveLayout(structure, view, availability, last?.layout), structure, view };
    }

    return last.layout;
  };
};
