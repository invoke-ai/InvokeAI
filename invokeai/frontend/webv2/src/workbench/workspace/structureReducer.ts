import type {
  FloatingWidgetGeometry,
  FloatingWidgetState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import {
  cascadeDefaultGeometry,
  getRegionOrder,
  nextStackOrder,
  toFloatingWidgetState,
  withoutFloatedInstances,
  writeRegionOrder,
  type RegionOrderSlot,
} from '@workbench/floatingWindows';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';

import type { LayoutAvailability, ResolvedLayout } from './resolveLayout';
import type { WorkspaceStructure } from './structure';
import type { SurfaceAvailability, WorkspaceView } from './view';

import { clampRegionSize, normalizeStructure, PANEL_OPEN_KEYS, SIDE_REGIONS } from './structure';
import { isSharedWithCenter } from './view';

export type StructureAction =
  /**
   * Places a widget: an instance of the type already here (unless `createNew`) — one in the region, else a window of
   * it, which docks, else one placed elsewhere — else the project's unplaced one, else a new one with `mintId`.
   * `index` counts docked members; absent appends.
   */
  | {
      type: 'addWidget';
      region: WidgetRegion;
      typeId: WidgetTypeId;
      mintId: WidgetInstanceId;
      createNew?: boolean;
      index?: number;
      unplacedInstanceId?: WidgetInstanceId;
    }
  | { type: 'removeWidget'; region: WidgetRegion; instanceId: WidgetInstanceId }
  /** `toIndex` counts docked members, wherever the window markers sit. */
  | {
      type: 'moveWidget';
      instanceId: WidgetInstanceId;
      fromRegion: WidgetRegion;
      toRegion: WidgetRegion;
      toIndex: number;
    }
  | { type: 'reorderWidgets'; region: WidgetRegion; instanceIds: readonly WidgetInstanceId[] }
  | { type: 'setAlignment'; region: WidgetRegion; instanceId: WidgetInstanceId; align: 'start' | 'end' }
  /** Without geometry the window cascades from the others. */
  | { type: 'floatWidget'; instanceId: WidgetInstanceId; region?: WidgetRegion; geometry?: FloatingWidgetGeometry }
  | { type: 'dockWindow'; instanceId: WidgetInstanceId }
  | { type: 'closeWindow'; instanceId: WidgetInstanceId };

type Availability = Pick<LayoutAvailability, 'getPresentation'>;

/** The instance an `addWidget` lands on, so the caller can front it. */
export const getAddedInstanceId = (
  structure: WorkspaceStructure,
  { createNew, mintId, region, typeId, unplacedInstanceId }: Extract<StructureAction, { type: 'addWidget' }>
): WidgetInstanceId => {
  if (createNew) {
    return mintId;
  }

  const isType = (instanceId: WidgetInstanceId) => structure.widgetInstances[instanceId]?.typeId === typeId;

  return (
    structure.widgetRegions[region].instanceIds.find(isType) ??
    Object.keys(structure.floatingWidgets ?? {}).find(isType) ??
    Object.keys(structure.widgetInstances).find(isType) ??
    unplacedInstanceId ??
    mintId
  );
};

const withRegion = (
  structure: WorkspaceStructure,
  regionId: WidgetRegion,
  update: (region: WidgetRegionState) => WidgetRegionState
): WorkspaceStructure => ({
  ...structure,
  widgetRegions: { ...structure.widgetRegions, [regionId]: update(structure.widgetRegions[regionId]) },
});

const withoutWindow = (
  floatingWidgets: WorkspaceStructure['floatingWidgets'],
  instanceId: WidgetInstanceId
): WorkspaceStructure['floatingWidgets'] => {
  const { [instanceId]: _removed, ...remaining } = floatingWidgets ?? {};

  return Object.keys(remaining).length > 0 ? remaining : undefined;
};

/** Edits members and window markers together so marker return indices stay in step; a dropped marker closes it. */
const updateRegionOrder = (
  structure: WorkspaceStructure,
  regionId: WidgetRegion,
  update: (slots: RegionOrderSlot[]) => RegionOrderSlot[]
): WorkspaceStructure => {
  const before = getRegionOrder(regionId, structure.widgetRegions[regionId].instanceIds, structure.floatingWidgets);
  const after = update(before);
  const markerIds = new Set(after.filter((slot) => slot.isFloating).map((slot) => slot.instanceId));
  const remaining = before
    .filter((slot) => slot.isFloating && !markerIds.has(slot.instanceId))
    .reduce((windows, slot) => withoutWindow(windows, slot.instanceId), structure.floatingWidgets);
  const order = writeRegionOrder(after, remaining);

  return withRegion({ ...structure, floatingWidgets: order.floatingWidgets }, regionId, (region) => ({
    ...region,
    instanceIds: order.instanceIds,
  }));
};

const withPanelCollapsed = (
  structure: WorkspaceStructure,
  regionId: WidgetRegion,
  collapsed: boolean
): WorkspaceStructure =>
  regionId === 'center'
    ? structure
    : {
        ...withRegion(structure, regionId, (region) => ({ ...region, isCollapsed: collapsed })),
        layout: {
          ...structure.layout,
          panels: { ...structure.layout.panels, [PANEL_OPEN_KEYS[regionId]]: !collapsed },
        },
      };

/** A region that gains its first member fronts it and, at a side, opens: an empty rail is collapsed only for being empty. */
const afterFirstMember = (
  structure: WorkspaceStructure,
  regionId: WidgetRegion,
  wasEmpty: boolean,
  instanceId: WidgetInstanceId
): WorkspaceStructure =>
  wasEmpty
    ? withPanelCollapsed(
        withRegion(structure, regionId, (region) => ({ ...region, activeInstanceId: instanceId })),
        regionId,
        false
      )
    : structure;

/** The front a region keeps after losing `instanceId`: its own front, else the first other member. */
const getFrontWithout = (region: WidgetRegionState, instanceId: WidgetInstanceId): WidgetInstanceId | null =>
  region.activeInstanceId !== instanceId
    ? region.activeInstanceId
    : (region.instanceIds.find((memberId) => memberId !== instanceId) ?? null);

const afterLosingMember = (
  structure: WorkspaceStructure,
  regionId: WidgetRegion,
  previous: WidgetRegionState,
  instanceId: WidgetInstanceId
): WorkspaceStructure =>
  withRegion(structure, regionId, (region) => ({
    ...region,
    activeInstanceId: getFrontWithout(previous, instanceId) ?? (regionId === 'center' ? region.activeInstanceId : ''),
    isCollapsed: regionId !== 'center' && (region.instanceIds.length === 0 || region.isCollapsed),
  }));

const insertAtDockedIndex = (
  slots: RegionOrderSlot[],
  instanceId: WidgetInstanceId,
  index: number | undefined
): RegionOrderSlot[] => {
  const others = slots.filter((slot) => slot.instanceId !== instanceId);
  const target = index === undefined ? undefined : others.filter((slot) => !slot.isFloating)[Math.max(0, index)];
  const position = target ? others.indexOf(target) : others.length;

  return [...others.slice(0, position), { instanceId, isFloating: false }, ...others.slice(position)];
};

/** Docks a window into its return slot. */
const dockWindow = (structure: WorkspaceStructure, instanceId: WidgetInstanceId): WorkspaceStructure => {
  const window = structure.floatingWidgets?.[instanceId];

  if (!window) {
    return structure;
  }

  const { returnRegion } = window;

  return afterFirstMember(
    updateRegionOrder(structure, returnRegion, (slots) =>
      slots.map((slot) => (slot.instanceId === instanceId ? { ...slot, isFloating: false } : slot))
    ),
    returnRegion,
    structure.widgetRegions[returnRegion].instanceIds.length === 0,
    instanceId
  );
};

const floatWidget = (
  structure: WorkspaceStructure,
  { geometry, instanceId, region }: Extract<StructureAction, { type: 'floatWidget' }>
): WorkspaceStructure => {
  const memberRegions = WIDGET_REGIONS.filter((regionId) =>
    structure.widgetRegions[regionId].instanceIds.includes(instanceId)
  );
  const returnRegion = region && memberRegions.includes(region) ? region : memberRegions[0];

  if (structure.floatingWidgets?.[instanceId] || !returnRegion) {
    return structure;
  }

  let floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined = {
    ...structure.floatingWidgets,
    [instanceId]: toFloatingWidgetState({
      ...(geometry ?? cascadeDefaultGeometry(Object.keys(structure.floatingWidgets ?? {}).length)),
      mode: 'windowed',
      returnRegion,
      stackOrder: nextStackOrder(structure.floatingWidgets),
    }),
  };
  const widgetRegions = { ...structure.widgetRegions };

  for (const regionId of memberRegions) {
    const memberRegion = structure.widgetRegions[regionId];
    const slots = getRegionOrder(regionId, memberRegion.instanceIds, structure.floatingWidgets);
    const order = writeRegionOrder(
      regionId === returnRegion
        ? slots.map((slot) => (slot.instanceId === instanceId ? { ...slot, isFloating: true } : slot))
        : slots.filter((slot) => slot.instanceId !== instanceId),
      floatingWidgets
    );

    floatingWidgets = order.floatingWidgets;
    widgetRegions[regionId] = withoutFloatedInstances(regionId, memberRegion, order.instanceIds);
  }

  return { ...structure, floatingWidgets, widgetRegions };
};

const addWidget = (
  structure: WorkspaceStructure,
  action: Extract<StructureAction, { type: 'addWidget' }>,
  availability: Availability
): WorkspaceStructure => {
  const instanceId = getAddedInstanceId(structure, action);
  const existing = structure.widgetInstances[instanceId];

  if (
    availability.getPresentation(action.typeId, action.region) === null ||
    (existing && existing.typeId !== action.typeId)
  ) {
    return structure;
  }

  const withInstance = existing
    ? structure
    : {
        ...structure,
        widgetInstances: { ...structure.widgetInstances, [instanceId]: { id: instanceId, typeId: action.typeId } },
      };
  const docked = dockWindow(withInstance, instanceId);

  if (docked.widgetRegions[action.region].instanceIds.includes(instanceId)) {
    return docked;
  }

  return afterFirstMember(
    updateRegionOrder(docked, action.region, (slots) => insertAtDockedIndex(slots, instanceId, action.index)),
    action.region,
    docked.widgetRegions[action.region].instanceIds.length === 0,
    instanceId
  );
};

const moveWidget = (
  structure: WorkspaceStructure,
  { fromRegion, instanceId, toIndex, toRegion }: Extract<StructureAction, { type: 'moveWidget' }>,
  availability: Availability
): WorkspaceStructure => {
  const from = structure.widgetRegions[fromRegion];
  const typeId = structure.widgetInstances[instanceId]?.typeId;

  if (
    !from.instanceIds.includes(instanceId) ||
    typeId === undefined ||
    availability.getPresentation(typeId, toRegion) === null
  ) {
    return structure;
  }

  const detached =
    fromRegion === toRegion
      ? structure
      : afterLosingMember(
          updateRegionOrder(structure, fromRegion, (slots) => slots.filter((slot) => slot.instanceId !== instanceId)),
          fromRegion,
          from,
          instanceId
        );

  return afterFirstMember(
    updateRegionOrder(detached, toRegion, (slots) => insertAtDockedIndex(slots, instanceId, toIndex)),
    toRegion,
    detached.widgetRegions[toRegion].instanceIds.length === 0,
    instanceId
  );
};

const applyAction = (
  structure: WorkspaceStructure,
  action: StructureAction,
  availability: Availability
): WorkspaceStructure => {
  switch (action.type) {
    case 'addWidget':
      return addWidget(structure, action, availability);
    case 'removeWidget': {
      const region = structure.widgetRegions[action.region];

      return region.instanceIds.includes(action.instanceId)
        ? afterLosingMember(
            updateRegionOrder(structure, action.region, (slots) =>
              slots.filter((slot) => slot.instanceId !== action.instanceId)
            ),
            action.region,
            region,
            action.instanceId
          )
        : structure;
    }
    case 'moveWidget':
      return moveWidget(structure, action, availability);
    case 'reorderWidgets': {
      const members = new Set(structure.widgetRegions[action.region].instanceIds);

      // Anything but a permutation of docked members is stale: it could re-add a floated instance.
      return action.instanceIds.length === members.size &&
        new Set(action.instanceIds).size === members.size &&
        action.instanceIds.every((instanceId) => members.has(instanceId))
        ? withRegion(structure, action.region, (region) => ({ ...region, instanceIds: [...action.instanceIds] }))
        : structure;
    }
    case 'setAlignment':
      return withRegion(structure, action.region, (region) => {
        const current = region.alignEndInstanceIds ?? [];

        return {
          ...region,
          alignEndInstanceIds:
            action.align === 'end'
              ? current.includes(action.instanceId)
                ? current
                : [...current, action.instanceId]
              : current.filter((instanceId) => instanceId !== action.instanceId),
        };
      });
    case 'floatWidget':
      return floatWidget(structure, action);
    case 'dockWindow':
      return dockWindow(structure, action.instanceId);
    case 'closeWindow': {
      const window = structure.floatingWidgets?.[action.instanceId];

      if (!window) {
        return structure;
      }

      return updateRegionOrder(structure, window.returnRegion, (slots) =>
        slots.filter((slot) => slot.instanceId !== action.instanceId)
      );
    }
  }
};

/** What "the center keeps a view" reads: a structure, a resolved layout, or the shell's placement projection. */
interface CenterPlacement {
  widgetInstances: Readonly<Record<WidgetInstanceId, { typeId: WidgetTypeId } | undefined>>;
  widgetRegions: { center: Pick<WidgetRegionState, 'instanceIds'> };
}

/** The center's views (`availability` says a type is a view there), leaving out `without`. */
const countCenterViews = (placement: CenterPlacement, availability: Availability, without?: WidgetInstanceId) =>
  placement.widgetRegions.center.instanceIds.filter((instanceId) => {
    const typeId = placement.widgetInstances[instanceId]?.typeId;

    return instanceId !== without && typeId !== undefined && availability.getPresentation(typeId, 'center') === 'view';
  }).length;

/**
 * The center keeps a view: whether taking `instanceId` out of the center (removing, floating or moving it) would take
 * its last one. `reduceStructure` refuses such an edit, so the shell offers no control that makes it.
 */
export const takesLastCenterView = (
  placement: CenterPlacement,
  instanceId: WidgetInstanceId,
  availability: Availability
): boolean =>
  placement.widgetRegions.center.instanceIds.includes(instanceId) &&
  countCenterViews(placement, availability) > 0 &&
  countCenterViews(placement, availability, instanceId) === 0;

/**
 * Every structural edit of one workspace, for Customize's draft and for rearranging without it. An edit that would take
 * the center's last member, or its last view (by `availability`), is refused. The same object when nothing
 * changes; every result is canonical (`normalizeStructure` leaves it unchanged).
 */
export const reduceStructure = (
  structure: WorkspaceStructure,
  action: StructureAction,
  availability: Availability
): WorkspaceStructure => {
  const next = applyAction(structure, action, availability);

  if (
    next === structure ||
    (next.widgetRegions.center.instanceIds.length === 0 && structure.widgetRegions.center.instanceIds.length > 0) ||
    (countCenterViews(next, availability) === 0 && countCenterViews(structure, availability) > 0)
  ) {
    return structure;
  }

  const normalized = normalizeStructure(next);

  return areJsonValuesStructurallyEqual(normalized, structure) ? structure : normalized;
};

/** A region's defaults set to what the view shows: its front the home front, its size and collapsed state. */
const captureRegion = (
  structure: WorkspaceStructure,
  region: WidgetRegion,
  view: WorkspaceView | undefined
): WorkspaceStructure => {
  const front = view?.fronts?.[region];
  let next = structure;

  if (front !== undefined && next.widgetRegions[region].instanceIds.includes(front)) {
    next = withRegion(next, region, (state) => ({ ...state, activeInstanceId: front }));
  }
  if (region === 'center') {
    return next;
  }

  const sizePx = view?.sizes?.[region];
  const collapsed = view?.collapsed?.[region];

  if (sizePx !== undefined && Number.isFinite(sizePx)) {
    next = withRegion(next, region, (state) => ({ ...state, sizePx: clampRegionSize(region, sizePx) }));
  }

  return collapsed === undefined ? next : withPanelCollapsed(next, region, collapsed);
};

/**
 * What Done saves: the structure with what the screen shows as its defaults, each front its region's home front (the
 * center's the stage), and the sizes, collapsed panels and window geometry. Where repair 4 gave a side region the
 * instance the center was asked for, the view the center showed instead becomes the stage. Canonical; the same
 * object when nothing changes.
 */
export const captureLayout = (
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  layout: Pick<ResolvedLayout, 'widgetRegions'>,
  surfaces: SurfaceAvailability
): WorkspaceStructure => {
  const asked = view?.fronts?.center ?? structure.widgetRegions.center.activeInstanceId;
  const shown = layout.widgetRegions.center.activeInstanceId;
  const yielded =
    shown !== asked &&
    structure.widgetRegions.center.instanceIds.includes(shown) &&
    SIDE_REGIONS.some(
      (region) =>
        !layout.widgetRegions[region].isCollapsed &&
        layout.widgetRegions[region].activeInstanceId === asked &&
        isSharedWithCenter(structure, region, asked, surfaces)
    );
  const captured = yielded ? { ...view, fronts: { ...view?.fronts, center: shown } } : view;
  let next = WIDGET_REGIONS.reduce((current, region) => captureRegion(current, region, captured), structure);
  const windows = Object.entries(captured?.windows ?? {}).filter(([instanceId]) =>
    Object.hasOwn(next.floatingWidgets ?? {}, instanceId)
  );

  if (windows.length > 0) {
    next = {
      ...next,
      floatingWidgets: {
        ...next.floatingWidgets,
        ...Object.fromEntries(
          windows.map(([instanceId, window]) => [
            instanceId,
            toFloatingWidgetState({ ...next.floatingWidgets![instanceId]!, ...window }),
          ])
        ),
      },
    };
  }

  const normalized = normalizeStructure(next);

  return areJsonValuesStructurallyEqual(normalized, structure) ? structure : normalized;
};
