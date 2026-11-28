import type {
  CenterViewId,
  LayoutPresetSnapshot,
  LayoutPresetWidgetInstanceSnapshot,
  PanelState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getRegionOrder, normalizeFloatingPlacement, writeRegionOrder } from '@workbench/floatingWindows';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';

import { getCenterFront, getRegionFront } from './workspaceDerivations';

/**
 * A workspace's structure: the stored account snapshot, read with new meaning. Members, order, end cluster and windows
 * are placement; each region's `activeInstanceId` is its home front (the center's is the stage); `sizePx` and the
 * collapsed state are defaults a project's view overrides; window geometry is a default too. `layout.centerViewId` is
 * ignored on read and kept consistent on write.
 */
export type WorkspaceStructure = LayoutPresetSnapshot;

export type SideRegion = Exclude<WidgetRegion, 'center'>;

export const SIDE_REGIONS: readonly SideRegion[] = ['left', 'right', 'bottom'];

export const PANEL_OPEN_KEYS = {
  bottom: 'isBottomOpen',
  left: 'isLeftOpen',
  right: 'isRightOpen',
} as const satisfies Record<SideRegion, keyof PanelState>;

const PANEL_SIZE_BOUNDS: Record<SideRegion, { max: number; min: number }> = {
  bottom: { max: 420, min: 96 },
  left: { max: 720, min: 350 },
  right: { max: 720, min: 350 },
};

/** How far past its minimum a dragged divider goes before the panel collapses instead. */
const PANEL_COLLAPSE_OVERSHOOT_PX = 80;

export const getRegionSizeBounds = (region: SideRegion): { max: number; min: number } => PANEL_SIZE_BOUNDS[region];

export const clampRegionSize = (region: SideRegion, sizePx: number): number =>
  Math.min(PANEL_SIZE_BOUNDS[region].max, Math.max(PANEL_SIZE_BOUNDS[region].min, sizePx));

/** The size a dragged divider collapses its panel at, for a panel shown `visibleSizePx` wide. */
export const getPanelCollapseThreshold = (region: SideRegion, visibleSizePx: number): number =>
  Math.min(PANEL_SIZE_BOUNDS[region].min - PANEL_COLLAPSE_OVERSHOOT_PX, visibleSizePx - PANEL_COLLAPSE_OVERSHOOT_PX);

export interface RegionDefaults {
  collapsed: boolean;
  sizePx: number;
}

/**
 * Stored apart, a closed panel (`layout.panels`) and a collapsed region (`isCollapsed`) are one fact, the one the shell
 * shows: collapsed when either says so. The center never collapses.
 */
export const getRegionDefaults = (structure: WorkspaceStructure, region: WidgetRegion): RegionDefaults => {
  const { isCollapsed, sizePx } = structure.widgetRegions[region];

  return {
    collapsed: region !== 'center' && (isCollapsed || !structure.layout.panels[PANEL_OPEN_KEYS[region]]),
    sizePx,
  };
};

/** Members the status bar shows in its trailing cluster; an aligned id that is not a member is inert. */
export const getEndCluster = (region: Pick<WidgetRegionState, 'alignEndInstanceIds' | 'instanceIds'>) =>
  region.instanceIds.filter((instanceId) => region.alignEndInstanceIds?.includes(instanceId) === true);

/** The stage's instance, or null when the center places no view. */
export const getStageInstance = (structure: WorkspaceStructure): LayoutPresetWidgetInstanceSnapshot | null => {
  const stage = getCenterFront(structure);

  return stage === null ? null : (structure.widgetInstances[stage] ?? null);
};

/** The center views a stored structure's `layout.centerViewId` may name. */
export const CENTER_VIEW_TYPE_IDS: ReadonlySet<string> = new Set<CenterViewId>([
  'canvas',
  'gallery',
  'preview',
  'workflow',
]);

const normalizeRegion = (
  regionId: WidgetRegion,
  region: WidgetRegionState,
  hasInstance: (instanceId: WidgetInstanceId) => boolean,
  isPanelOpen: boolean
): WidgetRegionState => {
  const instanceIds = [...new Set(region.instanceIds)].filter(hasInstance);
  const front = getRegionFront({ activeInstanceId: region.activeInstanceId, instanceIds });

  return {
    activeInstanceId: front ?? (regionId === 'center' ? region.activeInstanceId : ''),
    ...(region.alignEndInstanceIds ? { alignEndInstanceIds: region.alignEndInstanceIds } : {}),
    instanceIds,
    isCollapsed: regionId !== 'center' && (instanceIds.length === 0 || region.isCollapsed || !isPanelOpen),
    sizePx: region.sizePx,
  };
};

type Placement = ReturnType<typeof normalizeFloatingPlacement>;

/**
 * A center whose every member floats docks one window back, the one it fronted when it floats: Customize never lets the
 * center's last view float, and a structure stored before that rule may hold one floated away.
 */
const dockFloatedStage = (placement: Placement, hasInstance: (instanceId: WidgetInstanceId) => boolean): Placement => {
  const { floatingWidgets, widgetRegions } = placement;
  const center = widgetRegions.center;
  const slots = getRegionOrder('center', center.instanceIds, floatingWidgets);

  if (center.instanceIds.some(hasInstance) || !slots.some((slot) => slot.isFloating)) {
    return placement;
  }

  const windows = slots.filter((slot) => slot.isFloating).map((slot) => slot.instanceId);
  const docked = windows.includes(center.activeInstanceId) ? center.activeInstanceId : windows[0]!;
  const { [docked]: _window, ...others } = floatingWidgets ?? {};
  const order = writeRegionOrder(
    slots.map((slot) => (slot.instanceId === docked ? { instanceId: docked, isFloating: false } : slot)),
    Object.keys(others).length > 0 ? others : undefined
  );

  return {
    floatingWidgets: order.floatingWidgets,
    widgetRegions: {
      ...widgetRegions,
      center: { ...center, activeInstanceId: docked, instanceIds: order.instanceIds },
    },
  };
};

/**
 * The canonical form of a stored or imported structure, for shapes the account loader validated: placement normalized
 * as hydration reads it, the center holding a docked member whenever one returns to it, every front a member, collapsed
 * and open panels as one fact, the center view id following the stage, and only instances something places.
 * Idempotent; the same object when already canonical.
 */
export const normalizeStructure = (structure: WorkspaceStructure): WorkspaceStructure => {
  const hasInstance = (instanceId: WidgetInstanceId) => Object.hasOwn(structure.widgetInstances, instanceId);
  const placement = dockFloatedStage(
    normalizeFloatingPlacement(structure.widgetRegions, structure.floatingWidgets, hasInstance),
    hasInstance
  );
  const isOpen = (region: WidgetRegion) => region === 'center' || structure.layout.panels[PANEL_OPEN_KEYS[region]];
  const widgetRegions = Object.fromEntries(
    WIDGET_REGIONS.map((region) => [
      region,
      normalizeRegion(region, placement.widgetRegions[region], hasInstance, isOpen(region)),
    ])
  ) as Record<WidgetRegion, WidgetRegionState>;
  const referenced = new Set<WidgetInstanceId>([
    ...WIDGET_REGIONS.flatMap((region) => widgetRegions[region].instanceIds),
    ...Object.keys(placement.floatingWidgets ?? {}),
    widgetRegions.center.activeInstanceId,
  ]);
  const stageTypeId = structure.widgetInstances[getCenterFront({ widgetRegions }) ?? '']?.typeId;
  const next: WorkspaceStructure = {
    ...(placement.floatingWidgets ? { floatingWidgets: placement.floatingWidgets } : {}),
    layout: {
      ...structure.layout,
      centerViewId:
        stageTypeId !== undefined && CENTER_VIEW_TYPE_IDS.has(stageTypeId)
          ? (stageTypeId as CenterViewId)
          : structure.layout.centerViewId,
      panels: {
        isBottomOpen: !widgetRegions.bottom.isCollapsed,
        isLeftOpen: !widgetRegions.left.isCollapsed,
        isRightOpen: !widgetRegions.right.isCollapsed,
      },
    },
    widgetInstances: Object.fromEntries(
      Object.entries(structure.widgetInstances).filter(([instanceId]) => referenced.has(instanceId))
    ),
    widgetRegions,
  };

  return areJsonValuesStructurallyEqual(next, structure) ? structure : next;
};

/** Whether two structures are the same once canonical, whatever workspace each was saved as. */
export const isSameStructure = (left: WorkspaceStructure, right: WorkspaceStructure): boolean => {
  const canonical = (structure: WorkspaceStructure) =>
    normalizeStructure({ ...structure, layout: { ...structure.layout, presetId: '' } });

  return areJsonValuesStructurallyEqual(canonical(left), canonical(right));
};

const areListsEqual = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((value, index) => value === right[index]);

/**
 * Whether two structures place the same instances the same way: the instance set with types and titles, each region's
 * members and order, the status bar's end cluster, and which instances float and every region docking puts them back
 * in. Home fronts, default sizes, collapsed panels and window geometry are what a view overrides, so they are not
 * compared.
 */
export const isSamePlacement = (leftStructure: WorkspaceStructure, rightStructure: WorkspaceStructure): boolean => {
  const left = normalizeStructure(leftStructure);
  const right = normalizeStructure(rightStructure);
  const leftIds = Object.keys(left.widgetInstances).sort();
  const leftWindows = Object.keys(left.floatingWidgets ?? {}).sort();

  return (
    areListsEqual(leftIds, Object.keys(right.widgetInstances).sort()) &&
    leftIds.every(
      (instanceId) =>
        left.widgetInstances[instanceId]?.typeId === right.widgetInstances[instanceId]?.typeId &&
        left.widgetInstances[instanceId]?.title === right.widgetInstances[instanceId]?.title
    ) &&
    WIDGET_REGIONS.every(
      (region) =>
        areListsEqual(left.widgetRegions[region].instanceIds, right.widgetRegions[region].instanceIds) &&
        areListsEqual(getEndCluster(left.widgetRegions[region]), getEndCluster(right.widgetRegions[region]))
    ) &&
    areListsEqual(leftWindows, Object.keys(right.floatingWidgets ?? {}).sort()) &&
    leftWindows.every(
      (instanceId) =>
        left.floatingWidgets?.[instanceId]?.returnRegion === right.floatingWidgets?.[instanceId]?.returnRegion &&
        left.floatingWidgets?.[instanceId]?.returnIndex === right.floatingWidgets?.[instanceId]?.returnIndex &&
        areJsonValuesStructurallyEqual(
          left.floatingWidgets?.[instanceId]?.alsoIn ?? null,
          right.floatingWidgets?.[instanceId]?.alsoIn ?? null
        )
    )
  );
};
