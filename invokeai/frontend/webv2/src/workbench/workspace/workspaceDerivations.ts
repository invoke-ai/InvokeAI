import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { LayoutPresetRoute, PanelState, WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

/** The placement facts a workspace's arrangement and a live project both carry. */
export interface WorkspacePlacement {
  floatingWidgets?: Readonly<Record<WidgetInstanceId, unknown>>;
  layout: { panels: PanelState };
  widgetInstances: Readonly<Record<WidgetInstanceId, { typeId: WidgetTypeId }>>;
  widgetRegions: Readonly<
    Record<WidgetRegion, Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds' | 'isCollapsed'>>
  >;
}

/** The source a widget type runs, or null when it is none or its source is unavailable. */
export type GetAvailableSourceId = (typeId: WidgetTypeId) => InvocationSourceId | null;

/** How a workspace's sources are read: which source a type runs, and where a source's results go by default. */
export interface WorkspaceSources {
  getNaturalDestination: (sourceId: InvocationSourceId) => ResultDestination;
  getSourceId: GetAvailableSourceId;
}

/** A workspace as its derivations read it: the arrangement the account saved and its explicit route. */
export interface SavedWorkspace {
  defaultRoute?: LayoutPresetRoute;
  snapshot: WorkspacePlacement;
}

const JOB_SOURCE_REGION_ORDER: readonly WidgetRegion[] = ['center', 'left', 'right', 'bottom'];

/** Whether a widget type can be the center's view for this account: enabled, available, and not a toolbar item. */
export type IsCenterView = (typeId: WidgetTypeId) => boolean;

/** What the center shows in this arrangement now: its front while that is a member, else its first member. */
export const getCenterFront = (placement: WorkspacePlacement): WidgetInstanceId | null => {
  const { activeInstanceId, instanceIds } = placement.widgetRegions.center;

  return instanceIds.includes(activeInstanceId) ? activeInstanceId : (instanceIds[0] ?? null);
};

/** The center view the workspace is built around: the center front of the arrangement the account saved. */
export const getWorkspaceStage = (workspace: SavedWorkspace): WidgetInstanceId | null =>
  getCenterFront(workspace.snapshot);

export const getWorkspaceHomeFronts = (placement: WorkspacePlacement): Record<WidgetRegion, WidgetInstanceId> => ({
  bottom: placement.widgetRegions.bottom.activeInstanceId,
  center: placement.widgetRegions.center.activeInstanceId,
  left: placement.widgetRegions.left.activeInstanceId,
  right: placement.widgetRegions.right.activeInstanceId,
});

/** Sources the arrangement docks in any region. */
export const getPlacedSources = (
  placement: WorkspacePlacement,
  getSourceId: GetAvailableSourceId
): Set<InvocationSourceId> => {
  const sourceIds = new Set<InvocationSourceId>();

  for (const region of Object.values(placement.widgetRegions)) {
    for (const instanceId of region.instanceIds) {
      const typeId = placement.widgetInstances[instanceId]?.typeId;
      const sourceId = typeId ? getSourceId(typeId) : null;

      if (sourceId) {
        sourceIds.add(sourceId);
      }
    }
  }

  return sourceIds;
};

/** The first available source among the home fronts, center first. */
const getFrontSource = (
  placement: WorkspacePlacement,
  getSourceId: GetAvailableSourceId
): InvocationSourceId | null => {
  const fronts = getWorkspaceHomeFronts(placement);

  for (const region of JOB_SOURCE_REGION_ORDER) {
    const typeId = placement.widgetInstances[fronts[region]]?.typeId;
    const sourceId = typeId ? getSourceId(typeId) : null;

    if (sourceId) {
      return sourceId;
    }
  }

  return null;
};

/** The workspace's stage while the laid-out arrangement still places it in the center, else null. */
export const getPlacedStage = (workspace: SavedWorkspace, placement: WorkspacePlacement): WidgetInstanceId | null => {
  const stage = getWorkspaceStage(workspace);

  return stage !== null && placement.widgetRegions.center.instanceIds.includes(stage) ? stage : null;
};

/** The stage entry fronts: the placed stage while it is a center view, else the first center view placed. */
export const getEntryStage = (
  workspace: SavedWorkspace,
  placement: WorkspacePlacement,
  isCenterView: IsCenterView
): WidgetInstanceId | null => {
  const isView = (instanceId: WidgetInstanceId) => {
    const typeId = placement.widgetInstances[instanceId]?.typeId;

    return typeId !== undefined && isCenterView(typeId);
  };
  const stage = getPlacedStage(workspace, placement);

  return stage !== null && isView(stage) ? stage : (placement.widgetRegions.center.instanceIds.find(isView) ?? null);
};

/** The stage to go back to while the center shows something else (Away), else null. */
export const getAwayStage = (workspace: SavedWorkspace, placement: WorkspacePlacement): WidgetInstanceId | null => {
  const stage = getPlacedStage(workspace, placement);

  return stage !== null && getCenterFront(placement) !== stage ? stage : null;
};

/**
 * The source the workspace exists for, as laid out: its explicit route's while that source is placed, else the first
 * placed source among the saved home fronts, then among the laid-out ones. Null: the workspace has no job.
 */
export const getWorkspaceJobSource = (
  workspace: SavedWorkspace,
  placement: WorkspacePlacement,
  getSourceId: GetAvailableSourceId
): InvocationSourceId | null => {
  const placed = getPlacedSources(placement, getSourceId);

  if (workspace.defaultRoute && placed.has(workspace.defaultRoute.sourceId)) {
    return workspace.defaultRoute.sourceId;
  }

  const getPlacedSourceId: GetAvailableSourceId = (typeId) => {
    const sourceId = getSourceId(typeId);

    return sourceId && placed.has(sourceId) ? sourceId : null;
  };

  return getFrontSource(workspace.snapshot, getPlacedSourceId) ?? getFrontSource(placement, getPlacedSourceId);
};

/** The job as a route: the explicit one when it is the job, else the job's source to its natural destination. */
export const getWorkspaceJob = (
  workspace: SavedWorkspace,
  placement: WorkspacePlacement,
  { getNaturalDestination, getSourceId }: WorkspaceSources
): LayoutPresetRoute | null => {
  const sourceId = getWorkspaceJobSource(workspace, placement, getSourceId);

  if (!sourceId) {
    return null;
  }

  return workspace.defaultRoute?.sourceId === sourceId
    ? workspace.defaultRoute
    : { destination: getNaturalDestination(sourceId), sourceId };
};
