import type { InvocationSourceId } from '@workbench/invocationContracts';
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

const JOB_SOURCE_REGION_ORDER: readonly WidgetRegion[] = ['center', 'left', 'right', 'bottom'];

/** The center view the workspace is built around: the center's front while it is a member, else its first member. */
export const getWorkspaceStage = (placement: WorkspacePlacement): WidgetInstanceId | null => {
  const { activeInstanceId, instanceIds } = placement.widgetRegions.center;

  return instanceIds.includes(activeInstanceId) ? activeInstanceId : (instanceIds[0] ?? null);
};

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
export const getFrontSource = (
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

/** The source the workspace exists for: its explicit route while that source is placed, otherwise its front source. */
export const getWorkspaceJobSource = (
  defaultRoute: LayoutPresetRoute | undefined,
  placement: WorkspacePlacement,
  getSourceId: GetAvailableSourceId
): InvocationSourceId | null =>
  defaultRoute && getPlacedSources(placement, getSourceId).has(defaultRoute.sourceId)
    ? defaultRoute.sourceId
    : getFrontSource(placement, getSourceId);
