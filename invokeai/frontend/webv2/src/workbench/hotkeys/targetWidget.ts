import type { WorkbenchFocusTarget } from '@workbench/focusRegions';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  WidgetContributionSource,
  WidgetInstanceId,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/widgetContracts';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';

import { isWidgetRegion } from '@workbench/layoutContracts';

export const getHotkeyTargetWidget = (
  target: EventTarget | null
): { instanceId: WidgetInstanceId; region: WorkbenchRegion | null; typeId: WidgetTypeId } | null => {
  if (!(target instanceof Element)) {
    return null;
  }

  const widgetElement = target.closest('[data-hotkey-widget-instance-id][data-hotkey-widget-type-id]');
  const instanceId = widgetElement?.getAttribute('data-hotkey-widget-instance-id');
  const region = widgetElement?.getAttribute('data-hotkey-widget-region') ?? null;
  const typeId = widgetElement?.getAttribute('data-hotkey-widget-type-id');

  return instanceId && typeId
    ? { instanceId, region: region as WorkbenchRegion | null, typeId: typeId as WidgetTypeId }
    : null;
};

export interface HotkeyTarget {
  activeInstanceId: WidgetInstanceId | null;
  activeWidgetTypeId: WidgetTypeId | null;
  focusedRegion: WidgetRegion | 'floating' | null;
  /** The widget a scoped command runs against; a floating window's source carries `region: 'floating'`. */
  source: WidgetContributionSource | null;
}

/**
 * What a key press is aimed at. The widget under the event target wins; outside any widget's DOM the press falls
 * back to whatever holds workbench focus — the active floating window, or the active instance of the focused
 * docked region. A floating target is never used to index `widgetRegions`. A region's active pointer counts only
 * while it names one of the region's members: an emptied center keeps pointing at the view that floated out of
 * it, and that view answers as a window, not as the center.
 */
export const resolveHotkeyTarget = ({
  focusTarget,
  project,
  targetWidget,
}: {
  /** From the focus controller, which already drops a floating target whose window is gone. */
  focusTarget: WorkbenchFocusTarget | null;
  project: WidgetPlacementProject;
  targetWidget: ReturnType<typeof getHotkeyTargetWidget>;
}): HotkeyTarget => {
  // Where the press landed says more than where focus was last recorded; a popover or dialog is no region.
  const targetRegion = targetWidget?.region;
  const focusedRegion =
    targetRegion === 'floating' || isWidgetRegion(targetRegion)
      ? targetRegion
      : focusTarget === null
        ? null
        : focusTarget.kind === 'floating'
          ? 'floating'
          : focusTarget.region;
  const region = focusTarget?.kind === 'region' ? project.widgetRegions[focusTarget.region] : null;
  const focusedInstanceId =
    focusTarget?.kind === 'floating'
      ? focusTarget.instanceId
      : region?.instanceIds.includes(region.activeInstanceId)
        ? region.activeInstanceId
        : null;
  const activeInstanceId = targetWidget?.instanceId ?? focusedInstanceId;
  const activeWidgetTypeId = activeInstanceId
    ? (targetWidget?.typeId ?? project.widgetInstances[activeInstanceId]?.typeId ?? null)
    : null;
  const sourceRegion = targetWidget?.region ?? focusedRegion;

  return {
    activeInstanceId,
    activeWidgetTypeId,
    focusedRegion,
    source:
      activeInstanceId && activeWidgetTypeId && sourceRegion
        ? {
            instanceId: activeInstanceId,
            projectId: project.projectId ?? '',
            region: sourceRegion,
            typeId: activeWidgetTypeId,
          }
        : null,
  };
};
