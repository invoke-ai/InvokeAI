import type { WorkbenchFocusTarget } from '@workbench/focusRegions';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  WidgetContributionSource,
  WidgetInstanceId,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/widgetContracts';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';

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
 * docked region. A floating target is never used to index `widgetRegions`, and is ignored once its instance no
 * longer floats in this project.
 */
export const resolveHotkeyTarget = ({
  focusTarget,
  project,
  targetWidget,
}: {
  focusTarget: WorkbenchFocusTarget | null;
  project: WidgetPlacementProject;
  targetWidget: ReturnType<typeof getHotkeyTargetWidget>;
}): HotkeyTarget => {
  const focus =
    focusTarget?.kind === 'floating' && !project.floatingPlacements?.[focusTarget.instanceId] ? null : focusTarget;
  const focusedRegion = focus === null ? null : focus.kind === 'floating' ? 'floating' : focus.region;
  const focusedInstanceId =
    focus === null
      ? null
      : focus.kind === 'floating'
        ? focus.instanceId
        : project.widgetRegions[focus.region].activeInstanceId;
  const activeInstanceId = targetWidget?.instanceId ?? (focusedInstanceId || null);
  const activeWidgetTypeId = activeInstanceId
    ? (targetWidget?.typeId ?? project.widgetInstances[activeInstanceId]?.typeId ?? null)
    : null;
  const region = targetWidget?.region ?? focusedRegion;

  return {
    activeInstanceId,
    activeWidgetTypeId,
    focusedRegion,
    source:
      activeInstanceId && activeWidgetTypeId && region
        ? { instanceId: activeInstanceId, projectId: project.projectId ?? '', region, typeId: activeWidgetTypeId }
        : null,
  };
};
