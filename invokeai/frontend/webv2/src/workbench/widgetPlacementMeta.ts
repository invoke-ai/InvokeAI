import type { WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import type { FloatingWidgetPlacement } from './floatingWindows';
import type { WidgetPlacementProject } from './widgetPlacementCommands';
import type { WidgetPlacementMeta } from './widgetRegionViewModel';

export type WidgetPlacementRegionState = Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds' | 'isCollapsed'>;

/** Every instance the layout shows somewhere: region members, and windows, which belong to no region. */
export const getWidgetPlacementMeta = (
  project: Pick<Project, 'floatingWidgets' | 'widgetInstances' | 'widgetRegions'>
): WidgetPlacementMeta => {
  const placedInstanceIds = new Set<WidgetInstanceId>(Object.keys(project.floatingWidgets ?? {}));

  for (const region of Object.values(project.widgetRegions)) {
    for (const instanceId of region.instanceIds) {
      placedInstanceIds.add(instanceId);
    }
  }

  return Object.fromEntries(
    [...placedInstanceIds].flatMap((instanceId) => {
      const instance = project.widgetInstances[instanceId];

      return instance ? [[instanceId, { id: instance.id, title: instance.title, typeId: instance.typeId }]] : [];
    })
  );
};

const NO_FLOATING_PLACEMENTS: Record<WidgetInstanceId, FloatingWidgetPlacement> = {};

/**
 * Only where each window returns to. Geometry, mode, and stacking stay out, so moving, shading, or raising a
 * window leaves this projection equal and its subscribers — the rails among them — do not re-render.
 */
const getFloatingPlacements = (
  floatingWidgets: Project['floatingWidgets']
): Record<WidgetInstanceId, FloatingWidgetPlacement> =>
  floatingWidgets
    ? Object.fromEntries(
        Object.entries(floatingWidgets).map(([instanceId, { returnIndex, returnRegion }]) => [
          instanceId,
          { returnIndex, returnRegion },
        ])
      )
    : NO_FLOATING_PLACEMENTS;

export const getWidgetPlacementProject = (
  project: Pick<Project, 'floatingWidgets' | 'id' | 'widgetInstances' | 'widgetRegions'>
): WidgetPlacementProject => ({
  floatingPlacements: getFloatingPlacements(project.floatingWidgets),
  projectId: project.id,
  widgetInstances: getWidgetPlacementMeta(project),
  widgetRegions: project.widgetRegions,
});

const areFloatingPlacementsEqual = (
  left: WidgetPlacementProject['floatingPlacements'] = NO_FLOATING_PLACEMENTS,
  right: WidgetPlacementProject['floatingPlacements'] = NO_FLOATING_PLACEMENTS
): boolean => {
  const leftKeys = Object.keys(left);

  return (
    leftKeys.length === Object.keys(right).length &&
    leftKeys.every(
      (instanceId) =>
        left[instanceId].returnRegion === right[instanceId]?.returnRegion &&
        left[instanceId].returnIndex === right[instanceId]?.returnIndex
    )
  );
};

export const areWidgetPlacementMetaEqual = (left: WidgetPlacementMeta, right: WidgetPlacementMeta): boolean => {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);

  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key) => {
      const leftMeta = left[key];
      const rightMeta = right[key];

      return (
        rightMeta !== undefined &&
        leftMeta.id === rightMeta.id &&
        leftMeta.typeId === rightMeta.typeId &&
        leftMeta.title === rightMeta.title
      );
    })
  );
};

export const areWidgetPlacementProjectsEqual = (left: WidgetPlacementProject, right: WidgetPlacementProject): boolean =>
  left.projectId === right.projectId &&
  areWidgetPlacementMetaEqual(left.widgetInstances, right.widgetInstances) &&
  areFloatingPlacementsEqual(left.floatingPlacements, right.floatingPlacements) &&
  (Object.keys(left.widgetRegions) as WidgetRegion[]).every((region) => {
    const leftRegion = left.widgetRegions[region];
    const rightRegion = right.widgetRegions[region];

    return (
      rightRegion !== undefined &&
      leftRegion.activeInstanceId === rightRegion.activeInstanceId &&
      leftRegion.instanceIds.length === rightRegion.instanceIds.length &&
      leftRegion.instanceIds.every((instanceId, index) => instanceId === rightRegion.instanceIds[index])
    );
  });
