import type { Project, WidgetInstanceId, WidgetRegion, WidgetRegionState } from './types';
import type { WidgetPlacementProject } from './widgetPlacementCommands';
import type { WidgetPlacementMeta } from './widgetRegionViewModel';

export type WidgetPlacementRegionState = Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds' | 'isCollapsed'>;

export const getWidgetPlacementMeta = (
  project: Pick<Project, 'widgetInstances' | 'widgetRegions'>
): WidgetPlacementMeta => {
  const placedInstanceIds = new Set<WidgetInstanceId>();

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

export const getWidgetPlacementProject = (
  project: Pick<Project, 'id' | 'widgetInstances' | 'widgetRegions'>
): WidgetPlacementProject => ({
  projectId: project.id,
  widgetInstances: getWidgetPlacementMeta(project),
  widgetRegions: project.widgetRegions,
});

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
