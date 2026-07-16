import { closestCenter, pointerWithin, type CollisionDetection } from '@dnd-kit/core';
import { arrayMove } from '@dnd-kit/sortable';

import type { WidgetIconComponent, WidgetInstanceId, WidgetRegion, WidgetRegionState, WidgetTypeId } from './types';

export interface WidgetDndProject {
  widgetInstances: Record<WidgetInstanceId, { typeId: WidgetTypeId; title?: string }>;
  widgetRegions: Record<WidgetRegion, Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds'>>;
}

export interface ActiveWidgetDrag {
  fromRegion: WidgetRegion;
  icon: WidgetIconComponent;
  instanceId: WidgetInstanceId;
  label: string;
  typeId: WidgetTypeId;
}

export interface WidgetInstanceDragData {
  kind: 'widget-instance';
  instanceId: WidgetInstanceId;
  region: WidgetRegion;
  typeId: WidgetTypeId;
}

export interface WidgetRegionDropData {
  kind: 'widget-region';
  region: WidgetRegion;
}

export type WidgetDndData = WidgetInstanceDragData | WidgetRegionDropData;

export interface WidgetDndManifestLookupResult {
  manifest: {
    allowedRegions: WidgetRegion[];
    allowMultiple: boolean;
  };
}

export type GetWidgetDndManifest = (typeId: WidgetTypeId) => WidgetDndManifestLookupResult | undefined;

export interface WidgetRegionDropState {
  helperText: string;
  isActive: boolean;
  isAllowed: boolean;
}

export type WidgetDragEndResolution =
  | {
      activeInstanceId: WidgetInstanceId;
      instanceIds: WidgetInstanceId[];
      region: WidgetRegion;
      type: 'reorder';
    }
  | {
      fromRegion: WidgetRegion;
      instanceId: WidgetInstanceId;
      toIndex: number;
      toRegion: WidgetRegion;
      type: 'move';
    };

export const getWidgetInstanceDragId = (region: WidgetRegion, instanceId: WidgetInstanceId): string =>
  `widget-instance:${region}:${instanceId}`;

export const getWidgetRegionDropId = (region: WidgetRegion): string => `widget-region:${region}`;

export const getWidgetInstanceDragData = (
  region: WidgetRegion,
  instanceId: WidgetInstanceId,
  typeId: WidgetTypeId
): WidgetInstanceDragData => ({ kind: 'widget-instance', instanceId, region, typeId });

export const getWidgetRegionDropData = (region: WidgetRegion): WidgetRegionDropData => ({
  kind: 'widget-region',
  region,
});

export const isWidgetInstanceDragData = (data: unknown): data is WidgetInstanceDragData =>
  isRecord(data) &&
  data.kind === 'widget-instance' &&
  typeof data.instanceId === 'string' &&
  isWidgetRegion(data.region) &&
  typeof data.typeId === 'string';

export const isWidgetRegionDropData = (data: unknown): data is WidgetRegionDropData =>
  isRecord(data) && data.kind === 'widget-region' && isWidgetRegion(data.region);

export const isWidgetDndData = (data: unknown): data is WidgetDndData =>
  isWidgetInstanceDragData(data) || isWidgetRegionDropData(data);

export const regionHasWidgetType = (
  project: WidgetDndProject,
  region: WidgetRegion,
  typeId: WidgetTypeId,
  excludedInstanceId?: WidgetInstanceId
): boolean =>
  project.widgetRegions[region].instanceIds.some(
    (instanceId) => instanceId !== excludedInstanceId && project.widgetInstances[instanceId]?.typeId === typeId
  );

export const canMoveWidgetToRegion = (
  project: WidgetDndProject,
  region: WidgetRegion,
  typeId: WidgetTypeId,
  instanceId: WidgetInstanceId,
  getWidget: GetWidgetDndManifest
): boolean => {
  const widget = getWidget(typeId);

  if (
    !widget ||
    !widget.manifest.allowedRegions.includes(region) ||
    project.widgetRegions[region].instanceIds.includes(instanceId)
  ) {
    return false;
  }

  return widget.manifest.allowMultiple || !regionHasWidgetType(project, region, typeId);
};

export const getRegionDropState = (
  project: WidgetDndProject,
  activeDrag: ActiveWidgetDrag | null,
  region: WidgetRegion,
  getWidget: GetWidgetDndManifest
): WidgetRegionDropState => {
  const widget = activeDrag ? getWidget(activeDrag.typeId) : null;

  if (!activeDrag || !widget) {
    return { helperText: 'Unavailable', isActive: false, isAllowed: false };
  }

  if (!widget.manifest.allowedRegions.includes(region)) {
    return { helperText: 'Unavailable', isActive: true, isAllowed: false };
  }

  if (activeDrag.fromRegion !== region && project.widgetRegions[region].instanceIds.includes(activeDrag.instanceId)) {
    return { helperText: 'Already placed', isActive: true, isAllowed: false };
  }

  const excludedInstanceId = region === activeDrag.fromRegion ? activeDrag.instanceId : undefined;

  if (!widget.manifest.allowMultiple && regionHasWidgetType(project, region, activeDrag.typeId, excludedInstanceId)) {
    return { helperText: 'Already placed', isActive: true, isAllowed: false };
  }

  return { helperText: 'Drop here', isActive: true, isAllowed: true };
};

export const resolveWidgetDragEnd = (
  project: WidgetDndProject,
  activeData: WidgetInstanceDragData | null,
  overData: WidgetDndData | null,
  getWidget: GetWidgetDndManifest
): WidgetDragEndResolution | null => {
  if (!activeData || !overData) {
    return null;
  }

  const activeInstance = project.widgetInstances[activeData.instanceId];
  const widget = activeInstance ? getWidget(activeInstance.typeId) : undefined;
  const fromRegion = activeData.region;
  const overRegion = overData.region;

  if (
    !activeInstance ||
    !widget ||
    activeInstance.typeId !== activeData.typeId ||
    !project.widgetRegions[fromRegion].instanceIds.includes(activeData.instanceId) ||
    !widget.manifest.allowedRegions.includes(overRegion)
  ) {
    return null;
  }

  if (fromRegion === overRegion) {
    if (overData.kind !== 'widget-instance') {
      return null;
    }

    const overRegionState = project.widgetRegions[overRegion];
    const oldIndex = overRegionState.instanceIds.indexOf(activeData.instanceId);
    const toIndex = Math.max(0, overRegionState.instanceIds.indexOf(overData.instanceId));

    if (oldIndex === -1 || overData.instanceId === activeData.instanceId) {
      return null;
    }

    return {
      activeInstanceId: activeData.instanceId,
      instanceIds: arrayMove(overRegionState.instanceIds, oldIndex, toIndex),
      region: overRegion,
      type: 'reorder',
    };
  }

  if (!canMoveWidgetToRegion(project, overRegion, activeInstance.typeId, activeData.instanceId, getWidget)) {
    return null;
  }

  const overRegionState = project.widgetRegions[overRegion];
  const toIndex =
    overData.kind === 'widget-region'
      ? overRegionState.instanceIds.length
      : Math.max(0, overRegionState.instanceIds.indexOf(overData.instanceId));

  return {
    fromRegion,
    instanceId: activeData.instanceId,
    toIndex,
    toRegion: overRegion,
    type: 'move',
  };
};

export const widgetCollisionDetection: CollisionDetection = (args) => {
  const activeData = args.active.data.current;
  const collisionArgs = isWidgetInstanceDragData(activeData)
    ? args
    : {
        ...args,
        droppableContainers: args.droppableContainers.filter((container) => !isWidgetDndData(container.data.current)),
      };
  const pointerCollisions = pointerWithin(collisionArgs);

  if (pointerCollisions.length > 0) {
    const widgetItemCollisions = pointerCollisions.filter((collision) => {
      return isWidgetInstanceDragData(getCollisionData(args, collision.id));
    });

    if (widgetItemCollisions.length > 0) {
      return widgetItemCollisions;
    }

    if (isWidgetInstanceDragData(activeData)) {
      const widgetRegionCollisions = pointerCollisions.filter((collision) => {
        return isWidgetRegionDropData(getCollisionData(args, collision.id));
      });
      const nonSourceRegionCollisions = widgetRegionCollisions.filter((collision) => {
        const data = getCollisionData(args, collision.id);

        return isWidgetRegionDropData(data) && data.region !== activeData.region;
      });

      if (nonSourceRegionCollisions.length > 0) {
        return nonSourceRegionCollisions;
      }

      if (widgetRegionCollisions.length > 0) {
        return closestCenter({
          ...args,
          droppableContainers: args.droppableContainers.filter((container) => {
            const data = container.data.current;

            return (
              isWidgetInstanceDragData(data) &&
              data.region === activeData.region &&
              data.instanceId !== activeData.instanceId
            );
          }),
        });
      }
    }

    return pointerCollisions;
  }

  return args.pointerCoordinates ? [] : closestCenter(collisionArgs);
};

const getCollisionData = (args: Parameters<CollisionDetection>[0], id: unknown): unknown =>
  args.droppableContainers.find((container) => container.id === id)?.data.current;

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isWidgetRegion = (value: unknown): value is WidgetRegion =>
  value === 'left' || value === 'right' || value === 'center' || value === 'bottom';
