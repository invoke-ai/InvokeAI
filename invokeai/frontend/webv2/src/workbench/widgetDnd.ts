import type { WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type { WidgetIconComponent, WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import {
  closestCenter,
  getClientRect,
  pointerWithin,
  type Collision,
  type CollisionDetection,
  type ClientRect,
} from '@dnd-kit/core';
import { arrayMove } from '@dnd-kit/sortable';

const clipsOverflow = (value: string): boolean => value !== 'visible';

/**
 * Measures a droppable's rect clipped by its overflow ancestors — the part
 * of it a user can actually see. A fully hidden droppable collapses to a
 * zero-size rect no pointer can be within.
 *
 * This must NOT be used as `measuring.droppable.measure`: dnd-kit measures
 * droppables once at drag start and only shifts the cached rect by ancestor
 * scroll deltas (width/height stay frozen), so a row hidden at drag start
 * would keep a zero-size rect even after auto-scroll reveals it. It is
 * evaluated live at collision time instead (see widgetCollisionDetection).
 */
export const measureDroppableVisibleRect = (element: HTMLElement): ClientRect => {
  const rect = { ...getClientRect(element) };
  let ancestor = element.parentElement;

  while (ancestor && ancestor !== document.body) {
    const style = getComputedStyle(ancestor);

    if (clipsOverflow(style.overflowX) || clipsOverflow(style.overflowY)) {
      const clip = ancestor.getBoundingClientRect();

      if (clipsOverflow(style.overflowY)) {
        rect.top = Math.min(Math.max(rect.top, clip.top), clip.bottom);
        rect.bottom = Math.min(Math.max(rect.bottom, clip.top), clip.bottom);
      }
      if (clipsOverflow(style.overflowX)) {
        rect.left = Math.min(Math.max(rect.left, clip.left), clip.right);
        rect.right = Math.min(Math.max(rect.right, clip.left), clip.right);
      }
    }

    ancestor = ancestor.parentElement;
  }

  rect.width = Math.max(0, rect.right - rect.left);
  rect.height = Math.max(0, rect.bottom - rect.top);

  return rect;
};

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

/**
 * Drops pointer collisions with droppables the pointer is not visibly over.
 *
 * The rects `pointerWithin` tested against are cached at drag start and never
 * re-clipped, so a droppable scrolled out of its container extends invisibly
 * across whatever is rendered below — a gallery board row below the fold
 * overlays the entire image grid, catching drags and racing the auto-scroller.
 * Re-checking the pointer against the live visible rect removes those phantom
 * hits while still letting a row that auto-scroll reveals mid-drag be hit the
 * moment it appears.
 */
const dropOccludedPointerCollisions = (
  collisions: Collision[],
  args: Parameters<CollisionDetection>[0]
): Collision[] => {
  const pointer = args.pointerCoordinates;

  if (!pointer) {
    return collisions;
  }

  return collisions.filter((collision) => {
    const node = args.droppableContainers.find((container) => container.id === collision.id)?.node.current;

    if (!node) {
      return true;
    }

    const rect = measureDroppableVisibleRect(node);

    return pointer.x >= rect.left && pointer.x <= rect.right && pointer.y >= rect.top && pointer.y <= rect.bottom;
  });
};

export const widgetCollisionDetection: CollisionDetection = (args) => {
  const activeData = args.active.data.current;
  const collisionArgs = isWidgetInstanceDragData(activeData)
    ? args
    : {
        ...args,
        droppableContainers: args.droppableContainers.filter((container) => !isWidgetDndData(container.data.current)),
      };
  const pointerCollisions = dropOccludedPointerCollisions(pointerWithin(collisionArgs), args);

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
