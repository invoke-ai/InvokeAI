import type { WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type { WidgetIconComponent, WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import {
  closestCenter,
  getClientRect,
  pointerWithin,
  type Collision,
  type CollisionDetection,
  type ClientRect,
  type DndContextProps,
} from '@dnd-kit/core';
import { arrayMove } from '@dnd-kit/sortable';
import { isWidgetRegion as isKnownWidgetRegion } from '@workbench/layoutContracts';

/** Preserve board-list edge scrolling while non-sortable surfaces opt out. */
export const workbenchAutoScroll = {
  acceleration: 5,
  canScroll: (element) => element.getAttribute('data-dnd-auto-scroll') !== 'false',
} satisfies NonNullable<DndContextProps['autoScroll']>;

const clipsOverflow = (value: string): boolean => value !== 'visible';

/**
 * Clip against live overflow ancestors at collision time, never in dnd-kit's cached droppable measurement: hidden
 * rows must become hittable when scrolling reveals them.
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
  widgetRegions: Record<
    WidgetRegion,
    Pick<WidgetRegionState, 'activeInstanceId' | 'alignEndInstanceIds' | 'instanceIds'>
  >;
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

/** The trailing cluster of a strip (the flex spacer): dropping here aligns the widget to the end. */
export interface WidgetRegionEndDropData {
  kind: 'widget-region-end';
  region: WidgetRegion;
}

export type WidgetDndData = WidgetInstanceDragData | WidgetRegionDropData | WidgetRegionEndDropData;

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
      /** Cluster the drop landed in, when it differs from the widget's current one. */
      align?: 'start' | 'end';
    }
  | {
      fromRegion: WidgetRegion;
      instanceId: WidgetInstanceId;
      toIndex: number;
      toRegion: WidgetRegion;
      type: 'move';
      align?: 'start' | 'end';
    };

export const getWidgetInstanceDragId = (region: WidgetRegion, instanceId: WidgetInstanceId): string =>
  `widget-instance:${region}:${instanceId}`;

export const getWidgetRegionDropId = (region: WidgetRegion): string => `widget-region:${region}`;

export const getWidgetRegionEndDropId = (region: WidgetRegion): string => `widget-region-end:${region}`;

export const getWidgetRegionEndDropData = (region: WidgetRegion): WidgetRegionEndDropData => ({
  kind: 'widget-region-end',
  region,
});

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

export const isWidgetRegionEndDropData = (data: unknown): data is WidgetRegionEndDropData =>
  isRecord(data) && data.kind === 'widget-region-end' && isWidgetRegion(data.region);

export const isWidgetDndData = (data: unknown): data is WidgetDndData =>
  isWidgetInstanceDragData(data) || isWidgetRegionDropData(data) || isWidgetRegionEndDropData(data);

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
    const overRegionState = project.widgetRegions[overRegion];
    const alignEndIds = overRegionState.alignEndInstanceIds ?? [];
    const isActiveEnd = alignEndIds.includes(activeData.instanceId);

    // Dropping on the strip's trailing spacer joins the end cluster.
    if (overData.kind === 'widget-region-end') {
      const oldIndex = overRegionState.instanceIds.indexOf(activeData.instanceId);

      if (oldIndex === -1) {
        return null;
      }

      return {
        activeInstanceId: activeData.instanceId,
        ...(isActiveEnd ? {} : { align: 'end' }),
        instanceIds: arrayMove(overRegionState.instanceIds, oldIndex, overRegionState.instanceIds.length - 1),
        region: overRegion,
        type: 'reorder',
      };
    }

    if (overData.kind !== 'widget-instance') {
      return null;
    }

    const oldIndex = overRegionState.instanceIds.indexOf(activeData.instanceId);
    const toIndex = Math.max(0, overRegionState.instanceIds.indexOf(overData.instanceId));

    if (oldIndex === -1 || overData.instanceId === activeData.instanceId) {
      return null;
    }

    // Landing beside a widget adopts that widget's cluster.
    const isOverEnd = alignEndIds.includes(overData.instanceId);

    return {
      activeInstanceId: activeData.instanceId,
      ...(isOverEnd === isActiveEnd ? {} : { align: isOverEnd ? 'end' : 'start' }),
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
    overData.kind === 'widget-instance'
      ? Math.max(0, overRegionState.instanceIds.indexOf(overData.instanceId))
      : overRegionState.instanceIds.length;
  // Adopt the target cluster; discard remembered end alignment when dropping at the start.
  const targetIsEnd =
    overData.kind === 'widget-region-end' ||
    (overData.kind === 'widget-instance' && (overRegionState.alignEndInstanceIds ?? []).includes(overData.instanceId));
  const hasStaleEndAlignment = (overRegionState.alignEndInstanceIds ?? []).includes(activeData.instanceId);
  const align = targetIsEnd ? ('end' as const) : hasStaleEndAlignment ? ('start' as const) : undefined;

  return {
    ...(align ? { align } : {}),
    fromRegion,
    instanceId: activeData.instanceId,
    toIndex,
    toRegion: overRegion,
    type: 'move',
  };
};

/**
 * Recheck cached pointer hits against live clipped rects so scrolled-out rows cannot intercept drops over
 * neighboring content.
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
      // Prefer a direct spacer hit over its containing region to preserve trailing-cluster drops.
      const endZoneCollisions = pointerCollisions.filter((collision) =>
        isWidgetRegionEndDropData(getCollisionData(args, collision.id))
      );

      if (endZoneCollisions.length > 0) {
        return endZoneCollisions;
      }

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

const isWidgetRegion = (value: unknown): value is WidgetRegion => isKnownWidgetRegion(value);
