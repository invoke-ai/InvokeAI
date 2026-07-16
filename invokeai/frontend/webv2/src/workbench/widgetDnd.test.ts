import type { CollisionDetection, DroppableContainer } from '@dnd-kit/core';

import { describe, expect, it } from 'vitest';

import type { Project, WidgetTypeId, WorkbenchState } from './types';

import {
  getRegionDropState,
  getWidgetInstanceDragData,
  getWidgetInstanceDragId,
  getWidgetRegionDropData,
  resolveWidgetDragEnd,
  widgetCollisionDetection,
} from './widgetDnd';
import { getCanvasImageDropData } from './widgets/canvas/canvasImageDnd';
import { getGalleryImageDragData } from './widgets/gallery/galleryDnd';
import { createInitialWorkbenchState } from './workbenchState';

type CollisionArgs = Parameters<CollisionDetection>[0];

const getRect = (left: number, top: number, width: number, height: number) => ({
  bottom: top + height,
  height,
  left,
  right: left + width,
  top,
  width,
});

const createDroppable = (id: string, data: object, rect: ReturnType<typeof getRect>): DroppableContainer => ({
  data: { current: data as Record<string, unknown> },
  disabled: false,
  id,
  key: id,
  node: { current: null },
  rect: { current: rect },
});

const createCollisionArgs = (options: {
  activeData: object;
  collisionRect?: ReturnType<typeof getRect>;
  droppables: Array<{ data: object; id: string; rect: ReturnType<typeof getRect> }>;
  pointerCoordinates: { x: number; y: number } | null;
}): CollisionArgs => {
  const collisionRect = options.collisionRect ?? getRect(45, 45, 10, 10);

  return {
    active: {
      data: { current: options.activeData as Record<string, unknown> },
      id: 'active',
      rect: { current: { initial: collisionRect, translated: collisionRect } },
    },
    collisionRect,
    droppableContainers: options.droppables.map(({ data, id, rect }) => createDroppable(id, data, rect)),
    droppableRects: new Map(options.droppables.map(({ id, rect }) => [id, rect])),
    pointerCoordinates: options.pointerCoordinates,
  };
};

const getWidget = (typeId: WidgetTypeId) => {
  const widgets: Partial<
    Record<
      WidgetTypeId,
      { manifest: { allowedRegions: Array<'left' | 'right' | 'center' | 'bottom'>; allowMultiple: boolean } }
    >
  > = {
    canvas: { manifest: { allowedRegions: ['center'], allowMultiple: false } },
    preview: { manifest: { allowedRegions: ['center', 'right'], allowMultiple: false } },
  };

  return widgets[typeId];
};

const getActiveProject = (state: WorkbenchState): Project => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project as Project;
};

const createProjectWithPreviewInCenterAndRight = (): Project => {
  const project = getActiveProject(createInitialWorkbenchState());

  return {
    ...project,
    widgetRegions: {
      ...project.widgetRegions,
      center: { ...project.widgetRegions.center, activeInstanceId: 'preview', instanceIds: ['canvas', 'preview'] },
      right: { ...project.widgetRegions.right, activeInstanceId: 'preview', instanceIds: ['preview', 'layers'] },
    },
  };
};

describe('widget dnd helpers', () => {
  it('uses region-scoped draggable ids for duplicated widget instances', () => {
    expect(getWidgetInstanceDragId('center', 'preview')).not.toBe(getWidgetInstanceDragId('right', 'preview'));
  });

  it('does not treat a duplicate instance in another region as a same-region reorder', () => {
    const project = createProjectWithPreviewInCenterAndRight();
    const resolution = resolveWidgetDragEnd(
      project,
      getWidgetInstanceDragData('center', 'preview', 'preview'),
      getWidgetInstanceDragData('right', 'preview', 'preview'),
      getWidget
    );

    expect(resolution).toBeNull();
  });

  it('reorders only within the active source region', () => {
    const project = createProjectWithPreviewInCenterAndRight();
    const resolution = resolveWidgetDragEnd(
      project,
      getWidgetInstanceDragData('center', 'preview', 'preview'),
      getWidgetInstanceDragData('center', 'canvas', 'canvas'),
      getWidget
    );

    expect(resolution).toEqual({
      activeInstanceId: 'preview',
      instanceIds: ['preview', 'canvas'],
      region: 'center',
      type: 'reorder',
    });
  });

  it('disables a non-source region that already contains the dragged singleton instance', () => {
    const project = createProjectWithPreviewInCenterAndRight();
    const dropState = getRegionDropState(
      project,
      {
        fromRegion: 'center',
        icon: () => null,
        instanceId: 'preview',
        label: 'Preview',
        typeId: 'preview',
      },
      'right',
      getWidget
    );

    expect(dropState).toMatchObject({ helperText: 'Already placed', isActive: true, isAllowed: false });
  });

  it('ignores source region bar collisions so sorting only reacts to widget items', () => {
    const project = createProjectWithPreviewInCenterAndRight();
    const resolution = resolveWidgetDragEnd(
      project,
      getWidgetInstanceDragData('center', 'canvas', 'canvas'),
      getWidgetRegionDropData('center'),
      getWidget
    );

    expect(resolution).toBeNull();
  });

  it('moves to another eligible region when dropped on that region bar', () => {
    const project = getActiveProject(createInitialWorkbenchState());
    const projectWithoutRightPreview = {
      ...project,
      widgetRegions: {
        ...project.widgetRegions,
        right: {
          ...project.widgetRegions.right,
          instanceIds: project.widgetRegions.right.instanceIds.filter((instanceId) => instanceId !== 'preview'),
        },
      },
    };
    const resolution = resolveWidgetDragEnd(
      projectWithoutRightPreview,
      getWidgetInstanceDragData('center', 'preview', 'preview'),
      getWidgetRegionDropData('right'),
      getWidget
    );

    expect(resolution).toEqual({
      fromRegion: 'center',
      instanceId: 'preview',
      toIndex: projectWithoutRightPreview.widgetRegions.right.instanceIds.length,
      toRegion: 'right',
      type: 'move',
    });
  });
});

describe('widget collision detection', () => {
  const outerRect = getRect(0, 0, 100, 100);
  const innerRect = getRect(25, 25, 50, 50);
  const pointerCoordinates = { x: 50, y: 50 };

  it('routes a gallery image drag to an inner canvas target instead of its enclosing widget', () => {
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        activeData: getGalleryImageDragData([{ boardId: 'none', imageName: 'image.png' }]),
        droppables: [
          {
            data: getWidgetInstanceDragData('center', 'canvas', 'canvas'),
            id: 'canvas-widget',
            rect: outerRect,
          },
          { data: getWidgetRegionDropData('center'), id: 'center-region', rect: outerRect },
          { data: getCanvasImageDropData('raster'), id: 'canvas-raster-target', rect: innerRect },
          { data: { kind: 'future-target' }, id: 'future-target', rect: innerRect },
        ],
        pointerCoordinates,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['canvas-raster-target', 'future-target']);
  });

  it('keeps widget instances ahead of overlapping non-widget targets for widget drags', () => {
    const activeData = getWidgetInstanceDragData('center', 'preview', 'preview');
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        activeData,
        droppables: [
          { data: getWidgetInstanceDragData('center', 'canvas', 'canvas'), id: 'canvas-widget', rect: outerRect },
          { data: getCanvasImageDropData('raster'), id: 'canvas-raster-target', rect: innerRect },
        ],
        pointerCoordinates,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['canvas-widget']);
  });

  it('keeps widget regions ahead of overlapping non-widget targets for widget drags', () => {
    const activeData = getWidgetInstanceDragData('center', 'preview', 'preview');
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        activeData,
        droppables: [
          { data: getWidgetRegionDropData('right'), id: 'right-region', rect: outerRect },
          { data: getCanvasImageDropData('raster'), id: 'canvas-raster-target', rect: innerRect },
        ],
        pointerCoordinates,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['right-region']);
  });

  it('uses same-region widget geometry instead of the source-region bar', () => {
    const activeData = getWidgetInstanceDragData('center', 'preview', 'preview');
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        activeData,
        collisionRect: getRect(90, 90, 20, 20),
        droppables: [
          { data: activeData, id: 'active-widget', rect: getRect(0, 0, 20, 20) },
          {
            data: getWidgetInstanceDragData('center', 'canvas', 'canvas'),
            id: 'other-center-widget',
            rect: getRect(100, 100, 20, 20),
          },
          { data: getWidgetRegionDropData('center'), id: 'center-region', rect: getRect(40, 40, 20, 20) },
        ],
        pointerCoordinates,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['other-center-widget']);
  });

  it('treats malformed widget active data as a non-widget drag', () => {
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        activeData: { instanceId: 'preview', kind: 'widget-instance', region: 'center' },
        droppables: [
          { data: getWidgetInstanceDragData('center', 'canvas', 'canvas'), id: 'canvas-widget', rect: outerRect },
          { data: getCanvasImageDropData('raster'), id: 'canvas-raster-target', rect: innerRect },
        ],
        pointerCoordinates,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['canvas-raster-target']);
  });

  it('excludes widget droppables from the no-pointer closest-center fallback for non-widget drags', () => {
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        activeData: getGalleryImageDragData([{ boardId: 'none', imageName: 'image.png' }]),
        collisionRect: getRect(0, 0, 20, 20),
        droppables: [
          {
            data: getWidgetInstanceDragData('center', 'canvas', 'canvas'),
            id: 'canvas-widget',
            rect: getRect(0, 0, 20, 20),
          },
          {
            data: getWidgetRegionDropData('center'),
            id: 'center-region',
            rect: getRect(20, 20, 20, 20),
          },
          {
            data: getCanvasImageDropData('raster'),
            id: 'canvas-raster-target',
            rect: getRect(100, 100, 20, 20),
          },
          { data: { kind: 'future-target' }, id: 'future-target', rect: getRect(100, 100, 20, 20) },
        ],
        pointerCoordinates: null,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['canvas-raster-target', 'future-target']);
  });

  it('keeps the existing no-pointer closest-center behavior for widget drags', () => {
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        activeData: getWidgetInstanceDragData('center', 'preview', 'preview'),
        collisionRect: getRect(0, 0, 20, 20),
        droppables: [
          {
            data: getWidgetInstanceDragData('center', 'canvas', 'canvas'),
            id: 'canvas-widget',
            rect: getRect(0, 0, 20, 20),
          },
          {
            data: getWidgetRegionDropData('right'),
            id: 'right-region',
            rect: getRect(20, 20, 20, 20),
          },
          {
            data: getCanvasImageDropData('raster'),
            id: 'canvas-raster-target',
            rect: getRect(100, 100, 20, 20),
          },
        ],
        pointerCoordinates: null,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['canvas-widget', 'right-region', 'canvas-raster-target']);
  });
});
