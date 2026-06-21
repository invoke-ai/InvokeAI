import { describe, expect, it } from 'vitest';

import type { Project, WidgetTypeId, WorkbenchState } from './types';

import {
  getRegionDropState,
  getWidgetInstanceDragData,
  getWidgetInstanceDragId,
  getWidgetRegionDropData,
  resolveWidgetDragEnd,
} from './widgetDnd';
import { createInitialWorkbenchState } from './workbenchState';

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
