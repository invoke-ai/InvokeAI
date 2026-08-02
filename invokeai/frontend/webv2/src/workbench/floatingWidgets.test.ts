import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { describe, expect, it } from 'vitest';

import {
  cascadeDefaultGeometry,
  clampSizeToMinimum,
  clampWindowToViewport,
  FLOATING_MIN_HEIGHT_PX,
  FLOATING_MIN_WIDTH_PX,
  nextStackOrder,
} from './floatingWindows';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const getActiveProject = (state: WorkbenchState): Project => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  if (!project) {
    throw new Error('missing active project');
  }

  return project;
};

const floatGallery = (state = createInitialWorkbenchState()): WorkbenchState =>
  workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });

describe('floatWidget', () => {
  it('detaches the instance from its region and repairs the active instance', () => {
    const initial = createInitialWorkbenchState();
    const before = getActiveProject(initial).widgetRegions.right;
    expect(before.instanceIds).toContain('gallery');
    expect(before.activeInstanceId).toBe('gallery');

    const state = floatGallery(initial);
    const project = getActiveProject(state);

    expect(project.widgetRegions.right.instanceIds).not.toContain('gallery');
    expect(project.widgetRegions.right.activeInstanceId).not.toBe('gallery');
    expect(project.widgetRegions.right.instanceIds).toContain(project.widgetRegions.right.activeInstanceId);
    expect(project.floatingWidgets?.gallery).toMatchObject({ mode: 'windowed', returnRegion: 'right', stackOrder: 1 });
  });

  it('is a no-op for unknown or already-floating instances', () => {
    const floated = floatGallery();

    expect(workbenchReducer(floated, { instanceId: 'gallery', type: 'floatWidget' })).toBe(floated);
    expect(workbenchReducer(floated, { instanceId: 'no-such-widget', type: 'floatWidget' })).toBe(floated);
  });

  it('cascades geometry and stacks successive windows', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    const floating = getActiveProject(state).floatingWidgets;

    expect(floating?.queue.stackOrder).toBe(2);
    expect(floating?.queue.x).toBeGreaterThan(floating?.gallery.x ?? 0);
  });
});

describe('dockFloatingWidget', () => {
  it('returns the instance to its origin region as the active widget', () => {
    const state = workbenchReducer(floatGallery(), { instanceId: 'gallery', type: 'dockFloatingWidget' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.gallery).toBeUndefined();
    expect(project.widgetRegions.right.instanceIds).toContain('gallery');
    expect(project.widgetRegions.right.activeInstanceId).toBe('gallery');
    expect(project.layout.panels.isRightOpen).toBe(true);
    expect(project.widgetRegions.right.isCollapsed).toBe(false);
  });

  it('float -> dock -> float round-trips', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });

    expect(getActiveProject(state).floatingWidgets?.gallery).toBeDefined();
    expect(getActiveProject(state).widgetRegions.right.instanceIds).not.toContain('gallery');
  });

  it('is a no-op when nothing is floating', () => {
    const initial = createInitialWorkbenchState();

    expect(workbenchReducer(initial, { instanceId: 'gallery', type: 'dockFloatingWidget' })).toBe(initial);
  });
});

describe('geometry, mode, and stacking actions', () => {
  it('clamps committed sizes to the minimums', () => {
    const state = workbenchReducer(floatGallery(), {
      heightPx: 10,
      instanceId: 'gallery',
      type: 'setFloatingWidgetGeometry',
      widthPx: 10,
      x: 5,
      y: 7,
    });
    const floating = getActiveProject(state).floatingWidgets?.gallery;

    expect(floating).toMatchObject({ heightPx: FLOATING_MIN_HEIGHT_PX, widthPx: FLOATING_MIN_WIDTH_PX, x: 5, y: 7 });
  });

  it('switches modes and ignores repeats', () => {
    const state = workbenchReducer(floatGallery(), {
      instanceId: 'gallery',
      mode: 'maximized',
      type: 'setFloatingWidgetMode',
    });

    expect(getActiveProject(state).floatingWidgets?.gallery.mode).toBe('maximized');
    const repeat = workbenchReducer(state, { instanceId: 'gallery', mode: 'maximized', type: 'setFloatingWidgetMode' });
    expect(repeat).toBe(state);
  });

  it('focus raises a window to the top and no-ops when already topmost', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'focusFloatingWidget' });
    const floating = getActiveProject(state).floatingWidgets;
    expect(floating?.gallery.stackOrder).toBeGreaterThan(floating?.queue.stackOrder ?? 0);

    expect(workbenchReducer(state, { instanceId: 'gallery', type: 'focusFloatingWidget' })).toBe(state);
  });
});

describe('interaction with region placement', () => {
  it('openRegionWidget docks a floating instance instead of double-rendering it', () => {
    const floated = floatGallery();
    const state = workbenchReducer(floated, { region: 'right', type: 'openRegionWidget', widgetId: 'gallery' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.gallery).toBeUndefined();
    expect(project.widgetRegions.right.instanceIds).toContain('gallery');
  });
});

describe('pure helpers', () => {
  it('nextStackOrder starts at 1 and increments past the max', () => {
    expect(nextStackOrder(undefined)).toBe(1);
    expect(
      nextStackOrder({
        a: { heightPx: 1, mode: 'windowed', returnRegion: 'right', stackOrder: 4, widthPx: 1, x: 0, y: 0 },
      })
    ).toBe(5);
  });

  it('cascadeDefaultGeometry offsets successive windows', () => {
    expect(cascadeDefaultGeometry(1).x).toBe(cascadeDefaultGeometry(0).x + 32);
  });

  it('clampWindowToViewport keeps a grabbable sliver on screen', () => {
    const clamped = clampWindowToViewport(
      { heightPx: 300, widthPx: 400, x: 5000, y: 5000 },
      { height: 800, width: 1200 }
    );
    expect(clamped.x).toBeLessThanOrEqual(1200 - 48);
    expect(clamped.y).toBeLessThanOrEqual(800 - 48);

    const negative = clampWindowToViewport(
      { heightPx: 300, widthPx: 400, x: -5000, y: -50 },
      { height: 800, width: 1200 }
    );
    expect(negative.x).toBeGreaterThanOrEqual(48 - 400);
    expect(negative.y).toBe(0);
  });

  it('clampSizeToMinimum enforces the floor without moving the window', () => {
    expect(clampSizeToMinimum({ heightPx: 1, widthPx: 1, x: 9, y: 9 })).toEqual({
      heightPx: FLOATING_MIN_HEIGHT_PX,
      widthPx: FLOATING_MIN_WIDTH_PX,
      x: 9,
      y: 9,
    });
  });
});

describe('interaction with presets and undo', () => {
  it('applying a preset docks all floating widgets (no double render)', () => {
    const floated = floatGallery();
    const state = workbenchReducer(floated, { presetId: 'compose', type: 'applyPreset' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets ?? {}).toEqual({});
    expect(project.widgetRegions.right.instanceIds).toContain('gallery');
  });

  it('undo restores floating state together with the regions', () => {
    // Snapshot A: gallery docked. The undoable action (apply preset) captures it.
    let state = createInitialWorkbenchState();
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    // Float after the snapshot, then undo: gallery must return to docked-only.
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    let project = getActiveProject(state);

    expect(project.floatingWidgets ?? {}).toEqual({});
    expect(project.widgetRegions.right.instanceIds).toContain('gallery');

    // Mirror case: float, take a snapshot, dock, undo -> floating state returns
    // WITH the matching regions; the widget is exactly one of docked/floating.
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    const isFloating = Boolean(project.floatingWidgets?.gallery);
    const isDocked = Object.values(project.widgetRegions).some((region) => region.instanceIds.includes('gallery'));
    expect(isFloating !== isDocked).toBe(true);
  });
});
