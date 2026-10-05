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
import {
  areLayoutPresetSnapshotsEqual,
  createLayoutPresetSnapshot,
  doesProjectMatchLayoutPreset,
  resolveSavedLayoutPreset,
} from './layoutPresetSnapshots';
import { serializeProjectDocumentV3 } from './projects/projectDocument';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from './widgetPlacementMeta';
import { createWidgetRegionViewModelFromState } from './widgetRegionViewModel';
import { getWidgetsForRegion } from './widgetRegistry';
import { normalizeWorkbenchAccount, normalizeWorkbenchProject } from './workbenchState';
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

const getRegionsHolding = (project: Project, instanceId: string): string[] =>
  Object.entries(project.widgetRegions)
    .filter(([, region]) => region.instanceIds.includes(instanceId))
    .map(([regionId]) => regionId);

const withActiveProject = (state: WorkbenchState, update: (project: Project) => Project): WorkbenchState => ({
  ...state,
  projects: state.projects.map((project) => (project.id === state.activeProjectId ? update(project) : project)),
});

/** What a reload does to the active project: the autosaved JSON comes back through normalization. */
const reload = (state: WorkbenchState): WorkbenchState =>
  withActiveProject(state, (project) => normalizeWorkbenchProject(JSON.parse(JSON.stringify(project))));

/** The right rail exactly as the shell derives it: reducer state -> placement selector -> region view model. */
const getRightRail = (state: WorkbenchState): string[] => {
  const placement = getWidgetPlacementProject(getActiveProject(state));

  return createWidgetRegionViewModelFromState({
    floatingWidgets: placement.floatingPlacements,
    region: 'right',
    regionState: getActiveProject(state).widgetRegions.right,
    widgetInstances: placement.widgetInstances,
    widgets: getWidgetsForRegion('right'),
  }).placedItems.map((item) => (item.isFloating ? `${item.id}*` : item.id));
};

/** Video shows Preview only in the center; put it on the right rail too, then float it from there. */
const floatVideoPreviewFromRail = (): WorkbenchState => {
  let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'video', type: 'applyPreset' });
  state = workbenchReducer(state, { region: 'right', type: 'openRegionWidget', widgetId: 'preview' });

  return workbenchReducer(state, { instanceId: 'preview', region: 'right', type: 'floatWidget' });
};

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

  it('floats the last center view, leaving the surface to its fallback view', () => {
    let state = createInitialWorkbenchState();
    const centerInstanceIds = getActiveProject(state).widgetRegions.center.instanceIds;

    for (const instanceId of centerInstanceIds.slice(1)) {
      state = workbenchReducer(state, { region: 'center', type: 'toggleRegionWidget', widgetId: instanceId });
    }

    const lastCenterInstanceId = centerInstanceIds[0];
    expect(getActiveProject(state).widgetRegions.center.instanceIds).toEqual([lastCenterInstanceId]);

    state = workbenchReducer(state, { instanceId: lastCenterInstanceId, type: 'floatWidget' });
    const center = getActiveProject(state).widgetRegions.center;

    // An empty center retains the floated active instance as its dock-back target and uses its fallback view.
    expect(center.instanceIds).toEqual([]);
    expect(center.activeInstanceId).toBe(lastCenterInstanceId);
    expect(center.isCollapsed).toBe(false);
    expect(getActiveProject(state).floatingWidgets?.[lastCenterInstanceId]).toMatchObject({
      returnRegion: 'center',
    });
  });

  it('leaves every region that held the instance and records one return destination', () => {
    // Edit shows Preview in the center and on the right rail at once.
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'right', type: 'floatWidget' });
    const project = getActiveProject(state);

    expect(getRegionsHolding(project, 'preview')).toEqual([]);
    expect(project.floatingWidgets?.preview).toMatchObject({ returnIndex: 1, returnRegion: 'right' });
    // The center had another view, so it simply moves on.
    expect(project.widgetRegions.center).toMatchObject({ activeInstanceId: 'canvas', instanceIds: ['canvas'] });
  });

  it('returns to the region the float was asked from, not the first member region', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'center', type: 'floatWidget' });

    expect(getActiveProject(state).floatingWidgets?.preview).toMatchObject({ returnRegion: 'center' });

    const docked = getActiveProject(workbenchReducer(state, { instanceId: 'preview', type: 'dockFloatingWidget' }));

    expect(docked.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['canvas', 'preview'],
    });
    // The rail membership was given up on float; docking restores the one recorded origin.
    expect(docked.widgetRegions.right.instanceIds).not.toContain('preview');
  });

  it('picks a member region deterministically when the float carries no region hint', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', type: 'floatWidget' });

    // Rails come before the center, whatever order the regions happen to be stored in.
    expect(getActiveProject(state).floatingWidgets?.preview).toMatchObject({ returnRegion: 'right' });
  });

  it('collapses a rail it empties instead of leaving it open and blank', () => {
    let state = createInitialWorkbenchState();
    const rightInstanceIds = getActiveProject(state).widgetRegions.right.instanceIds;

    for (const instanceId of rightInstanceIds.slice(1)) {
      state = workbenchReducer(state, { region: 'right', type: 'toggleRegionWidget', widgetId: instanceId });
    }

    state = workbenchReducer(state, { region: 'right', type: 'setRegionWidgetCollapsed', isCollapsed: false });
    state = workbenchReducer(state, { instanceId: rightInstanceIds[0], type: 'floatWidget' });
    const right = getActiveProject(state).widgetRegions.right;

    expect(right.instanceIds).toEqual([]);
    expect(right.isCollapsed).toBe(true);
  });
});

describe('closeFloatingWidget', () => {
  it('drops the window without docking, leaving the rail and its disclosure untouched', () => {
    const floated = floatGallery();
    const rightBefore = getActiveProject(floated).widgetRegions.right;
    const state = workbenchReducer(floated, { instanceId: 'gallery', type: 'closeFloatingWidget' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets).toBeUndefined();
    expect(project.widgetRegions.right).toBe(rightBefore);
    expect(getRegionsHolding(project, 'gallery')).toEqual([]);
    // The instance survives for a later placement, as a closed tab's does.
    expect(project.widgetInstances.gallery).toBeDefined();
  });

  it('is a no-op for an instance that is not floating', () => {
    const state = createInitialWorkbenchState();

    expect(workbenchReducer(state, { instanceId: 'gallery', type: 'closeFloatingWidget' })).toBe(state);
  });

  it('frees the rail slot, closing the gap ahead of later markers', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    expect(getRightRail(state)).toEqual(['gallery*', 'image-map', 'queue*']);

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'closeFloatingWidget' });

    expect(getRightRail(state)).toEqual(['image-map', 'queue*']);
    expect(getActiveProject(state).floatingWidgets?.queue.returnIndex).toBe(1);
  });
});

describe('empty-center recovery', () => {
  it('Video: docking gives both the right rail and the emptied center their Preview back', () => {
    const floated = floatVideoPreviewFromRail();
    const emptied = getActiveProject(floated).widgetRegions.center;

    // The center keeps naming the view it lost; that pointer is the whole record of the debt.
    expect(emptied).toMatchObject({ activeInstanceId: 'preview', instanceIds: [], isCollapsed: false });

    const project = getActiveProject(workbenchReducer(floated, { instanceId: 'preview', type: 'dockFloatingWidget' }));

    expect(project.floatingWidgets).toBeUndefined();
    expect(project.widgetRegions.center).toMatchObject({ activeInstanceId: 'preview', instanceIds: ['preview'] });
    expect(project.widgetRegions.right).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['gallery', 'queue', 'preview'],
      isCollapsed: false,
    });
  });

  it('Video: removing gives the center its Preview back but restores no rail slot and opens no panel', () => {
    let floated = floatVideoPreviewFromRail();
    floated = workbenchReducer(floated, { region: 'right', type: 'setRegionWidgetCollapsed', isCollapsed: true });
    const before = getActiveProject(floated);
    const project = getActiveProject(workbenchReducer(floated, { instanceId: 'preview', type: 'closeFloatingWidget' }));

    expect(project.floatingWidgets).toBeUndefined();
    expect(project.widgetRegions.center).toMatchObject({ activeInstanceId: 'preview', instanceIds: ['preview'] });
    expect(project.widgetRegions.right).toBe(before.widgetRegions.right);
    expect(project.layout).toBe(before.layout);
  });

  it('Video: both recoveries still work after a reload', () => {
    const reloaded = reload(floatVideoPreviewFromRail());

    expect(getActiveProject(reloaded).widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: [],
    });

    const docked = getActiveProject(workbenchReducer(reloaded, { instanceId: 'preview', type: 'dockFloatingWidget' }));
    const removed = getActiveProject(
      workbenchReducer(reloaded, { instanceId: 'preview', type: 'closeFloatingWidget' })
    );

    expect(getRegionsHolding(docked, 'preview')).toEqual(['right', 'center']);
    expect(getRegionsHolding(removed, 'preview')).toEqual(['center']);
  });

  it('Edit: docking returns Preview to the rail only, because the center kept its other view', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    const railBefore = [...getActiveProject(state).widgetRegions.right.instanceIds];

    state = workbenchReducer(state, { instanceId: 'preview', region: 'right', type: 'floatWidget' });
    state = workbenchReducer(state, { instanceId: 'preview', type: 'dockFloatingWidget' });
    const project = getActiveProject(state);

    expect(project.widgetRegions.right.instanceIds).toEqual(railBefore);
    // Not a round trip: the center's Preview membership is gone for good.
    expect(project.widgetRegions.center.instanceIds).toEqual(['canvas']);
  });

  it.each(['dockFloatingWidget', 'closeFloatingWidget'] as const)(
    '%s leaves the center alone once it has gained another view',
    (type) => {
      let state = floatVideoPreviewFromRail();
      state = workbenchReducer(state, { region: 'center', type: 'openRegionWidget', widgetId: 'gallery' });
      const center = getActiveProject(state).widgetRegions.center;

      expect(center.instanceIds).not.toContain('preview');

      state = workbenchReducer(state, { instanceId: 'preview', type });

      expect(getActiveProject(state).widgetRegions.center).toBe(center);
    }
  );

  it.each(['dockFloatingWidget', 'closeFloatingWidget'] as const)(
    '%s leaves an empty center alone once its pointer names another view',
    (type) => {
      const state = withActiveProject(floatVideoPreviewFromRail(), (project) => ({
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          center: { ...project.widgetRegions.center, activeInstanceId: 'canvas' },
        },
      }));
      const center = getActiveProject(state).widgetRegions.center;
      const project = getActiveProject(workbenchReducer(state, { instanceId: 'preview', type }));

      expect(project.widgetRegions.center).toBe(center);
    }
  );

  it('does not duplicate the view when the center is itself the return region', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'video', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'center', type: 'floatWidget' });

    const docked = getActiveProject(workbenchReducer(state, { instanceId: 'preview', type: 'dockFloatingWidget' }));
    const removed = getActiveProject(workbenchReducer(state, { instanceId: 'preview', type: 'closeFloatingWidget' }));

    expect(docked.widgetRegions.center.instanceIds).toEqual(['preview']);
    expect(removed.widgetRegions.center.instanceIds).toEqual(['preview']);
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

  it.each([
    ['gallery', 'queue'],
    ['queue', 'gallery'],
  ])('docks %s then %s without moving either return destination', (first, second) => {
    const initial = createInitialWorkbenchState();
    const before = [...getActiveProject(initial).widgetRegions.right.instanceIds];
    expect(before).toEqual(['gallery', 'image-map', 'queue']);

    let state = workbenchReducer(initial, { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    expect(getRightRail(state)).toEqual(['gallery*', 'image-map', 'queue*']);

    state = workbenchReducer(state, { instanceId: first, type: 'dockFloatingWidget' });
    // The window still out keeps the slot the rail was already showing for it.
    expect(getRightRail(state).map((id) => id.replace('*', ''))).toEqual(before);

    state = workbenchReducer(state, { instanceId: second, type: 'dockFloatingWidget' });
    expect(getActiveProject(state).widgetRegions.right.instanceIds).toEqual(before);
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

  it('ignores a commit that lands on the geometry already stored', () => {
    const state = floatGallery();
    const floating = getActiveProject(state).floatingWidgets?.gallery;

    // What a pointer-down/up with no movement sends: the starting geometry.
    expect(
      workbenchReducer(state, {
        heightPx: floating?.heightPx ?? 0,
        instanceId: 'gallery',
        type: 'setFloatingWidgetGeometry',
        widthPx: floating?.widthPx ?? 0,
        x: floating?.x ?? 0,
        y: floating?.y ?? 0,
      })
    ).toBe(state);
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

  it('keeps stack orders compact instead of climbing on every raise', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });

    for (const instanceId of ['gallery', 'queue', 'gallery', 'queue', 'gallery']) {
      state = workbenchReducer(state, { instanceId, type: 'raiseFloatingWidget' });
    }

    const floating = getActiveProject(state).floatingWidgets;

    // Two windows, so two orders — no matter how many times they trade places.
    expect(
      Object.values(floating ?? {})
        .map((window) => window.stackOrder)
        .sort()
    ).toEqual([1, 2]);
    expect(floating?.gallery.stackOrder).toBe(2);
  });

  it('focus raises a window to the top and no-ops when already topmost', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'raiseFloatingWidget' });
    const floating = getActiveProject(state).floatingWidgets;
    expect(floating?.gallery.stackOrder).toBeGreaterThan(floating?.queue.stackOrder ?? 0);

    expect(workbenchReducer(state, { instanceId: 'gallery', type: 'raiseFloatingWidget' })).toBe(state);
  });
});

describe('interaction with region placement', () => {
  it('openRegionWidget docks a floating instance instead of double-rendering it', () => {
    const floated = floatGallery();
    const state = workbenchReducer(floated, { region: 'right', type: 'openRegionWidget', widgetId: 'gallery' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets).toBeUndefined();
    // Back at its return slot, not appended.
    expect(project.widgetRegions.right).toMatchObject({
      activeInstanceId: 'gallery',
      instanceIds: ['gallery', 'image-map', 'queue'],
    });
  });

  it('openRegionWidget restores the return placement before opening the requested one', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'right', type: 'floatWidget' });
    const layoutBefore = getActiveProject(state).layout;

    state = workbenchReducer(state, { region: 'center', type: 'openRegionWidget', widgetId: 'preview' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets).toBeUndefined();
    expect(project.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['canvas', 'preview'],
    });
    // The rail gets its tab back where it was, without being brought to the front or opened.
    expect(project.widgetRegions.right).toMatchObject({
      activeInstanceId: 'layers',
      instanceIds: ['layers', 'preview'],
    });
    expect(project.layout.panels).toEqual(layoutBefore.panels);
  });

  it('openRegionWidget routes once, to the widget it reveals', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'generate', type: 'floatWidget' });
    state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });

    state = workbenchReducer(state, { region: 'left', type: 'openRegionWidget', widgetId: 'generate' });

    expect(getActiveProject(state).invocation.sourceId).toBe('generate');
    expect(getActiveProject(state).widgetRegions.left.instanceIds).toEqual(['generate', 'upscale']);
  });

  it('openRegionWidget hands an emptied center its view back when it docks the window', () => {
    // Opening Preview on the left: the window returns to the right rail, the center it emptied gets it back,
    // and the requested rail shows it.
    const state = workbenchReducer(floatVideoPreviewFromRail(), {
      region: 'left',
      type: 'openRegionWidget',
      widgetId: 'preview',
    });
    const project = getActiveProject(state);

    expect(project.floatingWidgets).toBeUndefined();
    expect(getRegionsHolding(project, 'preview').sort()).toEqual(['center', 'left', 'right']);
    expect(project.widgetRegions.center).toMatchObject({ activeInstanceId: 'preview', instanceIds: ['preview'] });
    expect(project.widgetRegions.left.activeInstanceId).toBe('preview');
  });

  it('a stale move or reorder cannot place a floating instance back in a region', () => {
    const state = floatGallery();
    const moved = workbenchReducer(state, {
      fromRegion: 'right',
      instanceId: 'gallery',
      toIndex: 0,
      toRegion: 'left',
      type: 'moveWidgetInstance',
    });
    const reordered = workbenchReducer(state, {
      instanceIds: ['gallery', 'image-map', 'queue'],
      region: 'right',
      type: 'reorderWidgetInstances',
    });

    expect(moved).toBe(state);
    expect(reordered).toBe(state);
  });

  it('a stale selection of a floating instance changes nothing and keeps the window', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'right', type: 'floatWidget' });

    for (const region of ['right', 'center'] as const) {
      expect(workbenchReducer(state, { region, type: 'selectRegionWidget', widgetId: 'preview' })).toBe(state);
    }
  });

  it('a stale toggle cannot enable a floating instance in a region', () => {
    const state = floatGallery();

    expect(workbenchReducer(state, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' })).toBe(state);
  });
});

describe('rail order around floating markers', () => {
  // The default right rail is gallery, image-map, queue; image-map floats, leaving a marker between the others.
  const floatImageMap = (): WorkbenchState =>
    workbenchReducer(createInitialWorkbenchState(), { instanceId: 'image-map', type: 'floatWidget' });
  const dockImageMap = (state: WorkbenchState): string[] =>
    getActiveProject(workbenchReducer(state, { instanceId: 'image-map', type: 'dockFloatingWidget' })).widgetRegions
      .right.instanceIds;

  it('shows the marker where the reducer will dock the window', () => {
    const state = floatImageMap();

    expect(getRightRail(state)).toEqual(['gallery', 'image-map*', 'queue']);
    expect(dockImageMap(state)).toEqual(['gallery', 'image-map', 'queue']);
  });

  it('does not re-derive the rail when a window only moves, shades, or is raised', () => {
    let state = floatImageMap();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    const placement = getWidgetPlacementProject(getActiveProject(state));

    state = workbenchReducer(state, {
      heightPx: 300,
      instanceId: 'image-map',
      type: 'setFloatingWidgetGeometry',
      widthPx: 300,
      x: 400,
      y: 300,
    });
    state = workbenchReducer(state, { instanceId: 'image-map', mode: 'shaded', type: 'setFloatingWidgetMode' });
    state = workbenchReducer(state, { instanceId: 'image-map', type: 'raiseFloatingWidget' });

    // The shell's selector equality: an equal projection keeps its previous snapshot, so the rails do not render.
    expect(areWidgetPlacementProjectsEqual(placement, getWidgetPlacementProject(getActiveProject(state)))).toBe(true);

    state = workbenchReducer(state, { instanceId: 'queue', type: 'dockFloatingWidget' });

    expect(areWidgetPlacementProjectsEqual(placement, getWidgetPlacementProject(getActiveProject(state)))).toBe(false);
  });

  it('closes the gap when a member ahead of the marker is removed', () => {
    const state = workbenchReducer(floatImageMap(), {
      region: 'right',
      type: 'toggleRegionWidget',
      widgetId: 'gallery',
    });

    expect(getRightRail(state)).toEqual(['image-map*', 'queue']);
    expect(dockImageMap(state)).toEqual(['image-map', 'queue']);
  });

  it('appends an added member after the marker', () => {
    const state = workbenchReducer(floatImageMap(), { region: 'right', type: 'openRegionWidget', widgetId: 'preview' });

    expect(getRightRail(state)).toEqual(['gallery', 'image-map*', 'queue', 'preview']);
    expect(dockImageMap(state)).toEqual(['gallery', 'image-map', 'queue', 'preview']);
  });

  it('keeps the marker stationary while the docked members reorder around it', () => {
    const state = workbenchReducer(floatImageMap(), {
      instanceIds: ['queue', 'gallery'],
      region: 'right',
      type: 'reorderWidgetInstances',
    });

    expect(getRightRail(state)).toEqual(['queue', 'image-map*', 'gallery']);
    expect(dockImageMap(state)).toEqual(['queue', 'image-map', 'gallery']);
  });

  it('lands a dragged-in widget in front of the member it was dropped on, wherever the marker sits', () => {
    // `toIndex` counts docked members: 1 is "in front of queue", which sits after the marker.
    const beforeQueue = workbenchReducer(floatImageMap(), {
      fromRegion: 'center',
      instanceId: 'preview',
      toIndex: 1,
      toRegion: 'right',
      type: 'moveWidgetInstance',
    });
    const beforeGallery = workbenchReducer(floatImageMap(), {
      fromRegion: 'center',
      instanceId: 'preview',
      toIndex: 0,
      toRegion: 'right',
      type: 'moveWidgetInstance',
    });

    expect(getRightRail(beforeQueue)).toEqual(['gallery', 'image-map*', 'preview', 'queue']);
    expect(getRightRail(beforeGallery)).toEqual(['preview', 'gallery', 'image-map*', 'queue']);
    expect(dockImageMap(beforeGallery)).toEqual(['preview', 'gallery', 'image-map', 'queue']);
  });

  it('keeps markers through a reload', () => {
    let state = floatImageMap();
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    const reloaded = reload(state);

    expect(getRightRail(reloaded)).toEqual(['gallery*', 'image-map*', 'queue']);
    expect(getActiveProject(reloaded).floatingWidgets).toEqual(getActiveProject(state).floatingWidgets);
  });
});

describe('revealFloatingWidget', () => {
  it('raises the window and expands it when it is shaded', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    state = workbenchReducer(state, { instanceId: 'gallery', mode: 'shaded', type: 'setFloatingWidgetMode' });

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'revealFloatingWidget' });
    const floating = getActiveProject(state).floatingWidgets;

    expect(floating?.gallery).toMatchObject({ mode: 'windowed', stackOrder: 2 });
    expect(floating?.queue.stackOrder).toBe(1);
  });

  it('leaves a maximized window maximized', () => {
    let state = floatGallery();
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    state = workbenchReducer(state, { instanceId: 'gallery', mode: 'maximized', type: 'setFloatingWidgetMode' });

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'revealFloatingWidget' });

    expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject({ mode: 'maximized', stackOrder: 2 });
  });

  it('is a no-op for a window already expanded on top, and for an instance that is not floating', () => {
    const state = floatGallery();

    expect(workbenchReducer(state, { instanceId: 'gallery', type: 'revealFloatingWidget' })).toBe(state);
    expect(workbenchReducer(state, { instanceId: 'queue', type: 'revealFloatingWidget' })).toBe(state);
  });
});

describe('remembered window geometry', () => {
  const PLACED = { heightPx: 360, widthPx: 480, x: 700, y: 220 };
  const moveGallery = (state: WorkbenchState): WorkbenchState =>
    workbenchReducer(state, { instanceId: 'gallery', type: 'setFloatingWidgetGeometry', ...PLACED });

  it('reopens a docked window where it last floated instead of cascading it again', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });

    expect(getActiveProject(state).lastFloatingGeometry).toEqual({ gallery: PLACED });

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.gallery).toMatchObject({ ...PLACED, mode: 'windowed' });
    // While the window is open its geometry is its own state; nothing stale is kept beside it.
    expect(project.lastFloatingGeometry).toBeUndefined();
  });

  it('reopens a removed window there too, once its widget is back on the rail and floated', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'closeFloatingWidget' });
    state = workbenchReducer(state, { region: 'right', type: 'openRegionWidget', widgetId: 'gallery' });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });

    expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject(PLACED);
  });

  it('remembers each window on its own, and gives a first float the next cascade slot', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.queue).toMatchObject(cascadeDefaultGeometry(0));
    expect(project.lastFloatingGeometry).toEqual({ gallery: PLACED });
  });

  it('survives a save and reload, through the durable project document', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    const document = JSON.parse(JSON.stringify(serializeProjectDocumentV3(getActiveProject(state)))) as Project;
    const reloaded = withActiveProject(state, () => normalizeWorkbenchProject(document));

    expect(getActiveProject(reloaded).lastFloatingGeometry).toEqual({ gallery: PLACED });
    expect(
      getActiveProject(workbenchReducer(reloaded, { instanceId: 'gallery', type: 'floatWidget' })).floatingWidgets
        ?.gallery
    ).toMatchObject(PLACED);
  });

  it('keeps only well-formed memory for instances that exist and are not floating now', () => {
    const project = getActiveProject(floatGallery());
    const normalized = normalizeWorkbenchProject({
      ...project,
      lastFloatingGeometry: {
        // Floating right now: the window's own geometry is the truth.
        gallery: PLACED,
        'no-such-widget': PLACED,
        preview: { ...PLACED, x: Number.NaN },
        queue: { ...PLACED, heightPx: 1, widthPx: 1 },
      },
    } as Project);

    expect(normalized.lastFloatingGeometry).toEqual({
      queue: { ...PLACED, heightPx: FLOATING_MIN_HEIGHT_PX, widthPx: FLOATING_MIN_WIDTH_PX },
    });
  });

  it('reopens windowed at the windowed rectangle, whatever mode the window was closed in', () => {
    for (const mode of ['maximized', 'shaded'] as const) {
      let state = moveGallery(floatGallery());
      state = workbenchReducer(state, { instanceId: 'gallery', mode, type: 'setFloatingWidgetMode' });
      state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
      state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });

      expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject({ ...PLACED, mode: 'windowed' });
    }
  });

  it('remembers the latest place a window was closed, not an earlier one', () => {
    const LATER = { heightPx: 300, widthPx: 420, x: 120, y: 80 };
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'setFloatingWidgetGeometry', ...LATER });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'closeFloatingWidget' });

    expect(getActiveProject(state).lastFloatingGeometry).toEqual({ gallery: LATER });
  });

  it('remembers a window that an explicit open docks', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { region: 'right', type: 'openRegionWidget', widgetId: 'gallery' });

    expect(getActiveProject(state).floatingWidgets).toBeUndefined();
    expect(getActiveProject(state).lastFloatingGeometry).toEqual({ gallery: PLACED });
  });

  it('brings a window remembered in a larger viewport back on screen when it floats into a smaller one', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    // Remembered at x=700 with a 480px width; the viewport is now 720px wide (the same window at 200% zoom).
    state = workbenchReducer(state, {
      instanceId: 'gallery',
      type: 'floatWidget',
      viewport: { height: 450, width: 720 },
    });

    expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject({
      heightPx: 360,
      widthPx: 480,
      x: 720 - 480,
      y: 450 - 360,
    });
  });

  it('remembers a window a preset closes, and forgets the memory of one a preset opens', () => {
    // Saved with Gallery floating at its first place; then the window is moved and the plain layout applied.
    let state = workbenchReducer(floatGallery(), { presetId: 'compose', type: 'saveLayoutPreset' });
    const savedPlace = { ...cascadeDefaultGeometry(0) };
    state = moveGallery(state);
    state = workbenchReducer(state, { presetId: 'edit', type: 'applyPreset' });

    expect(getActiveProject(state).floatingWidgets).toBeUndefined();
    expect(getActiveProject(state).lastFloatingGeometry).toEqual({ gallery: PLACED });

    // Back to the layout that floats it: the window returns where this project left it (Compose's working copy),
    // and no memory is kept beside it.
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });

    expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject(PLACED);
    expect(getActiveProject(state).lastFloatingGeometry).toBeUndefined();

    // Reverting to the saved preset moves the open window to the saved place; still nothing is remembered beside it.
    state = workbenchReducer(state, { presetId: 'compose', type: 'revertLayoutPreset' });

    expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject(savedPlace);
    expect(getActiveProject(state).lastFloatingGeometry).toBeUndefined();
  });

  it('remembers a window an undo closes, and forgets the memory of one an undo reopens', () => {
    // The undoable step is a preset switch taken while Gallery floats.
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { presetId: 'edit', type: 'applyPreset' });
    expect(getActiveProject(state).lastFloatingGeometry).toEqual({ gallery: PLACED });

    state = workbenchReducer(state, { type: 'undoProjectChange' });

    // Undo reopened the window, so its geometry is the window's own again.
    expect(getActiveProject(state).floatingWidgets?.gallery).toMatchObject(PLACED);
    expect(getActiveProject(state).lastFloatingGeometry).toBeUndefined();

    state = workbenchReducer(state, { type: 'redoProjectChange' });

    expect(getActiveProject(state).floatingWidgets).toBeUndefined();
    expect(getActiveProject(state).lastFloatingGeometry).toEqual({ gallery: PLACED });
  });

  it('always saves windows and their memory the way a reload would read them back', () => {
    // Persistence compares what it sent with the document as a reload reads it, as JSON. Anything a reload would
    // drop or reorder in these fields turns a lost response into a conflict.
    const placement = (project: Project): string => {
      const { floatingWidgets, lastFloatingGeometry } = serializeProjectDocumentV3(project);

      return JSON.stringify({ floatingWidgets, lastFloatingGeometry });
    };
    const savesCanonically = (state: WorkbenchState) => {
      const saved = serializeProjectDocumentV3(getActiveProject(state));
      const reloaded = normalizeWorkbenchProject(JSON.parse(JSON.stringify(saved)) as Project);

      expect(placement(reloaded)).toBe(placement(getActiveProject(state)));
    };
    let state = workbenchReducer(floatGallery(), { presetId: 'compose', type: 'saveLayoutPreset' });
    state = moveGallery(state);

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    savesCanonically(state);
    // A preset that floats the docked window again.
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    savesCanonically(state);
    // A preset that closes it, then an undo that reopens it, then a redo that closes it again.
    state = workbenchReducer(state, { presetId: 'edit', type: 'applyPreset' });
    savesCanonically(state);
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    savesCanonically(state);
    state = workbenchReducer(state, { type: 'redoProjectChange' });
    savesCanonically(state);

    // An open window, from the moment it floats and through everything that rewrites its state.
    state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'image-map', type: 'floatWidget' });
    savesCanonically(state);
    // Removing a member ahead of the marker re-indexes it; a second float re-indexes around both.
    state = workbenchReducer(state, { region: 'right', type: 'toggleRegionWidget', widgetId: 'gallery' });
    savesCanonically(state);
    state = workbenchReducer(state, { instanceId: 'queue', type: 'floatWidget' });
    savesCanonically(state);
    state = workbenchReducer(state, { instanceId: 'image-map', type: 'setFloatingWidgetGeometry', ...PLACED });
    state = workbenchReducer(state, { instanceId: 'image-map', mode: 'shaded', type: 'setFloatingWidgetMode' });
    state = workbenchReducer(state, { instanceId: 'image-map', type: 'raiseFloatingWidget' });
    savesCanonically(state);
  });

  it('is not layout: floating, moving, and docking back leaves the saved preset matching', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    const project = getActiveProject(state);

    expect(project.lastFloatingGeometry).toBeDefined();
    expect(doesProjectMatchLayoutPreset(project, resolveSavedLayoutPreset(state.account, 'compose'))).toBe(true);
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

describe('normalization of persisted floating windows', () => {
  it('keeps a floated widget floating across a reload instead of re-docking it', () => {
    // The bottom rail migration re-adds `queue-status` to any rail that reads as
    // a pre-queue-status default — which is exactly the shape floating it leaves.
    const state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'queue-status', type: 'floatWidget' });
    const project = normalizeWorkbenchProject(getActiveProject(state));

    expect(project.floatingWidgets?.['queue-status']).toBeDefined();
    expect(getRegionsHolding(project, 'queue-status')).toEqual([]);
  });

  it('survives a second normalization pass unchanged', () => {
    const state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'queue-status', type: 'floatWidget' });
    const once = normalizeWorkbenchProject(getActiveProject(state));
    const twice = normalizeWorkbenchProject(once);

    expect(twice.floatingWidgets).toEqual(once.floatingWidgets);
    expect(twice.widgetRegions).toEqual(once.widgetRegions);
  });

  it('drops entries a hand-edited or foreign project file could carry', () => {
    const project = getActiveProject(floatGallery());
    const normalized = normalizeWorkbenchProject({
      ...project,
      floatingWidgets: {
        // Would crash `dockFloatingWidget` on `widgetRegions[returnRegion]`.
        queue: { ...project.floatingWidgets!.gallery, returnRegion: 'nowhere' },
        // Would reach the window's fixed-position CSS as `NaNpx`.
        preview: { ...project.floatingWidgets!.gallery, x: Number.NaN },
        // No such instance to render.
        'no-such-widget': { ...project.floatingWidgets!.gallery },
        notifications: { ...project.floatingWidgets!.gallery, mode: 'iconified' },
      } as unknown as Project['floatingWidgets'],
    });

    expect(normalized.floatingWidgets).toBeUndefined();
    for (const instanceId of ['queue', 'preview', 'notifications']) {
      expect(getRegionsHolding(normalized, instanceId)).not.toEqual([]);
    }
  });

  it('clamps persisted geometry below the minimum size', () => {
    const project = getActiveProject(floatGallery());
    const normalized = normalizeWorkbenchProject({
      ...project,
      floatingWidgets: { gallery: { ...project.floatingWidgets!.gallery, heightPx: 1, widthPx: 1 } },
    });

    expect(normalized.floatingWidgets?.gallery).toMatchObject({
      heightPx: FLOATING_MIN_HEIGHT_PX,
      widthPx: FLOATING_MIN_WIDTH_PX,
    });
  });

  it('reads return indices stored before markers were counted the same way on every load', () => {
    // An older build stored each index against the docked members only, so two windows can share one.
    const stored = withActiveProject(floatGallery(), (project) => ({
      ...project,
      floatingWidgets: {
        queue: { ...project.floatingWidgets!.gallery, returnIndex: 0, stackOrder: 1 },
        gallery: { ...project.floatingWidgets!.gallery, returnIndex: 0, stackOrder: 2 },
      },
      widgetRegions: {
        ...project.widgetRegions,
        right: { ...project.widgetRegions.right, activeInstanceId: 'image-map', instanceIds: ['image-map'] },
      },
    }));
    const restacked = withActiveProject(stored, (project) => ({
      ...project,
      floatingWidgets: {
        gallery: { ...project.floatingWidgets!.gallery, stackOrder: 1 },
        queue: { ...project.floatingWidgets!.queue, stackOrder: 2 },
      },
    }));
    const once = reload(stored);

    // Ties break by instance id; the order the windows were floated in is not recoverable and is not claimed.
    expect(getRightRail(once)).toEqual(['gallery*', 'queue*', 'image-map']);
    expect(getRightRail(reload(restacked))).toEqual(getRightRail(once));
    expect(getActiveProject(once).floatingWidgets).toMatchObject({
      gallery: { returnIndex: 0 },
      queue: { returnIndex: 1 },
    });
    expect(getActiveProject(reload(once))).toEqual(getActiveProject(once));
  });

  it('appends a window whose stored return index is missing', () => {
    const stored = withActiveProject(floatGallery(), (project) => {
      const { returnIndex: _dropped, ...gallery } = project.floatingWidgets!.gallery;

      return { ...project, floatingWidgets: { gallery } };
    });
    const reloaded = reload(stored);

    expect(getRightRail(reloaded)).toEqual(['image-map', 'queue', 'gallery*']);
    expect(
      getActiveProject(workbenchReducer(reloaded, { instanceId: 'gallery', type: 'dockFloatingWidget' })).widgetRegions
        .right.instanceIds
    ).toEqual(['image-map', 'queue', 'gallery']);
  });

  it('lets a sole center view keep floating across a reload', () => {
    // An explicitly empty center with a floated active pointer must survive normalization without default views
    // being added.
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'video', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'center', type: 'floatWidget' });
    const project = getActiveProject(state);
    expect(project.widgetRegions.center.instanceIds).toEqual([]);

    // The reducer output is exactly what autosave persists.
    const normalized = normalizeWorkbenchProject(JSON.parse(JSON.stringify(project)));

    expect(normalized.floatingWidgets?.preview).toMatchObject({ returnRegion: 'center' });
    expect(normalized.widgetRegions.center.instanceIds).toEqual([]);
    expect(normalized.widgetRegions.center.activeInstanceId).toBe('preview');

    // Docking after the reload restores the view.
    const docked = workbenchReducer(
      { ...state, projects: state.projects.map((p) => (p.id === normalized.id ? normalized : p)) },
      { instanceId: 'preview', type: 'dockFloatingWidget' }
    );

    expect(getActiveProject(docked).widgetRegions.center.instanceIds).toEqual(['preview']);
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

  it('a preset saved while a widget floats restores the window, not nothing', () => {
    // The rail is customized first so the reload migration is not what keeps
    // the widget alive here — only the preset can.
    let state = workbenchReducer(createInitialWorkbenchState(), {
      region: 'right',
      type: 'toggleRegionWidget',
      widgetId: 'project',
    });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'saveLayoutPreset' });
    // Dock it, then revert to the preset that was saved with it floating.
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.gallery).toBeDefined();
    expect(getRegionsHolding(project, 'gallery')).toEqual([]);
  });

  it('a preset saved with the sole center view floating restores the emptied surface, not a phantom view', () => {
    // Snapshot normalization must preserve an intentionally empty center.
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'video', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'center', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'video', type: 'saveLayoutPreset' });
    // Dock it, then revert to the preset that was saved with it floating.
    state = workbenchReducer(state, { instanceId: 'preview', type: 'dockFloatingWidget' });
    state = workbenchReducer(state, { presetId: 'video', type: 'applyPreset' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.preview).toBeDefined();
    expect(project.widgetRegions.center.instanceIds).toEqual([]);
  });

  it('carries a saved preset’s floating window through account rehydration', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'saveLayoutPreset' });

    const rehydrated = normalizeWorkbenchAccount(state.account);

    expect(rehydrated.layoutPresetOverrides?.compose?.floatingWidgets?.gallery).toMatchObject({
      mode: 'windowed',
      returnRegion: 'right',
    });
  });

  it('validates a preset’s floating windows the way a persisted project’s are', () => {
    // Preset bodies come from account storage, which never passes through
    // `normalizeWorkbenchProject`.
    let state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'saveLayoutPreset' });

    const saved = state.account.layoutPresetOverrides!.compose!;
    const account = normalizeWorkbenchAccount({
      ...state.account,
      layoutPresetOverrides: {
        compose: {
          ...saved,
          floatingWidgets: {
            // Would throw in `dockFloatingWidget` on widgetRegions[returnRegion].
            gallery: { ...saved.floatingWidgets!.gallery, returnRegion: 'nowhere' },
            // Docked by the same preset, so honouring this would double-render.
            queue: { ...saved.floatingWidgets!.gallery },
          },
        },
      },
    } as unknown);
    state = { ...state, account };
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.gallery).toBeUndefined();
    expect(getRegionsHolding(project, 'queue')).toEqual([]);
    expect(project.floatingWidgets?.queue).toBeDefined();
    // The dropped window's instance is placed nowhere, on either side of the comparison.
    expect(doesProjectMatchLayoutPreset(project, resolveSavedLayoutPreset(state.account, 'compose'))).toBe(true);
  });

  it('reads a floated window as unsaved layout drift until it is saved', () => {
    const floated = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'gallery', type: 'floatWidget' });

    expect(
      doesProjectMatchLayoutPreset(getActiveProject(floated), resolveSavedLayoutPreset(floated.account, 'compose'))
    ).toBe(false);

    const saved = workbenchReducer(floated, { presetId: 'compose', type: 'saveLayoutPreset' });

    expect(
      doesProjectMatchLayoutPreset(getActiveProject(saved), resolveSavedLayoutPreset(saved.account, 'compose'))
    ).toBe(true);
  });

  it('reads a moved return slot as layout drift', () => {
    const snapshot = createLayoutPresetSnapshot(getActiveProject(floatGallery()));
    const moved = {
      ...snapshot,
      floatingWidgets: { gallery: { ...snapshot.floatingWidgets!.gallery, returnIndex: 2 } },
    };

    expect(snapshot.floatingWidgets?.gallery.returnIndex).toBe(0);
    expect(areLayoutPresetSnapshotsEqual(snapshot, moved)).toBe(false);
  });

  it('matches a freshly applied preset against a snapshot stored by an older build', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'right', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'edit', type: 'saveLayoutPreset' });
    const saved = state.account.layoutPresetOverrides!.edit!;
    const { returnIndex: _dropped, ...preview } = saved.floatingWidgets!.preview;
    // As an older build stored it: no return index, and the window still a member of the center it also showed in.
    const legacy = {
      ...saved,
      floatingWidgets: { preview },
      widgetRegions: {
        ...saved.widgetRegions,
        center: { ...saved.widgetRegions.center, instanceIds: ['canvas', 'preview'] },
      },
    };

    for (const account of [
      // Read straight from account storage, and as account hydration rewrites it.
      { ...state.account, layoutPresetOverrides: { edit: legacy } },
      normalizeWorkbenchAccount({ ...state.account, layoutPresetOverrides: { edit: legacy } }),
    ]) {
      const applied = workbenchReducer({ ...state, account }, { presetId: 'edit', type: 'applyPreset' });
      const project = getActiveProject(applied);

      expect(getRegionsHolding(project, 'preview')).toEqual([]);
      expect(project.floatingWidgets?.preview).toMatchObject({ returnIndex: 1, returnRegion: 'right' });
      expect(doesProjectMatchLayoutPreset(project, resolveSavedLayoutPreset(applied.account, 'edit'))).toBe(true);
    }
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

    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    const isFloating = Boolean(project.floatingWidgets?.gallery);
    const isDocked = Object.values(project.widgetRegions).some((region) => region.instanceIds.includes('gallery'));
    expect(isFloating !== isDocked).toBe(true);
  });
});
