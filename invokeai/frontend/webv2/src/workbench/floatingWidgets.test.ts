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
import { doesProjectMatchLayoutPreset, resolveSavedLayoutPreset } from './layoutPresetSnapshots';
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

  it('refuses to float the last center view, which would blank the work surface', () => {
    let state = createInitialWorkbenchState();
    const centerInstanceIds = getActiveProject(state).widgetRegions.center.instanceIds;

    for (const instanceId of centerInstanceIds.slice(1)) {
      state = workbenchReducer(state, { region: 'center', type: 'toggleRegionWidget', widgetId: instanceId });
    }

    const lastCenterInstanceId = centerInstanceIds[0];
    expect(getActiveProject(state).widgetRegions.center.instanceIds).toEqual([lastCenterInstanceId]);
    expect(workbenchReducer(state, { instanceId: lastCenterInstanceId, type: 'floatWidget' })).toBe(state);
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

describe('normalization of persisted floating windows', () => {
  it('keeps a floated widget floating across a reload instead of re-docking it', () => {
    // The right rail migration re-adds `image-map` to any rail that reads as a
    // pre-image-map default — which is exactly the shape floating it leaves.
    const state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'image-map', type: 'floatWidget' });
    const project = normalizeWorkbenchProject(getActiveProject(state));

    expect(project.floatingWidgets?.['image-map']).toBeDefined();
    expect(getRegionsHolding(project, 'image-map')).toEqual([]);
  });

  it('survives a second normalization pass unchanged', () => {
    const state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'image-map', type: 'floatWidget' });
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
        'image-map': { ...project.floatingWidgets!.gallery, returnRegion: 'nowhere' },
        // Would reach the window's fixed-position CSS as `NaNpx`.
        preview: { ...project.floatingWidgets!.gallery, x: Number.NaN },
        // No such instance to render.
        'no-such-widget': { ...project.floatingWidgets!.gallery },
        queue: { ...project.floatingWidgets!.gallery, mode: 'iconified' },
      } as unknown as Project['floatingWidgets'],
    });

    expect(normalized.floatingWidgets).toBeUndefined();
    for (const instanceId of ['image-map', 'preview', 'queue']) {
      expect(getRegionsHolding(normalized, instanceId)).toContain('right');
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

  it('docks rather than empties the center region', () => {
    const project = getActiveProject(createInitialWorkbenchState());
    const [onlyCenterInstanceId] = project.widgetRegions.center.instanceIds;
    const normalized = normalizeWorkbenchProject({
      ...project,
      floatingWidgets: {
        [onlyCenterInstanceId]: {
          heightPx: 300,
          mode: 'windowed',
          returnRegion: 'center',
          stackOrder: 1,
          widthPx: 400,
          x: 10,
          y: 10,
        },
      },
      widgetRegions: {
        ...project.widgetRegions,
        center: { ...project.widgetRegions.center, instanceIds: [onlyCenterInstanceId] },
      },
    });

    expect(normalized.floatingWidgets).toBeUndefined();
    expect(normalized.widgetRegions.center.instanceIds).toEqual([onlyCenterInstanceId]);
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

  it('carries a saved preset’s floating window through account rehydration', () => {
    // Account presets are rebuilt by `cloneLayoutPresetSnapshot` on every load;
    // a field it forgets is a field the preset loses on the next reload.
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
