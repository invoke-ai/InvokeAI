import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { describe, expect, it } from 'vitest';

import { FLOATING_MIN_HEIGHT_PX, FLOATING_MIN_WIDTH_PX } from './floatingWindows';
import { doesProjectMatchLayoutPreset, resolveSavedLayoutPreset } from './layoutPresetSnapshots';
import { serializeProjectDocumentV3 } from './projects/projectDocument';
import { getWidgetPlacementProject } from './widgetPlacementMeta';
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

  it('always saves windows and their memory the way a reload would read them back', () => {
    // Persistence compares what it sent with the document as a reload reads it, as JSON. Anything a reload would drop
    // or reorder in these fields turns a lost response into a conflict.
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
});

describe('normalization of persisted floating windows', () => {
  it('keeps a floated widget floating across a reload instead of re-docking it', () => {
    // The bottom rail migration re-adds `queue-status` to any rail that reads as a pre-queue-status default — which is
    // exactly the shape floating it leaves.
    const state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'queue-status', type: 'floatWidget' });
    const project = normalizeWorkbenchProject(getActiveProject(state));

    expect(project.floatingWidgets?.['queue-status']).toBeDefined();
    expect(getRegionsHolding(project, 'queue-status')).toEqual([]);
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
});

describe('interaction with presets and undo', () => {
  it('reverting to a preset docks all floating widgets (no double render)', () => {
    const floated = floatGallery();
    const state = workbenchReducer(floated, { presetId: 'compose', type: 'revertLayoutPreset' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets ?? {}).toEqual({});
    expect(project.widgetRegions.right.instanceIds).toContain('gallery');
  });

  it('a preset saved while a widget floats restores the window, not nothing', () => {
    // The rail is customized first so the reload migration is not what keeps the widget alive here — only the preset
    // can.
    let state = workbenchReducer(createInitialWorkbenchState(), {
      region: 'right',
      type: 'toggleRegionWidget',
      widgetId: 'diagnostics',
    });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'saveLayoutPreset' });
    // Dock it, then revert to the preset that was saved with it floating.
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'revertLayoutPreset' });
    const project = getActiveProject(state);

    expect(project.floatingWidgets?.gallery).toBeDefined();
    expect(getRegionsHolding(project, 'gallery')).toEqual([]);
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
