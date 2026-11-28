import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { describe, expect, it } from 'vitest';

import { FLOATING_MIN_HEIGHT_PX, FLOATING_MIN_WIDTH_PX } from './floatingWindows';
import { serializeProjectDocumentV3 } from './projects/projectDocument';
import { getWidgetPlacementProject } from './widgetPlacementMeta';
import { createWidgetRegionViewModelFromState } from './widgetRegionViewModel';
import { getWidgetsForRegion } from './widgetRegistry';
import { normalizeWorkbenchProject } from './workbenchState';
import {
  asStoredBeforeSplit,
  createInitialWorkbenchState,
  getActiveProject,
  getLayoutOf,
  workbenchReducer,
} from './workbenchState.testing';
import { getWorkspaceStructure } from './workspace/accountWorkspaces';
import { storedOn } from './workspace/storedLayoutFixtures.testing';

const floatGallery = (state = createInitialWorkbenchState()): WorkbenchState =>
  workbenchReducer(state, { instanceId: 'gallery', type: 'floatWidget' });

const getRegionsHolding = (state: WorkbenchState, instanceId: string): string[] =>
  Object.entries(getLayoutOf(state).widgetRegions)
    .filter(([, region]) => region.instanceIds.includes(instanceId))
    .map(([regionId]) => regionId);

/** The right rail exactly as the shell derives it: resolved layout -> placement selector -> region view model. */
const getRightRail = (state: WorkbenchState): string[] => {
  const activeLayout = getLayoutOf(state);
  const placement = getWidgetPlacementProject({ activeLayout, activeProject: getActiveProject(state) });

  return createWidgetRegionViewModelFromState({
    floatingWidgets: placement.floatingPlacements,
    region: 'right',
    regionState: activeLayout.widgetRegions.right,
    widgetInstances: placement.widgetInstances,
    widgets: getWidgetsForRegion('right'),
  }).placedItems.map((item) => (item.isFloating ? `${item.id}*` : item.id));
};

describe('floating a placed widget', () => {
  it('detaches it from its region into a window of the workspace', () => {
    const initial = createInitialWorkbenchState();
    const state = floatGallery(initial);
    const layout = getLayoutOf(state);

    expect(layout.widgetRegions.right.instanceIds).not.toContain('gallery');
    expect(layout.widgetRegions.right.instanceIds).toContain(layout.widgetRegions.right.activeInstanceId);
    expect(layout.floatingWidgets?.gallery).toMatchObject({ mode: 'windowed', returnRegion: 'right', stackOrder: 1 });
    expect(getWorkspaceStructure(state.account, 'compose').floatingWidgets?.gallery).toBeDefined();
  });

  it('returns to the region the float was asked from, and to every other region it left', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'preview', region: 'center', type: 'floatWidget' });

    expect(getLayoutOf(state).floatingWidgets?.preview).toMatchObject({ returnRegion: 'center' });

    const docked = getLayoutOf(workbenchReducer(state, { instanceId: 'preview', type: 'dockFloatingWidget' }));

    expect(docked.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['canvas', 'preview'],
    });
    // Floating took it from the rail too; docking puts it back there, behind the rail's own front.
    expect(docked.widgetRegions.right).toMatchObject({
      activeInstanceId: 'layers',
      instanceIds: ['layers', 'preview'],
    });
  });

  it('closes a window without docking, leaving the rail and the instance’s values', () => {
    const floated = floatGallery();
    const rightBefore = getLayoutOf(floated).widgetRegions.right;
    const state = workbenchReducer(floated, { instanceId: 'gallery', type: 'closeFloatingWidget' });

    expect(getLayoutOf(state).floatingWidgets).toBeUndefined();
    expect(getLayoutOf(state).widgetRegions.right).toEqual(rightBefore);
    expect(getRegionsHolding(state, 'gallery')).toEqual([]);
    expect(getActiveProject(state).widgetInstances.gallery).toBeDefined();
  });

  it('docks a window back as its region’s shown widget', () => {
    const state = workbenchReducer(floatGallery(), { instanceId: 'gallery', type: 'dockFloatingWidget' });
    const layout = getLayoutOf(state);

    expect(layout.floatingWidgets).toBeUndefined();
    expect(layout.widgetRegions.right).toMatchObject({ activeInstanceId: 'gallery', isCollapsed: false });
  });

  it('docks a floating instance an open asks for instead of double-rendering it', () => {
    const state = workbenchReducer(floatGallery(), { region: 'right', type: 'openRegionWidget', widgetId: 'gallery' });

    expect(getLayoutOf(state).floatingWidgets).toBeUndefined();
    // Back at its return slot, not appended.
    expect(getLayoutOf(state).widgetRegions.right).toMatchObject({
      activeInstanceId: 'gallery',
      instanceIds: ['gallery', 'image-map', 'queue'],
    });
  });

  it('refuses a stale move or reorder that would place a floating instance back in a region', () => {
    const state = floatGallery();

    expect(
      workbenchReducer(state, {
        fromRegion: 'right',
        instanceId: 'gallery',
        toIndex: 0,
        toRegion: 'left',
        type: 'moveWidgetInstance',
      })
    ).toBe(state);
    expect(
      workbenchReducer(state, {
        instanceIds: ['gallery', 'image-map', 'queue'],
        region: 'right',
        type: 'reorderWidgetInstances',
      })
    ).toBe(state);
  });

  it('shows a window’s marker in its rail where docking puts it back', () => {
    const state = workbenchReducer(createInitialWorkbenchState(), { instanceId: 'image-map', type: 'floatWidget' });

    expect(getRightRail(state)).toEqual(['gallery', 'image-map*', 'queue']);
    expect(
      getLayoutOf(workbenchReducer(state, { instanceId: 'image-map', type: 'dockFloatingWidget' })).widgetRegions.right
        .instanceIds
    ).toEqual(['gallery', 'image-map', 'queue']);
  });

  it('undoes a float by putting the workspace back as it was', () => {
    const before = createInitialWorkbenchState();
    const undone = workbenchReducer(floatGallery(before), { type: 'undoProjectChange' });

    expect(getLayoutOf(undone).floatingWidgets).toBeUndefined();
    expect(getLayoutOf(undone).widgetRegions.right).toEqual(getLayoutOf(before).widgetRegions.right);
  });
});

describe('how a project looks at a window', () => {
  it('moves, resizes and shades a window in its view, leaving the workspace and other projects alone', () => {
    const floated = floatGallery();
    const withOther = workbenchReducer(floated, { type: 'createProject' });
    const otherId = withOther.activeProjectId;
    const back = workbenchReducer(withOther, { projectId: floated.activeProjectId, type: 'switchProject' });
    const moved = workbenchReducer(back, {
      heightPx: 10,
      instanceId: 'gallery',
      type: 'setFloatingWidgetGeometry',
      widthPx: 10,
      x: 5,
      y: 7,
    });
    const shaded = workbenchReducer(moved, { instanceId: 'gallery', mode: 'shaded', type: 'setFloatingWidgetMode' });

    expect(getLayoutOf(shaded).floatingWidgets?.gallery).toMatchObject({
      heightPx: FLOATING_MIN_HEIGHT_PX,
      mode: 'shaded',
      widthPx: FLOATING_MIN_WIDTH_PX,
      x: 5,
      y: 7,
    });
    expect(shaded.account).toBe(floated.account);
    expect(
      getLayoutOf(
        shaded,
        shaded.projects.find(({ id }) => id === otherId)
      ).floatingWidgets?.gallery
    ).toEqual(getLayoutOf(floated).floatingWidgets?.gallery);
  });

  it('keeps stack orders compact, and a reveal raises and expands a shaded window', () => {
    let state = workbenchReducer(floatGallery(), { instanceId: 'queue', type: 'floatWidget' });

    for (const instanceId of ['gallery', 'queue', 'gallery', 'queue']) {
      state = workbenchReducer(state, { instanceId, type: 'raiseFloatingWidget' });
    }
    state = workbenchReducer(state, { instanceId: 'gallery', mode: 'shaded', type: 'setFloatingWidgetMode' });
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'revealFloatingWidget' });

    const floating = getLayoutOf(state).floatingWidgets;

    expect(floating?.gallery).toMatchObject({ mode: 'windowed', stackOrder: 2 });
    expect(floating?.queue.stackOrder).toBe(1);
  });
});

describe('remembered window geometry', () => {
  const PLACED = { heightPx: 360, widthPx: 480, x: 700, y: 220 };
  const moveGallery = (state: WorkbenchState): WorkbenchState =>
    workbenchReducer(state, { instanceId: 'gallery', type: 'setFloatingWidgetGeometry', ...PLACED });

  it('reopens a docked window where it last floated, through a save and reload', () => {
    let state = moveGallery(floatGallery());
    state = workbenchReducer(state, { instanceId: 'gallery', type: 'dockFloatingWidget' });

    expect(getActiveProject(state).lastFloatingGeometry).toEqual({ gallery: PLACED });

    const document = JSON.parse(JSON.stringify(serializeProjectDocumentV3(getActiveProject(state)))) as Project;
    const reloaded = {
      ...state,
      projects: state.projects.map((project) =>
        project.id === state.activeProjectId
          ? { ...normalizeWorkbenchProject(document), undoRedo: { future: [], past: [] } }
          : project
      ),
    };
    const floated = workbenchReducer(reloaded, { instanceId: 'gallery', type: 'floatWidget' });

    expect(getLayoutOf(floated).floatingWidgets?.gallery).toMatchObject({ ...PLACED, mode: 'windowed' });
    // While the window is open its geometry is the layout's; nothing stale is kept beside it.
    expect(getActiveProject(floated).lastFloatingGeometry).toBeUndefined();
  });

  it('keeps only well-formed memory for instances that exist', () => {
    const project = getActiveProject(createInitialWorkbenchState());
    const normalized = normalizeWorkbenchProject({
      ...project,
      lastFloatingGeometry: {
        'no-such-widget': PLACED,
        preview: { ...PLACED, x: Number.NaN },
        queue: { ...PLACED, heightPx: 1, widthPx: 1 },
      },
    } as Project);

    expect(normalized.lastFloatingGeometry).toEqual({
      queue: { ...PLACED, heightPx: FLOATING_MIN_HEIGHT_PX, widthPx: FLOATING_MIN_WIDTH_PX },
    });
  });
});

describe('windows stored before structure and view split', () => {
  it('drops entries a hand-edited or foreign project file could carry', () => {
    const window = {
      heightPx: 300,
      mode: 'windowed',
      returnIndex: 0,
      returnRegion: 'right',
      stackOrder: 1,
      widthPx: 400,
      x: 10,
      y: 10,
    } as const;
    const loaded = asStoredBeforeSplit(createInitialWorkbenchState().projects[0]!, {
      ...storedOn('compose'),
      floatingWidgets: {
        // Would crash docking on `widgetRegions[returnRegion]`.
        queue: { ...window, returnRegion: 'nowhere' },
        // Would reach the window's fixed-position CSS as `NaNpx`.
        preview: { ...window, x: Number.NaN },
        'no-such-widget': window,
        notifications: { ...window, mode: 'iconified' },
      } as unknown as NonNullable<Project['legacyLayout']>['floatingWidgets'],
    });

    expect(loaded.legacyLayout?.floatingWidgets).toBeUndefined();
    expect(loaded.legacyLayout?.widgetRegions.right.instanceIds).toContain('queue');
  });
});
