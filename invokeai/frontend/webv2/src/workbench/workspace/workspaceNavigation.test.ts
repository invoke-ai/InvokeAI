import type { WorkbenchState } from '@workbench/projectContracts';

import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createInitialWorkbenchState, withArrangement, workbenchReducer } from '@workbench/workbenchState.testing';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { describe, expect, it, vi } from 'vitest';

import { getWorkspaceStructure } from './accountWorkspaces';
import {
  getWorkspaceOffer,
  goBackToPreviousWorkspace,
  goBackToStage,
  keepGuestInWorkspace,
  openInWorkspace,
  runWorkspaceCommand,
  showWorkflowForm,
  toggleCenterPreview,
} from './workspaceNavigation';

const createNavigation = ({
  arrange = (state) => state,
  presetId = 'compose',
  warm = true,
}: { arrange?: (state: WorkbenchState) => WorkbenchState; presetId?: string; warm?: boolean } = {}) => {
  const pendingLoads: (() => void)[] = [];
  const store = createWorkbenchStore(
    workbenchReducer(arrange(createInitialWorkbenchState()), { presetId, type: 'applyPreset' }),
    {
      isLoaded: () => warm,
      loadLayoutWidgets: () =>
        new Promise<void>((resolve) => {
          pendingLoads.push(resolve);
        }),
    }
  );
  const focus = { focusFloating: vi.fn(), focusRegion: vi.fn() };
  const ports = { commands: store.commands, focus, getWidgetsForRegion, queries: store.queries };
  const project = () => store.queries.getSnapshot().activeProject;
  const layout = () => store.queries.getSnapshot().activeLayout;
  const notices = () => store.queries.getSnapshot().notifications;
  const placedTypes = () =>
    Object.values(layout().widgetRegions).flatMap(({ instanceIds }) =>
      instanceIds.map((id) => layout().widgetInstances[id]?.typeId)
    );

  return {
    finishLoading: () => pendingLoads.forEach((resolve) => resolve()),
    focus,
    layout,
    notices,
    placedTypes,
    ports,
    project,
    store,
  };
};

describe('openInWorkspace', () => {
  it('lets only the first intent of a gesture change workspace; what the new one lacks shows there as a guest', () => {
    const { layout, placedTypes, ports, project, store } = createNavigation();
    const account = store.getSnapshot().account;

    expect(openInWorkspace(ports, [{ surface: 'stage', typeId: 'canvas' }, { typeId: 'diagnostics' }])).toEqual({
      changedWorkspace: true,
      ok: true,
      region: 'center',
      workspaceId: 'edit',
    });

    expect(project().workspaceId).toBe('edit');
    expect(placedTypes()).not.toContain('diagnostics');
    expect(Object.values(layout().guests)).toEqual(['window']);
    expect(getWorkspaceStructure(store.getSnapshot().account, 'edit')).toBe(getWorkspaceStructure(account, 'edit'));
  });

  it('focuses the guest a repeated open already shows instead of making another', () => {
    const { focus, layout, ports } = createNavigation();

    expect(openInWorkspace(ports, { typeId: 'diagnostics' })).toMatchObject({ changedWorkspace: false, ok: true });
    openInWorkspace(ports, { typeId: 'diagnostics' });

    const guests = Object.keys(layout().guests);
    expect(guests).toHaveLength(1);
    expect(focus.focusFloating.mock.calls).toEqual([[guests[0]], [guests[0]]]);
  });

  it('shows the rest of a gesture when its first widget is unavailable, focusing the one it names', () => {
    const { focus, layout, ports } = createNavigation();
    const withoutPreview = (region: Parameters<typeof getWidgetsForRegion>[0]) =>
      getWidgetsForRegion(region).filter(({ manifest }) => manifest.id !== 'preview');

    openInWorkspace(
      { ...ports, getWidgetsForRegion: withoutPreview },
      [{ surface: 'stage', typeId: 'preview' }, { typeId: 'queue' }],
      { focus: 'queue' }
    );

    expect(layout().widgetRegions.right.activeInstanceId).toBe('queue');
    expect(focus.focusRegion).toHaveBeenCalledExactlyOnceWith('right', 'queue', 'queue');
  });

  it('fronts what this workspace places for a background source, and leaves the offer to its notice', () => {
    const { focus, layout, notices, ports, project } = createNavigation();
    const regionsBefore = layout().widgetRegions;

    expect(openInWorkspace(ports, { typeId: 'canvas' }, { background: true })).toEqual({
      ok: false,
      reason: 'unavailable',
    });
    expect(project().workspaceId).toBe('compose');
    expect(layout().widgetRegions).toBe(regionsBefore);
    expect(layout().guests).toEqual({});
    expect(notices()).toEqual([]);
    expect(getWorkspaceOffer(ports, { typeId: 'canvas' })).toEqual([{ kind: 'open-widget', widgetId: 'canvas' }]);

    openInWorkspace(ports, { typeId: 'upscale' }, { background: true });
    expect(layout().widgetRegions.left).toMatchObject({
      activeInstanceId: 'upscale',
      instanceIds: ['generate', 'upscale'],
    });
    expect(focus.focusRegion).not.toHaveBeenCalled();
  });

  it('runs follow-up work once a cold entry lands, and never for one superseded before it does', async () => {
    const landed = createNavigation({ presetId: 'automate', warm: false });
    const afterLanding = vi.fn(() => landed.project().workspaceId);

    openInWorkspace(landed.ports, { surface: 'panel', typeId: 'generate' }, { afterShown: afterLanding });
    expect(afterLanding).not.toHaveBeenCalled();
    landed.finishLoading();
    await vi.waitFor(() => expect(afterLanding).toHaveReturnedWith('compose'));

    const dropped = createNavigation({ presetId: 'automate', warm: false });
    const afterDropped = vi.fn();

    expect(
      openInWorkspace(dropped.ports, { surface: 'panel', typeId: 'upscale' }, { afterShown: afterDropped })
    ).toMatchObject({ changedWorkspace: true, workspaceId: 'compose' });
    const edit = dropped.store.commands.layout.activatePreset('edit');
    dropped.finishLoading();

    await expect(edit).resolves.toBe('edit');
    await Promise.resolve();
    expect(dropped.layout().widgetRegions.left.activeInstanceId).toBe('generate');
    expect(afterDropped).not.toHaveBeenCalled();
    expect(goBackToPreviousWorkspace(dropped.ports)).toBe(false);
  });
});

describe('navigation while customizing', () => {
  it('drops a workspace switch still loading when Customize begins, so it cannot leave the workspace', async () => {
    const { finishLoading, project, store } = createNavigation({ warm: false });
    const switching = store.commands.layout.activatePreset('video');

    store.commands.layout.beginCustomize();
    finishLoading();
    await switching;

    expect(project().workspaceId).toBe('compose');
    expect(store.getSnapshot().customizing?.[project().id]?.presetId).toBe('compose');
  });

  it('stays in the workspace being customized: an open shows a guest outside the draft, and switching is refused', async () => {
    const { layout, ports, project, store } = createNavigation();

    store.commands.layout.beginCustomize();
    const draft = store.getSnapshot().customizing?.[project().id]?.structure;

    expect(openInWorkspace(ports, { surface: 'stage', typeId: 'workflow' })).toMatchObject({
      changedWorkspace: false,
      ok: true,
      region: 'center',
    });
    expect(Object.values(layout().guests)).toEqual(['center']);
    expect(store.getSnapshot().customizing?.[project().id]?.structure).toBe(draft);
    await expect(store.commands.layout.activatePreset('edit')).resolves.toBe('compose');
    expect(project().workspaceId).toBe('compose');

    store.commands.layout.finishCustomize();
    const saved = getWorkspaceStructure(store.getSnapshot().account, 'compose');
    expect(Object.values(saved.widgetInstances).map(({ typeId }) => typeId)).not.toContain('workflow');
    expect(Object.values(layout().guests)).toEqual(['center']);
  });
});

describe('guests', () => {
  it('belong to the workspace’s view: hidden on leaving, back on return, forgotten on close, gone with Reset view', () => {
    const { layout, ports, project, store } = createNavigation();
    const guestOf = () => Object.keys(layout().guests)[0];

    openInWorkspace(ports, { typeId: 'diagnostics' });
    const guestId = guestOf()!;
    store.commands.widgets.patchInstanceValues(guestId, { kept: true });

    void store.commands.layout.activatePreset('edit');
    expect(layout().guests).toEqual({});
    void store.commands.layout.activatePreset('compose');
    expect(guestOf()).toBe(guestId);

    store.commands.widgets.closeGuest(guestId);
    expect(layout().guests).toEqual({});
    expect(project().widgetInstances[guestId]?.state.values).toMatchObject({ kept: true });

    openInWorkspace(ports, { typeId: 'diagnostics' });
    store.commands.layout.resetWorkspaceView('compose');
    expect(layout().guests).toEqual({});
  });

  it('shows a guest beside a placement the workspace cannot show (Layers stored in the left rail)', () => {
    const store = createWorkbenchStore(
      withArrangement(
        workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' }),
        'compose',
        { left: ['generate', 'layers'] }
      )
    );
    const focus = { focusFloating: vi.fn(), focusRegion: vi.fn() };
    const layout = () => store.queries.getSnapshot().activeLayout;

    expect(layout().widgetRegions.left.instanceIds).not.toContain('layers');
    expect(
      openInWorkspace(
        { commands: store.commands, focus, getWidgetsForRegion, queries: store.queries },
        { typeId: 'layers' },
        { mode: 'stay' }
      )
    ).toMatchObject({ ok: true });
    expect(Object.values(layout().guests)).toEqual(['window']);
  });

  it('a center guest is Away: Back to the stage forgets it', () => {
    const { layout, ports, project } = createNavigation({ presetId: 'edit' });

    openInWorkspace(ports, { surface: 'stage', typeId: 'workflow' }, { mode: 'stay' });
    expect(layout().widgetInstances[layout().widgetRegions.center.activeInstanceId]?.typeId).toBe('workflow');

    expect(goBackToStage(ports)).toBe(true);
    expect(layout().widgetRegions.center.activeInstanceId).toBe('canvas');
    expect(layout().guests).toEqual({});
    expect(project().views.edit?.guests).toBeUndefined();
  });

  it('Keep in workspace adds a locked guest to a Customize draft, kept on Cancel, and to the workspace while rearranging', () => {
    const { layout, ports, project, store } = createNavigation();
    const typesIn = (structure: ReturnType<typeof getWorkspaceStructure>) =>
      Object.values(structure.widgetInstances).map(({ typeId }) => typeId);

    openInWorkspace(ports, { typeId: 'diagnostics' });
    const guestId = Object.keys(layout().guests)[0]!;
    const window = layout().floatingWidgets?.[guestId];

    keepGuestInWorkspace(ports, guestId, { canEditStructure: false });
    const draft = store.getSnapshot().customizing?.[project().id]?.structure;
    expect(draft?.floatingWidgets?.[guestId]).toMatchObject({ x: window?.x, y: window?.y });
    expect(layout().guests).toEqual({});

    store.commands.layout.cancelCustomize();
    expect(typesIn(getWorkspaceStructure(store.getSnapshot().account, 'compose'))).not.toContain('diagnostics');
    expect(Object.keys(layout().guests)).toEqual([guestId]);

    keepGuestInWorkspace(ports, guestId, { canEditStructure: true });
    expect(store.getSnapshot().customizing?.[project().id]).toBeUndefined();
    expect(getWorkspaceStructure(store.getSnapshot().account, 'compose').floatingWidgets?.[guestId]).toBeDefined();
    expect(layout().guests).toEqual({});
  });

  it('Keep in workspace places the fallback stage of a center with nothing available, like any other guest', () => {
    const { layout, ports, project, store } = createNavigation({
      arrange: (state) => withArrangement(state, 'edit', { center: ['retired:center'] }),
      presetId: 'edit',
    });
    const centerOf = () => getWorkspaceStructure(store.getSnapshot().account, 'edit').widgetRegions.center.instanceIds;
    const stage = layout().widgetRegions.center.activeInstanceId;
    expect(layout().guests).toEqual({ [stage]: 'center' });

    keepGuestInWorkspace(ports, stage, { canEditStructure: false });
    expect(store.getSnapshot().customizing?.[project().id]?.structure.widgetRegions.center.instanceIds).toEqual([
      'retired:center',
      stage,
    ]);
    expect(layout().guests).toEqual({});
    expect(layout().widgetRegions.center.activeInstanceId).toBe(stage);

    store.commands.layout.cancelCustomize();
    expect(centerOf()).toEqual(['retired:center']);
    expect(layout().guests).toEqual({ [stage]: 'center' });

    keepGuestInWorkspace(ports, stage, { canEditStructure: true });
    expect(centerOf()).toEqual(['retired:center', stage]);
    expect(layout().guests).toEqual({});
  });
});

describe('showWorkflowForm', () => {
  it('shows the form as a guest window where the only workflow is the graph in the status bar panel', () => {
    const { layout, ports } = createNavigation({
      arrange: (state) => withArrangement(state, 'edit', { bottom: ['workflow'] }),
      presetId: 'edit',
    });

    showWorkflowForm(ports);

    const [guestId, surface] = Object.entries(layout().guests)[0] ?? [];
    expect(surface).toBe('window');
    expect(layout().widgetInstances[guestId!]?.typeId).toBe('workflow');
  });

  it('fronts the form the left rail places, behind another tab', () => {
    const { layout, ports } = createNavigation({
      arrange: (state) => withArrangement(state, 'edit', { left: ['generate', 'workflow'] }),
      presetId: 'edit',
    });

    showWorkflowForm(ports);

    expect(layout().widgetRegions.left.activeInstanceId).toBe('workflow');
    expect(layout().guests).toEqual({});
  });
});

describe('goBackToPreviousWorkspace', () => {
  it('returns once to the workspace the last navigation left', () => {
    const { ports, project } = createNavigation();

    openInWorkspace(ports, { surface: 'stage', typeId: 'workflow' });
    expect(project().workspaceId).toBe('automate');

    expect(goBackToPreviousWorkspace(ports)).toBe(true);
    expect(project().workspaceId).toBe('compose');
    expect(goBackToPreviousWorkspace(ports)).toBe(false);
  });

  it('forgets the navigation once the user switches workspace by hand, even back to where it landed', () => {
    const { ports, project, store } = createNavigation();

    openInWorkspace(ports, { surface: 'stage', typeId: 'workflow' });
    void store.commands.layout.activatePreset('video');
    void store.commands.layout.activatePreset('automate');

    expect(goBackToPreviousWorkspace(ports)).toBe(false);
    expect(project().workspaceId).toBe('automate');
  });
});

describe('toggleCenterPreview', () => {
  it('swaps Preview in and back to the stage, without leaving the workspace', () => {
    const { layout, ports, project, store } = createNavigation({ presetId: 'edit' });

    store.commands.widgets.select({ projectId: project().id, region: 'center', widgetId: 'preview' });
    expect(toggleCenterPreview(ports)).toBe(true);
    expect(layout().widgetRegions.center.activeInstanceId).toBe('canvas');
    expect(toggleCenterPreview(ports)).toBe(true);
    expect(layout().widgetRegions.center.activeInstanceId).toBe('preview');
    expect(project().workspaceId).toBe('edit');
  });

  it('swaps a Preview stage with the center’s other view', () => {
    const { layout, ports } = createNavigation({ presetId: 'compose' });

    expect(toggleCenterPreview(ports)).toBe(true);
    expect(layout().widgetRegions.center.activeInstanceId).toBe('gallery:center');
    expect(toggleCenterPreview(ports)).toBe(true);
    expect(layout().widgetRegions.center.activeInstanceId).toBe('preview');
  });
});

describe('runWorkspaceCommand', () => {
  it('leaves the workspace only while not customizing', () => {
    const { ports, store } = createNavigation();
    const leave = vi.fn();
    const run = () =>
      runWorkspaceCommand(ports, { commandId: 'app.selectEditPreset', rearrangeWithoutCustomize: false }, leave);

    store.commands.layout.beginCustomize();
    run();
    expect(leave).not.toHaveBeenCalled();

    store.commands.layout.cancelCustomize();
    run();
    expect(leave).toHaveBeenCalledOnce();
  });
});
