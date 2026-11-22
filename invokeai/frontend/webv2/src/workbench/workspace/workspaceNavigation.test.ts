import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { describe, expect, it, vi } from 'vitest';

import {
  getWorkspaceOffer,
  goBackToPreviousWorkspace,
  openInWorkspace,
  toggleCenterPreview,
} from './workspaceNavigation';

const createNavigation = ({ presetId = 'compose', warm = true }: { presetId?: string; warm?: boolean } = {}) => {
  const pendingLoads: (() => void)[] = [];
  const store = createWorkbenchStore(
    workbenchReducer(createInitialWorkbenchState(), { presetId, type: 'applyPreset' }),
    {
      isLoaded: () => warm,
      loadLayoutPresetWidgets: () =>
        new Promise<void>((resolve) => {
          pendingLoads.push(resolve);
        }),
    }
  );
  const focus = { focusFloating: vi.fn(), focusRegion: vi.fn() };
  const ports = { commands: store.commands, focus, getWidgetsForRegion, queries: store.queries };
  const project = () => store.queries.getSnapshot().activeProject;
  const notices = () => store.queries.getSnapshot().notifications;
  const placedTypes = () =>
    Object.values(project().widgetRegions).flatMap(({ instanceIds }) =>
      instanceIds.map((id) => project().widgetInstances[id]?.typeId)
    );

  return {
    finishLoading: () => pendingLoads.forEach((resolve) => resolve()),
    focus,
    notices,
    placedTypes,
    ports,
    project,
    store,
  };
};

describe('openInWorkspace', () => {
  it('lets only the first intent of a gesture change workspace, and offers to add what the new one lacks', () => {
    const { notices, placedTypes, ports, project } = createNavigation();

    expect(
      openInWorkspace(ports, [
        { surface: 'stage', typeId: 'canvas' },
        { surface: 'stage', typeId: 'workflow' },
      ])
    ).toEqual({ changedWorkspace: true, ok: true, region: 'center', workspaceId: 'edit' });

    expect(project().layout.presetId).toBe('edit');
    expect(placedTypes()).not.toContain('workflow');
    expect(notices()[0]).toMatchObject({
      actions: [{ kind: 'add-widget', surface: 'stage', widgetId: 'workflow' }],
      titleKey: 'shell.workspace.notHere',
    });
  });

  it('coalesces repeated offers for one widget into the newest notice', () => {
    const { notices, ports } = createNavigation();

    openInWorkspace(ports, { typeId: 'diagnostics' });
    openInWorkspace(ports, { typeId: 'diagnostics' });

    expect(notices()).toHaveLength(1);
    expect(notices()[0]).toMatchObject({ occurrenceCount: 2, titleKey: 'shell.workspace.notHere' });
  });

  it('shows the rest of a gesture when its first widget is unavailable, focusing the one it names', () => {
    const { focus, ports, project } = createNavigation();
    const withoutPreview = (region: Parameters<typeof getWidgetsForRegion>[0]) =>
      getWidgetsForRegion(region).filter(({ manifest }) => manifest.id !== 'preview');

    openInWorkspace(
      { ...ports, getWidgetsForRegion: withoutPreview },
      [{ surface: 'stage', typeId: 'preview' }, { typeId: 'queue' }],
      { focus: 'queue' }
    );

    expect(project().widgetRegions.right.activeInstanceId).toBe('queue');
    expect(focus.focusRegion).toHaveBeenCalledExactlyOnceWith('right', 'queue', 'queue');
  });

  it('fronts what this workspace places for a background source, and leaves the offer to its notice', () => {
    const { focus, notices, ports, project } = createNavigation();
    const regionsBefore = project().widgetRegions;

    expect(openInWorkspace(ports, { typeId: 'canvas' }, { background: true })).toEqual({
      ok: false,
      reason: 'not-placed',
    });
    expect(project().layout.presetId).toBe('compose');
    expect(project().widgetRegions).toBe(regionsBefore);
    expect(notices()).toEqual([]);
    expect(getWorkspaceOffer(ports, { typeId: 'canvas' })).toEqual([{ kind: 'open-widget', widgetId: 'canvas' }]);

    openInWorkspace(ports, { typeId: 'upscale' }, { background: true });
    expect(project().widgetRegions.left).toMatchObject({
      activeInstanceId: 'upscale',
      instanceIds: ['generate', 'upscale'],
    });
    expect(focus.focusRegion).not.toHaveBeenCalled();
  });

  it('runs follow-up work once a cold entry lands, and never for one superseded before it does', async () => {
    const landed = createNavigation({ presetId: 'automate', warm: false });
    const afterLanding = vi.fn(() => landed.project().layout.presetId);

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
    expect(dropped.project().widgetRegions.left.activeInstanceId).toBe('generate');
    expect(afterDropped).not.toHaveBeenCalled();
    expect(goBackToPreviousWorkspace(dropped.ports)).toBe(false);
  });
});

describe('goBackToPreviousWorkspace', () => {
  it('returns once to the workspace the last navigation left', () => {
    const { ports, project } = createNavigation();

    openInWorkspace(ports, { surface: 'stage', typeId: 'workflow' });
    expect(project().layout.presetId).toBe('automate');

    expect(goBackToPreviousWorkspace(ports)).toBe(true);
    expect(project().layout.presetId).toBe('compose');
    expect(goBackToPreviousWorkspace(ports)).toBe(false);
  });

  it('forgets the navigation once the user switches workspace by hand, even back to where it landed', () => {
    const { ports, project, store } = createNavigation();

    openInWorkspace(ports, { surface: 'stage', typeId: 'workflow' });
    void store.commands.layout.activatePreset('video');
    void store.commands.layout.activatePreset('automate');

    expect(goBackToPreviousWorkspace(ports)).toBe(false);
    expect(project().layout.presetId).toBe('automate');
  });
});

describe('toggleCenterPreview', () => {
  it('swaps Preview into the center and back to the view it replaced, without leaving the workspace', () => {
    const { ports, project } = createNavigation({ presetId: 'automate' });

    expect(toggleCenterPreview(ports)).toBe(true);
    expect(project().widgetRegions.center.activeInstanceId).toBe('preview');
    expect(toggleCenterPreview(ports)).toBe(true);
    expect(project().widgetRegions.center.activeInstanceId).toBe('workflow:center');
  });
});
