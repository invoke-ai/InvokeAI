import type { RegisteredWidget } from '@workbench/widgetContracts';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { bringRecallWidgetToFront } from './recallEventRuntime';

/** The registry's view of the Video widget: placeable on the left, nothing to preload. */
const videoWidget = {
  implementation: { preload: () => undefined },
  manifest: { allowedRegions: ['left', 'right'], id: 'video' },
  status: 'enabled',
} as unknown as RegisteredWidget;

const testRevealContext = (isEditingText = false) => ({
  getWidgetsForRegion: (region: string) => (region === 'left' ? [videoWidget] : []),
  isEditingText: () => isEditingText,
});

type Store = ReturnType<typeof createWorkbenchStore>;

const videoInstanceId = (store: Store, projectId: string) =>
  Object.values(store.queries.getProject(projectId)!.widgetInstances).find((instance) => instance.typeId === 'video')
    ?.id;

/** Where the Video widget is visible: the active tab of an expanded region whose panel is open, or floating. */
const videoWidgetPlacement = (store: Store, projectId: string): 'hidden' | 'floating' | 'shown' => {
  const project = store.queries.getProject(projectId)!;
  const id = videoInstanceId(store, projectId);

  if (id && project.floatingWidgets?.[id]) {
    return 'floating';
  }
  const left = project.widgetRegions.left;

  return id && left.activeInstanceId === id && !left.isCollapsed && project.layout.panels.isLeftOpen
    ? 'shown'
    : 'hidden';
};

const setup = () => {
  const store = createWorkbenchStore();
  const projectId = store.queries.getSnapshot().activeProject.id;
  const reveal = (options: { isEditingText?: boolean; projectId?: string } = {}) =>
    bringRecallWidgetToFront({
      commands: store.commands,
      owner: captureAccountScope(),
      projectId: options.projectId ?? projectId,
      queries: store.queries,
      reveal: testRevealContext(options.isEditingText),
      typeId: 'video',
    });

  return { projectId, reveal, store };
};

describe('bringRecallWidgetToFront', () => {
  beforeEach(() => {
    accountLifecycle.activate('owner');
  });

  it('opens the widget when the layout has none', () => {
    const { projectId, reveal, store } = setup();

    reveal();

    expect(videoWidgetPlacement(store, projectId)).toBe('shown');
  });

  it('selects its tab when another widget is in front of it', () => {
    const { projectId, reveal, store } = setup();
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'video' });
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'generate' });
    expect(videoWidgetPlacement(store, projectId)).toBe('hidden');

    reveal();

    expect(videoWidgetPlacement(store, projectId)).toBe('shown');
  });

  it('expands its collapsed region, and never collapses a shown one', () => {
    const { projectId, reveal, store } = setup();
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'video' });
    const id = videoInstanceId(store, projectId)!;
    // Selecting the active, expanded tab collapses its region.
    store.commands.widgets.select({ projectId, region: 'left', widgetId: id });
    expect(videoWidgetPlacement(store, projectId)).toBe('hidden');

    reveal();
    expect(videoWidgetPlacement(store, projectId)).toBe('shown');

    reveal();
    expect(videoWidgetPlacement(store, projectId)).toBe('shown');
  });

  it('raises a floating widget without docking it', () => {
    const { projectId, reveal, store } = setup();
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'video' });
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'generate' });
    const videoId = videoInstanceId(store, projectId)!;
    store.commands.widgets.float(videoId);
    const generateId = store.queries.getProject(projectId)!.widgetRegions.left.activeInstanceId!;
    store.commands.widgets.float(generateId);
    const stackOrder = () => store.queries.getProject(projectId)!.floatingWidgets![videoId]!.stackOrder;
    expect(stackOrder()).toBe(1);

    reveal();

    expect(videoWidgetPlacement(store, projectId)).toBe('floating');
    expect(stackOrder()).toBe(2);
  });

  it('unshades a floating widget rolled up to its title bar', () => {
    const { projectId, reveal, store } = setup();
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'video' });
    const videoId = videoInstanceId(store, projectId)!;
    store.commands.widgets.float(videoId);
    store.commands.widgets.setFloatingMode(videoId, 'shaded');

    reveal();

    expect(store.queries.getProject(projectId)!.floatingWidgets![videoId]!.mode).toBe('windowed');
  });

  it('opens the panel of an active, expanded tab whose panel is closed', () => {
    const { projectId, store } = setup();
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'video' });
    // Only layout presets close a panel; stand in for one having done so.
    const queries = {
      getSnapshot: () => {
        const snapshot = store.queries.getSnapshot();
        const { activeProject } = snapshot;
        return {
          ...snapshot,
          activeProject: {
            ...activeProject,
            layout: { ...activeProject.layout, panels: { ...activeProject.layout.panels, isLeftOpen: false } },
          },
        };
      },
    };
    const open = vi.spyOn(store.commands.widgets, 'open');

    bringRecallWidgetToFront({
      commands: store.commands,
      owner: captureAccountScope(),
      projectId,
      queries,
      reveal: testRevealContext(),
      typeId: 'video',
    });

    expect(open).toHaveBeenCalledWith({ projectId, region: 'left', widgetId: 'video' });
  });

  it('does nothing for an account that has since signed out', () => {
    const { projectId, store } = setup();
    const owner = captureAccountScope();
    const before = store.queries.getProject(projectId);
    accountLifecycle.activate('someone-else');

    bringRecallWidgetToFront({
      commands: store.commands,
      owner,
      projectId,
      queries: store.queries,
      reveal: testRevealContext(),
      typeId: 'video',
    });

    expect(store.queries.getProject(projectId)).toBe(before);
  });

  it('leaves the layout alone while the user is typing', () => {
    const { projectId, reveal, store } = setup();
    const before = store.queries.getProject(projectId);

    reveal({ isEditingText: true });

    expect(store.queries.getProject(projectId)).toBe(before);
  });

  it('leaves a project that is no longer on screen alone', () => {
    const { projectId, reveal, store } = setup();
    const before = store.queries.getProject(projectId);
    store.commands.projects.create();

    reveal({ projectId });

    expect(store.queries.getProject(projectId)).toBe(before);
  });
});
