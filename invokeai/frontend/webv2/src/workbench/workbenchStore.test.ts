import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { CanvasLayerContract } from './canvas-engine/contracts';
import type { LayoutPreset, LayoutPresetRoute } from './layoutContracts';
import type { WorkbenchInternalStore, WorkbenchSnapshot } from './workbenchStore';

import { publishLayerPanelSelection, readLayerPanelState, toggleLayerStackCollapsed } from './layerPanelState';
import { getActiveProjectGraph } from './projectWorkflows';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from './widgetPlacementMeta';
import { getProjectWidgetValues } from './widgetState';
import { createInitialWorkbenchState } from './workbenchState';
import { createWorkbenchStore } from './workbenchStore';

const overlays = vi.hoisted(() => ({ closeWidgetOverlays: vi.fn() }));
vi.mock('@platform/ui/widgetOverlayRegistry', () => overlays);

const paintLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const watchSelector = <Selected>(
  store: WorkbenchInternalStore,
  selector: (snapshot: WorkbenchSnapshot) => Selected,
  isEqual: (left: Selected, right: Selected) => boolean = Object.is
) => {
  let current = selector(store.getSnapshot());
  let changeCount = 0;

  const unsubscribe = store.subscribe(() => {
    const next = selector(store.getSnapshot());

    if (!isEqual(current, next)) {
      current = next;
      changeCount += 1;
    }
  });

  return {
    get changeCount() {
      return changeCount;
    },
    get current() {
      return current;
    },
    unsubscribe,
  };
};

describe('createWorkbenchStore', () => {
  beforeEach(() => {
    resetLogging();
    configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'trace' });
  });

  it('exposes stable capability interfaces and initializes snapshot metadata', () => {
    const store = createWorkbenchStore();
    const snapshot = store.getSnapshot();

    expect(store.commands).toBe(store.commands);
    expect(store.queries).toBe(store.queries);
    expect(snapshot.hasHydrated).toBe(false);
    expect(snapshot.projects).toHaveLength(1);
  });

  it('blocks persistence retargets only for queue work that has reached the backend', () => {
    const runningState = createInitialWorkbenchState();
    const source = runningState.projects[0]!;
    runningState.projects[0] = {
      ...source,
      queue: {
        items: [{ backendItemIds: [11], id: 'run-1', status: 'running' } as (typeof source.queue.items)[number]],
      },
    };
    const runningStore = createWorkbenchStore(runningState);
    const payload = {
      boardId: 'copy-board',
      name: `${source.name} (copy)`,
      project: { ...source, id: 'copy-id' },
      projectId: source.id,
      sourceName: source.name,
      targetProjectId: 'copy-id',
    };

    expect(runningStore.internal.persistence.retargetProject(payload)).toEqual({
      ok: false,
      reason: 'active-queue-runs',
    });

    for (const status of ['completed', 'failed', 'cancelled'] as const) {
      const terminalStore = createWorkbenchStore({
        ...runningState,
        projects: [
          {
            ...runningState.projects[0]!,
            queue: {
              items: [
                {
                  backendItemIds: [11],
                  id: 'run-1',
                  status,
                } as (typeof source.queue.items)[number],
              ],
            },
          },
        ],
      });
      expect(terminalStore.internal.persistence.retargetProject(payload)).toEqual({ ok: true });
    }

    const pendingState = createInitialWorkbenchState();
    pendingState.projects[0] = {
      ...pendingState.projects[0]!,
      queue: { items: [{ id: 'pending-1', status: 'pending' } as (typeof source.queue.items)[number]] },
    };
    const pendingStore = createWorkbenchStore(pendingState);
    const pendingSource = pendingState.projects[0]!;
    expect(
      pendingStore.internal.persistence.retargetProject({
        ...payload,
        project: { ...pendingSource, id: 'copy-id' },
        projectId: pendingSource.id,
        sourceName: pendingSource.name,
      })
    ).toEqual({ ok: true });
  });

  it('blocks persistence retargets while durable cancellation is pending', () => {
    const state = createInitialWorkbenchState();
    const source = state.projects[0]!;
    state.projects[0] = {
      ...source,
      queue: {
        items: [
          {
            backendItemIds: [11],
            cancellationPending: true,
            id: 'run-1',
            status: 'cancelled',
          } as (typeof source.queue.items)[number],
        ],
      },
    };
    const store = createWorkbenchStore(state);

    expect(
      store.internal.persistence.retargetProject({
        boardId: 'copy-board',
        name: `${source.name} (copy)`,
        project: { ...source, id: 'copy-id' },
        projectId: source.id,
        sourceName: source.name,
        targetProjectId: 'copy-id',
      })
    ).toEqual({ ok: false, reason: 'active-queue-runs' });
  });

  it("keeps each project's transient layer multi-selection across project switches", () => {
    const store = createWorkbenchStore();
    const firstProjectId = store.getSnapshot().activeProject.id;
    store.commands.canvas.apply(firstProjectId, {
      anchor: stackTopAnchor(firstProjectId),
      layer: paintLayer('a'),
      type: 'addCanvasLayer',
    });
    store.commands.canvas.apply(firstProjectId, {
      anchor: stackTopAnchor(firstProjectId),
      layer: paintLayer('b'),
      type: 'addCanvasLayer',
    });
    publishLayerPanelSelection({ primaryId: 'b', projectId: firstProjectId, selectedIds: ['a', 'b'] });

    const secondProject = store.commands.projects.create();
    store.commands.projects.switchTo(firstProjectId);

    expect(readLayerPanelState(firstProjectId, 'b').selectedIds).toEqual(['a', 'b']);
    expect(readLayerPanelState(secondProject.id, null).selectedIds).toEqual([]);
  });

  it("clears every project's panel state on hydration", () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;
    store.commands.canvas.apply(projectId, {
      anchor: stackTopAnchor(projectId),
      layer: paintLayer('a'),
      type: 'addCanvasLayer',
    });
    store.commands.canvas.apply(projectId, {
      anchor: stackTopAnchor(projectId),
      layer: paintLayer('b'),
      type: 'addCanvasLayer',
    });
    publishLayerPanelSelection({ primaryId: 'b', projectId, selectedIds: ['a', 'b'] });
    toggleLayerStackCollapsed(projectId, 'b', 'raster');
    expect(readLayerPanelState(projectId, 'b')).toMatchObject({ collapsedStacks: ['raster'], selectedIds: ['a', 'b'] });

    store.internal.persistence.hydrate(createInitialWorkbenchState());

    expect(readLayerPanelState(projectId, 'b')).toMatchObject({ collapsedStacks: [], selectedIds: ['b'] });
  });

  it('does not resurrect stale secondaries after external primary changes in the same project', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;
    for (const id of ['a', 'b', 'c']) {
      store.commands.canvas.apply(projectId, {
        anchor: stackTopAnchor(projectId),
        layer: paintLayer(id),
        type: 'addCanvasLayer',
      });
    }
    // Model a panel-originated multi-selection: it publishes before dispatching
    // its new primary, so the store preserves the selected set.
    publishLayerPanelSelection({ primaryId: 'c', projectId, selectedIds: ['a', 'c'] });
    store.commands.canvas.apply(projectId, { id: 'c', type: 'setCanvasSelectedLayer' });
    expect(readLayerPanelState(projectId, 'c').selectedIds).toEqual(['a', 'c']);

    store.commands.canvas.apply(projectId, { id: 'b', type: 'setCanvasSelectedLayer' });
    store.commands.canvas.apply(projectId, { id: 'c', type: 'setCanvasSelectedLayer' });

    expect(readLayerPanelState(projectId, 'c').selectedIds).toEqual(['c']);
  });

  it('coordinates layout preset activation across every command caller', async () => {
    const deferred = new Map<string, { promise: Promise<void>; resolve: () => void }>();
    const loadLayoutPresetWidgets = (preset: LayoutPreset): Promise<void> => {
      let resolve!: () => void;
      const promise = new Promise<void>((next) => {
        resolve = next;
      });

      deferred.set(preset.id, { promise, resolve });
      return promise;
    };
    const store = createWorkbenchStore(createInitialWorkbenchState(), { loadLayoutPresetWidgets });

    const stripActivation = store.commands.layout.activatePreset('compose');
    const hotkeyActivation = store.commands.layout.activatePreset('edit');

    deferred.get('edit')?.resolve();
    await hotkeyActivation;
    deferred.get('compose')?.resolve();
    await stripActivation;

    expect(store.getSnapshot().activeProject.layout.presetId).toBe('edit');
    expect(store.getSnapshot().activeProject.invocation).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });
  });

  it('discards a pending preset when project focus changes before loading finishes', async () => {
    let resolve!: () => void;
    const loadLayoutPresetWidgets = (): Promise<void> =>
      new Promise<void>((next) => {
        resolve = next;
      });
    const store = createWorkbenchStore(createInitialWorkbenchState(), { loadLayoutPresetWidgets });
    const firstProjectId = store.getSnapshot().activeProject.id;
    const secondProject = store.commands.projects.create();

    store.commands.projects.switchTo(firstProjectId);
    const activation = store.commands.layout.activatePreset('edit');
    store.commands.projects.switchTo(secondProject.id);
    store.commands.projects.switchTo(firstProjectId);
    resolve();
    await activation;

    expect(store.getSnapshot().activeProject.layout.presetId).toBe('compose');
  });

  it('keeps a synchronous preset application newer than a pending activation', async () => {
    let resolve!: () => void;
    const loadLayoutPresetWidgets = (): Promise<void> =>
      new Promise<void>((next) => {
        resolve = next;
      });
    const store = createWorkbenchStore(createInitialWorkbenchState(), { loadLayoutPresetWidgets });

    const activation = store.commands.layout.activatePreset('edit');
    store.commands.layout.applyPreset('automate');
    resolve();
    await activation;

    expect(store.getSnapshot().activeProject.layout.presetId).toBe('automate');
    expect(store.getSnapshot().activeProject.invocation).toMatchObject({
      destination: 'gallery',
      sourceId: 'workflow',
    });
  });

  it('discards a pending custom preset after its saved snapshot is removed', async () => {
    let resolve!: () => void;
    const loadLayoutPresetWidgets = (): Promise<void> =>
      new Promise<void>((next) => {
        resolve = next;
      });
    const store = createWorkbenchStore(createInitialWorkbenchState(), { loadLayoutPresetWidgets });

    store.commands.layout.applyPreset('edit');
    store.commands.layout.createPreset('custom-pending', 'Pending');
    const activation = store.commands.layout.activatePreset('custom-pending');
    store.commands.layout.deletePreset('custom-pending');
    resolve();
    await activation;

    expect(store.getSnapshot().activeProject.layout.presetId).toBe('edit');
  });

  it('closes widget overlays only for changes that hide or replace a shown widget', () => {
    const store = createWorkbenchStore();
    overlays.closeWidgetOverlays.mockClear();
    const project = store.getSnapshot().activeProject;
    const [region, regionState] = Object.entries(project.widgetRegions).find(
      ([, state]) => state.instanceIds.length > 1
    )!;
    const other = regionState.instanceIds.find((id) => id !== regionState.activeInstanceId)!;

    store.commands.projects.rename(project.id, 'Renamed');
    expect(overlays.closeWidgetOverlays).not.toHaveBeenCalled();

    store.commands.widgets.select({ projectId: project.id, region: region as never, widgetId: other });
    expect(overlays.closeWidgetOverlays).toHaveBeenCalledTimes(1);

    store.commands.widgets.select({ projectId: project.id, region: region as never, widgetId: other });
    expect(overlays.closeWidgetOverlays).toHaveBeenCalledTimes(1);

    store.commands.layout.applyPreset('edit');
    expect(overlays.closeWidgetOverlays).toHaveBeenCalledTimes(2);

    store.commands.projects.create();
    expect(overlays.closeWidgetOverlays).toHaveBeenCalledTimes(3);
  });

  it('notifies subscribers once for reducer changes and not for no-op reducer results', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();

    store.subscribe(listener);

    const projectId = store.getSnapshot().activeProject.id;

    store.commands.projects.rename(projectId, 'Renamed');

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().activeProject.name).toBe('Renamed');

    store.commands.projects.rename(projectId, '   ');

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().activeProject.name).toBe('Renamed');
  });

  it('updates hydration metadata without changing durable workbench state', () => {
    const store = createWorkbenchStore();
    const initialState = store.getState();
    const initialPersistedRevision = store.getPersistedRevision();
    const listener = vi.fn();

    store.subscribe(listener);

    store.setHasHydrated(true);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().hasHydrated).toBe(true);
    expect(store.getState()).toBe(initialState);
    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.setHasHydrated(true);

    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('bumps persisted revision only for autosaved state changes', () => {
    const store = createWorkbenchStore();
    const initialPersistedRevision = store.getPersistedRevision();

    store.commands.queue.setConnectionStatus({ status: 'connected' });

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.commands.notifications.add({ kind: 'info', title: 'Global notice' });

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.commands.projects.rename(store.getSnapshot().activeProject.id, 'Persisted rename');

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision + 1);
  });

  it('forwards payload-form commands into reducer actions', () => {
    const store = createWorkbenchStore();

    store.commands.notifications.add({ kind: 'info', title: 'Payload form' });

    expect(store.getSnapshot().notifications[0]?.title).toBe('Payload form');
  });

  it('maps positional-form commands onto their action payloads', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    store.commands.gallery.setSearchTerm('positional form', projectId);

    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'gallery').searchTerm).toBe('positional form');
  });

  it('edits a preset default route through the layout capability', () => {
    const store = createWorkbenchStore();

    store.commands.layout.setPresetRoute('compose', { destination: 'canvas', sourceId: 'upscale' });

    expect(store.getSnapshot().account.layoutPresetRouteOverrides?.compose).toEqual({
      destination: 'canvas',
      sourceId: 'upscale',
    });
  });

  it('reorders presets through the layout capability', () => {
    const store = createWorkbenchStore();

    store.commands.layout.reorderPresets('compose', 'edit');

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['edit', 'compose', 'video', 'automate']);
  });

  it('creates a custom preset with an explicitly selected default route in one command', () => {
    const store = createWorkbenchStore();
    const defaultRoute: LayoutPresetRoute = { destination: 'gallery', sourceId: 'workflow' };

    store.commands.layout.createPreset('custom-explicit-route', 'Workflow layout', 'workflow', defaultRoute);
    defaultRoute.destination = 'canvas';

    expect(store.getSnapshot().account.customLayoutPresets?.[0]?.defaultRoute).toEqual({
      destination: 'gallery',
      sourceId: 'workflow',
    });
  });

  it('dispatches nullary commands with no payload', () => {
    const store = createWorkbenchStore();

    store.commands.notifications.add({ kind: 'info', title: 'To clear' });
    store.commands.notifications.clear();

    expect(store.getSnapshot().notifications).toHaveLength(0);
  });

  it('records recordError actions into project diagnostics', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    const error = new Error('socket closed');

    store.commands.notifications.reportError({
      area: 'queue-runtime',
      context: { error, itemId: 'item-1' },
      message: 'Queue failed',
      namespace: 'queue',
    });

    expect(getLogSnapshot().entries).toMatchObject([
      {
        context: { itemId: 'item-1' },
        error: { message: 'socket closed', name: 'Error' },
        level: 'error',
        message: 'Queue failed',
        name: 'queue.queue-runtime',
        namespace: 'queue',
        source: { area: 'queue-runtime', namespace: 'queue', projectId },
      },
    ]);
    expect(store.getSnapshot().notifications[0]).toMatchObject({
      message: 'Queue failed: socket closed',
      title: 'Error',
    });
  });

  it('records autosave outcomes and queue item transitions with the history owner as the only reporter', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    store.internal.persistence.saveStarted();
    store.internal.persistence.savePending('Autosave requires your attention.');
    store.internal.persistence.saveFailed('Backend rejected the revision');
    store.commands.queue.setStatus({ error: 'Out of memory', projectId, queueItemId: 'missing', status: 'failed' });

    expect(getLogSnapshot().entries).toMatchObject([
      {
        context: { queueItemId: 'missing', reason: 'Out of memory', status: 'failed' },
        level: 'error',
        message: 'Queue item failed: Out of memory',
        name: 'queue.item-failed',
        source: { area: 'history', namespace: 'queue', projectId },
      },
      {
        context: { reason: 'Backend rejected the revision' },
        level: 'error',
        name: 'persistence.autosave-failed',
        source: { area: 'autosave', namespace: 'persistence', projectId },
      },
      { level: 'warn', name: 'persistence.autosave-pending' },
      { level: 'debug', name: 'persistence.autosave-started' },
    ]);
  });

  it('records accepted widget failures into diagnostics once', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;
    const failure = {
      details: 'Widget stack',
      message: 'Widget failed',
      occurredAt: '2026-06-29T00:00:00.000Z',
      widgetId: 'workflow' as const,
    };

    store.commands.notifications.recordWidgetFailure(failure);
    store.commands.notifications.recordWidgetFailure(failure);

    expect(getLogSnapshot().entries).toMatchObject([
      {
        context: { details: 'Widget stack', widgetId: 'workflow' },
        level: 'error',
        message: 'Widget failed',
        name: 'widget.registration-failed',
        namespace: 'system',
        source: { area: 'widget-failure', namespace: 'system', projectId },
      },
    ]);
  });

  it('stops notifying unsubscribed listeners', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);

    unsubscribe();
    store.commands.projects.create();

    expect(listener).not.toHaveBeenCalled();
  });

  it('keeps active-project selectors stable across unrelated global state updates', () => {
    const store = createWorkbenchStore();
    const activeProjectWatcher = watchSelector(store, (snapshot) => snapshot.activeProject);
    const activeProjectIdWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.id);
    const notificationCountWatcher = watchSelector(store, (snapshot) => snapshot.notifications.length);

    store.commands.notifications.add({ kind: 'info', title: 'Global notice' });

    expect(notificationCountWatcher.current).toBe(1);
    expect(notificationCountWatcher.changeCount).toBe(1);
    expect(activeProjectWatcher.changeCount).toBe(0);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
  });

  it('lets narrow selectors observe only their own state changes', () => {
    const store = createWorkbenchStore();
    const backendStatusWatcher = watchSelector(store, (snapshot) => snapshot.backendConnection.status);
    const activeProjectIdWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.id);
    const projectCountWatcher = watchSelector(store, (snapshot) => snapshot.projects.length);
    const batchCountWatcher = watchSelector(store, (snapshot) =>
      Number(getProjectWidgetValues(snapshot.activeProject, 'generate').batchCount ?? 1)
    );

    store.commands.queue.setConnectionStatus({ status: 'connected' });

    expect(backendStatusWatcher.current).toBe('connected');
    expect(backendStatusWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
    expect(projectCountWatcher.changeCount).toBe(0);
    expect(batchCountWatcher.changeCount).toBe(0);

    store.commands.generation.setBatchCount(3);

    expect(batchCountWatcher.current).toBe(3);
    expect(batchCountWatcher.changeCount).toBe(1);
    expect(backendStatusWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
    expect(projectCountWatcher.changeCount).toBe(0);

    store.commands.projects.create();

    expect(projectCountWatcher.current).toBe(2);
    expect(projectCountWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.current).toBe(store.getSnapshot().activeProject.id);
    expect(activeProjectIdWatcher.changeCount).toBe(1);
    expect(backendStatusWatcher.changeCount).toBe(1);
  });

  it('supports custom equality for derived selector objects', () => {
    const store = createWorkbenchStore();
    const statusObjectWatcher = watchSelector(
      store,
      (snapshot) => ({ status: snapshot.backendConnection.status }),
      (left, right) => left.status === right.status
    );

    store.commands.notifications.add({ kind: 'info', title: 'Unrelated notice' });

    expect(statusObjectWatcher.current).toEqual({ status: 'connecting' });
    expect(statusObjectWatcher.changeCount).toBe(0);

    store.commands.queue.setConnectionStatus({ status: 'connected' });

    expect(statusObjectWatcher.current).toEqual({ status: 'connected' });
    expect(statusObjectWatcher.changeCount).toBe(1);
  });

  it('keeps widget placement metadata stable across widget value updates', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );
    const widgetInstancesWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.widgetInstances);

    store.commands.widgets.patchInstanceValues('generate', { prompt: 'updated' });

    expect(widgetInstancesWatcher.changeCount).toBe(1);
    expect(placementWatcher.changeCount).toBe(0);
  });

  it('changes the rail-facing placement only when a window floats or docks, not while it moves, shades, or is raised', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );
    const windowsWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.floatingWidgets);

    store.commands.widgets.float('image-map');
    store.commands.widgets.float('gallery');
    expect(placementWatcher.changeCount).toBe(2);

    for (let step = 1; step <= 20; step += 1) {
      store.commands.widgets.setFloatingGeometry('image-map', { heightPx: 300, widthPx: 320, x: 100 + step, y: 100 });
    }
    store.commands.widgets.setFloatingMode('image-map', 'shaded');
    store.commands.widgets.raiseFloating('image-map');
    store.commands.widgets.revealFloating('gallery');

    // 23 window writes, none of which reach the rails.
    expect(windowsWatcher.changeCount).toBe(2 + 23);
    expect(placementWatcher.changeCount).toBe(2);
    // Raising is ordinary focus: it leaves a shaded window shaded. Only the explicit reveal expands one.
    expect(store.getSnapshot().activeProject.floatingWidgets?.['image-map'].mode).toBe('shaded');

    store.commands.widgets.dockFloating('gallery');
    expect(placementWatcher.changeCount).toBe(3);
  });

  it('notifies no subscriber when the window already on top is raised again', () => {
    const store = createWorkbenchStore();

    store.commands.widgets.float('image-map');
    store.commands.widgets.float('gallery');
    const stateWatcher = watchSelector(store, (snapshot) => snapshot.activeProject);

    // Every press inside a window asks for a raise. For the topmost window the reducer returns the same state, and
    // the store must turn that into silence: no notification, so no re-render and no autosave.
    for (let repeat = 0; repeat < 5; repeat += 1) {
      store.commands.widgets.raiseFloating('gallery');
    }

    expect(stateWatcher.changeCount).toBe(0);
  });

  it('treats identical widget placement in a different project as a placement change', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );
    const firstProjectId = placementWatcher.current.projectId;

    store.commands.projects.create();

    expect(placementWatcher.current.projectId).not.toBe(firstProjectId);
    expect(placementWatcher.changeCount).toBe(1);
  });

  it('does not notify subscribers for equivalent generate and project settings writes', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();

    store.subscribe(listener);

    store.commands.generation.setBatchCount(1);
    store.commands.account.updateProjectPreferences({ useCpuNoise: true });
    store.commands.layout.setRegionCollapsed('left', false);
    store.commands.layout.setRegionSize('left', store.getSnapshot().activeProject.widgetRegions.left.sizePx);
    store.commands.widgets.patchValues('generate', {});
    store.commands.queue.setConnectionStatus({ status: 'connecting' });

    expect(listener).not.toHaveBeenCalled();
  });

  it('composes same-turn functional widget patches from the latest values', () => {
    const store = createWorkbenchStore();
    store.commands.widgets.patchValues('upscale', { weights: { first: 0.75, second: 0.5 }, steps: 30 });

    store.commands.widgets.patchValues('upscale', (current) => ({
      weights: { ...(current.weights as Record<string, number>), first: 0.8 },
    }));
    store.commands.widgets.patchValues('upscale', (current) => ({
      weights: { ...(current.weights as Record<string, number>), second: 0.55 },
    }));

    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'upscale')).toMatchObject({
      steps: 30,
      weights: { first: 0.8, second: 0.55 },
    });
  });

  it('reads and writes the fenced project after a switch and skips a closed project', () => {
    const store = createWorkbenchStore();
    const firstProjectId = store.getSnapshot().activeProject.id;
    store.commands.widgets.patchValues('video', { steps: 12 });
    const lateUpdate = () =>
      store.commands.widgets.patchValues('video', (current) => ({ steps: Number(current.steps) + 1 }), firstProjectId);

    store.commands.projects.create();
    store.commands.widgets.patchValues('video', { steps: 50 });
    lateUpdate();

    const firstProject = store.getSnapshot().projects.find((project) => project.id === firstProjectId)!;
    expect(getProjectWidgetValues(firstProject, 'video').steps).toBe(13);
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'video').steps).toBe(50);

    store.commands.projects.close(firstProjectId);
    const update = vi.fn(() => ({ steps: 99 }));
    store.commands.widgets.patchValues('video', update, firstProjectId);
    expect(update).not.toHaveBeenCalled();
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'video').steps).toBe(50);
  });

  it('does not notify subscribers for equivalent queue status writes', () => {
    const store = createWorkbenchStore();
    const project = store.getSnapshot().activeProject;
    const listener = vi.fn();

    store.subscribe(listener);

    store.commands.queue.setStatus({ projectId: project.id, queueItemId: 'missing-queue-item', status: 'pending' });

    expect(listener).not.toHaveBeenCalled();
  });

  it('exposes queue behavior through namespaced commands', () => {
    const initialState = createInitialWorkbenchState();
    const initialProject = initialState.projects[0];
    const store = createWorkbenchStore({
      ...initialState,
      projects: [
        {
          ...initialProject,
          queue: {
            items: [
              {
                cancellable: true,
                id: 'queue-item-1',
                snapshot: {} as (typeof initialProject.queue.items)[number]['snapshot'],
                status: 'pending',
              },
              {
                cancellable: false,
                id: 'queue-item-complete',
                snapshot: {} as (typeof initialProject.queue.items)[number]['snapshot'],
                status: 'completed',
              },
            ],
          },
        },
      ],
    });
    const projectId = store.getSnapshot().activeProject.id;

    store.commands.queue.setConnectionStatus({ status: 'connected' });
    store.commands.queue.cancel(projectId, 'queue-item-1');

    expect(store.getState().backendConnection.status).toBe('connected');
    expect(store.getSnapshot().activeProject.queue.items[0]?.status).toBe('cancelled');
    expect(store.getSnapshot().notifications[0]).toMatchObject({
      kind: 'info',
      title: 'Invocation cancellation requested',
    });

    store.commands.queue.clearCompleted();

    expect(store.getSnapshot().activeProject.queue.items.map((item) => item.id)).toEqual(['queue-item-1']);
  });

  it('enforces project lifecycle invariants through observable command results', () => {
    const store = createWorkbenchStore();
    const firstProjectId = store.getSnapshot().activeProject.id;

    expect(store.commands.projects.switchTo('missing')).toEqual({ ok: false, reason: 'project-not-found' });
    expect(store.getSnapshot().activeProject.id).toBe(firstProjectId);

    const secondProject = store.commands.projects.create();

    expect(store.getSnapshot().activeProject.id).toBe(secondProject.id);
    expect(store.commands.projects.switchTo(firstProjectId)).toEqual({ ok: true });
    expect(store.getSnapshot().activeProject.id).toBe(firstProjectId);
    expect(store.commands.projects.close(firstProjectId)).toEqual({ ok: true });
    expect(store.getSnapshot().activeProject.id).toBe(secondProject.id);
    expect(store.commands.projects.close(secondProject.id)).toEqual({ ok: false, reason: 'last-project' });
    expect(store.getSnapshot().notifications).toEqual([]);
  });

  it.each([
    { cancellationPending: undefined, status: 'pending' as const },
    { cancellationPending: undefined, status: 'running' as const },
    { cancellationPending: true, status: 'cancelled' as const },
  ])('keeps a project open while queue work is active ($status)', ({ cancellationPending, status }) => {
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    state.projects[0] = {
      ...project,
      queue: {
        items: [
          {
            cancellable: true,
            ...(cancellationPending === undefined ? {} : { cancellationPending }),
            id: 'active-run',
            snapshot: {} as (typeof project.queue.items)[number]['snapshot'],
            status,
          },
        ],
      },
    };
    const store = createWorkbenchStore(state);

    expect(store.commands.projects.close(project.id)).toEqual({ ok: false, reason: 'active-queue-runs' });
    store.commands.projects.create();

    expect(store.commands.projects.close(project.id)).toEqual({ ok: false, reason: 'active-queue-runs' });
    expect(store.getSnapshot().projects.some((candidate) => candidate.id === project.id)).toBe(true);
  });

  it('applies Canvas and Workflow edits without exposing aggregate reducer actions', () => {
    const store = createWorkbenchStore();
    const project = store.getSnapshot().activeProject;
    const bbox = project.canvas.document.bbox;

    expect(store.commands.canvas.apply(project.id, { bbox: { ...bbox, x: bbox.x + 10 }, type: 'setCanvasBbox' })).toBe(
      true
    );
    expect(store.getSnapshot().activeProject.canvas.document.bbox.x).toBe(bbox.x + 10);

    store.commands.workflows.editGraph({ patch: { name: 'Command-owned workflow' }, type: 'setMetadata' });

    expect(getActiveProjectGraph(store.getSnapshot().activeProject).name).toBe('Command-owned workflow');
  });

  it('keeps placement selectors stable across generate and settings changes', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );

    store.commands.generation.setBatchCount(2);
    store.commands.account.updateProjectPreferences({ useCpuNoise: false });

    expect(placementWatcher.changeCount).toBe(0);
  });
});
