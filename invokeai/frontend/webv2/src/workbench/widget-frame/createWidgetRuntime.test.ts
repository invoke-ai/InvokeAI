import type { WidgetInstanceContract, WidgetTypeId } from '@workbench/widgetContracts';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';
import type { WorkbenchAction } from '@workbench/workbenchState.testing';
import type { WorkbenchWidgetCommands } from '@workbench/workbenchStore';

import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { cloneWidgetRuntimeState, createWidgetRuntime, getProjectWidgetRuntimeState } from './createWidgetRuntime';

const createInstance = (id = 'alpha', typeId: WidgetTypeId = 'alpha'): WidgetInstanceContract => ({
  createdAt: '2026-01-01T00:00:00.000Z',
  id,
  state: { id: typeId, label: typeId, values: { current: true }, version: 1 },
  typeId,
});

const createDispatch = () => {
  const actions: WorkbenchAction[] = [];
  const dispatch = (action: WorkbenchAction): void => {
    actions.push(action);
  };

  const widgets: WorkbenchWidgetCommands = {
    clearInstanceValues: (instanceId, keys, projectId) =>
      dispatch({ instanceId, keys, projectId, type: 'clearWidgetInstanceValues' }),
    dockFloating: (instanceId) => dispatch({ instanceId, type: 'dockFloatingWidget' }),
    closeFloating: (instanceId) => dispatch({ instanceId, type: 'closeFloatingWidget' }),
    closeGuest: (instanceId) => dispatch({ instanceId, type: 'closeGuest' }),
    keepGuest: (instanceId, allowMultiple) => dispatch({ allowMultiple, instanceId, type: 'keepGuest' }),
    showGuest: (options) => dispatch({ ...options, type: 'showGuest' }),
    front: (options) => dispatch({ ...options, type: 'frontRegionWidget' }),
    float: (instanceId, region) =>
      dispatch(region ? { instanceId, region, type: 'floatWidget' } : { instanceId, type: 'floatWidget' }),
    raiseFloating: (instanceId) => dispatch({ instanceId, type: 'raiseFloatingWidget' }),
    move: (options) => dispatch({ ...options, type: 'moveWidgetInstance' }),
    setAlignment: (options) => dispatch({ ...options, type: 'setWidgetInstanceAlignment' }),
    open: (options) => dispatch({ ...options, type: 'openRegionWidget' }),
    patchInstanceValues: (instanceId, values, projectId) =>
      dispatch({ instanceId, projectId, type: 'patchWidgetInstanceValues', values }),
    patchValues: vi.fn(),
    reorder: (options) => dispatch({ ...options, type: 'reorderWidgetInstances' }),
    revealFloating: (instanceId) => dispatch({ instanceId, type: 'revealFloatingWidget' }),
    select: (options) => dispatch({ ...options, type: 'selectRegionWidget' }),
    setFloatingGeometry: (instanceId, geometry) =>
      dispatch({ instanceId, type: 'setFloatingWidgetGeometry', ...geometry }),
    setFloatingMode: (instanceId, mode) => dispatch({ instanceId, mode, type: 'setFloatingWidgetMode' }),
    setInstanceValues: (instanceId, values, projectId) =>
      dispatch({ instanceId, projectId, type: 'setWidgetInstanceValues', values }),
    toggle: (options) => dispatch({ ...options, type: 'toggleRegionWidget' }),
  };

  return { actions, widgets };
};

const createStateApi = () => ({
  getSnapshot: () => ({ current: true }),
  patch: vi.fn(),
  set: vi.fn(),
  useSelector: vi.fn(),
});

const createPlacementProject = (
  overrides: Partial<Pick<WidgetPlacementProject, 'widgetInstances' | 'widgetRegions'>> = {}
): WidgetPlacementProject => ({
  projectId: 'project-1',
  widgetInstances: {
    alpha: createInstance('alpha'),
    beta: createInstance('beta', 'beta'),
    ...overrides.widgetInstances,
  },
  widgetRegions: {
    bottom: { activeInstanceId: 'alpha', instanceIds: [] },
    center: { activeInstanceId: 'alpha', instanceIds: [] },
    left: { activeInstanceId: 'alpha', instanceIds: [] },
    right: { activeInstanceId: 'alpha', instanceIds: ['alpha'] },
    ...overrides.widgetRegions,
  },
});

describe('createWidgetRuntime', () => {
  beforeEach(() => {
    resetLogging();
    configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'trace', namespaces: ['workflows'] });
  });

  it('deep clones runtime state snapshots', () => {
    const values = { nested: { mutable: 'before' }, tags: ['before'] };
    const snapshot = cloneWidgetRuntimeState(values);

    values.nested.mutable = 'after';
    values.tags[0] = 'after';

    expect(snapshot).toEqual({ nested: { mutable: 'before' }, tags: ['before'] });
  });

  it('returns an empty runtime state snapshot when the scoped project is missing', () => {
    expect(getProjectWidgetRuntimeState({ projects: [], scopedDefaults: {} }, 'missing-project', 'generate')).toEqual(
      {}
    );
  });

  it('exposes widget state reads and writes through the state namespace', () => {
    const { actions, widgets } = createDispatch();
    const state = createStateApi();
    const runtime = createWidgetRuntime({
      extensions: createExtensionRegistry(),
      widgets,
      open: vi.fn(),
      instance: createInstance(),
      project: createPlacementProject(),
      region: 'right',
      state,
    });

    runtime.state.patch({ next: true });
    runtime.state.set({ replaced: true });

    expect(actions).toEqual([]);
    expect(state.getSnapshot()).toEqual({ current: true });
    expect(state.patch).toHaveBeenCalledWith({ next: true });
    expect(state.set).toHaveBeenCalledWith({ replaced: true });
  });

  it('binds diagnostics to the current widget source', () => {
    const { widgets } = createDispatch();
    const runtime = createWidgetRuntime({
      extensions: createExtensionRegistry(),
      widgets,
      open: vi.fn(),
      instance: createInstance('workflow:center', 'workflow'),
      project: createPlacementProject(),
      region: 'center',
      state: createStateApi(),
    });

    runtime.diagnostics.logger('workflows').info('Widget rendered');

    expect(getLogSnapshot().entries).toMatchObject([
      {
        level: 'info',
        message: 'Widget rendered',
        namespace: 'workflows',
        source: {
          area: 'workflow',
          namespace: 'workflows',
          projectId: 'project-1',
          widget: { instanceId: 'workflow:center', region: 'center', typeId: 'workflow' },
        },
      },
    ]);
  });

  it('reveals an instance of its own region', () => {
    const { actions, widgets } = createDispatch();
    const runtime = createWidgetRuntime({
      extensions: createExtensionRegistry(),
      widgets,
      open: vi.fn(),
      instance: createInstance(),
      project: createPlacementProject({
        widgetRegions: {
          bottom: { activeInstanceId: 'alpha', instanceIds: ['alpha', 'beta'] },
          center: { activeInstanceId: 'alpha', instanceIds: [] },
          left: { activeInstanceId: 'alpha', instanceIds: [] },
          right: { activeInstanceId: 'alpha', instanceIds: [] },
        },
      }),
      region: 'bottom',
      state: createStateApi(),
    });

    expect(runtime.workbench.revealWidgetInstance('beta')).toEqual({ ok: true, region: 'bottom' });
    expect(actions).toEqual([
      { projectId: 'project-1', region: 'bottom', type: 'selectRegionWidget', widgetId: 'beta' },
    ]);
  });

  it('returns unsupported for a reveal in another region', () => {
    const { actions, widgets } = createDispatch();
    const runtime = createWidgetRuntime({
      extensions: createExtensionRegistry(),
      widgets,
      open: vi.fn(),
      instance: createInstance(),
      project: createPlacementProject(),
      region: 'right',
      state: createStateApi(),
    });

    expect(runtime.workbench.revealWidgetInstance('beta')).toEqual({ ok: false, reason: 'unsupported' });
    expect(actions).toEqual([]);
  });

  it('executes commands registered by the current widget source', async () => {
    const { widgets } = createDispatch();
    const extensions = createExtensionRegistry();
    const alphaRuntime = createWidgetRuntime({
      extensions,
      widgets,
      open: vi.fn(),
      instance: createInstance('alpha', 'test-widget'),
      project: createPlacementProject(),
      region: 'right',
      state: createStateApi(),
    });
    const betaRuntime = createWidgetRuntime({
      extensions,
      widgets,
      open: vi.fn(),
      instance: createInstance('beta', 'test-widget'),
      project: createPlacementProject(),
      region: 'right',
      state: createStateApi(),
    });
    const disposeAlpha = alphaRuntime.commands.register({
      handler: () => 'alpha',
      id: 'test-widget.shared-command',
      title: 'Shared command',
    });
    const disposeBeta = betaRuntime.commands.register({
      handler: () => 'beta',
      id: 'test-widget.shared-command',
      title: 'Shared command',
    });

    await expect(alphaRuntime.commands.execute('test-widget.shared-command')).resolves.toBe('alpha');
    await expect(betaRuntime.commands.execute('test-widget.shared-command')).resolves.toBe('beta');

    disposeAlpha();
    disposeBeta();
  });

  it('does not execute commands registered by the same widget instance in another project', async () => {
    const { widgets } = createDispatch();
    const extensions = createExtensionRegistry();
    const projectOneRuntime = createWidgetRuntime({
      extensions,
      widgets,
      open: vi.fn(),
      instance: createInstance('alpha', 'test-widget'),
      project: createPlacementProject({}),
      region: 'right',
      state: createStateApi(),
    });
    const projectTwoRuntime = createWidgetRuntime({
      extensions,
      widgets,
      open: vi.fn(),
      instance: createInstance('alpha', 'test-widget'),
      project: { ...createPlacementProject({}), projectId: 'project-2' },
      region: 'right',
      state: createStateApi(),
    });
    const disposeOne = projectOneRuntime.commands.register({
      handler: () => 'project-1',
      id: 'test-widget.project-command',
      title: 'Project command',
    });
    const disposeTwo = projectTwoRuntime.commands.register({
      handler: () => 'project-2',
      id: 'test-widget.project-command',
      title: 'Project command',
    });

    await expect(projectOneRuntime.commands.execute('test-widget.project-command')).resolves.toBe('project-1');
    await expect(projectTwoRuntime.commands.execute('test-widget.project-command')).resolves.toBe('project-2');

    disposeOne();
    disposeTwo();
  });

  it('scopes palette and search registrations to the widget source', () => {
    const { widgets } = createDispatch();
    const extensions = createExtensionRegistry();
    const runtime = createWidgetRuntime({
      extensions,
      widgets,
      open: vi.fn(),
      instance: createInstance('alpha', 'test-widget'),
      project: createPlacementProject(),
      region: 'right',
      state: createStateApi(),
    });

    runtime.palette.register({ commandId: 'test-widget.palette-command', title: 'Palette entry' });
    runtime.search.registerProvider({ id: 'test-widget.search', label: 'Search', search: () => [] });

    const expectedSource = expect.objectContaining({ instanceId: 'alpha', typeId: 'test-widget' });
    expect(extensions.stores.palette.list()).toEqual([expect.objectContaining({ source: expectedSource })]);
    expect(extensions.stores.search.list()).toEqual([expect.objectContaining({ source: expectedSource })]);
  });
});
