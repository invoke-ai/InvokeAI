import type {
  Project,
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetInstanceContract,
  WidgetRegion,
  WidgetTypeId,
} from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { clearProjectDiagnostics, configureDiagnostics, getProjectDiagnostics } from '@workbench/diagnostics/logger';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { cloneWidgetRuntimeState, createWidgetRuntime, getProjectWidgetRuntimeState } from './createWidgetRuntime';

const TestIcon = () => null;
const TestView = () => null;

const createWidget = (
  overrides: Partial<NormalizedWidgetManifest> & Pick<NormalizedWidgetManifest, 'id' | 'label'>
): RegisteredWidget => ({
  manifest: {
    apiVersion: 1,
    allowMultiple: false,
    allowedRegions: ['left', 'right', 'center', 'bottom'],
    bottomPanel: 'expandable',
    centerPlacement: 'view',
    failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    icon: TestIcon,
    state: { createInitial: () => ({}), persistence: 'project', version: 1 },
    version: 1,
    view: TestView,
    ...overrides,
  },
  status: 'enabled',
});

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

  return { actions, dispatch };
};

const createStateApi = () => ({
  getSnapshot: () => ({ current: true }),
  patch: vi.fn(),
  set: vi.fn(),
  useSelector: vi.fn(),
});

const createRegistry =
  (widgetsByRegion: Partial<Record<WidgetRegion, RegisteredWidget[]>>) =>
  (region: WidgetRegion): RegisteredWidget[] =>
    widgetsByRegion[region] ?? [];

const createPlacementProject = (
  overrides: Partial<Pick<Project, 'widgetInstances' | 'widgetRegions'>> = {}
): Pick<Project, 'widgetInstances' | 'widgetRegions'> & { projectId: string } => ({
  projectId: 'project-1',
  widgetInstances: {
    alpha: createInstance('alpha'),
    beta: createInstance('beta', 'beta'),
    ...overrides.widgetInstances,
  },
  widgetRegions: {
    bottom: { activeInstanceId: 'alpha', instanceIds: [], isCollapsed: false, sizePx: 240 },
    center: { activeInstanceId: 'alpha', instanceIds: [], isCollapsed: false, sizePx: 240 },
    left: { activeInstanceId: 'alpha', instanceIds: [], isCollapsed: false, sizePx: 240 },
    right: { activeInstanceId: 'alpha', instanceIds: ['alpha'], isCollapsed: false, sizePx: 240 },
    ...overrides.widgetRegions,
  },
});

describe('createWidgetRuntime', () => {
  beforeEach(() => {
    clearProjectDiagnostics('project-1');
    configureDiagnostics({
      enabled: true,
      level: 'trace',
      namespaces: ['workflows'],
      performanceTimingsEnabled: false,
    });
  });

  it('deep clones runtime state snapshots', () => {
    const values = { nested: { mutable: 'before' }, tags: ['before'] };
    const snapshot = cloneWidgetRuntimeState(values);

    values.nested.mutable = 'after';
    values.tags[0] = 'after';

    expect(snapshot).toEqual({ nested: { mutable: 'before' }, tags: ['before'] });
  });

  it('returns an empty runtime state snapshot when the scoped project is missing', () => {
    expect(getProjectWidgetRuntimeState([], 'missing-project', 'generate')).toEqual({});
  });

  it('exposes widget state reads and writes through the state namespace', () => {
    const { actions, dispatch } = createDispatch();
    const state = createStateApi();
    const runtime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
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
    const { dispatch } = createDispatch();
    const runtime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
      instance: createInstance('workflow:center', 'workflow'),
      project: createPlacementProject(),
      region: 'center',
      state: createStateApi(),
    });

    runtime.diagnostics.logger('workflows').info('Widget rendered');

    expect(getProjectDiagnostics('project-1')).toMatchObject([
      {
        level: 'info',
        message: 'Widget rendered',
        namespace: 'workflows',
        source: {
          instanceId: 'workflow:center',
          kind: 'widget',
          projectId: 'project-1',
          region: 'center',
          typeId: 'workflow',
        },
      },
    ]);
  });

  it('opens widgets through canonical validation and seeded initial values', () => {
    const { actions, dispatch } = createDispatch();
    const createInitial = vi.fn(() => ({ seeded: true }));
    const runtime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({
        bottom: [createWidget({ bottomPanel: 'tooltip', id: 'tooltip-widget', label: 'Tooltip Widget' })],
        center: [createWidget({ centerPlacement: 'toolbar', id: 'toolbar-widget', label: 'Toolbar Widget' })],
        right: [
          createWidget({
            id: 'panel-widget',
            label: 'Panel Widget',
            state: { createInitial, persistence: 'project', version: 1 },
          }),
        ],
      }),
      instance: createInstance(),
      project: createPlacementProject(),
      region: 'right',
      state: createStateApi(),
    });

    expect(runtime.workbench.openWidget('panel-widget', { preferredRegions: ['right'] })).toEqual({
      ok: true,
      region: 'right',
    });
    expect(runtime.workbench.openWidget('tooltip-widget', { preferredRegions: ['bottom'] })).toEqual({
      ok: false,
      reason: 'unavailable',
    });
    expect(runtime.workbench.openWidget('toolbar-widget', { preferredRegions: ['center'] })).toEqual({
      ok: false,
      reason: 'unavailable',
    });
    expect(runtime.workbench.openWidget('missing-widget', { preferredRegions: ['right'] })).toEqual({
      ok: false,
      reason: 'unavailable',
    });

    expect(createInitial).toHaveBeenCalledTimes(1);
    expect(actions).toEqual([
      {
        createNew: undefined,
        initialValues: { seeded: true },
        projectId: 'project-1',
        region: 'right',
        type: 'openRegionWidget',
        widgetId: 'panel-widget',
      },
    ]);
  });

  it('dispatches own-region close and reveal actions', () => {
    const { actions, dispatch } = createDispatch();
    const runtime = createWidgetRuntime({
      dispatch,
      getWidgetById: (typeId) => createWidget({ id: typeId, label: typeId }),
      getWidgetsForRegion: createRegistry({}),
      instance: createInstance(),
      project: createPlacementProject({
        widgetRegions: {
          bottom: { activeInstanceId: 'alpha', instanceIds: ['alpha', 'beta'], isCollapsed: false, sizePx: 240 },
          center: { activeInstanceId: 'alpha', instanceIds: [], isCollapsed: false, sizePx: 240 },
          left: { activeInstanceId: 'alpha', instanceIds: [], isCollapsed: false, sizePx: 240 },
          right: { activeInstanceId: 'alpha', instanceIds: [], isCollapsed: false, sizePx: 240 },
        },
      }),
      region: 'bottom',
      state: createStateApi(),
    });

    expect(runtime.workbench.revealWidgetInstance('beta')).toEqual({ ok: true, region: 'bottom' });
    expect(runtime.workbench.closeWidgetInstance('beta')).toEqual({ ok: true, region: 'bottom' });
    expect(actions).toEqual([
      { projectId: 'project-1', region: 'bottom', type: 'selectRegionWidget', widgetId: 'beta' },
      { projectId: 'project-1', region: 'bottom', type: 'toggleRegionWidget', widgetId: 'beta' },
    ]);
  });

  it('returns unsupported for cross-region close and reveal', () => {
    const { actions, dispatch } = createDispatch();
    const runtime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
      instance: createInstance(),
      project: createPlacementProject(),
      region: 'right',
      state: createStateApi(),
    });

    expect(runtime.workbench.revealWidgetInstance('beta')).toEqual({ ok: false, reason: 'unsupported' });
    expect(runtime.workbench.closeWidgetInstance('beta')).toEqual({ ok: false, reason: 'unsupported' });
    expect(actions).toEqual([]);
  });

  it('executes commands registered by the current widget source', async () => {
    const { dispatch } = createDispatch();
    const alphaRuntime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
      instance: createInstance('alpha', 'test-widget'),
      project: createPlacementProject(),
      region: 'right',
      state: createStateApi(),
    });
    const betaRuntime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
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
    const { dispatch } = createDispatch();
    const projectOneRuntime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
      instance: createInstance('alpha', 'test-widget'),
      project: createPlacementProject({}),
      region: 'right',
      state: createStateApi(),
    });
    const projectTwoRuntime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
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
});
