import type {
  Project,
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetInstanceContract,
  WidgetRegion,
  WidgetTypeId,
} from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { describe, expect, it, vi } from 'vitest';

import { createWidgetRuntime } from './createWidgetRuntime';

const TestIcon = () => null;
const TestView = () => null;

const createWidget = (
  overrides: Partial<NormalizedWidgetManifest> & Pick<NormalizedWidgetManifest, 'id' | 'labelText'>
): RegisteredWidget => ({
  manifest: {
    apiVersion: 1,
    allowMultiple: false,
    allowedRegions: ['left', 'right', 'center', 'bottom'],
    bottomPanel: 'expandable',
    centerPlacement: 'view',
    failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    icon: TestIcon,
    label: overrides.labelText,
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

const createRegistry =
  (widgetsByRegion: Partial<Record<WidgetRegion, RegisteredWidget[]>>) =>
  (region: WidgetRegion): RegisteredWidget[] =>
    widgetsByRegion[region] ?? [];

const createPlacementProject = (
  overrides: Partial<Pick<Project, 'widgetInstances' | 'widgetRegions'>> = {}
): Pick<Project, 'widgetInstances' | 'widgetRegions'> => ({
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
  it('dispatches patchState and setState to the current widget instance', () => {
    const { actions, dispatch } = createDispatch();
    const runtime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({}),
      instance: createInstance(),
      project: createPlacementProject(),
      region: 'right',
    });

    runtime.patchState({ next: true });
    runtime.setState({ replaced: true });

    expect(actions).toEqual([
      { instanceId: 'alpha', type: 'patchWidgetInstanceValues', values: { next: true } },
      { instanceId: 'alpha', type: 'setWidgetInstanceValues', values: { replaced: true } },
    ]);
  });

  it('opens widgets through canonical validation and seeded initial values', () => {
    const { actions, dispatch } = createDispatch();
    const createInitial = vi.fn(() => ({ seeded: true }));
    const runtime = createWidgetRuntime({
      dispatch,
      getWidgetById: () => undefined,
      getWidgetsForRegion: createRegistry({
        bottom: [createWidget({ bottomPanel: 'tooltip', id: 'tooltip-widget', labelText: 'Tooltip Widget' })],
        center: [createWidget({ centerPlacement: 'toolbar', id: 'toolbar-widget', labelText: 'Toolbar Widget' })],
        right: [
          createWidget({
            id: 'panel-widget',
            labelText: 'Panel Widget',
            state: { createInitial, persistence: 'project', version: 1 },
          }),
        ],
      }),
      instance: createInstance(),
      project: createPlacementProject(),
      region: 'right',
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
      getWidgetById: (typeId) => createWidget({ id: typeId, labelText: typeId }),
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
    });

    expect(runtime.workbench.revealWidgetInstance('beta')).toEqual({ ok: true, region: 'bottom' });
    expect(runtime.workbench.closeWidgetInstance('beta')).toEqual({ ok: true, region: 'bottom' });
    expect(actions).toEqual([
      { region: 'bottom', type: 'selectRegionWidget', widgetId: 'beta' },
      { region: 'bottom', type: 'toggleRegionWidget', widgetId: 'beta' },
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
    });

    expect(runtime.workbench.revealWidgetInstance('beta')).toEqual({ ok: false, reason: 'unsupported' });
    expect(runtime.workbench.closeWidgetInstance('beta')).toEqual({ ok: false, reason: 'unsupported' });
    expect(actions).toEqual([]);
  });
});
