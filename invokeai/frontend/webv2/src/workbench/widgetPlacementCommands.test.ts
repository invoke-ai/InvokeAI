import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { NormalizedWidgetManifest, RegisteredWidget } from '@workbench/widgetContracts';

import { describe, expect, it, vi } from 'vitest';

import type { WorkbenchAction } from './workbenchState.testing';
import type { WorkbenchWidgetCommands } from './workbenchStore';

import { createWidgetImplementationResource } from './widgetImplementationResource';
import {
  activateRailPlacement,
  addWidgetPlacement,
  closeWidgetPlacement,
  cycleRegionWidget,
  dockFloatingPlacement,
  revealWidgetPlacement,
} from './widgetPlacementCommands';
import { getWidgetPlacementProject } from './widgetPlacementMeta';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const TestIcon = () => null;
const TestView = () => null;

const getActiveProject = (state: WorkbenchState): Project => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project as Project;
};

const createWidget = (
  overrides: Partial<NormalizedWidgetManifest> & Pick<NormalizedWidgetManifest, 'id' | 'label'>
): RegisteredWidget => ({
  implementation: createWidgetImplementationResource(overrides.id, () => Promise.resolve({ view: TestView })),
  manifest: {
    apiVersion: 1,
    allowMultiple: false,
    allowedRegions: ['left', 'right', 'center', 'bottom'],
    bottomPanel: 'expandable',
    centerPlacement: 'view',
    failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    icon: TestIcon,
    load: () => Promise.resolve({ view: TestView }),
    state: { createInitial: () => ({}), persistence: 'project', version: 1 },
    version: 1,
    ...overrides,
  },
  status: 'enabled',
});

const createRegistry =
  (widgetsByRegion: Partial<Record<WidgetRegion, RegisteredWidget[]>>) =>
  (region: WidgetRegion): RegisteredWidget[] =>
    widgetsByRegion[region] ?? [];

const applyCommand = (state: WorkbenchState, command: (widgets: WorkbenchWidgetCommands) => void): WorkbenchState => {
  let nextState = state;
  const dispatch = (action: WorkbenchAction): void => {
    nextState = workbenchReducer(nextState, action);
  };

  command(createWidgetCommands(dispatch));

  return nextState;
};

const createWidgetCommands = (dispatch: (action: WorkbenchAction) => void): WorkbenchWidgetCommands => ({
  clearInstanceValues: (instanceId, keys, projectId) =>
    dispatch({ instanceId, keys, projectId, type: 'clearWidgetInstanceValues' }),
  dockFloating: (instanceId) => dispatch({ instanceId, type: 'dockFloatingWidget' }),
  closeFloating: (instanceId) => dispatch({ instanceId, type: 'closeFloatingWidget' }),
  float: (instanceId) => dispatch({ instanceId, type: 'floatWidget' }),
  front: (options) => dispatch({ ...options, type: 'frontRegionWidget' }),
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
});

describe('widget placement commands', () => {
  it('cycles eligible docked widgets in both directions, wrapping without moving or selecting other regions', () => {
    const initial = createInitialWorkbenchState();
    const project = getActiveProject(initial);
    project.widgetRegions.left = {
      ...project.widgetRegions.left,
      activeInstanceId: 'generate',
      instanceIds: ['generate', 'missing', 'upscale', 'queue'],
      isCollapsed: false,
    };
    const registry = createRegistry({
      left: [
        createWidget({ id: 'generate', label: 'Generate' }),
        { ...createWidget({ id: 'upscale', label: 'Upscale' }), status: 'disabled' },
        createWidget({ id: 'queue', label: 'Queue' }),
      ],
    });
    const cycle = (state: WorkbenchState, direction: -1 | 1) =>
      applyCommand(state, (widgets) => {
        cycleRegionWidget({
          direction,
          getWidgetsForRegion: registry,
          project: getWidgetPlacementProject(getActiveProject(state)),
          region: 'left',
          widgets,
        });
      });
    const next = cycle(initial, 1);
    expect(getActiveProject(next).widgetRegions.left.activeInstanceId).toBe('queue');
    expect(getActiveProject(next).widgetRegions.left.instanceIds).toEqual(['generate', 'missing', 'upscale', 'queue']);
    expect(getActiveProject(next).widgetRegions.right).toBe(project.widgetRegions.right);
    const wrapped = cycle(next, 1);
    expect(getActiveProject(wrapped).widgetRegions.left.activeInstanceId).toBe('generate');
    expect(getActiveProject(cycle(wrapped, -1)).widgetRegions.left.activeInstanceId).toBe('queue');
  });

  it('opens a singleton already in the target region without duplicating it', () => {
    const state = createInitialWorkbenchState();
    const widget = createWidget({ id: 'queue', label: 'Queue' });
    const nextState = applyCommand(state, (widgets) => {
      expect(
        addWidgetPlacement({
          widgets,
          getWidgetsForRegion: createRegistry({ right: [widget] }),
          regions: ['right'],
          typeId: 'queue',
        })
      ).toEqual({ ok: true, region: 'right' });
    });
    const rightRegion = getActiveProject(nextState).widgetRegions.right;

    expect(rightRegion.instanceIds.filter((instanceId) => instanceId === 'queue')).toHaveLength(1);
    expect(rightRegion.activeInstanceId).toBe('queue');
    expect(widget.implementation.getStatus()).toBe('loading');
  });

  it('does not create a new instance for singleton widgets even when requested', () => {
    const actions: WorkbenchAction[] = [];
    const widgets = createWidgetCommands((action) => actions.push(action));
    const widget = createWidget({ allowMultiple: false, id: 'queue', label: 'Queue' });

    addWidgetPlacement({
      createNew: true,
      widgets,
      getWidgetsForRegion: createRegistry({ right: [widget] }),
      regions: ['right'],
      typeId: 'queue',
    });

    expect(actions).toMatchObject([
      { createNew: undefined, region: 'right', type: 'openRegionWidget', widgetId: 'queue' },
    ]);
  });

  it('rejects invalid regions and unrenderable placements', () => {
    const actions: WorkbenchAction[] = [];
    const dispatch = (action: WorkbenchAction): void => {
      actions.push(action);
    };
    const widgets = createWidgetCommands(dispatch);

    expect(
      addWidgetPlacement({
        widgets,
        getWidgetsForRegion: createRegistry({
          left: [createWidget({ allowedRegions: ['center'], id: 'models', label: 'Models' })],
        }),
        regions: ['left'],
        typeId: 'models',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(
      addWidgetPlacement({
        widgets,
        getWidgetsForRegion: createRegistry({
          bottom: [createWidget({ bottomPanel: 'tooltip', id: 'server-status', label: 'Server Status' })],
        }),
        regions: ['bottom'],
        typeId: 'server-status',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(
      addWidgetPlacement({
        widgets,
        getWidgetsForRegion: createRegistry({
          center: [createWidget({ centerPlacement: 'toolbar', id: 'toolbar-tools', label: 'Toolbar Tools' })],
        }),
        regions: ['center'],
        typeId: 'toolbar-tools',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(actions).toEqual([]);
  });

  it('prevents removing the last center view', () => {
    const actions: WorkbenchAction[] = [];
    const widgets = createWidgetCommands((action) => actions.push(action));
    const project = getActiveProject(createInitialWorkbenchState());

    expect(
      closeWidgetPlacement({
        widgets,
        getWidgetById: (typeId) => createWidget({ id: typeId, label: typeId }),
        instanceId: 'canvas',
        project: {
          ...project,
          widgetRegions: {
            ...project.widgetRegions,
            center: { ...project.widgetRegions.center, activeInstanceId: 'canvas', instanceIds: ['canvas'] },
          },
        },
        region: 'center',
      })
    ).toEqual({ ok: false, reason: 'unsupported' });
    expect(actions).toEqual([]);
  });

  it('reveals a placement and uncollapses its region', () => {
    const state = createInitialWorkbenchState();
    const project = getActiveProject(state);
    const collapsedState: WorkbenchState = {
      ...state,
      projects: state.projects.map((candidate) =>
        candidate.id === project.id
          ? {
              ...candidate,
              widgetRegions: {
                ...candidate.widgetRegions,
                right: { ...candidate.widgetRegions.right, activeInstanceId: 'layers', isCollapsed: true },
              },
            }
          : candidate
      ),
    };
    const nextState = applyCommand(collapsedState, (widgets) => {
      revealWidgetPlacement({
        widgets,
        instanceId: 'queue',
        project: getActiveProject(collapsedState),
        region: 'right',
      });
    });

    expect(getActiveProject(nextState).widgetRegions.right).toMatchObject({
      activeInstanceId: 'queue',
      isCollapsed: false,
    });
  });
});

describe('floating window rail commands', () => {
  /** Run a rail command against the real reducer, with the placement projection the shell hands it. */
  const run = <Result>(
    state: WorkbenchState,
    command: (widgets: WorkbenchWidgetCommands, project: ReturnType<typeof getWidgetPlacementProject>) => Result
  ): { project: Project; result: Result } => {
    let result!: Result;
    const next = applyCommand(state, (widgets) => {
      result = command(widgets, getWidgetPlacementProject(getActiveProject(state)));
    });

    return { project: getActiveProject(next), result };
  };
  const float = (state: WorkbenchState, instanceId: string, region?: WidgetRegion): WorkbenchState =>
    workbenchReducer(state, region ? { instanceId, region, type: 'floatWidget' } : { instanceId, type: 'floatWidget' });

  it('activates a marker by raising and expanding its window, not by docking it', () => {
    let state = float(float(createInitialWorkbenchState(), 'image-map'), 'gallery');
    state = workbenchReducer(state, { instanceId: 'image-map', mode: 'shaded', type: 'setFloatingWidgetMode' });
    const railBefore = getActiveProject(state).widgetRegions.right;

    const { project, result } = run(state, (widgets, placement) =>
      activateRailPlacement({ instanceId: 'image-map', project: placement, region: 'right', widgets })
    );

    expect(result).toBe('window');
    expect(project.floatingWidgets?.['image-map']).toMatchObject({ mode: 'windowed', stackOrder: 2 });
    expect(project.widgetRegions.right).toBe(railBefore);
  });

  it('docks a window where it came from and names that region', () => {
    const state = float(createInitialWorkbenchState(), 'image-map');
    const docked = run(state, (widgets, placement) =>
      dockFloatingPlacement({ instanceId: 'image-map', project: placement, widgets })
    );
    const stale = run(createInitialWorkbenchState(), (widgets, placement) =>
      dockFloatingPlacement({ instanceId: 'image-map', project: placement, widgets })
    );

    expect(docked.result).toBe('right');
    expect(docked.project.floatingWidgets).toBeUndefined();
    expect(docked.project.widgetRegions.right.instanceIds).toEqual(['gallery', 'image-map', 'queue']);
    // Nothing floats there: no dispatch, no region to move focus to.
    expect(stale.result).toBeNull();
  });
});
