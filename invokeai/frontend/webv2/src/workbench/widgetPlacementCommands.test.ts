import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WorkbenchState } from '@workbench/projectContracts';
import type { NormalizedWidgetManifest, RegisteredWidget } from '@workbench/widgetContracts';
import type { ResolvedLayout } from '@workbench/workspace/resolveLayout';

import { describe, expect, it, vi } from 'vitest';

import type { WorkbenchAction } from './workbenchState.testing';
import type { WorkbenchWidgetCommands } from './workbenchStore';

import { createWidgetImplementationResource } from './widgetImplementationResource';
import {
  activateRailPlacement,
  addWidgetPlacement,
  cycleRegionWidget,
  dockFloatingPlacement,
  revealWidgetPlacement,
} from './widgetPlacementCommands';
import { getWidgetPlacementProject } from './widgetPlacementMeta';
import { createInitialWorkbenchState, getActiveProject, getLayoutOf, workbenchReducer } from './workbenchState.testing';

const TestIcon = () => null;
const TestView = () => null;

/** The placement projection the shell hands the commands, from the reducer's state. */
const placementOf = (state: WorkbenchState) =>
  getWidgetPlacementProject({ activeLayout: getLayoutOf(state), activeProject: getActiveProject(state) });

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
  closeGuest: (instanceId) => dispatch({ instanceId, type: 'closeGuest' }),
  keepGuest: (instanceId) => dispatch({ instanceId, type: 'keepGuest' }),
  showGuest: (options) => dispatch({ ...options, type: 'showGuest' }),
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
  it('cycles eligible docked widgets in both directions, wrapping and skipping what cannot show', () => {
    const select = vi.fn();
    const widgets = { select } as unknown as WorkbenchWidgetCommands;
    const left = (activeInstanceId: string) => ({
      activeInstanceId,
      instanceIds: ['generate', 'missing', 'upscale', 'queue'],
    });
    const placement = (activeInstanceId: string) => ({
      ...placementOf(createInitialWorkbenchState()),
      widgetInstances: {
        generate: { id: 'generate', typeId: 'generate' },
        queue: { id: 'queue', typeId: 'queue' },
        upscale: { id: 'upscale', typeId: 'upscale' },
      },
      widgetRegions: { ...placementOf(createInitialWorkbenchState()).widgetRegions, left: left(activeInstanceId) },
    });
    const registry = createRegistry({
      left: [
        createWidget({ id: 'generate', label: 'Generate' }),
        { ...createWidget({ id: 'upscale', label: 'Upscale' }), status: 'disabled' },
        createWidget({ id: 'queue', label: 'Queue' }),
      ],
    });
    const cycle = (activeInstanceId: string, direction: -1 | 1) =>
      cycleRegionWidget({
        direction,
        getWidgetsForRegion: registry,
        project: placement(activeInstanceId),
        region: 'left',
        widgets,
      });

    expect(cycle('generate', 1)).toBe('queue');
    expect(cycle('queue', 1)).toBe('generate');
    expect(cycle('generate', -1)).toBe('queue');
    expect(select.mock.calls.map(([options]) => (options as { region: string }).region)).toEqual([
      'left',
      'left',
      'left',
    ]);
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
    const rightRegion = getLayoutOf(nextState).widgetRegions.right;

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

  it('reveals a placement and uncollapses its region', () => {
    const collapsedState = workbenchReducer(createInitialWorkbenchState(), {
      isCollapsed: true,
      region: 'right',
      type: 'setRegionWidgetCollapsed',
    });
    const nextState = applyCommand(collapsedState, (widgets) => {
      revealWidgetPlacement({ widgets, instanceId: 'queue', project: placementOf(collapsedState), region: 'right' });
    });

    expect(getLayoutOf(nextState).widgetRegions.right).toMatchObject({ activeInstanceId: 'queue', isCollapsed: false });
  });
});

describe('floating window rail commands', () => {
  /** Run a rail command against the real reducer, with the placement projection the shell hands it. */
  const run = <Result>(
    state: WorkbenchState,
    command: (widgets: WorkbenchWidgetCommands, project: ReturnType<typeof getWidgetPlacementProject>) => Result
  ): { project: ResolvedLayout; result: Result } => {
    let result!: Result;
    const next = applyCommand(state, (widgets) => {
      result = command(widgets, placementOf(state));
    });

    return { project: getLayoutOf(next), result };
  };
  const float = (state: WorkbenchState, instanceId: string, region?: WidgetRegion): WorkbenchState =>
    workbenchReducer(state, region ? { instanceId, region, type: 'floatWidget' } : { instanceId, type: 'floatWidget' });

  it('activates a marker by raising and expanding its window, not by docking it', () => {
    let state = float(float(createInitialWorkbenchState(), 'image-map'), 'gallery');
    state = workbenchReducer(state, { instanceId: 'image-map', mode: 'shaded', type: 'setFloatingWidgetMode' });
    const railBefore = getLayoutOf(state).widgetRegions.right;

    const { project, result } = run(state, (widgets, placement) =>
      activateRailPlacement({ instanceId: 'image-map', project: placement, region: 'right', widgets })
    );

    expect(result).toBe('window');
    expect(project.floatingWidgets?.['image-map']).toMatchObject({ mode: 'windowed', stackOrder: 2 });
    expect(project.widgetRegions.right).toEqual(railBefore);
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
