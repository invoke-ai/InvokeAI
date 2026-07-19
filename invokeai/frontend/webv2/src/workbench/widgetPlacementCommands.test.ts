import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { NormalizedWidgetManifest, RegisteredWidget } from '@workbench/widgetContracts';

import { describe, expect, it } from 'vitest';

import type { WorkbenchAction } from './workbenchState.testing';
import type { WorkbenchWidgetCommands } from './workbenchStore';

import { createWidgetImplementationResource } from './widgetImplementationResource';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from './widgetPlacementCommands';
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
  move: (options) => dispatch({ ...options, type: 'moveWidgetInstance' }),
  open: (options) => dispatch({ ...options, type: 'openRegionWidget' }),
  patchInstanceValues: (instanceId, values, projectId) =>
    dispatch({ instanceId, projectId, type: 'patchWidgetInstanceValues', values }),
  patchValues: (widgetId, values, projectId) => dispatch({ projectId, type: 'patchWidgetValues', values, widgetId }),
  reorder: (options) => dispatch({ ...options, type: 'reorderWidgetInstances' }),
  select: (options) => dispatch({ ...options, type: 'selectRegionWidget' }),
  setInstanceValues: (instanceId, values, projectId) =>
    dispatch({ instanceId, projectId, type: 'setWidgetInstanceValues', values }),
  toggle: (options) => dispatch({ ...options, type: 'toggleRegionWidget' }),
});

describe('widget placement commands', () => {
  it('opens a singleton already in the target region without duplicating it', () => {
    const state = createInitialWorkbenchState();
    const widget = createWidget({ id: 'queue', label: 'Queue' });
    const nextState = applyCommand(state, (widgets) => {
      expect(
        openWidgetPlacement({
          widgets,
          getWidgetsForRegion: createRegistry({ right: [widget] }),
          options: { preferredRegions: ['right'] },
          typeId: 'queue',
        })
      ).toEqual({ ok: true, region: 'right' });
    });
    const rightRegion = getActiveProject(nextState).widgetRegions.right;

    expect(rightRegion.instanceIds.filter((instanceId) => instanceId === 'queue')).toHaveLength(1);
    expect(rightRegion.activeInstanceId).toBe('queue');
    expect(widget.implementation.getStatus()).toBe('loading');
  });

  it('opens an existing singleton instance from another region', () => {
    const state = createInitialWorkbenchState();
    const widget = createWidget({ id: 'preview', label: 'Preview' });
    const nextState = applyCommand(state, (widgets) => {
      openWidgetPlacement({
        widgets,
        getWidgetsForRegion: createRegistry({ right: [widget] }),
        options: { preferredRegions: ['right'] },
        typeId: 'preview',
      });
    });
    const rightRegion = getActiveProject(nextState).widgetRegions.right;

    expect(rightRegion.instanceIds).toContain('preview');
    expect(rightRegion.activeInstanceId).toBe('preview');
  });

  it('passes allowMultiple through as createNew', () => {
    const actions: WorkbenchAction[] = [];
    const widgets = createWidgetCommands((action) => actions.push(action));
    const widget = createWidget({ allowMultiple: true, id: 'queue', label: 'Queue' });

    openWidgetPlacement({
      widgets,
      getWidgetsForRegion: createRegistry({ right: [widget] }),
      options: { createNew: true, preferredRegions: ['right'] },
      typeId: 'queue',
    });

    expect(actions).toMatchObject([{ createNew: true, region: 'right', type: 'openRegionWidget', widgetId: 'queue' }]);
  });

  it('does not create a new instance for singleton widgets even when requested', () => {
    const actions: WorkbenchAction[] = [];
    const widgets = createWidgetCommands((action) => actions.push(action));
    const widget = createWidget({ allowMultiple: false, id: 'queue', label: 'Queue' });

    openWidgetPlacement({
      widgets,
      getWidgetsForRegion: createRegistry({ right: [widget] }),
      options: { createNew: true, preferredRegions: ['right'] },
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
      openWidgetPlacement({
        widgets,
        getWidgetsForRegion: createRegistry({
          left: [createWidget({ allowedRegions: ['center'], id: 'models', label: 'Models' })],
        }),
        options: { preferredRegions: ['left'] },
        typeId: 'models',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(
      openWidgetPlacement({
        widgets,
        getWidgetsForRegion: createRegistry({
          bottom: [createWidget({ bottomPanel: 'tooltip', id: 'diagnostics', label: 'Diagnostics' })],
        }),
        options: { preferredRegions: ['bottom'] },
        typeId: 'diagnostics',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(
      openWidgetPlacement({
        widgets,
        getWidgetsForRegion: createRegistry({
          center: [createWidget({ centerPlacement: 'toolbar', id: 'toolbar-tools', label: 'Toolbar Tools' })],
        }),
        options: { preferredRegions: ['center'] },
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

  it('closes a placement and preserves reducer fallback active item behavior', () => {
    const state = createInitialWorkbenchState();
    const nextState = applyCommand(state, (widgets) => {
      closeWidgetPlacement({
        widgets,
        getWidgetById: (typeId) => createWidget({ id: typeId, label: typeId }),
        instanceId: 'layers',
        project: getActiveProject(state),
        region: 'right',
      });
    });
    const rightRegion = getActiveProject(nextState).widgetRegions.right;

    expect(rightRegion.instanceIds).not.toContain('layers');
    expect(rightRegion.activeInstanceId).toBe('gallery');
  });
});
