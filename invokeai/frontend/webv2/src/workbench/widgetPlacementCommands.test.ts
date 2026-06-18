import { describe, expect, it } from 'vitest';

import type { NormalizedWidgetManifest, Project, RegisteredWidget, WidgetRegion, WorkbenchState } from './types';
import type { WorkbenchAction } from './workbenchState';

import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from './widgetPlacementCommands';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState';

const TestIcon = () => null;
const TestView = () => null;

const getActiveProject = (state: WorkbenchState): Project => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project as Project;
};

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

const createRegistry =
  (widgetsByRegion: Partial<Record<WidgetRegion, RegisteredWidget[]>>) =>
  (region: WidgetRegion): RegisteredWidget[] =>
    widgetsByRegion[region] ?? [];

const applyCommand = (
  state: WorkbenchState,
  command: (dispatch: (action: WorkbenchAction) => void) => void
): WorkbenchState => {
  let nextState = state;
  const dispatch = (action: WorkbenchAction): void => {
    nextState = workbenchReducer(nextState, action);
  };

  command(dispatch);

  return nextState;
};

describe('widget placement commands', () => {
  it('opens a singleton already in the target region without duplicating it', () => {
    const state = createInitialWorkbenchState();
    const widget = createWidget({ id: 'queue', labelText: 'Queue' });
    const nextState = applyCommand(state, (dispatch) => {
      expect(
        openWidgetPlacement({
          dispatch,
          getWidgetsForRegion: createRegistry({ right: [widget] }),
          options: { preferredRegions: ['right'] },
          typeId: 'queue',
        })
      ).toEqual({ ok: true, region: 'right' });
    });
    const rightRegion = getActiveProject(nextState).widgetRegions.right;

    expect(rightRegion.instanceIds.filter((instanceId) => instanceId === 'queue')).toHaveLength(1);
    expect(rightRegion.activeInstanceId).toBe('queue');
  });

  it('opens an existing singleton instance from another region', () => {
    const state = createInitialWorkbenchState();
    const widget = createWidget({ id: 'preview', labelText: 'Preview' });
    const nextState = applyCommand(state, (dispatch) => {
      openWidgetPlacement({
        dispatch,
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
    const widget = createWidget({ allowMultiple: true, id: 'queue', labelText: 'Queue' });

    openWidgetPlacement({
      dispatch: (action) => actions.push(action),
      getWidgetsForRegion: createRegistry({ right: [widget] }),
      options: { createNew: true, preferredRegions: ['right'] },
      typeId: 'queue',
    });

    expect(actions).toMatchObject([{ createNew: true, region: 'right', type: 'openRegionWidget', widgetId: 'queue' }]);
  });

  it('rejects invalid regions and unrenderable placements', () => {
    const actions: WorkbenchAction[] = [];
    const dispatch = (action: WorkbenchAction): void => {
      actions.push(action);
    };

    expect(
      openWidgetPlacement({
        dispatch,
        getWidgetsForRegion: createRegistry({
          left: [createWidget({ allowedRegions: ['center'], id: 'models', labelText: 'Models' })],
        }),
        options: { preferredRegions: ['left'] },
        typeId: 'models',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(
      openWidgetPlacement({
        dispatch,
        getWidgetsForRegion: createRegistry({
          bottom: [createWidget({ bottomPanel: 'tooltip', id: 'diagnostics', labelText: 'Diagnostics' })],
        }),
        options: { preferredRegions: ['bottom'] },
        typeId: 'diagnostics',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(
      openWidgetPlacement({
        dispatch,
        getWidgetsForRegion: createRegistry({
          center: [createWidget({ centerPlacement: 'toolbar', id: 'layout-actions', labelText: 'Layout Actions' })],
        }),
        options: { preferredRegions: ['center'] },
        typeId: 'layout-actions',
      })
    ).toEqual({ ok: false, reason: 'unavailable' });
    expect(actions).toEqual([]);
  });

  it('prevents removing the last center view', () => {
    const actions: WorkbenchAction[] = [];
    const project = getActiveProject(createInitialWorkbenchState());

    expect(
      closeWidgetPlacement({
        dispatch: (action) => actions.push(action),
        getWidgetById: (typeId) => createWidget({ id: typeId, labelText: typeId }),
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
    const nextState = applyCommand(collapsedState, (dispatch) => {
      revealWidgetPlacement({
        dispatch,
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
    const nextState = applyCommand(state, (dispatch) => {
      closeWidgetPlacement({
        dispatch,
        getWidgetById: (typeId) => createWidget({ id: typeId, labelText: typeId }),
        instanceId: 'layers',
        project: getActiveProject(state),
        region: 'right',
      });
    });
    const rightRegion = getActiveProject(nextState).widgetRegions.right;

    expect(rightRegion.instanceIds).not.toContain('layers');
    expect(rightRegion.activeInstanceId).toBe('queue');
  });
});
