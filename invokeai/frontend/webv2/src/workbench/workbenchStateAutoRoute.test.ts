import { describe, expect, it } from 'vitest';

import { getActiveProjectGraph } from './projectWorkflows';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const preferenceOff = { autoSwitchInvocationRoute: false };

describe('auto invocation route switching with the preference off', () => {
  it('leaves the route alone for workflow, generate, upscale, and Canvas edits', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(
      state,
      {
        action: {
          node: {
            data: {
              inputs: {},
              isIntermediate: true,
              isOpen: true,
              label: '',
              nodePack: 'invokeai',
              notes: '',
              type: 'add',
              useCache: true,
              version: '1.0.0',
            },
            id: 'node-1',
            position: { x: 0, y: 0 },
            type: 'invocation',
          },
          type: 'addNode',
        },
        type: 'applyWorkflowAction',
      },
      preferenceOff
    );
    state = workbenchReducer(state, { type: 'patchGenerateSettings', values: { steps: 25 } }, preferenceOff);
    state = workbenchReducer(
      state,
      {
        type: 'patchWidgetValues',
        values: { inputImage: { height: 512, image_name: 'input.png', width: 768 } },
        widgetId: 'upscale',
      },
      preferenceOff
    );
    state = workbenchReducer(
      state,
      {
        sourceId: 'upscale',
        type: 'patchProjectPromptDraft',
        values: { positivePrompt: 'shared prompt' },
      },
      preferenceOff
    );
    state = workbenchReducer(
      state,
      {
        mutation: {
          bbox: { height: 768, width: 768, x: 0, y: 0 },
          type: 'setCanvasBbox',
        },
        projectId: state.activeProjectId,
        type: 'applyCanvasProjectMutation',
      },
      preferenceOff
    );
    const projectGraph = getActiveProjectGraph(state.projects.find((project) => project.id === state.activeProjectId)!);

    state = workbenchReducer(
      state,
      {
        document: { ...projectGraph, id: 'replacement-graph' },
        label: 'Preference-off replacement',
        type: 'addProjectWorkflow',
      },
      preferenceOff
    );
    state = workbenchReducer(
      state,
      {
        intent: { kind: 'paint' },
        projectId: state.activeProjectId,
        type: 'commitCanvasEdit',
      },
      preferenceOff
    );

    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

    expect(getActiveProjectGraph(project!).nodes).toHaveLength(1);
    expect(project?.invocation).toMatchObject({ destination: 'gallery', sourceId: 'generate' });
  });
});

type State = ReturnType<typeof createInitialWorkbenchState>;

const getProject = (state: State) => state.projects.find((project) => project.id === state.activeProjectId)!;
const getInvocation = (state: State) => getProject(state).invocation;
const getRegion = (state: State, region: 'left' | 'right' | 'center' | 'bottom') =>
  getProject(state).widgetRegions[region];

describe('auto invocation route switching on widget reveal', () => {
  it('follows the rail tab that is brought to the front', () => {
    let state = createInitialWorkbenchState();

    expect(getInvocation(state)).toMatchObject({ sourceId: 'generate' });

    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'video' });

    expect(getInvocation(state)).toMatchObject({ destination: 'gallery', sourceId: 'video' });

    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'generate' });

    expect(getInvocation(state)).toMatchObject({ destination: 'gallery', sourceId: 'generate' });
  });

  it('routes when a collapsed rail is expanded back onto a graph widget, but not when it is collapsed', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'video' });
    // Collapsing the rail puts nothing in front, so the route must not move.
    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'video' });

    expect(getRegion(state, 'left').isCollapsed).toBe(true);

    state = workbenchReducer(state, { sourceId: 'generate', type: 'setInvocationSource' });
    // Expanding it puts the Video panel back on screen — that is a reveal.
    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'video' });

    expect(getRegion(state, 'left').isCollapsed).toBe(false);
    expect(getInvocation(state)).toMatchObject({ sourceId: 'video' });

    state = workbenchReducer(state, { sourceId: 'generate', type: 'setInvocationSource' });
    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'video' });

    expect(getRegion(state, 'left').isCollapsed).toBe(true);
    expect(getInvocation(state)).toMatchObject({ sourceId: 'generate' });
  });

  it('leaves the route alone when a background tab is closed', () => {
    let state = createInitialWorkbenchState();

    // Place Video (defaults no longer carry it), then push it to the background.
    state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId: 'video' });
    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'generate' });
    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    // `video` is not the front tab, so closing it reveals nothing.
    state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId: 'video' });

    expect(getRegion(state, 'left').instanceIds).not.toContain('video');
    expect(getInvocation(state)).toMatchObject({ sourceId: 'workflow' });
  });

  it('follows the tab promoted when the front tab is closed', () => {
    let state = createInitialWorkbenchState();

    // Placing Video reveals it, which is also the route-follow being tested.
    state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId: 'video' });

    expect(getInvocation(state)).toMatchObject({ sourceId: 'video' });

    state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId: 'video' });

    const promoted = getRegion(state, 'left').activeInstanceId;

    expect(promoted).not.toBe('video');
    expect(getInvocation(state)).toMatchObject({ sourceId: promoted });
  });

  it('never routes to a widget that is left dangling by the last removal', () => {
    let state = createInitialWorkbenchState();

    for (const widgetId of getRegion(state, 'left').instanceIds) {
      state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId });
    }

    const region = getRegion(state, 'left');

    expect(region.instanceIds).toEqual([]);
    // Leave the route unchanged; validation separately rejects an unmounted source.
    expect(region.instanceIds).not.toContain(getInvocation(state).sourceId);
  });

  it('keeps a refused close a no-op down to object identity', () => {
    let state = createInitialWorkbenchState();
    const centerIds = [...getRegion(state, 'center').instanceIds];

    for (const widgetId of centerIds.slice(1)) {
      state = workbenchReducer(state, { region: 'center', type: 'toggleRegionWidget', widgetId });
    }

    state = workbenchReducer(state, { sourceId: 'video', type: 'setInvocationSource' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });

    const before = state;
    const beforeProject = getProject(state);

    state = workbenchReducer(state, { region: 'center', type: 'toggleRegionWidget', widgetId: centerIds[0]! });

    expect(state).toBe(before);
    expect(getProject(state)).toBe(beforeProject);
    expect(getInvocation(state)).toMatchObject({ destination: 'gallery', sourceId: 'video' });
  });

  it('does not route when a widget is added to a collapsed rail', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { region: 'bottom', type: 'setRegionWidgetCollapsed', isCollapsed: true });
    state = workbenchReducer(state, { region: 'bottom', type: 'toggleRegionWidget', widgetId: 'workflow:bottom' });

    expect(getRegion(state, 'bottom').isCollapsed).toBe(true);
    expect(getInvocation(state)).toMatchObject({ sourceId: 'generate' });
  });

  it('never lets a revealed generate widget steal the route from Canvas', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'canvas', type: 'setInvocationSource' });
    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'generate' });

    expect(getInvocation(state)).toMatchObject({ sourceId: 'canvas' });
  });

  it('ignores reveals of widgets that are not invocation sources', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'video' });
    state = workbenchReducer(state, { region: 'right', type: 'selectRegionWidget', widgetId: 'gallery' });

    expect(getInvocation(state)).toMatchObject({ sourceId: 'video' });
  });

  it('respects the source lock and the preference', () => {
    let locked = createInitialWorkbenchState();

    locked = workbenchReducer(locked, { type: 'toggleSourceLock' });
    locked = workbenchReducer(locked, { region: 'left', type: 'selectRegionWidget', widgetId: 'video' });

    expect(getInvocation(locked)).toMatchObject({ sourceId: 'generate' });

    let preferenceOffState = createInitialWorkbenchState();

    preferenceOffState = workbenchReducer(
      preferenceOffState,
      { region: 'left', type: 'selectRegionWidget', widgetId: 'video' },
      preferenceOff
    );

    expect(getInvocation(preferenceOffState)).toMatchObject({ sourceId: 'generate' });
  });

  it('follows a widget opened into a region', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { region: 'left', type: 'openRegionWidget', widgetId: 'video' });

    expect(getInvocation(state)).toMatchObject({ destination: 'gallery', sourceId: 'video' });
  });

  it('follows a widget floated out, raised, or docked back', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId: 'video' });
    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'generate' });
    state = workbenchReducer(state, { instanceId: 'video', type: 'floatWidget' });

    expect(getInvocation(state)).toMatchObject({ destination: 'gallery', sourceId: 'video' });

    state = workbenchReducer(state, { instanceId: 'upscale', type: 'floatWidget' });
    state = workbenchReducer(state, { sourceId: 'generate', type: 'setInvocationSource' });
    state = workbenchReducer(state, { instanceId: 'video', type: 'focusFloatingWidget' });

    expect(getInvocation(state)).toMatchObject({ sourceId: 'video' });

    state = workbenchReducer(state, { sourceId: 'generate', type: 'setInvocationSource' });
    state = workbenchReducer(state, { instanceId: 'video', type: 'dockFloatingWidget' });

    expect(getInvocation(state)).toMatchObject({ sourceId: 'video' });
  });

  it('does not re-route or dirty the project when the already-raised window is touched', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { instanceId: 'upscale', type: 'floatWidget' });
    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });

    const before = state;

    // Bound to `onPointerDownCapture`, so this fires on every scroll and click
    // inside the window, not just on a raise.
    state = workbenchReducer(state, { instanceId: 'upscale', type: 'focusFloatingWidget' });

    expect(state).toBe(before);
    expect(getInvocation(state)).toMatchObject({ sourceId: 'workflow' });
  });

  it('follows a tab dragged into another region, and a reorder only when it changes the front tab', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      fromRegion: 'left',
      instanceId: 'video',
      toIndex: 0,
      toRegion: 'right',
      type: 'moveWidgetInstance',
    });

    expect(getRegion(state, 'right').activeInstanceId).toBe('video');
    expect(getInvocation(state)).toMatchObject({ sourceId: 'video' });

    const reordered = workbenchReducer(state, {
      instanceIds: [...getRegion(state, 'left').instanceIds].reverse(),
      region: 'left',
      type: 'reorderWidgetInstances',
    });

    expect(getInvocation(reordered)).toMatchObject({ sourceId: 'video' });

    const reselected = workbenchReducer(state, {
      activeInstanceId: 'upscale',
      instanceIds: [...getRegion(state, 'left').instanceIds],
      region: 'left',
      type: 'reorderWidgetInstances',
    });

    expect(getInvocation(reselected)).toMatchObject({ sourceId: 'upscale' });
  });
});
