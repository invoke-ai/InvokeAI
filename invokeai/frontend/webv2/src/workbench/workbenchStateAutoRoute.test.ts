import { describe, expect, it } from 'vitest';

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
        type: 'applyProjectGraphAction',
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
    state = workbenchReducer(state, { type: 'saveProjectGraphSnapshot' }, preferenceOff);
    const snapshotId = state.projects.find((project) => project.id === state.activeProjectId)?.graphHistory[0]?.id;
    const projectGraph = state.projects.find((project) => project.id === state.activeProjectId)?.projectGraph;

    expect(projectGraph).toBeDefined();

    state = workbenchReducer(
      state,
      {
        document: { ...projectGraph!, id: 'replacement-graph' },
        label: 'Preference-off replacement',
        type: 'replaceProjectGraph',
      },
      preferenceOff
    );
    state = workbenchReducer(
      state,
      { snapshotId: snapshotId ?? '', type: 'restoreProjectGraphSnapshot' },
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

    expect(project?.projectGraph.nodes).toHaveLength(1);
    expect(project?.invocation).toMatchObject({ destination: 'gallery', sourceId: 'generate' });
  });
});
