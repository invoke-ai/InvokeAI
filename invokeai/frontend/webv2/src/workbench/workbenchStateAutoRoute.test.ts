import { describe, expect, it, vi } from 'vitest';

import type * as settingsStore from './settings/store';

import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

/**
 * With the autoSwitchInvocationRoute preference off, no surface edit — not
 * even the pre-existing workflow rule — may steer the Invoke route. The
 * default-on behavior is covered in workbenchState.test.ts, whose un-hydrated
 * store resolves to DEFAULT_PREFERENCES.
 */

vi.mock('./settings/store', async (importOriginal) => {
  const original = await importOriginal<typeof settingsStore>();

  return {
    ...original,
    getWorkbenchPreferences: () => ({ ...original.DEFAULT_PREFERENCES, autoSwitchInvocationRoute: false }),
  };
});

describe('auto invocation route switching with the preference off', () => {
  it('leaves the route alone for workflow, generate, and canvas edits', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
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
    });
    state = workbenchReducer(state, { type: 'patchGenerateSettings', values: { steps: 25 } });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { inputImage: { height: 512, image_name: 'input.png', width: 768 } },
      widgetId: 'upscale',
    });

    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

    expect(project?.projectGraph.nodes).toHaveLength(1);
    expect(project?.invocation).toMatchObject({ destination: 'canvas', sourceId: 'generate' });
  });
});
