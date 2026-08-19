import type * as workflowReactModule from '@features/workflow/react';
import type { WorkbenchState } from '@workbench/projectContracts';

// Feature-private document builders: the dependency policy exempts test files,
// and hand-rolling a compilable graph literal here would only duplicate them.
import { buildInvocationNode, createProjectGraph, projectGraphReducer } from '@features/workflow/core/document';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * The workflow compile site stamps the project graph's library binding onto the
 * compiled submission, which is the only place a completed run can learn which
 * library record produced it. Templates are read imperatively by both the route
 * resolver and the compiler, so they are stubbed as loaded here.
 */

const stringField = { batch: false, cardinality: 'SINGLE' as const, name: 'StringField' };

const template = {
  category: 'test',
  classification: 'stable' as const,
  description: '',
  inputs: {
    value: {
      default: 'hello',
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      input: 'any' as const,
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'value',
      options: null,
      required: true,
      title: 'value',
      type: stringField,
      uiChoiceLabels: null,
      uiComponent: null,
      uiHidden: false,
      uiModelBase: null,
      uiModelFormat: null,
      uiModelType: null,
      uiOrder: null,
    },
  },
  nodePack: 'invokeai',
  outputs: { out: { description: '', name: 'out', title: 'Out', type: stringField } },
  outputType: 'source_output',
  tags: [],
  title: 'source',
  type: 'source',
  useCache: true,
  version: '1.0.0',
};

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<typeof workflowReactModule>()),
  getInvocationTemplatesSnapshot: () => ({ error: null, status: 'loaded', templates: { source: template } }),
}));

const buildReadyGraph = (libraryWorkflowId?: string) => {
  let document = createProjectGraph('run-binding');

  document = projectGraphReducer(document, { node: buildInvocationNode(template, { x: 0, y: 0 }), type: 'addNode' });

  return { ...document, ...(libraryWorkflowId ? { libraryWorkflowId } : {}) };
};

/** Mounts the workflow widget so the workflow source resolves as available. */
const withWorkflowWidgetMounted = (state: WorkbenchState): WorkbenchState => ({
  ...state,
  projects: state.projects.map((project) => ({
    ...project,
    widgetRegions: {
      ...project.widgetRegions,
      center: { ...project.widgetRegions.center, activeInstanceId: 'workflow', instanceIds: ['workflow'] },
    },
  })),
});

const submitWorkflow = (libraryWorkflowId?: string) => {
  let state = withWorkflowWidgetMounted(createInitialWorkbenchState());

  state = workbenchReducer(state, {
    document: buildReadyGraph(libraryWorkflowId),
    label: 'Load workflow',
    type: 'replaceProjectGraph',
  });
  state = workbenchReducer(state, {
    backendSupportsCancellation: true,
    route: { destination: 'gallery', destinationLocked: false, sourceId: 'workflow', sourceLocked: true },
    type: 'submitResolvedInvocationSnapshot',
  });

  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  return project?.queue.items[0];
};

describe('library binding on the compiled workflow submission', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('stamps the bound library record onto the run it submits', () => {
    expect(submitWorkflow('library-workflow-1')?.snapshot.backendSubmission).toMatchObject({
      kind: 'workflow',
      libraryWorkflowId: 'library-workflow-1',
    });
  });

  // An ad-hoc workflow belongs to no library record, so a completed run of it
  // must have nothing to write back to.
  it('leaves an unbound workflow run unstamped', () => {
    expect(submitWorkflow()?.snapshot.backendSubmission).not.toHaveProperty('libraryWorkflowId');
  });
});
