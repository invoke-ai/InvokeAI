import { describe, expect, it } from 'vitest';

import type { GenerateWidgetValues, MainModelConfig } from './generation/types';
import type { GraphContract, Project, QueueItem } from './types';

import { getDefaultGenerateSettings } from './generation/baseGenerationPolicies';
import { createQueueItemBackendSubmission } from './WorkbenchRuntime';
import { createInitialWorkbenchState } from './workbenchState';

const model: MainModelConfig = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' };

const createValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(model),
  model,
  modelKey: model.key,
  negativePrompt: 'low quality',
  positivePrompt: 'a canvas prompt',
  seed: 123,
  shouldRandomizeSeed: true,
  ...overrides,
});

const graph: GraphContract = {
  backendGraph: { edges: [], id: 'backend-graph', nodes: {} },
  edges: [],
  id: 'graph-1',
  label: 'Canvas',
  nodes: [],
  updatedAt: '2026-06-09T00:00:00.000Z',
  version: 1,
};

const createProject = (): Project => createInitialWorkbenchState().projects[0]!;

const createQueueItem = (project: Project, overrides: Partial<QueueItem['snapshot']>): QueueItem => ({
  cancellable: true,
  id: 'local-1',
  snapshot: {
    canvas: project.canvas,
    destination: 'canvas',
    graph,
    sourceId: 'canvas',
    submittedAt: '2026-06-09T00:00:00.000Z',
    widgetInstances: project.widgetInstances,
    widgetStates: Object.fromEntries(
      Object.values(project.widgetInstances).map((instance) => [instance.typeId, instance.state])
    ),
    ...overrides,
  },
  status: 'pending',
});

describe('createQueueItemBackendSubmission', () => {
  it('submits canvas queue items through the generate batch path with resolved prompt and seed metadata', () => {
    const project = createProject();
    const values = createValues({
      batchCount: 4,
      negativePrompt: 'avoid blur',
      positivePrompt: 'inpaint prompt',
      seed: 987,
    });
    const queueItem = createQueueItem(project, {
      generate: {
        negativePromptNodeId: 'canvas_negative_prompt',
        positivePromptNodeId: 'canvas_positive_prompt',
        seedNodeId: 'canvas_seed',
        values,
      },
      sourceId: 'canvas',
    });

    const submission = createQueueItemBackendSubmission(project, queueItem);

    expect(submission).toEqual({
      kind: 'generate',
      request: {
        batchCount: 4,
        destination: 'canvas',
        graph: graph.backendGraph,
        negativePrompt: 'avoid blur',
        negativePromptNodeId: 'canvas_negative_prompt',
        positivePrompt: 'inpaint prompt',
        positivePromptNodeId: 'canvas_positive_prompt',
        projectId: project.id,
        seed: 987,
        seedNodeId: 'canvas_seed',
        shouldRandomizeSeed: true,
        sourceQueueItemId: 'local-1',
      },
    });
  });

  it('falls back to legacy canvas widget-state metadata when a persisted snapshot has no generate payload', () => {
    const project = createProject();
    const values = createValues({
      batchCount: 2,
      negativePrompt: 'legacy negative',
      positivePrompt: 'legacy prompt',
      seed: 321,
      shouldRandomizeSeed: false,
    });
    const queueItem = createQueueItem(project, {
      sourceId: 'canvas',
      widgetStates: {
        ...createQueueItem(project, {}).snapshot.widgetStates,
        generate: { id: 'generate', label: 'Generate', values: { ...values } as Record<string, unknown>, version: 1 },
      },
    });

    const submission = createQueueItemBackendSubmission(project, queueItem);

    expect(submission).toEqual({
      kind: 'generate',
      request: {
        batchCount: 2,
        destination: 'canvas',
        graph: graph.backendGraph,
        negativePrompt: 'legacy negative',
        negativePromptNodeId: 'negative_prompt',
        positivePrompt: 'legacy prompt',
        positivePromptNodeId: 'positive_prompt',
        projectId: project.id,
        seed: 321,
        seedNodeId: 'seed',
        shouldRandomizeSeed: false,
        sourceQueueItemId: 'local-1',
      },
    });
  });

  it('keeps workflow queue items on the workflow path', () => {
    const project = createProject();
    const queueItem = createQueueItem(project, { destination: 'gallery', sourceId: 'workflow' });

    const submission = createQueueItemBackendSubmission(project, queueItem);

    expect(submission).toEqual({
      kind: 'workflow',
      request: {
        batchCount: 1,
        destination: 'gallery',
        graph: graph.backendGraph,
        projectId: project.id,
        sourceQueueItemId: 'local-1',
      },
    });
  });
});
