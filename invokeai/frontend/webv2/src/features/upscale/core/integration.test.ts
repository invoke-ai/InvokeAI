import type { ModelConfig } from '@features/models';

import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { describe, expect, it, vi } from 'vitest';

import { createDefaultUpscaleWidgetValues } from './settings';

const model = (key: string, type: string, base: string, name = key): ModelConfig => ({
  base,
  file_size: 1,
  format: 'checkpoint',
  hash: `${key}-hash`,
  key,
  name,
  path: key,
  source: key,
  source_type: 'path',
  type,
});

const MODELS = [
  model('sdxl', 'main', 'sdxl'),
  model('spandrel', 'spandrel_image_to_image', 'any'),
  model('tile', 'controlnet', 'sdxl', 'SDXL Tile ControlNet'),
];

describe('Upscale snapshot submission', () => {
  it('captures a resolved immutable snapshot, prompt history, graph preview, and generate-style backend request', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.25);
    let state = createInitialWorkbenchState();
    const projectId = state.activeProjectId;
    const values = {
      ...createDefaultUpscaleWidgetValues(MODELS),
      batchCount: 3,
      inputImage: { height: 100, image_name: 'input.png', width: 200 },
      negativePrompt: 'stale Upscale prompt',
      positivePrompt: 'stale Upscale prompt',
    };

    state = workbenchReducer(state, {
      projectId,
      type: 'patchWidgetValues',
      values: { ...values },
      widgetId: 'upscale',
    });
    state = workbenchReducer(state, {
      projectId,
      type: 'patchProjectPromptDraft',
      values: { negativePrompt: 'blur', positivePrompt: 'fine detail' },
    });
    state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });
    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      models: MODELS,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'upscale', sourceLocked: false },
      type: 'submitResolvedInvocationSnapshot',
    });
    state = workbenchReducer(state, {
      projectId,
      type: 'patchProjectPromptDraft',
      values: { negativePrompt: 'changed later', positivePrompt: 'changed later' },
    });

    const project = state.projects.find((candidate) => candidate.id === projectId)!;
    const queueItem = project.queue.items[0]!;
    const snapshotValues = queueItem.snapshot.widgetStates.upscale?.values;

    expect(queueItem.snapshot.sourceId).toBe('upscale');
    expect(snapshotValues).toMatchObject({
      batchCount: 3,
      inputImage: { image_name: 'input.png' },
      negativePrompt: 'blur',
      positivePrompt: 'fine detail',
      seed: Math.floor(0.25 * 4_294_967_295),
    });
    expect(project.promptHistory[0]).toEqual({ negativePrompt: 'blur', positivePrompt: 'fine detail' });
    expect(project.widgetGraphs.upscale?.backendGraph?.nodes.upscale_output).toMatchObject({
      is_intermediate: false,
      type: 'l2i',
    });
    expect(queueItem.snapshot.resultNodeIds).toEqual(['upscale_output']);
    expect(queueItem.snapshot.backendSubmission).toMatchObject({
      batchCount: 3,
      kind: 'generate',
      negativePrompt: 'blur',
      positivePrompt: 'fine detail',
      seedNodeId: 'seed',
    });
    expect(getProjectPromptValues(project)).toMatchObject({
      negativePrompt: 'changed later',
      positivePrompt: 'changed later',
    });
  });
});

const getProjectPromptValues = (project: ReturnType<typeof createInitialWorkbenchState>['projects'][number]) =>
  project.widgetInstances.generate?.state.values;
