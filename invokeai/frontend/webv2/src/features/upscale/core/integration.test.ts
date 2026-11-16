import type { ModelConfig } from '@features/models';
import type { WorkbenchState } from '@workbench/projectContracts';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDynamicPromptsExpansionSource } from '@workbench/invocation';
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

// Upscale, like Generate, does not submit without the served architecture table.
seedArchitectureCapabilities();

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
      sourceId: 'upscale',
      type: 'patchProjectPromptDraft',
      values: { negativePrompt: 'blur', positivePrompt: 'fine detail' },
    });
    state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });
    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      models: MODELS,
      projectId: state.activeProjectId,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'upscale', sourceLocked: false },
      type: 'submitResolvedInvocationSnapshot',
    });
    state = workbenchReducer(state, {
      projectId,
      sourceId: 'upscale',
      type: 'patchProjectPromptDraft',
      values: { negativePrompt: 'changed later', positivePrompt: 'changed later' },
    });

    const project = state.projects.find((candidate) => candidate.id === projectId)!;
    const queueItem = project.queue.items[0]!;

    expect(queueItem.snapshot.sourceId).toBe('upscale');
    expect(queueItem.snapshot.recall).toBeUndefined();
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

const CINEMATIC = {
  id: 'cinematic',
  name: 'Cinematic',
  negativePrompt: '{prompt}, lowres',
  positivePrompt: '{prompt}, film grain',
};

/** An Upscale ready to Refine, with the shared prompt draft patched as the Upscale widget would. */
const prepareUpscale = (draft: Record<string, unknown>, values: Record<string, unknown> = {}): WorkbenchState => {
  let state = createInitialWorkbenchState();

  state = workbenchReducer(state, {
    type: 'patchWidgetValues',
    values: {
      ...createDefaultUpscaleWidgetValues(MODELS),
      inputImage: { height: 100, image_name: 'input.png', width: 200 },
      ...values,
    },
    widgetId: 'upscale',
  });
  state = workbenchReducer(state, { sourceId: 'upscale', type: 'patchProjectPromptDraft', values: draft });

  return workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });
};

const activeProject = (state: WorkbenchState) =>
  state.projects.find((candidate) => candidate.id === state.activeProjectId)!;

/** Submits as `submitResolvedInvocation` does: with the expansion of what the project expands from, if anything. */
const submitUpscale = (state: WorkbenchState, expansion?: { prompts: string[]; sampleSeed: number | null }) => {
  const source = getDynamicPromptsExpansionSource(activeProject(state), 'upscale');

  return workbenchReducer(state, {
    backendSupportsCancellation: true,
    ...(source && expansion
      ? {
          expandedFrom: {
            combinatorial: source.combinatorial,
            maxPrompts: source.maxPrompts,
            positivePrompt: source.positivePrompt,
          },
          positivePrompts: expansion.prompts,
          positivePromptsSampleSeed: expansion.sampleSeed,
        }
      : {}),
    models: MODELS,
    projectId: state.activeProjectId,
    route: { destination: 'gallery', destinationLocked: false, sourceId: 'upscale', sourceLocked: false },
    type: 'submitResolvedInvocationSnapshot',
  });
};

describe('Upscale with the prompt template and dynamic prompts it shares with Generate', () => {
  it('queues the prompt inside the template, recording the prompt as written in history', () => {
    const state = submitUpscale(
      prepareUpscale({ negativePrompt: 'blur', positivePrompt: 'fine detail', promptTemplate: CINEMATIC })
    );
    const project = activeProject(state);
    const queueItem = project.queue.items[0]!;

    expect(queueItem.snapshot.backendSubmission).toMatchObject({
      kind: 'generate',
      negativePrompt: 'blur, lowres',
      positivePrompt: 'fine detail, film grain',
    });
    expect(queueItem.snapshot.presentation.positivePrompt).toBe('fine detail, film grain');
    expect(project.promptHistory[0]).toEqual({ negativePrompt: 'blur', positivePrompt: 'fine detail' });
  });

  it('expands the templated prompt and queues one upscale per prompt and iteration', () => {
    const prepared = prepareUpscale(
      { dynamicPromptsMaxPrompts: 10, positivePrompt: 'a {red|green} fox', promptTemplate: CINEMATIC },
      { batchCount: 2 }
    );

    expect(getDynamicPromptsExpansionSource(activeProject(prepared), 'upscale')).toMatchObject({
      combinatorial: true,
      maxPrompts: 10,
      positivePrompt: 'a {red|green} fox, film grain',
    });

    const queueItem = activeProject(
      submitUpscale(prepared, { prompts: ['a red fox, film grain', 'a green fox, film grain'], sampleSeed: null })
    ).queue.items[0]!;

    expect(queueItem.snapshot.backendSubmission).toMatchObject({
      batchCount: 2,
      positivePrompts: ['a red fox, film grain', 'a green fox, film grain'],
      seedBehaviour: 'per-iteration',
    });
    expect(queueItem.snapshot.presentation.batchCount).toBe(4);
  });

  it("rotates Generate's random sample after drawing from it, as a Generate run would", () => {
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);

    try {
      const prepared = prepareUpscale({
        dynamicPromptsCombinatorial: false,
        dynamicPromptsSampleSeed: 7,
        positivePrompt: 'a {red|green} fox',
      });
      const project = activeProject(submitUpscale(prepared, { prompts: ['a green fox'], sampleSeed: 7 }));

      expect(project.queue.items[0]!.snapshot.backendSubmission).toMatchObject({ positivePrompts: ['a green fox'] });
      expect(getProjectPromptValues(project)?.dynamicPromptsSampleSeed).toBe(Math.floor(0.5 * 4_294_967_295));
      expect(project.widgetInstances.upscale?.state.values).not.toHaveProperty('dynamicPromptsSampleSeed');
    } finally {
      random.mockRestore();
    }
  });

  it('runs the prompt as typed with dynamic prompts off, template still applied', () => {
    const prepared = prepareUpscale({
      dynamicPromptsEnabled: false,
      positivePrompt: 'a {red|green} fox',
      promptTemplate: CINEMATIC,
    });

    expect(getDynamicPromptsExpansionSource(activeProject(prepared), 'upscale')).toBeNull();
    expect(activeProject(submitUpscale(prepared)).queue.items[0]!.snapshot.backendSubmission).toMatchObject({
      positivePrompt: 'a {red|green} fox, film grain',
    });
  });

  it('leaves Model only, which reads no prompt, unexpanded', () => {
    const prepared = prepareUpscale(
      { positivePrompt: 'a {red|green} fox', promptTemplate: CINEMATIC },
      { mode: 'model-only' }
    );

    expect(getDynamicPromptsExpansionSource(activeProject(prepared), 'upscale')).toBeNull();
  });
});
