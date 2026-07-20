import type { GenerateWidgetValues } from '@features/generation/contracts';

import { describe, expect, it } from 'vitest';

import { normalizeWorkbenchQueueHistory } from './queueHistoryNormalization';
import { createInitialWorkbenchState } from './workbenchState.testing';

const backendGraph = { edges: [], id: 'backend-graph', nodes: {} };

const createGenerateValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues =>
  ({
    aspectRatioId: '1:1',
    aspectRatioIsLocked: true,
    aspectRatioValue: 1,
    batchCount: 3,
    cfgRescaleMultiplier: 0,
    cfgScale: 7,
    clipEmbedModel: null,
    clipGEmbedModel: null,
    clipLEmbedModel: null,
    clipSkip: 0,
    colorCompensation: false,
    componentSourceModel: null,
    height: 768,
    loras: [],
    model: { base: 'sdxl', key: 'model', name: 'Model', type: 'main' },
    modelKey: 'model',
    negativePrompt: 'fog',
    negativePromptEnabled: true,
    negativePromptHeightPx: 80,
    positivePrompt: 'a lighthouse',
    positivePromptHeightPx: 80,
    qwen3EncoderModel: null,
    qwenVLEncoderModel: null,
    referenceImages: [],
    scheduler: 'euler',
    seamlessXAxis: false,
    seamlessYAxis: false,
    seed: 42,
    shouldRandomizeSeed: false,
    steps: 20,
    t5EncoderModel: null,
    vae: null,
    vaePrecision: 'fp32',
    width: 1024,
    ...overrides,
  }) as GenerateWidgetValues;

const createContext = () => {
  const project = createInitialWorkbenchState().projects[0]!;
  return { canvas: project.canvas, widgetInstances: project.widgetInstances };
};

const createLegacyItem = (
  sourceId: string,
  options: {
    generate?: Record<string, unknown>;
    status?: string;
    widgetStates?: Record<string, unknown>;
  } = {}
) => ({
  cancellable: true,
  id: `legacy-${sourceId}`,
  snapshot: {
    canvas: createContext().canvas,
    destination: 'gallery',
    ...(options.generate ? { generate: options.generate } : {}),
    graph: { backendGraph, edges: [], id: 'graph', label: 'Legacy', nodes: [], updatedAt: '', version: 1 },
    sourceId,
    submittedAt: '2026-07-01T00:00:00.000Z',
    widgetInstances: createContext().widgetInstances,
    widgetStates: options.widgetStates ?? {},
  },
  status: options.status ?? 'pending',
});

const state = (id: string, values: unknown) => ({ id, label: id, values, version: 1 });

describe('normalizeWorkbenchQueueHistory', () => {
  it('preserves already-current queue snapshots by identity', () => {
    const current = {
      ...createLegacyItem('workflow'),
      snapshot: {
        ...createLegacyItem('workflow').snapshot,
        backendSubmission: { batchCount: 1, graph: backendGraph, kind: 'workflow' },
        filterIntermediateResults: true,
        galleryBoardId: null,
        presentation: { batchCount: 1, height: 512, width: 512 },
      },
    };
    const queue = { items: [current] };

    expect(normalizeWorkbenchQueueHistory(queue, createContext())).toBe(queue);
  });

  it.each(['pending', 'completed'] as const)('upgrades legacy Generate items with %s status', (status) => {
    const values = createGenerateValues();
    const item = createLegacyItem('generate', {
      status,
      widgetStates: {
        gallery: state('gallery', { selectedBoardId: 'board-1' }),
        generate: state('generate', values),
      },
    });

    const normalized = normalizeWorkbenchQueueHistory({ items: [item] }, createContext()).items[0]!;

    expect(normalized.status).toBe(status);
    expect(normalized.snapshot.backendSubmission).toEqual({
      batchCount: 3,
      graph: backendGraph,
      kind: 'generate',
      negativePrompt: 'fog',
      negativePromptNodeId: 'negative_prompt',
      positivePrompt: 'a lighthouse',
      positivePromptNodeId: 'positive_prompt',
      seed: 42,
      seedNodeId: 'seed',
      shouldRandomizeSeed: false,
    });
    expect(normalized.snapshot.presentation).toEqual({
      batchCount: 3,
      height: 768,
      positivePrompt: 'a lighthouse',
      width: 1024,
    });
    expect(normalized.snapshot).toMatchObject({
      filterIntermediateResults: false,
      galleryBoardId: 'board-1',
      resultNodeIds: ['canvas_output'],
    });
  });

  it('upgrades Canvas items both before and after the captured generate payload was added', () => {
    const widgetValues = createGenerateValues({ positivePrompt: 'widget prompt', seed: 10 });
    const capturedValues = createGenerateValues({ positivePrompt: 'captured prompt', seed: 99 });
    const withoutCapture = createLegacyItem('canvas', {
      widgetStates: { generate: state('generate', widgetValues) },
    });
    const withCapture = createLegacyItem('canvas', {
      generate: {
        negativePromptNodeId: 'captured-negative',
        positivePromptNodeId: 'captured-positive',
        seedNodeId: 'captured-seed',
        values: capturedValues,
      },
      widgetStates: { generate: state('generate', widgetValues) },
    });

    const [before, after] = normalizeWorkbenchQueueHistory(
      { items: [withoutCapture, withCapture] },
      createContext()
    ).items;

    expect(before?.snapshot.backendSubmission).toMatchObject({ positivePrompt: 'widget prompt', seed: 10 });
    expect(after?.snapshot.backendSubmission).toMatchObject({
      negativePromptNodeId: 'captured-negative',
      positivePrompt: 'captured prompt',
      positivePromptNodeId: 'captured-positive',
      seed: 99,
      seedNodeId: 'captured-seed',
    });
  });

  it('upgrades Upscale dimensions and Workflow filtering with source-specific submission rules', () => {
    const upscale = createLegacyItem('upscale', {
      widgetStates: {
        upscale: state('upscale', {
          batchCount: 2,
          inputImage: { height: 101, image_name: 'input.png', width: 203 },
          positivePrompt: 'more detail',
          scale: 2.5,
          seed: 7,
          shouldRandomizeSeed: true,
        }),
      },
    });
    const workflow = createLegacyItem('project-graph', {
      widgetStates: { generate: state('generate', { batchCount: 4 }) },
    });

    const [upscaleResult, workflowResult] = normalizeWorkbenchQueueHistory(
      { items: [upscale, workflow] },
      createContext()
    ).items;

    expect(upscaleResult?.snapshot.backendSubmission).toMatchObject({
      batchCount: 2,
      kind: 'generate',
      positivePrompt: 'more detail',
      seed: 7,
      shouldRandomizeSeed: true,
    });
    expect(upscaleResult?.snapshot.presentation).toEqual({ batchCount: 2, height: 248, width: 504 });
    expect(upscaleResult?.snapshot.resultNodeIds).toEqual(['upscale_output']);
    expect(workflowResult?.snapshot).toMatchObject({
      backendSubmission: { batchCount: 4, graph: backendGraph, kind: 'workflow' },
      filterIntermediateResults: true,
      sourceId: 'workflow',
    });
    expect(workflowResult?.snapshot.resultNodeIds).toBeUndefined();
  });

  it('turns unrecoverable and malformed snapshots into safe invalid submissions', () => {
    const malformed = { id: 'broken', snapshot: { sourceId: 'generate' }, status: 'pending' };
    const normalized = normalizeWorkbenchQueueHistory({ items: [malformed, null] }, createContext()).items;

    expect(normalized[0]?.snapshot.backendSubmission).toEqual({
      error: 'Legacy generate queue item is missing a compiled backend graph.',
      kind: 'invalid',
    });
    expect(normalized[0]?.snapshot.presentation.batchCount).toBe(1);
    expect(normalized[1]).toMatchObject({
      id: 'invalid-legacy-queue-item-1',
      snapshot: { backendSubmission: { kind: 'invalid' }, presentation: { batchCount: 1 } },
      status: 'failed',
    });
  });
});
