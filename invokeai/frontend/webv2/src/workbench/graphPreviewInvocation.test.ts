import type {
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
  ComponentModelConfig,
} from '@features/generation/contracts';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { describe, expect, it, vi } from 'vitest';

import { resolveAndSubmitGraphPreviewInvocation } from './graphPreviewInvocation';

const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const animaVae: VaeModelConfig = { base: 'qwen-image', key: 'anima-vae', name: 'Anima VAE', type: 'vae' };
const qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3-encoder',
  name: 'Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};

const createGenerateValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(animaModel),
  model: animaModel,
  modelKey: animaModel.key,
  positivePrompt: 'an anima prompt',
  qwen3EncoderModel: qwen3Encoder,
  vae: animaVae,
  ...overrides,
});

const getActiveProject = (values: GenerateWidgetValues) => {
  const state = workbenchReducer(createInitialWorkbenchState(), { type: 'setGenerateSettings', values });
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project!;
};

// Mounts the canvas widget into the center region so `resolveInvocationRoute`
// treats a `canvas` source as available (mirrors invocation.test.ts's
// `canvasInputFor` mountedWidgetIds override, but here as an actual project
// mutation since `resolveAndSubmitGraphPreviewInvocation` takes a real Project).
const withCanvasWidgetMounted = (
  project: ReturnType<typeof getActiveProject>
): ReturnType<typeof getActiveProject> => ({
  ...project,
  widgetRegions: {
    ...project.widgetRegions,
    center: {
      ...project.widgetRegions.center,
      activeInstanceId: 'canvas',
      instanceIds: ['canvas'],
    },
  },
});

describe('resolveAndSubmitGraphPreviewInvocation', () => {
  it('routes a canvas source through prepareCanvasInvocation and does not dispatch a resolved snapshot', () => {
    const project = withCanvasWidgetMounted(getActiveProject(createGenerateValues()));
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    const submitted = resolveAndSubmitGraphPreviewInvocation({
      commands,
      models: undefined,
      prepareCanvasInvocation,
      project,
      sourceId: 'canvas',
    });

    expect(submitted).toBe(true);
    expect(submitResolved).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).toHaveBeenCalledTimes(1);
    expect(prepareCanvasInvocation.mock.calls[0]?.[0]).toMatchObject({ projectId: project.id });
  });

  it('dispatches submitResolvedInvocationSnapshot for a non-canvas source and never prepares the canvas', () => {
    const project = getActiveProject(createGenerateValues());
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    const submitted = resolveAndSubmitGraphPreviewInvocation({
      commands,
      models: undefined,
      prepareCanvasInvocation,
      project,
      sourceId: 'generate',
    });

    expect(submitted).toBe(true);
    expect(prepareCanvasInvocation).not.toHaveBeenCalled();
    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({
      route: expect.objectContaining({ sourceId: 'generate' }),
    });
  });

  it('returns false and does not submit when there is no sourceId', () => {
    const project = getActiveProject(createGenerateValues());
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    const submitted = resolveAndSubmitGraphPreviewInvocation({
      commands,
      models: undefined,
      prepareCanvasInvocation,
      project,
      sourceId: undefined,
    });

    expect(submitted).toBe(false);
    expect(submitResolved).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).not.toHaveBeenCalled();
  });

  it('returns false and does not submit when the resolved route is invalid', () => {
    const project = getActiveProject(createGenerateValues({ qwen3EncoderModel: undefined }));
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    const submitted = resolveAndSubmitGraphPreviewInvocation({
      commands,
      models: undefined,
      prepareCanvasInvocation,
      project,
      sourceId: 'generate',
    });

    expect(submitted).toBe(false);
    expect(submitResolved).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).not.toHaveBeenCalled();
  });
});
