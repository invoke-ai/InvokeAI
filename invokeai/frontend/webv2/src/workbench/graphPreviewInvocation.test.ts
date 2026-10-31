import type {
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
  ComponentModelConfig,
} from '@features/generation/contracts';
import type { CanvasRegionalGuidanceLayerContract } from '@workbench/canvas-engine/api';

import { galleryKeys } from '@features/gallery/queries';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { queryClient } from '@platform/query/client';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { resolveAndSubmitGraphPreviewInvocation } from './graphPreviewInvocation';
import { beginInvocationPreparation, endInvocationPreparation } from './invocationPreparation';

// Production words rejections from the locale; tests only need the structured rejection to reach the notice.
const formatControlLayerError = ({ code, layerName }: { code: string; layerName: string }) => `${layerName}: ${code}`;
const formatUpscaleMetadataFailure = (name: string) => `metadataFailed: ${name}`;

// Seed capabilities to match app boot; submission fails closed without them.
seedArchitectureCapabilities();

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

// Mount Canvas in the project so resolveInvocationRoute considers that source available.
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

/** A fresh listing of every readable board, as `ensureGalleryBoardDestinations` leaves it: the boards named are destinations. */
const listGalleryBoardsInCache = (...boardIds: string[]) =>
  queryClient.setQueryData(
    galleryKeys.boards(captureAccountScope(), {
      includeArchived: true,
      includeDateBoards: false,
      orderBy: 'created_at',
      orderDir: 'DESC',
    }),
    boardIds.map((id) => ({
      archived: false,
      assetCount: 0,
      assetVideoCount: 0,
      id,
      imageCount: 0,
      isInbox: false,
      kind: 'board' as const,
      name: id,
      projectId: null,
      videoCount: 0,
    }))
  );

describe('resolveAndSubmitGraphPreviewInvocation', () => {
  beforeEach(() => {
    queryClient.clear();
    listGalleryBoardsInCache();
  });

  it('routes a canvas source through prepareCanvasInvocation and does not dispatch a resolved snapshot', () => {
    const project = withCanvasWidgetMounted(getActiveProject(createGenerateValues()));
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    const submitted = resolveAndSubmitGraphPreviewInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation,
      project,
      sourceId: 'canvas',
    });

    expect(submitted).toBe(true);
    expect(submitResolved).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).toHaveBeenCalledTimes(1);
    expect(prepareCanvasInvocation.mock.calls[0]?.[0]).toMatchObject({ projectId: project.id });
  });

  it('prepares nothing for a Canvas preview of a blank prompt around a layout region while Generate is selected', () => {
    const ideogramModel: MainModelConfig = {
      base: 'ideogram-4',
      format: 'diffusers',
      key: 'ideogram-4-model',
      name: 'Ideogram 4',
      type: 'main',
    };
    const region: CanvasRegionalGuidanceLayerContract = {
      autoNegative: false,
      blendMode: 'normal',
      id: 'region',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: { height: 64, imageName: 'region.png', width: 64 }, fill: { color: '#00ff00', style: 'solid' } },
      name: 'Region',
      negativePrompt: null,
      opacity: 1,
      positivePrompt: 'a red kite',
      referenceImages: [],
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 32, y: 32 },
      type: 'regional_guidance',
    };
    const previewWithPrompt = (positivePrompt: string) => {
      const project = withCanvasWidgetMounted(
        getActiveProject({
          ...getDefaultGenerateSettings(ideogramModel),
          model: ideogramModel,
          modelKey: ideogramModel.key,
          positivePrompt,
        })
      );
      const prepareCanvasInvocation = vi.fn();
      const submitted = resolveAndSubmitGraphPreviewInvocation({
        commands: createWorkbenchStore().commands,
        formatControlLayerError,
        formatUpscaleMetadataFailure,
        models: undefined,
        owner: captureAccountScope(),
        prepareCanvasInvocation,
        project: {
          ...project,
          canvas: {
            ...project.canvas,
            document: {
              ...project.canvas.document,
              bbox: { height: 512, width: 512, x: 0, y: 0 },
              stacks: { ...project.canvas.document.stacks, regional_guidance: [region] },
            },
          },
          invocation: { ...project.invocation, sourceId: 'generate' },
        },
        sourceId: 'canvas',
      });
      return { prepared: prepareCanvasInvocation.mock.calls.length, submitted };
    };

    // Nothing uploads for a submission the compiler would reject; with a prompt the same preview goes ahead.
    expect(previewWithPrompt('')).toEqual({ prepared: 0, submitted: false });
    expect(previewWithPrompt('a beach at noon')).toEqual({ prepared: 1, submitted: true });
  });

  it('dispatches submitResolvedInvocationSnapshot for a non-canvas source and never prepares the canvas', () => {
    const project = getActiveProject(createGenerateValues());
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    const submitted = resolveAndSubmitGraphPreviewInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      models: undefined,
      owner: captureAccountScope(),
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

  it('returns false and does not submit while another submission for the project is preparing', () => {
    const project = getActiveProject(createGenerateValues());
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const heldLease = beginInvocationPreparation(project.id);
    expect(heldLease).not.toBeNull();

    try {
      const submitted = resolveAndSubmitGraphPreviewInvocation({
        commands,
        formatControlLayerError,
        formatUpscaleMetadataFailure,
        models: undefined,
        owner: captureAccountScope(),
        prepareCanvasInvocation: vi.fn(),
        project,
        sourceId: 'generate',
      });

      expect(submitted).toBe(false);
      expect(submitResolved).not.toHaveBeenCalled();
    } finally {
      endInvocationPreparation(heldLease!);
    }
  });

  it('returns false and does not submit when there is no sourceId', () => {
    const project = getActiveProject(createGenerateValues());
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    const submitted = resolveAndSubmitGraphPreviewInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      models: undefined,
      owner: captureAccountScope(),
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
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation,
      project,
      sourceId: 'generate',
    });

    expect(submitted).toBe(false);
    expect(submitResolved).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).not.toHaveBeenCalled();
  });
});
