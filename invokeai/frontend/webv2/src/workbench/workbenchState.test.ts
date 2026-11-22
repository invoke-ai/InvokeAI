import type { GalleryImageItem, GalleryVideoItem, GeneratedImageContract } from '@features/gallery';
import type { GeneratedVideoContract } from '@features/gallery/contracts';
import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/api';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
} from '@workbench/canvas-engine/contracts';
import type { GraphContract } from '@workbench/graphContracts';
import type { ExpandedPositivePrompts, InvocationSourceId } from '@workbench/invocationContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { StoredProjectLayout } from '@workbench/workspace/migrateProjectLayout';
import type { TFunction } from 'i18next';

import {
  GALLERY_RECENT_IMAGE_LIMIT,
  legacyGeneratedImageToGalleryItem,
  registerImageCluster,
} from '@features/gallery/contracts';
import { galleryKeys } from '@features/gallery/queries';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { getWorkflowGeneratorKey } from '@features/workflow/utility';
import { queryClient } from '@platform/query/client';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { getDocumentLeaves } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { createEmptyCanvasDocument } from './canvasMigration';
import { MAX_CANVAS_SNAPSHOTS } from './canvasProjectMutations';
import { getCanvasStagingCandidateFingerprint, getCanvasStagingSlots } from './canvasStagingView';
import { getDynamicPromptsExpansionSource } from './invocation';
import { captureInvocationRuntimeInputs } from './invocationPreparation';
import { layoutPresets } from './layoutPresets';
import { serializeProjectDocumentV3 } from './projects/projectDocument';
import { deserializeProjectDocument } from './projects/projectHydration';
import { getActiveProjectGraph } from './projectWorkflows';
import { resolveNotificationText } from './shell/notifications/notificationToasts';
import { getProjectWidgetValues } from './widgetState';
import { normalizeWorkbenchAccount, normalizeWorkbenchProject } from './workbenchState';
import {
  addCustomWorkspace,
  asStoredBeforeSplit,
  createInitialWorkbenchState,
  getLayoutOf,
  nextLayerName,
  type WorkbenchAction,
  workbenchReducer as reduceWorkbench,
} from './workbenchState.testing';
import { getWorkspaceStructure, setWorkspaceRoute } from './workspace/accountWorkspaces';
import { storedOn, structuralLayouts, viewOnlyLayouts } from './workspace/storedLayoutFixtures.testing';

seedArchitectureCapabilities();

const preparedExpansionSource = (state: WorkbenchState, sourceId: InvocationSourceId) => {
  const source = getDynamicPromptsExpansionSource(getActiveProject(state), sourceId);

  return source
    ? { combinatorial: source.combinatorial, maxPrompts: source.maxPrompts, positivePrompt: source.positivePrompt }
    : undefined;
};

const generationDeviceMock = vi.hoisted(() => ({
  options: [] as { device: string; name: string }[],
}));

vi.mock('@features/queue/devices', async (importOriginal) => {
  const original = (await importOriginal()) as Record<string, unknown>;

  return {
    ...original,
    getGenerationDevicesSnapshot: () => ({
      error: null,
      loadState: 'loaded' as const,
      options: generationDeviceMock.options,
      setting: 'auto' as const,
    }),
  };
});

const workflowTemplatesMock = vi.hoisted(() => ({
  snapshot: { error: null, status: 'idle', templates: {} } as {
    error: string | null;
    status: 'idle' | 'loading' | 'loaded' | 'error';
    templates: Record<string, unknown>;
  },
}));

vi.mock('@features/workflow/react', async (importOriginal) => {
  const original = (await importOriginal()) as Record<string, unknown>;

  return { ...original, getInvocationTemplatesSnapshot: () => workflowTemplatesMock.snapshot };
});

afterEach(() => {
  workflowTemplatesMock.snapshot = { error: null, status: 'idle', templates: {} };
  generationDeviceMock.options = [];
});

type LegacyCanvasMutation = CanvasProjectMutation & { projectId?: string };
const CANVAS_MUTATION_TYPES = new Set<CanvasProjectMutation['type']>([
  'commitStagedImage',
  'addCanvasLayer',
  'applyCanvasLayerStackMutation',
  'clearCanvasStaging',
  'convertCanvasLayer',
  'cycleStagedImage',
  'deleteCanvasSnapshot',
  'discardAllStagedImages',
  'discardSelectedStagedImage',
  'mergeCanvasLayersDown',
  'removeCanvasLayers',
  'reorderCanvasSiblings',
  'replaceCanvasDocument',
  'replaceCanvasLayer',
  'replaceCanvasLayers',
  'rollbackStagedImageCommit',
  'resizeCanvasDocument',
  'restoreCanvasSnapshot',
  'restoreStagedImages',
  'saveCanvasSnapshot',
  'setCanvasBbox',
  'setCanvasLayersEnabled',
  'setCanvasLayersHidden',
  'setCanvasSelectedLayer',
  'setCanvasLayerPositions',
  'setCanvasStagingAutoSwitch',
  'setStagedImageIndex',
  'toggleCanvasStagingThumbnailsVisibility',
  'toggleCanvasStagingVisibility',
  'updateCanvasLayer',
  'updateCanvasLayerConfig',
  'updateCanvasLayerSource',
]);

const workbenchReducer = (state: WorkbenchState, action: WorkbenchAction | LegacyCanvasMutation): WorkbenchState => {
  if (CANVAS_MUTATION_TYPES.has(action.type as CanvasProjectMutation['type'])) {
    const { projectId = state.activeProjectId, ...mutation } = action as LegacyCanvasMutation;
    return reduceWorkbench(state, {
      mutation: mutation as CanvasProjectMutation,
      projectId,
      type: 'applyCanvasProjectMutation',
    });
  }
  return reduceWorkbench(state, action as WorkbenchAction);
};

const commitSelectedStagedImage = (state: WorkbenchState, projectId = state.activeProjectId): WorkbenchState => {
  const project = state.projects.find((candidate) => candidate.id === projectId);
  const slot = project
    ? getCanvasStagingSlots(project.canvas, project.queue.items)[project.canvas.stagingArea.selectedImageIndex]
    : undefined;
  if (!project || slot?.kind !== 'candidate') {
    return state;
  }
  const candidate = slot.candidate;
  const { placement } = candidate;
  const layer: CanvasRasterLayerContractV2 = {
    blendMode: 'normal',
    id: `accepted-${candidate.imageName}`,
    isEnabled: true,
    isLocked: false,
    name: `Layer ${getDocumentLeaves(project.canvas.document).length + 1}`,
    opacity: placement.opacity,
    source: {
      image: { height: candidate.height, imageName: candidate.imageName, width: candidate.width },
      type: 'image',
    },
    transform: {
      rotation: 0,
      scaleX: candidate.width === 0 ? 1 : placement.width / candidate.width,
      scaleY: candidate.height === 0 ? 1 : placement.height / candidate.height,
      x: placement.x,
      y: placement.y,
    },
    type: 'raster',
  };
  return reduceWorkbench(state, {
    mutation: {
      anchor: stackTopAnchor(projectId),
      candidateFingerprint: getCanvasStagingCandidateFingerprint(candidate),
      continueStaging: false,
      event: {
        createdAt: '2026-07-16T00:00:00.000Z',
        id: `event-${candidate.imageName}`,
        summary: `Accepted ${candidate.imageName} into a new raster layer`,
        type: 'canvas-layer-accepted',
      },
      layer,
      selectedImageIndex: project.canvas.stagingArea.selectedImageIndex,
      type: 'commitStagedImage',
    },
    projectId,
    type: 'applyCanvasProjectMutation',
  });
};

const model: MainModelConfig = {
  base: 'sdxl',
  key: 'test-model',
  name: 'Test Model',
  type: 'main',
};

const createGenerateValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  aspectRatioId: '1:1',
  aspectRatioIsLocked: false,
  aspectRatioValue: 1,
  batchCount: 1,
  pidMode: 'off',
  pidDecoderModel: null,
  gemma2EncoderModel: null,
  pidSteps: 4,
  refinerModel: null,
  refinerSteps: 20,
  refinerCfgScale: 7.5,
  refinerScheduler: 'euler',
  refinerPositiveAestheticScore: 6,
  refinerNegativeAestheticScore: 2.5,
  refinerStart: 0.8,
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  clipEmbedModel: null,
  clipGEmbedModel: null,
  clipLEmbedModel: null,
  clipSkip: 0,
  colorCompensation: false,
  hiDiffusionEnabled: false,
  hiDiffusionRauNetEnabled: true,
  hiDiffusionT1Ratio: 0.4,
  hiDiffusionT2Ratio: 0,
  hiDiffusionWindowAttentionEnabled: true,
  dynamicPromptsCombinatorial: true,
  dynamicPromptsEnabled: true,
  dynamicPromptsMaxPrompts: 100,
  dynamicPromptsSampleSeed: 0,
  dynamicPromptsResample: true,
  dynamicPromptsSeedBehaviour: 'per-iteration',
  componentSourceModel: null,
  height: 1024,
  loras: [],
  model,
  modelKey: model.key,
  negativePromptEnabled: true,
  negativePrompt: '',
  negativePromptHeightPx: 56,
  positivePrompt: 'first prompt',
  positivePromptHeightPx: 96,
  expandPromptModelKey: null,
  imageToPromptModelKey: null,
  promptTemplate: null,
  promptTemplateViewMode: false,
  mistralEncoderModel: null,
  qwen3EncoderModel: null,
  qwenVLEncoderModel: null,
  qwen3VLEncoderModel: null,
  qwen3VLVisionTowerModel: null,
  qwen35EncoderModel: null,
  animaInpaintAdapterModel: null,
  animaInpaintAdapterEnabled: true,
  animaInpaintAdapterWeight: 1,
  flux3PromptUpsamplerEnabled: false,
  flux3PromptUpsamplerModel: null,
  wanT5EncoderModel: null,
  wanLowNoiseModel: null,
  ideogram4UnconditionalModel: null,
  wanGuidanceScaleLowNoise: null,
  ideogram4SamplerPreset: 'V4_QUALITY_48',
  ideogram4Steps: null,
  ideogram4GuidanceScale: null,
  ideogram4Mu: null,
  ideogram4ColorPalette: [],
  externalImageSize: null,
  openaiQuality: 'auto',
  openaiBackground: 'auto',
  openaiInputFidelity: null,
  geminiTemperature: null,
  geminiThinkingLevel: null,
  seedreamWatermark: false,
  seedreamOptimizePrompt: false,
  krea2RebalanceEnabled: false,
  krea2RebalanceMultiplier: 4,
  krea2RebalanceWeights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0',
  krea2SeedVarianceEnabled: false,
  krea2SeedVarianceStrength: 0.1,
  krea2SeedVarianceRandomizePercent: 50,
  krea2ReferenceKvCache: false,
  fluxDypePreset: 'off',
  fluxDypeScale: 2,
  fluxDypeExponent: 2,
  zImageShift: null,
  zImageSeedVarianceEnabled: false,
  zImageSeedVarianceStrength: 0.1,
  zImageSeedVarianceRandomizePercent: 50,
  qwenImageShift: null,
  qwenImageEncoderQuantization: 'none',
  mossImageUncondSkipLayers: [22],
  referenceImages: [],
  scheduler: 'euler_a',
  seamlessXAxis: false,
  seamlessYAxis: false,
  seed: 123,
  seedMode: 'fixed',
  steps: 30,
  t5EncoderModel: null,
  vae: null,
  vaePrecision: 'fp32',
  width: 1024,
  ...overrides,
});

const createImage = (imageName: string, sourceQueueItemId: string): GeneratedImageContract => ({
  height: 768,
  imageName,
  imageUrl: `/api/v1/images/i/${imageName}/full`,
  queuedAt: '2026-06-09T00:00:00.000Z',
  sourceQueueItemId,
  thumbnailUrl: `/api/v1/images/i/${imageName}/thumbnail`,
  width: 512,
});

const createVideo = (videoName: string, sourceQueueItemId: string, boardId?: string): GeneratedVideoContract => ({
  ...(boardId === undefined ? {} : { boardId }),
  category: 'general',
  durationSeconds: 5,
  height: 768,
  isIntermediate: false,
  queuedAt: '2026-06-09T00:00:00.000Z',
  sourceQueueItemId,
  thumbnailUrl: `/api/v1/videos/i/${videoName}/thumbnail`,
  videoName,
  videoUrl: `/api/v1/videos/i/${videoName}/full`,
  width: 512,
});

const createGalleryImageItem = (name: string, boardId = 'none'): GalleryImageItem => ({
  boardId,
  category: 'general',
  createdAt: '2026-07-30T00:00:00.000Z',
  fullUrl: `/api/v1/images/i/${name}/full`,
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name,
  sourceQueueItemId: 'backend-gallery',
  starred: false,
  thumbnailUrl: `/api/v1/images/i/${name}/thumbnail`,
  width: 512,
});

const createGalleryVideoItem = (name: string, boardId = 'none'): GalleryVideoItem => ({
  ...createGalleryImageItem(name, boardId),
  durationSeconds: 3,
  fullUrl: `/api/v1/videos/i/${name}/full`,
  kind: 'video',
  thumbnailUrl: `/api/v1/videos/i/${name}/thumbnail`,
});

const galleryItemToRecentImage = (item: GalleryImageItem): GeneratedImageContract & { boardId: string } => ({
  boardId: item.boardId,
  height: item.height,
  imageName: item.name,
  imageUrl: item.fullUrl,
  queuedAt: item.createdAt,
  sourceQueueItemId: item.sourceQueueItemId ?? 'backend-gallery',
  thumbnailUrl: item.thumbnailUrl,
  width: item.width,
});

const createStagingCandidate = (
  imageName: string,
  sourceQueueItemId: string,
  placement: CanvasStagingCandidateContract['placement']
): CanvasStagingCandidateContract => ({
  ...createImage(imageName, sourceQueueItemId),
  placement,
});

const getProject = (state: WorkbenchState, projectId: string): Project => {
  const project = state.projects.find((candidate) => candidate.id === projectId);

  expect(project).toBeDefined();

  return project as Project;
};

const getActiveProject = (state: WorkbenchState): Project => getProject(state, state.activeProjectId);

type CanvasLayer = CanvasLayerContract;

const getRasterLayerImageName = (layer: CanvasLayer | undefined): string | undefined =>
  layer?.type === 'raster' && layer.source.type === 'image' ? layer.source.image.imageName : undefined;

const getRasterLayerPlacement = (layer: CanvasLayer | undefined) => {
  if (!layer || layer.type !== 'raster' || layer.source.type !== 'image') {
    return undefined;
  }

  const { image } = layer.source;

  return {
    height: image.height * layer.transform.scaleY,
    opacity: layer.opacity,
    width: image.width * layer.transform.scaleX,
    x: layer.transform.x,
    y: layer.transform.y,
  };
};

const createRasterLayer = (id: string, imageName = `${id}.png`): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 64, imageName, width: 64 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const createControlLayer = (id: string): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: false,
});

const withEmptyCanvas = (state: WorkbenchState): WorkbenchState =>
  workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });

/** Adds layers top-to-bottom in array order (each `addCanvasLayer` inserts at index 0). */
const withCanvasLayers = (state: WorkbenchState, layers: readonly CanvasLayerContract[]): WorkbenchState =>
  workbenchReducer(state, {
    document: {
      ...createEmptyCanvasDocument(),
      stacks: stacksFrom([...layers]),
      selectedLayerId: layers[0]?.id ?? null,
    },
    type: 'replaceCanvasDocument',
  });

const getCanvas = (state: WorkbenchState) => getActiveProject(state).canvas;

const getLayerIds = (state: WorkbenchState): string[] =>
  getDocumentLeaves(getCanvas(state).document).map((layer) => layer.id);

const withInvocationDestination = (state: WorkbenchState, destination: 'canvas' | 'gallery'): WorkbenchState => ({
  ...state,
  projects: state.projects.map((project) =>
    project.id === state.activeProjectId ? { ...project, invocation: { ...project.invocation, destination } } : project
  ),
});

const primeGenerate = (
  state = withInvocationDestination(createInitialWorkbenchState(), 'canvas'),
  overrides: Partial<GenerateWidgetValues> = {}
) => workbenchReducer(state, { type: 'setGenerateSettings', values: createGenerateValues(overrides) });

const submitGenerate = (state: WorkbenchState) =>
  workbenchReducer(state, {
    backendSupportsCancellation: true,
    projectId: state.activeProjectId,
    route: getActiveProject(state).invocation,
    type: 'submitResolvedInvocationSnapshot',
  });

describe('Invoke Front', () => {
  const getQueuedSnapshot = (state: WorkbenchState) => getActiveProject(state).queue.items[0]?.snapshot;

  it('records front priority on resolved submissions only when the command asks for it', () => {
    const submitResolved = (prepend?: boolean) => {
      const state = primeGenerate();

      return workbenchReducer(state, {
        backendSupportsCancellation: true,
        ...(prepend === undefined ? {} : { prepend }),
        projectId: state.activeProjectId,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        type: 'submitResolvedInvocationSnapshot',
      });
    };

    expect(getQueuedSnapshot(submitResolved())).not.toHaveProperty('prepend');
    expect(getQueuedSnapshot(submitResolved(true))?.prepend).toBe(true);
  });
});

const getQueuedOutputNode = (state: WorkbenchState) => {
  const submission = getActiveProject(state).queue.items[0]?.snapshot.backendSubmission;
  const nodes = submission?.kind === 'invalid' ? {} : (submission?.graph.nodes ?? {});

  return nodes.canvas_output;
};

const getQueuedRandDevice = (state: WorkbenchState): unknown => {
  const submission = getActiveProject(state).queue.items[0]?.snapshot.backendSubmission;
  const nodes = submission?.kind === 'invalid' ? {} : (submission?.graph.nodes ?? {});
  const metadata = Object.values(nodes).find((node) => node.type === 'core_metadata');

  return metadata?.rand_device;
};

const createUpscaleModel = (key: string, type: string, base: string, name = key): ModelConfig => ({
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

describe('explicit submission runtime inputs', () => {
  it('keeps graph, device, destination and seed planning independent of later cache changes', () => {
    let state = primeGenerate(undefined, { batchCount: 3, seed: 42, seedMode: 'increment' });
    state = workbenchReducer(state, { settings: { useCpuNoise: false }, type: 'setActiveProjectSettings' });
    state = workbenchReducer(state, { settings: { autoAddBoardId: 'captured-board' }, type: 'updateGallerySettings' });
    listGalleryBoardsInCache('captured-board');
    generationDeviceMock.options = [{ device: 'xpu:0', name: 'Arc' }];
    const runtime = captureInvocationRuntimeInputs('generate');
    const action = {
      backendSupportsCancellation: true,
      projectId: state.activeProjectId,
      route: getActiveProject(state).invocation,
      runtime,
      type: 'submitResolvedInvocationSnapshot' as const,
    };
    const time = vi.spyOn(Date, 'now').mockReturnValue(1_000);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.25);
    let before: WorkbenchState;
    let after: WorkbenchState;
    try {
      before = workbenchReducer(state, action);
      generationDeviceMock.options = [{ device: 'cuda:0', name: 'RTX' }];
      listGalleryBoardsInCache('different-board');
      after = workbenchReducer(state, action);
    } finally {
      time.mockRestore();
      random.mockRestore();
    }

    expect(getQueuedRandDevice(after)).toBe('xpu');
    const first = getActiveProject(before).queue.items[0]!.snapshot;
    const second = getActiveProject(after).queue.items[0]!.snapshot;
    expect(second.galleryBoardId).toBe('captured-board');
    expect(second.backendSubmission).toEqual(first.backendSubmission);
    expect(second.backendSubmission).toMatchObject({ kind: 'generate', seed: 42, seedStep: 1, batchCount: 3 });
    expect(getProjectWidgetValues(getActiveProject(after), 'generate').seed).toBe(45);
    expect(getActiveProject(after).queue.items).toHaveLength(1);

    const refreshed = workbenchReducer(after, { ...action, runtime: captureInvocationRuntimeInputs('generate') });
    expect(getQueuedRandDevice(refreshed)).toBe('cuda');
    expect(getActiveProject(refreshed).queue.items[0]!.snapshot.galleryBoardId).toBe('none');
    expect(getProjectWidgetValues(getActiveProject(refreshed), 'generate').seed).toBe(48);
  });
});

describe('project output filters', () => {
  it('queues the NSFW checker as the generate output when the project turns it on', () => {
    const initial = workbenchReducer(createInitialWorkbenchState(), {
      settings: { useNsfwChecker: true },
      type: 'setActiveProjectSettings',
    });
    const state = workbenchReducer(primeGenerate(initial), {
      backendSupportsCancellation: true,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
      projectId: initial.activeProjectId,
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getQueuedOutputNode(state)).toMatchObject({ is_intermediate: false, type: 'img_nsfw' });
  });
});

describe("the user's defaults for the generation flags", () => {
  const getQueuedNodeTypes = (state: WorkbenchState): string[] => {
    const submission = getActiveProject(state).queue.items[0]?.snapshot.backendSubmission;
    const nodes = submission?.kind === 'invalid' ? {} : (submission?.graph.nodes ?? {});
    return Object.values(nodes).map((node) => node.type);
  };
  const submitWithDefaults = (state: WorkbenchState) =>
    workbenchReducer(
      {
        ...primeGenerate(state),
        scopedDefaults: { generate: { useCpuNoise: false, useNsfwChecker: true, useWatermark: true } },
      },
      {
        backendSupportsCancellation: true,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        projectId: state.activeProjectId,
        type: 'submitResolvedInvocationSnapshot',
      }
    );

  it('filters and stamps a project that pins nothing, and draws its noise as the default says', () => {
    const state = submitWithDefaults(createInitialWorkbenchState());

    expect(getQueuedNodeTypes(state)).toEqual(expect.arrayContaining(['img_nsfw', 'img_watermark']));
    expect(getQueuedRandDevice(state)).toBe('cuda');
  });
});

describe('generation-device orchestration metadata', () => {
  it('passes the XPU runtime snapshot through the generate submission boundary', () => {
    generationDeviceMock.options = [{ device: 'xpu:0', name: 'Intel Arc' }];
    const initial = workbenchReducer(createInitialWorkbenchState(), {
      settings: { useCpuNoise: false },
      type: 'setActiveProjectSettings',
    });

    const state = workbenchReducer(primeGenerate(initial), {
      backendSupportsCancellation: true,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
      projectId: initial.activeProjectId,
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getQueuedRandDevice(state)).toBe('xpu');
  });
});

describe('model-only upscale submissions', () => {
  const upscaler = {
    ...createUpscaleModel('ultrasharp', 'spandrel_image_to_image', 'any', '4x-UltraSharp'),
    scale: 4,
    type: 'spandrel_image_to_image' as const,
  };
  const input = { height: 64, image_name: 'input.png', width: 48 };
  const sourceMetadata = { height: 512, positive_prompt: 'a lighthouse', seed: 42, width: 384 };
  const getQueued = (state: WorkbenchState) => {
    const snapshot = getActiveProject(state).queue.items[0]?.snapshot;
    const submission = snapshot?.backendSubmission;

    return { nodes: submission?.kind === 'invalid' ? {} : (submission?.graph.nodes ?? {}), snapshot, submission };
  };
  const primeModelOnly = () => {
    let state = workbenchReducer(createInitialWorkbenchState(), {
      type: 'patchWidgetValues',
      values: {
        ...createDefaultUpscaleWidgetValues([upscaler]),
        batchCount: 3,
        inputImage: input,
        mode: 'model-only',
        seed: 10,
        seedMode: 'increment',
      },
      widgetId: 'upscale',
    });

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { positivePrompt: 'the Generate draft' },
      widgetId: 'generate',
    });

    return state;
  };
  const submitRoute = (state: WorkbenchState, imageName = input.image_name) =>
    workbenchReducer(state, {
      backendSupportsCancellation: true,
      models: [upscaler],
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'upscale', sourceLocked: false },
      projectId: state.activeProjectId,
      type: 'submitResolvedInvocationSnapshot',
      upscaleSourceMetadata: { imageName, metadata: sourceMetadata },
    });

  it('queues the upscaler alone, once, with the input metadata, no main model, and leaves the Refine seed', () => {
    const state = submitRoute(primeModelOnly());
    const { nodes, snapshot, submission } = getQueued(state);

    expect(Object.values(nodes).map((node) => node.type)).toEqual([
      'string',
      'string',
      'integer',
      'spandrel_image_to_image_autoscale',
    ]);
    expect(nodes.upscale_output).toMatchObject({
      is_intermediate: false,
      metadata: { ...sourceMetadata, upscale_scale: 4 },
      scale: 4,
    });
    expect(submission).toMatchObject({
      batchCount: 1,
      kind: 'generate',
      negativePrompt: '',
      positivePrompt: '',
      seedStep: 0,
    });
    expect(snapshot?.presentation).toEqual({ batchCount: 1, height: 256, width: 192 });
    expect(getProjectWidgetValues(getActiveProject(state), 'upscale').seed).toBe(10);
  });

  it("queues the image menu's upscale to the gallery without touching the widget or the route", () => {
    listGalleryBoardsInCache('auto-board');
    let state = workbenchReducer(primeModelOnly(), {
      settings: { autoAddBoardId: 'auto-board' },
      type: 'updateGallerySettings',
    });
    state = workbenchReducer(state, { destination: 'canvas', type: 'setInvocationDestination' });
    const before = getActiveProject(state);

    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      // Odd sizes: 4x gives 404 x 324, which the result floors to multiples of 8 as the Upscale widget's does.
      image: { height: 101, image_name: 'menu.png', width: 81 },
      projectId: state.activeProjectId,
      scale: 'native',
      sourceMetadata,
      type: 'submitModelOnlyUpscale',
      upscaleModel: upscaler,
    });

    const after = getActiveProject(state);
    const { nodes, snapshot, submission } = getQueued(state);

    expect(after.queue.items).toHaveLength(1);
    expect(snapshot).toMatchObject({ destination: 'gallery', galleryBoardId: 'auto-board', sourceId: 'upscale' });
    expect(snapshot?.presentation).toEqual({ batchCount: 1, height: 400, width: 320 });
    expect(submission).toMatchObject({ batchCount: 1, kind: 'generate', positivePrompt: '' });
    expect(nodes.upscale_output).toMatchObject({
      fit_to_multiple_of_8: true,
      image: { image_name: 'menu.png' },
      metadata: {
        ...sourceMetadata,
        upscale_initial_image: { height: 101, image_name: 'menu.png', width: 81 },
        upscale_scale: 4,
      },
      scale: 4,
      type: 'spandrel_image_to_image_autoscale',
    });
    expect(after.invocation).toEqual({ ...before.invocation, lastSubmittedRunId: after.queue.items[0]?.id });
    expect(after.widgetInstances).toEqual(before.widgetInstances);
    expect(after.widgetGraphs).toEqual(before.widgetGraphs);
  });
});

describe('stored layouts from before the split', () => {
  const legacyAccount = { activeLayoutPresetId: 'compose' } satisfies WorkbenchState['account'];
  const hydrate = (state: WorkbenchState) =>
    workbenchReducer(createInitialWorkbenchState(), { state, type: 'hydrateWorkbench' });
  const sessionOf = (projects: Project[], account: WorkbenchState['account'] = legacyAccount): WorkbenchState => ({
    ...createInitialWorkbenchState(),
    account,
    activeProjectId: projects[0]!.id,
    projects,
  });
  const storedProject = (id: string, stored: StoredProjectLayout, storedAt?: string): Project =>
    asStoredBeforeSplit({ ...createInitialWorkbenchState().projects[0]!, id }, { ...stored, storedAt });
  const arrangementNotices = (state: WorkbenchState) =>
    state.notifications.filter(({ titleKey }) => titleKey === 'shell.workspace.arrangementDroppedTitle');

  it('shows what a project showed when it changed only how it looked, with nothing to tell', () => {
    for (const [name, stored] of Object.entries(viewOnlyLayouts)) {
      const state = hydrate(sessionOf([storedProject('viewed', stored)], createInitialWorkbenchState().account));
      const project = getActiveProject(state);

      expect(project.legacyLayout, name).toBeUndefined();
      expect(state.account.layoutPresetOverrides, name).toEqual({});
      expect(arrangementNotices(state), name).toEqual([]);
      expect(Object.keys(project.views).length, name).toBeGreaterThan(0);
    }
  });

  it('adopts the opening project’s arrangement as its workspace once, and drops the others’ with one notice each', () => {
    const adopting = storedProject('adopting', structuralLayouts['removed a widget'], '2026-01-01T00:00:00Z');
    const other = storedProject('other', structuralLayouts['moved a tab to another rail'], '2026-02-01T00:00:00Z');
    const state = hydrate(sessionOf([adopting, other]));

    expect(state.account.workspaceLayoutsAdopted).toBe(true);
    expect(getWorkspaceStructure(state.account, 'compose').widgetRegions.right.instanceIds).toEqual([
      'gallery',
      'queue',
    ]);
    expect(getActiveProject(state).views).toEqual({});
    expect(getLayoutOf(state).widgetRegions.right.instanceIds).toEqual(['gallery', 'queue']);
    expect(arrangementNotices(state)).toEqual([]);
    // The other project shows the adopted workspace, and is told once it is looked at.
    expect(getLayoutOf(state, getProject(state, 'other')).widgetRegions.left.instanceIds).toEqual([
      'generate',
      'upscale',
    ]);

    const switched = workbenchReducer(state, { projectId: 'other', type: 'switchProject' });

    expect(arrangementNotices(switched)).toMatchObject([
      { actions: [{ kind: 'customize-workspace' }], messageValues: { workspaces: 'Compose' }, projectId: 'other' },
    ]);
    // Shown in the reader's language, workspace names included.
    const german = ((key: string, values?: { workspaces?: string }) =>
      key === 'topbar.presets.builtInLabels.compose' ? 'Komponieren' : `${key}: ${values?.workspaces}`) as TFunction;

    expect(resolveNotificationText(arrangementNotices(switched)[0]!, german).message).toBe(
      'shell.workspace.arrangementDroppedMessage: Komponieren'
    );
    expect(
      arrangementNotices(
        workbenchReducer(workbenchReducer(switched, { projectId: 'adopting', type: 'switchProject' }), {
          projectId: 'other',
          type: 'switchProject',
        })
      )
    ).toHaveLength(1);
    // Hydrating again adopts nothing more: the account is marked.
    const again = hydrate({
      ...state,
      projects: [storedProject('late', structuralLayouts['floated a widget'], '2027-01-01T00:00:00Z')],
    });

    expect(again.account.layoutPresetOverrides).toEqual(state.account.layoutPresetOverrides);
  });

  it('tells a project only about arrangements of its own, and keeps telling it until it is active', () => {
    const adopting = storedProject('adopting', structuralLayouts['removed a widget'], '2026-02-01T00:00:00Z');
    const untouched = storedProject('untouched', viewOnlyLayouts['fronted other tabs'], '2026-01-01T00:00:00Z');
    const own = storedProject('own', structuralLayouts['moved a tab to another rail'], '2026-01-01T00:00:00Z');
    const state = hydrate(sessionOf([adopting, untouched, own]));

    // Its Compose matched Compose as it was, so the adopted one replaced nothing of its own.
    expect(getProject(state, 'untouched').droppedArrangements).toBeUndefined();
    expect(getProject(state, 'own').droppedArrangements).toEqual(['compose']);

    // Stored with the project, the notice survives a reload until the project is next active.
    const reloaded = hydrate({ ...state, projects: state.projects });

    expect(getProject(reloaded, 'own').droppedArrangements).toEqual(['compose']);

    const told = workbenchReducer(reloaded, { projectId: 'own', type: 'switchProject' });

    expect(arrangementNotices(told)).toMatchObject([
      { actions: [{ kind: 'customize-workspace', workspaceId: 'compose' }], projectId: 'own' },
    ]);
    expect(getProject(told, 'own').droppedArrangements).toBeUndefined();
  });

  it('holds what it migrated in the stored shape until the account that adopted is saved', () => {
    const state = hydrate(
      sessionOf([storedProject('adopting', structuralLayouts['removed a widget'], '2026-02-01T00:00:00Z')])
    );
    const document = (current: WorkbenchState) =>
      serializeProjectDocumentV3(getActiveProject(current)) as unknown as Record<string, unknown>;

    // Written back as stored, a load that never saved the account migrates and adopts the same way again.
    expect(document(state)).toMatchObject({ layout: { presetId: 'compose' } });
    expect(document(state)).not.toHaveProperty('workspaceId');
    const reloaded = deserializeProjectDocument(document(state));

    expect(reloaded.status === 'loaded' && hydrate(sessionOf([reloaded.project])).account).toEqual(state.account);

    // A project opened before that save migrated against the same unsaved adoption: it is held too.
    const opened = workbenchReducer(state, {
      project: storedProject('arriving', viewOnlyLayouts['fronted other tabs']),
      type: 'openProject',
    });

    expect(document(opened)).not.toHaveProperty('workspaceId');

    const released = workbenchReducer(opened, { type: 'releaseHeldLayouts' });

    expect(document(released)).toMatchObject({ workspaceId: 'compose' });
    expect(document(released)).not.toHaveProperty('layout');
    expect(document({ ...released, activeProjectId: getProject(released, 'adopting').id })).toMatchObject({
      views: {},
      workspaceId: 'compose',
    });
    // Once nothing is held, an arrival writes the new shape at once.
    expect(
      document(
        workbenchReducer(released, {
          project: storedProject('later', viewOnlyLayouts['fronted other tabs']),
          type: 'openProject',
        })
      )
    ).toHaveProperty('workspaceId', 'compose');
  });

  it('adopts the most recently stored project when the one opening has nothing stored', () => {
    const fresh = { ...createInitialWorkbenchState().projects[0]!, id: 'fresh' };
    const older = storedProject('older', structuralLayouts['removed a widget'], '2026-01-01T00:00:00Z');
    const newer = storedProject('newer', structuralLayouts['moved a tab to another rail'], '2026-03-01T00:00:00Z');
    const state = hydrate(sessionOf([fresh, older, newer]));

    expect(getWorkspaceStructure(state.account, 'compose').widgetRegions.left.instanceIds).toEqual([
      'generate',
      'image-map',
      'upscale',
    ]);
    expect(getProject(state, 'older').droppedArrangements).toEqual(['compose']);
  });

  it('never adopts into a fresh account, and drops what a project opened later stored', () => {
    const fresh = hydrate(
      sessionOf([storedProject('stored', structuralLayouts['removed a widget'])], createInitialWorkbenchState().account)
    );

    expect(fresh.account.layoutPresetOverrides).toEqual({});
    expect(arrangementNotices(fresh)).toHaveLength(1);

    const opened = workbenchReducer(fresh, {
      project: storedProject('later', structuralLayouts['floated a widget']),
      type: 'openProject',
    });

    expect(opened.account).toBe(fresh.account);
    expect(getLayoutOf(opened).floatingWidgets).toBeUndefined();
    expect(getActiveProject(opened).views.compose).toEqual({ fronts: { right: 'image-map' }, sizes: { left: 600 } });
    expect(arrangementNotices(opened)).toHaveLength(2);
  });

  it('builds the draft of an empty session from the cached account, not the shipped defaults', () => {
    const initial = createInitialWorkbenchState();
    const compose = layoutPresets[0]!.snapshot;
    const customized = {
      ...compose,
      widgetRegions: { ...compose.widgetRegions, left: { ...compose.widgetRegions.left, instanceIds: ['generate'] } },
    };
    const hydrated = hydrate({
      ...initial,
      account: { ...initial.account, layoutPresetOverrides: { compose: customized } },
      activeProjectId: '',
      projects: [],
    });

    expect(getLayoutOf(hydrated).widgetRegions.left.instanceIds).toEqual(['generate']);
  });

  it('repairs an active project id that names no hydrated project', () => {
    const initial = createInitialWorkbenchState();
    const hydrated = hydrate({ ...initial, activeProjectId: 'project-that-was-refused' });

    expect(hydrated.activeProjectId).toBe(hydrated.projects[0]?.id);
  });

  it('brings a historical default rail up to date, so the project changed nothing', () => {
    const stored = storedProject(
      'legacy-rail',
      storedOn('compose', { regions: { right: { instanceIds: ['queue', 'gallery', 'layers'] } } })
    );
    const state = hydrate(sessionOf([stored], createInitialWorkbenchState().account));

    expect(getLayoutOf(state).widgetRegions.right.instanceIds).toEqual(['gallery', 'image-map', 'queue']);
    expect(arrangementNotices(state)).toEqual([]);
  });

  it('migrates a legacy Upscale prompt only when Generate has no prompt content', () => {
    const initial = createInitialWorkbenchState();
    const withPrompts = (generatePrompt: string): WorkbenchState => ({
      ...initial,
      projects: initial.projects.map((project) => {
        const generate = project.widgetInstances.generate!;
        const upscale = project.widgetInstances.upscale!;

        return {
          ...project,
          widgetInstances: {
            ...project.widgetInstances,
            generate: {
              ...generate,
              state: { ...generate.state, values: { ...generate.state.values, positivePrompt: generatePrompt } },
            },
            upscale: {
              ...upscale,
              state: {
                ...upscale.state,
                values: {
                  ...upscale.state.values,
                  negativePrompt: 'legacy negative',
                  negativePromptEnabled: false,
                  positivePrompt: 'legacy positive',
                },
              },
            },
          },
        };
      }),
    });

    const migrated = workbenchReducer(initial, { state: withPrompts(''), type: 'hydrateWorkbench' });
    const preserved = workbenchReducer(initial, { state: withPrompts('generate positive'), type: 'hydrateWorkbench' });

    expect(getProjectWidgetValues(getActiveProject(migrated), 'generate')).toMatchObject({
      negativePrompt: 'legacy negative',
      negativePromptEnabled: false,
      positivePrompt: 'legacy positive',
    });
    expect(getProjectWidgetValues(getActiveProject(migrated), 'upscale')).toMatchObject({
      negativePrompt: '',
      negativePromptEnabled: true,
      positivePrompt: '',
    });
    expect(getProjectWidgetValues(getActiveProject(preserved), 'generate')).toMatchObject({
      positivePrompt: 'generate positive',
    });
  });
});

describe('a project’s view of its workspace', () => {
  it('remembers fronts, sizes and collapsed panels per project and per workspace, never for another project', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    const firstId = state.activeProjectId;

    state = workbenchReducer(state, { type: 'createProject' });
    const secondId = state.activeProjectId;

    state = workbenchReducer(state, { region: 'left', sizePx: 500, type: 'setRegionWidgetSize' });
    state = workbenchReducer(state, { isCollapsed: true, region: 'right', type: 'setRegionWidgetCollapsed' });
    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: 'upscale' });

    expect(getProject(state, secondId).views).toEqual({
      compose: { collapsed: { right: true }, fronts: { left: 'upscale' }, sizes: { left: 500 } },
    });
    expect(getProject(state, firstId).views).toEqual({});
    expect(state.account.layoutPresetOverrides).toBeUndefined();

    state = workbenchReducer(state, { presetId: 'edit', type: 'applyPreset' });
    expect(getLayoutOf(state).widgetRegions.left).toMatchObject({ activeInstanceId: 'generate', sizePx: 450 });

    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    expect(getLayoutOf(state).widgetRegions.left).toMatchObject({ sizePx: 500 });
    expect(getLayoutOf(state).widgetRegions.right.isCollapsed).toBe(true);
  });

  it('reopens a collapsed region at the size it was collapsed from', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { region: 'left', sizePx: 500, type: 'setRegionWidgetSize' });

    const activeInstanceId = getLayoutOf(state).widgetRegions.left.activeInstanceId;

    state = workbenchReducer(state, { isCollapsed: true, region: 'left', type: 'setRegionWidgetCollapsed' });
    expect(getLayoutOf(state).widgetRegions.left).toMatchObject({ isCollapsed: true, sizePx: 500 });

    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: activeInstanceId });
    expect(getLayoutOf(state).widgetRegions.left).toMatchObject({ isCollapsed: false, sizePx: 500 });
  });
});

describe('rearranging without Customize', () => {
  it('undoes a workspace step only while the account still holds what it left, and never onto a deleted one', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    const firstId = state.activeProjectId;

    state = workbenchReducer(state, { region: 'right', type: 'toggleRegionWidget', widgetId: 'image-map' });
    state = workbenchReducer(state, { type: 'createProject' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    state = workbenchReducer(state, { region: 'right', type: 'toggleRegionWidget', widgetId: 'queue' });
    state = workbenchReducer(state, { projectId: firstId, type: 'switchProject' });

    // The other project's later change stays for everyone; this project is told why its Undo did not reach it.
    const undone = workbenchReducer(state, { type: 'undoProjectChange' });

    expect(getWorkspaceStructure(undone.account, 'compose').widgetRegions.right.instanceIds).toEqual(['gallery']);
    expect(undone.notifications[0]).toMatchObject({ titleKey: 'shell.layoutUndo.workspaceChangedTitle' });

    let custom = addCustomWorkspace(createInitialWorkbenchState(), { id: 'mine', label: 'Mine' });

    custom = workbenchReducer(custom, { presetId: 'edit', type: 'applyPreset' });
    custom = workbenchReducer(custom, { presetId: 'mine', type: 'deleteLayoutPreset' });

    expect(getActiveProject(workbenchReducer(custom, { type: 'undoProjectChange' })).workspaceId).toBe('edit');
  });

  it('writes a structural gesture to the workspace for every project on it, and nothing else', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    const firstId = state.activeProjectId;

    state = workbenchReducer(state, { type: 'createProject' });
    state = workbenchReducer(state, { presetId: 'edit', type: 'applyPreset' });
    const onEditId = state.activeProjectId;

    state = workbenchReducer(state, { projectId: firstId, type: 'switchProject' });
    state = workbenchReducer(state, { region: 'right', type: 'toggleRegionWidget', widgetId: 'image-map' });

    expect(getWorkspaceStructure(state.account, 'compose').widgetRegions.right.instanceIds).toEqual([
      'gallery',
      'queue',
    ]);
    expect(state.account.layoutPresetOverrides?.edit).toBeUndefined();
    expect(getLayoutOf(state, getProject(state, onEditId)).widgetRegions.right.instanceIds).toEqual([
      'layers',
      'preview',
    ]);

    state = workbenchReducer(state, { type: 'createProject' });
    expect(getLayoutOf(state).widgetRegions.right.instanceIds).toEqual(['gallery', 'queue']);
  });

  it('undoes a structural gesture by putting the workspace back in the account', () => {
    const before = workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });
    const moved = workbenchReducer(before, {
      fromRegion: 'right',
      instanceId: 'image-map',
      toIndex: 0,
      toRegion: 'left',
      type: 'moveWidgetInstance',
    });

    expect(getLayoutOf(moved).widgetRegions.left).toMatchObject({ activeInstanceId: 'image-map' });

    const undone = workbenchReducer(moved, { type: 'undoProjectChange' });

    // Back at the shipped arrangement, the built-in follows what later releases ship again.
    expect(undone.account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(getLayoutOf(undone).widgetRegions).toEqual(getLayoutOf(before).widgetRegions);

    const redone = workbenchReducer(undone, { type: 'redoProjectChange' });

    expect(getWorkspaceStructure(redone.account, 'compose').widgetRegions.left.instanceIds).toContain('image-map');
  });

  it('opens a widget in a collapsed status bar, expanding it', () => {
    const state = workbenchReducer(createInitialWorkbenchState(), {
      region: 'bottom',
      type: 'openRegionWidget',
      widgetId: 'workflow',
    });
    const bottom = getLayoutOf(state).widgetRegions.bottom;

    expect(bottom).toMatchObject({ isCollapsed: false });
    expect(getActiveProject(state).widgetInstances[bottom.activeInstanceId]?.typeId).toBe('workflow');
    expect(bottom.instanceIds).toContain(bottom.activeInstanceId);
  });

  it('shows an instance opened in the center once, the rail holding it fronting its neighbour', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { region: 'right', type: 'selectRegionWidget', widgetId: 'preview' });
    state = workbenchReducer(state, { region: 'center', type: 'openRegionWidget', widgetId: 'preview' });
    const layout = getLayoutOf(state);

    expect(layout.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['canvas', 'preview'],
    });
    expect(layout.widgetRegions.right).toMatchObject({ activeInstanceId: 'layers', isCollapsed: false });
    expect(
      Object.values(getActiveProject(state).widgetInstances).filter(({ typeId }) => typeId === 'preview')
    ).toHaveLength(1);
  });
});

describe('fronting a placed widget', () => {
  const edit = () => {
    const state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });

    return workbenchReducer(state, { region: 'right', type: 'selectRegionWidget', widgetId: 'preview' });
  };

  it('never adds: an instance its region does not hold changes nothing', () => {
    const before = edit();

    expect(workbenchReducer(before, { instanceId: 'queue', region: 'right', type: 'frontRegionWidget' })).toBe(before);
  });

  it('expands a collapsed panel to front its member', () => {
    const collapsed = workbenchReducer(edit(), { isCollapsed: true, region: 'left', type: 'setRegionWidgetCollapsed' });
    const state = workbenchReducer(collapsed, { instanceId: 'upscale', region: 'left', type: 'frontRegionWidget' });

    expect(getLayoutOf(state).widgetRegions.left).toMatchObject({
      activeInstanceId: 'upscale',
      instanceIds: ['generate', 'upscale'],
      isCollapsed: false,
    });
    expect(state.account).toBe(collapsed.account);
  });

  it('fronts a center alternate while the rail showing the same instance fronts its neighbour', () => {
    const layout = getLayoutOf(
      workbenchReducer(edit(), { instanceId: 'preview', region: 'center', type: 'frontRegionWidget' })
    );

    expect(layout.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['canvas', 'preview'],
    });
    expect(layout.widgetRegions.right).toMatchObject({
      activeInstanceId: 'layers',
      instanceIds: ['layers', 'preview'],
    });
  });
});

describe('adopting a project from another realm', () => {
  const galleryProject = (values: Record<string, unknown>) => {
    const project = createInitialWorkbenchState().projects[0]!;
    const galleryInstance = Object.values(project.widgetInstances).find((instance) => instance.typeId === 'gallery')!;
    const galleryInstanceId = Object.keys(project.widgetInstances).find(
      (instanceId) => project.widgetInstances[instanceId]?.typeId === 'gallery'
    )!;

    return normalizeWorkbenchProject({
      ...project,
      widgetInstances: {
        ...project.widgetInstances,
        [galleryInstanceId]: {
          ...galleryInstance,
          state: { ...galleryInstance.state, values: { ...galleryInstance.state.values, ...values } },
        },
      },
    });
  };
  const galleryValuesOf = (project: Project) =>
    Object.values(project.widgetInstances).find((instance) => instance.typeId === 'gallery')!.state.values;

  it('drops legacy graph history and session events at ingestion', () => {
    const project = createInitialWorkbenchState().projects[0]!;
    const legacyProject = {
      ...project,
      events: [{ createdAt: 'now', id: 'legacy-event', summary: 'legacy', type: 'project-created' }],
      graphHistory: [{ document: getActiveProjectGraph(project), id: 'legacy-snapshot' }],
      queue: { items: [{}] },
    } as unknown as Project;

    const normalized = normalizeWorkbenchProject(legacyProject);

    expect(normalized.events).toEqual([]);
    expect('graphHistory' in normalized).toBe(false);
    expect(normalized.queue.items).toEqual([]);
  });

  it('loads a project saved before Auto-switch with it Off and without the retired pause stamp', () => {
    const project = createInitialWorkbenchState().projects[0]!;
    const { viewerAutoSwitch: _mode, ...settings } = project.settings;
    const galleryInstance = Object.values(project.widgetInstances).find((instance) => instance.typeId === 'gallery')!;
    const legacyProject = {
      ...project,
      settings: { ...settings, showProgressImagesInViewer: true },
      widgetInstances: {
        ...project.widgetInstances,
        [galleryInstance.id]: {
          ...galleryInstance,
          state: {
            ...galleryInstance.state,
            values: { ...galleryInstance.state.values, liveFollowPausedAt: '2026-06-10T00:00:00.000Z' },
          },
        },
      },
    } as unknown as Project;

    const normalized = normalizeWorkbenchProject(legacyProject);

    expect(normalized.settings.viewerAutoSwitch).toBe('off');
    expect(normalized.settings).not.toHaveProperty('showProgressImagesInViewer');
    expect(getProjectWidgetValues(normalized, 'gallery')).not.toHaveProperty('liveFollowPausedAt');
  });

  it('carries the iteration count workflow runs borrowed from Generate over to the workflow widget once', () => {
    const workflowValuesOf = (candidate: Project) => getProjectWidgetValues(candidate, 'workflow');
    const withoutWorkflowCount = (candidate: Project): Project => {
      const { batchCount: _, ...values } = workflowValuesOf(candidate);
      const instance = candidate.widgetInstances.workflow!;

      return {
        ...candidate,
        widgetInstances: {
          ...candidate.widgetInstances,
          workflow: { ...instance, state: { ...instance.state, values } },
        },
      };
    };
    let state = workbenchReducer(createInitialWorkbenchState(), {
      type: 'patchWidgetValues',
      values: { batchCount: 4 },
      widgetId: 'generate',
    });
    const fresh = getActiveProject(state);

    state = workbenchReducer(state, { type: 'patchWidgetValues', values: { batchCount: 2 }, widgetId: 'workflow' });
    const owned = getActiveProject(state);

    expect(workflowValuesOf(fresh).batchCount).toBe(1);
    expect(workflowValuesOf(normalizeWorkbenchProject(fresh)).batchCount).toBe(1);
    expect(workflowValuesOf(normalizeWorkbenchProject(withoutWorkflowCount(fresh))).batchCount).toBe(4);
    expect(workflowValuesOf(normalizeWorkbenchProject(owned)).batchCount).toBe(2);
  });

  it('keeps the failed submission of a run until the backend accepts it or the run ends', () => {
    let state = submitGenerate(primeGenerate());
    const projectId = state.activeProjectId;
    const queueItemId = getActiveProject(state).queue.items[0]!.id;
    const recordRetry = (current: WorkbenchState) =>
      workbenchReducer(current, {
        projectId,
        queueItemId,
        retry: {
          failedAt: 2_000,
          failureCount: 1,
          message: 'unavailable',
          reason: 'unavailable',
          serverErrorCount: 0,
          serverErrorSince: null,
          status: 503,
        },
        type: 'recordQueueItemSubmissionRetry',
      });

    state = recordRetry(state);
    expect(getActiveProject(state).queue.items[0]?.submissionRetry).toMatchObject({
      reason: 'unavailable',
      status: 503,
    });
    expect(recordRetry(state)).toBe(state);

    const accepted = workbenchReducer(state, {
      backendItemIds: [42],
      projectId,
      queueItemId,
      type: 'markQueueItemBackendSubmitted',
    });
    expect(getActiveProject(accepted).queue.items[0]).not.toHaveProperty('submissionRetry');
    expect(recordRetry(accepted)).toBe(accepted);

    const failed = workbenchReducer(state, {
      error: 'graph validation crashed',
      projectId,
      queueItemId,
      status: 'failed',
      type: 'setQueueItemStatus',
    });
    const cancelled = workbenchReducer(state, { notice: null, projectId, queueItemId, type: 'cancelQueueItem' });
    for (const [ended, status] of [
      [failed, 'failed'],
      [cancelled, 'cancelled'],
    ] as const) {
      expect(getActiveProject(ended).queue.items[0]).toMatchObject({ status });
      expect(getActiveProject(ended).queue.items[0]).not.toHaveProperty('submissionRetry');
    }
  });

  it('preserves and routes live queue work when server content replaces a project', () => {
    let state = submitGenerate(primeGenerate());
    const liveProject = getActiveProject(state);
    const queueItem = liveProject.queue.items[0]!;

    state = workbenchReducer(state, {
      backendItemIds: [42],
      projectId: liveProject.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    const liveQueue = getActiveProject(state).queue;

    state = workbenchReducer(state, {
      project: { ...liveProject, name: 'Server version', queue: { items: [] } },
      projectId: liveProject.id,
      type: 'replaceProjectFromServer',
    });
    expect(getActiveProject(state).queue).toBe(liveQueue);

    state = workbenchReducer(state, {
      images: [createImage('server-replaced-result.png', queueItem.id)],
      projectId: liveProject.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    const replacedProject = getActiveProject(state);
    expect(replacedProject.name).toBe('Server version');
    expect(replacedProject.queue.items[0]?.status).toBe('completed');
    expect(replacedProject.queue.items[0]?.resultImages?.map((image) => image.imageName)).toEqual([
      'server-replaced-result.png',
    ]);
    expect(replacedProject.canvas.stagingArea.pendingImageIds).toEqual([]);
  });

  it('drops a session-scoped search and the rank pages set against it', () => {
    const values = galleryValuesOf(
      galleryProject({
        galleryPage: 3,
        selectedImagePage: 3,
        selectedImageQuery: {
          boardId: 'none',
          galleryView: 'images',
          imageOrderDir: 'DESC',
          page: 3,
          paginationMode: 'paginated',
          searchTerm: '',
        },
        semanticImageQuery: { fileId: 'file-1', kind: 'file', label: 'dropped.png' },
      })
    );

    expect(values.semanticImageQuery).toBeNull();
    expect(values.galleryPage).toBe(0);
    expect(values.selectedImagePage).toBe(0);
    expect((values.selectedImageQuery as { page: number }).page).toBe(0);
  });

  it('retargets a project without disturbing its live session state', () => {
    const clusterId = registerImageCluster(['image:a.png', 'image:b.png'], 'beaches');
    let state = createInitialWorkbenchState();
    const project = getActiveProject(state);
    const staleCopy = project;

    state = workbenchReducer(state, {
      projectId: project.id,
      type: 'patchWidgetValues',
      values: {
        galleryPage: 7,
        paginationMode: 'infinite',
        semanticImageQuery: { clusterId, kind: 'cluster', label: 'beaches' },
      },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, {
      name: 'Renamed while copying',
      projectId: project.id,
      type: 'renameProject',
    });
    state = workbenchReducer(state, {
      boardId: 'board-copy',
      name: `${project.name} (copy)`,
      projectId: project.id,
      project: { ...staleCopy, id: `${project.id}-copy`, name: `${project.name} (copy)` },
      targetProjectId: `${project.id}-copy`,
      sourceName: project.name,
      type: 'retargetProject',
    });

    const fork = getActiveProject(state);
    const values = getProjectWidgetValues(fork, 'gallery');

    expect(fork.id).toBe(`${project.id}-copy`);
    expect(fork.name).toBe('Renamed while copying');
    expect(fork.events).toEqual(project.events);
    expect(values.galleryPage).toBe(7);
    expect(values.semanticImageQuery).toEqual({ clusterId, kind: 'cluster', label: 'beaches' });
  });
});

describe('new projects', () => {
  it('take a global UUID id, since another account may hold any other', () => {
    const initial = createInitialWorkbenchState();
    const first = workbenchReducer(initial, { type: 'createProject' });
    const second = workbenchReducer(first, { type: 'createProject' });
    const ids = second.projects.slice(-2).map((project) => project.id);

    for (const id of ids) {
      expect(id).toMatch(/^project-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    }
    expect(new Set(ids).size).toBe(2);
  });
});

describe('project settings', () => {
  it('patches an open project that is not active, leaving the active one', () => {
    const initial = createInitialWorkbenchState();
    const firstId = initial.activeProjectId;
    const state = workbenchReducer(workbenchReducer(initial, { type: 'createProject' }), {
      projectId: firstId,
      settings: { useNsfwChecker: true },
      type: 'setActiveProjectSettings',
    });
    const first = state.projects.find((project) => project.id === firstId)!;
    const active = state.projects.find((project) => project.id === state.activeProjectId)!;
    expect(state.activeProjectId).not.toBe(firstId);
    expect(first.settings.useNsfwChecker).toBe(true);
    expect(Object.hasOwn(active.settings, 'useNsfwChecker')).toBe(false);
  });
});

describe('workbench widget alignment', () => {
  it('moves a bottom widget between the strip clusters and back', () => {
    const initial = createInitialWorkbenchState();
    const instanceId = getLayoutOf(initial).widgetRegions.bottom.instanceIds[0]!;

    const alignedEnd = workbenchReducer(initial, {
      align: 'end',
      instanceId,
      region: 'bottom',
      type: 'setWidgetInstanceAlignment',
    });

    expect(getLayoutOf(alignedEnd).widgetRegions.bottom.alignEndInstanceIds).toEqual([
      ...(getLayoutOf(initial).widgetRegions.bottom.alignEndInstanceIds ?? []),
      instanceId,
    ]);
    expect(getLayoutOf(alignedEnd).widgetRegions.bottom.instanceIds).toEqual(
      getLayoutOf(initial).widgetRegions.bottom.instanceIds
    );

    const alignedStart = workbenchReducer(alignedEnd, {
      align: 'start',
      instanceId,
      region: 'bottom',
      type: 'setWidgetInstanceAlignment',
    });

    expect(getLayoutOf(alignedStart).widgetRegions.bottom.alignEndInstanceIds).toEqual(
      getLayoutOf(initial).widgetRegions.bottom.alignEndInstanceIds ?? []
    );
  });
});

describe('workbench widget state updates', () => {
  it('clears pins from the instance in the named project only, and is a no-op for keys it lacks', () => {
    let state = createInitialWorkbenchState();
    const firstProjectId = state.activeProjectId;

    state = workbenchReducer(state, {
      instanceId: 'gallery',
      type: 'patchWidgetInstanceValues',
      values: { showDateBoards: false, thumbnailFit: 'aspect' },
    });
    state = workbenchReducer(state, { type: 'createProject' });
    state = workbenchReducer(state, {
      instanceId: 'gallery',
      type: 'patchWidgetInstanceValues',
      values: { showDateBoards: false },
    });
    const secondProjectId = state.activeProjectId;

    state = workbenchReducer(state, {
      instanceId: 'gallery',
      keys: ['showDateBoards', 'showArchivedBoards'],
      projectId: firstProjectId,
      type: 'clearWidgetInstanceValues',
    });

    const firstValues = getProject(state, firstProjectId).widgetInstances.gallery?.state.values ?? {};
    expect(firstValues).not.toHaveProperty('showDateBoards');
    expect(firstValues.thumbnailFit).toBe('aspect');
    expect(getProject(state, secondProjectId).widgetInstances.gallery?.state.values.showDateBoards).toBe(false);
    expect(
      workbenchReducer(state, {
        instanceId: 'gallery',
        keys: ['showDateBoards'],
        projectId: firstProjectId,
        type: 'clearWidgetInstanceValues',
      })
    ).toBe(state);
    expect(
      workbenchReducer(state, {
        instanceId: 'gallery',
        keys: ['showDateBoards'],
        projectId: 'missing-project',
        type: 'clearWidgetInstanceValues',
      })
    ).toBe(state);
  });
});

describe('workbench layout presets', () => {
  it("starts new projects from the account's saved Compose preset", () => {
    const initial = createInitialWorkbenchState();
    const compose = layoutPresets[0]!;
    const customized: WorkbenchState = {
      ...initial,
      account: {
        ...initial.account,
        layoutPresetOverrides: {
          compose: {
            ...compose.snapshot,
            widgetRegions: {
              ...compose.snapshot.widgetRegions,
              center: { ...compose.snapshot.widgetRegions.center, activeInstanceId: 'gallery:center' },
            },
          },
        },
        layoutPresetRouteOverrides: {
          compose: { destination: 'canvas', sourceId: 'upscale' },
        },
      },
    };

    const withNewProject = workbenchReducer(customized, { type: 'createProject' });
    const project = getProject(withNewProject, withNewProject.activeProjectId);

    expect(getLayoutOf(withNewProject).widgetRegions.center.activeInstanceId).toBe('gallery:center');
    expect(project.invocation).toEqual({
      destination: 'canvas',
      destinationLocked: false,
      sourceId: 'upscale',
      sourceLocked: false,
    });
    expect(project.workspaceId).toBe('compose');
  });

  it('applies each built-in preset default route with its layout', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'generate' });

    state = workbenchReducer(state, { presetId: 'edit', type: 'applyPreset' });
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });

    state = workbenchReducer(state, { presetId: 'automate', type: 'applyPreset' });
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });
  });

  it('aims Invoke at the front source when a custom preset no longer places its explicit route’s', () => {
    let seeded = createInitialWorkbenchState();
    seeded = addCustomWorkspace(seeded, { id: 'custom-generate-only', label: 'Generate only' });
    const preset = seeded.account.customLayoutPresets?.[0];
    expect(preset).toBeDefined();
    if (!preset) {
      return;
    }

    const generateOnlyPreset = {
      ...preset,
      defaultRoute: { destination: 'gallery' as const, sourceId: 'workflow' as const },
      snapshot: {
        ...preset.snapshot,
        widgetRegions: {
          ...preset.snapshot.widgetRegions,
          center: { ...preset.snapshot.widgetRegions.center, activeInstanceId: 'preview', instanceIds: ['preview'] },
          left: { ...preset.snapshot.widgetRegions.left, activeInstanceId: 'generate', instanceIds: ['generate'] },
          right: { ...preset.snapshot.widgetRegions.right, activeInstanceId: 'gallery', instanceIds: ['gallery'] },
          bottom: {
            ...preset.snapshot.widgetRegions.bottom,
            instanceIds: preset.snapshot.widgetRegions.bottom.instanceIds.filter(
              (instanceId) => instanceId !== 'workflow:bottom'
            ),
          },
        },
      },
    };
    seeded = {
      ...seeded,
      account: { ...seeded.account, customLayoutPresets: [generateOnlyPreset] },
    };

    let state = workbenchReducer(seeded, { sourceId: 'canvas', type: 'setInvocationSource' });
    state = workbenchReducer(state, { destination: 'canvas', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { presetId: generateOnlyPreset.id, type: 'applyPreset' });

    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'generate' });
  });

  it('applies the Compose preset as a full widget-region layout', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });

    const project = getLayoutOf(state);

    expect(project.panels).toEqual({ isBottomOpen: false, isLeftOpen: true, isRightOpen: true });
    expect(project.widgetRegions.left).toMatchObject({
      activeInstanceId: 'generate',
      instanceIds: ['generate', 'upscale'],
      isCollapsed: false,
      sizePx: 450,
    });
    expect(project.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['preview', 'gallery:center'],
      isCollapsed: false,
      sizePx: 0,
    });
    expect(project.widgetRegions.right).toMatchObject({
      activeInstanceId: 'gallery',
      instanceIds: ['gallery', 'image-map', 'queue'],
      isCollapsed: false,
      sizePx: 450,
    });
    expect(project.widgetRegions.bottom).toMatchObject({
      activeInstanceId: 'gallery:bottom',
      instanceIds: ['server-status', 'queue-status', 'gallery:bottom', 'shortcuts', 'notifications', 'autosave-status'],
      isCollapsed: true,
      sizePx: 180,
    });
  });

  it('persists one account-wide order through reordering and custom preset deletion', () => {
    let state = createInitialWorkbenchState();

    state = addCustomWorkspace(state, { id: 'custom-layout-1', label: 'Custom' });

    expect(state.account.layoutPresetOrder).toEqual(['compose', 'edit', 'video', 'automate', 'custom-layout-1']);

    state = workbenchReducer(state, {
      activeId: 'custom-layout-1',
      overId: 'edit',
      type: 'reorderLayoutPresets',
    });

    expect(state.account.layoutPresetOrder).toEqual(['compose', 'custom-layout-1', 'edit', 'video', 'automate']);

    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'deleteLayoutPreset' });

    expect(state.account.layoutPresetOrder).toEqual(['compose', 'edit', 'video', 'automate']);
  });

  it('normalizes stale and duplicate preset ids when hydrating an account', () => {
    const initial = createInitialWorkbenchState();
    const custom = {
      id: 'custom-layout-1',
      label: 'Custom',
      snapshot: layoutPresets[0].snapshot,
    };
    const state = workbenchReducer(initial, {
      state: {
        ...initial,
        account: {
          ...initial.account,
          customLayoutPresets: [custom],
          layoutPresetOrder: ['automate', 'missing', 'automate'],
        },
      },
      type: 'hydrateWorkbench',
    });

    expect(state.account.layoutPresetOrder).toEqual(['compose', 'edit', 'video', 'automate', 'custom-layout-1']);
  });

  it('drops preset snapshots with invalid widget and layout references', () => {
    const snapshot = layoutPresets[0]!.snapshot;
    const invalidSnapshots = [
      { ...snapshot, layout: { ...snapshot.layout, centerViewId: 'retired-view' } },
      {
        ...snapshot,
        widgetInstances: { ...snapshot.widgetInstances, generate: true },
      },
      {
        ...snapshot,
        widgetRegions: {
          ...snapshot.widgetRegions,
          left: {
            ...snapshot.widgetRegions.left,
            instanceIds: [...snapshot.widgetRegions.left.instanceIds, 'missing'],
          },
        },
      },
      {
        ...snapshot,
        widgetRegions: {
          ...snapshot.widgetRegions,
          right: { ...snapshot.widgetRegions.right, sizePx: Number.POSITIVE_INFINITY },
        },
      },
    ];

    for (const invalidSnapshot of invalidSnapshots) {
      expect(
        normalizeWorkbenchAccount({ layoutPresetOverrides: { compose: invalidSnapshot } }).layoutPresetOverrides
      ).toEqual({});
    }
  });

  it('resets the view of one workspace as one undo step, keeping the project’s other views', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });

    state = workbenchReducer(state, { region: 'right', sizePx: 600, type: 'setRegionWidgetSize' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });
    state = workbenchReducer(state, { region: 'right', type: 'selectRegionWidget', widgetId: 'queue' });
    state = workbenchReducer(state, { region: 'left', sizePx: 600, type: 'setRegionWidgetSize' });

    const before = getActiveProject(state);
    const reset = workbenchReducer(state, { type: 'resetWorkspaceView', workspaceId: 'compose' });

    expect(getActiveProject(reset).views.compose).toBeUndefined();
    expect(getActiveProject(reset).views.edit).toBe(before.views.edit);
    expect(getLayoutOf(reset).widgetRegions.right.activeInstanceId).toBe('gallery');
    expect(getLayoutOf(reset).widgetRegions.left.sizePx).toBe(450);
    expect(getActiveProject(reset).undoRedo.past).toHaveLength(before.undoRedo.past.length + 1);
    expect(getActiveProject(reset).undoRedo.past.at(-1)?.notice).toEqual({ kind: 'resetView', presetId: 'compose' });
    expect(getActiveProject(workbenchReducer(reset, { type: 'undoProjectChange' })).views).toEqual(before.views);
    expect(
      getActiveProject(workbenchReducer(state, { type: 'resetWorkspaceView', workspaceId: 'edit' })).views
    ).toEqual({ compose: before.views.compose });
  });

  it('moves every project off a deleted custom layout preset', () => {
    let state = createInitialWorkbenchState();
    const firstProjectId = state.activeProjectId;

    state = addCustomWorkspace(state, { id: 'custom-layout-1', label: 'Shared custom layout' });
    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'applyPreset' });
    state = workbenchReducer(state, { type: 'createProject' });
    const secondProjectId = state.activeProjectId;
    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'applyPreset' });

    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'deleteLayoutPreset' });

    expect(getProject(state, firstProjectId).workspaceId).toBe('compose');
    expect(getProject(state, secondProjectId).workspaceId).toBe('compose');
    expect(getProject(state, firstProjectId).views).not.toHaveProperty('custom-layout-1');
    expect(state.account.activeLayoutPresetId).toBe('compose');
  });

  it('restores a built-in’s shipped default for every project on it', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });

    state = workbenchReducer(state, { region: 'right', type: 'toggleRegionWidget', widgetId: 'image-map' });
    state = {
      ...state,
      account: setWorkspaceRoute(state.account, 'compose', { destination: 'canvas', sourceId: 'upscale' }),
    };

    expect(getLayoutOf(state).widgetRegions.right.instanceIds).toEqual(['gallery', 'queue']);

    state = workbenchReducer(state, { presetId: 'compose', type: 'restoreLayoutPresetDefault' });

    expect(state.account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(state.account.layoutPresetRouteOverrides?.compose).toBeUndefined();
    expect(getLayoutOf(state).widgetRegions.right.instanceIds).toEqual(['gallery', 'image-map', 'queue']);
  });

  it('resolves retired built-in preset ids onto the three shipped presets', () => {
    const initial = createInitialWorkbenchState();
    const hydrated = workbenchReducer(initial, {
      state: {
        ...initial,
        account: { activeLayoutPresetId: 'workflow' },
        projects: initial.projects.map((project) => ({ ...project, workspaceId: 'canvas' })),
      } as unknown as WorkbenchState,
      type: 'hydrateWorkbench',
    });

    expect(hydrated.account.activeLayoutPresetId).toBe('automate');
    expect(getActiveProject(hydrated).workspaceId).toBe('edit');
  });
});

describe('entering a workspace', () => {
  const run = (state: WorkbenchState, ...actions: WorkbenchAction[]) => actions.reduce(workbenchReducer, state);
  const leaveEditOnPreview = (state: WorkbenchState) =>
    run(
      state,
      { presetId: 'edit', type: 'applyPreset' },
      { region: 'center', type: 'selectRegionWidget', widgetId: 'preview' },
      { region: 'right', sizePx: 433, type: 'setRegionWidgetSize' },
      { isCollapsed: true, region: 'left', type: 'setRegionWidgetCollapsed' }
    );

  it('restores the stage and the job, keeping the view’s sizes and collapsed panels', () => {
    const state = run(
      leaveEditOnPreview(createInitialWorkbenchState()),
      { presetId: 'compose', type: 'applyPreset' },
      { presetId: 'edit', type: 'applyPreset' }
    );
    const layout = getLayoutOf(state);

    expect(layout.widgetRegions.center.activeInstanceId).toBe('canvas');
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });
    expect(layout.widgetRegions.right.sizePx).toBe(433);
    expect(layout.widgetRegions.left.isCollapsed).toBe(true);
    expect(getActiveProject(state).views.edit?.fronts).toBeUndefined();
  });

  it('goes home on the active workspace without adding an undo step', () => {
    const away = leaveEditOnPreview(createInitialWorkbenchState());
    const state = run(away, { presetId: 'edit', type: 'applyPreset' });
    const layout = getLayoutOf(state);

    expect(layout.widgetRegions.center.activeInstanceId).toBe('canvas');
    expect(layout.widgetRegions.right.sizePx).toBe(433);
    expect(layout.widgetRegions.left.isCollapsed).toBe(true);
    expect(getActiveProject(state).undoRedo.past).toEqual(getActiveProject(away).undoRedo.past);
  });

  it('enters a project on its first activation in the session, not on returning to it', () => {
    const first = leaveEditOnPreview(createInitialWorkbenchState());
    const withSecond = leaveEditOnPreview(run(first, { type: 'createProject' }));
    const [firstId, secondId] = withSecond.projects.map(({ id }) => id) as [string, string];
    const centerOf = (state: WorkbenchState, projectId: string) =>
      getLayoutOf(state, getProject(state, projectId)).widgetRegions.center.activeInstanceId;
    let state = run(withSecond, { state: withSecond, type: 'hydrateWorkbench' });

    expect(centerOf(state, secondId)).toBe('canvas');
    expect(centerOf(state, firstId)).toBe('preview');

    state = run(state, { projectId: firstId, type: 'switchProject' });
    expect(centerOf(state, firstId)).toBe('canvas');

    state = run(
      state,
      { region: 'center', type: 'selectRegionWidget', widgetId: 'preview' },
      { projectId: secondId, type: 'switchProject' },
      { projectId: firstId, type: 'switchProject' }
    );
    expect(centerOf(state, firstId)).toBe('preview');
  });
});

describe('workbenchReducer Phase 5 generation flow', () => {
  it('restores valid active journal items while live items win duplicate ids', () => {
    const submittedState = submitGenerate(primeGenerate());
    const liveItem = getActiveProject(submittedState).queue.items[0]!;
    const restoredItem = {
      ...liveItem,
      id: 'restored-item',
      snapshot: { ...liveItem.snapshot, submittedAt: '2026-09-04T00:00:00.000Z' },
    };
    const duplicate = { ...liveItem, status: 'running' as const };

    const state = workbenchReducer(submittedState, {
      items: [duplicate, restoredItem],
      projectId: getActiveProject(submittedState).id,
      type: 'restoreQueueItemsFromJournal',
    });

    expect(getActiveProject(state).queue.items.map((item) => item.id)).toEqual([liveItem.id, restoredItem.id]);
    expect(getActiveProject(state).queue.items[0]).toEqual({ ...liveItem, localRecoveryState: 'durable' });
    expect(getActiveProject(state).queue.items[0]).not.toBe(liveItem);
    expect(getActiveProject(state).queue.items[1]).toEqual({ ...restoredItem, localRecoveryState: 'durable' });
    expect(getActiveProject(state).queue.items[1]).not.toBe(restoredItem);
  });

  it('rejects malformed and terminal journal items at restoration', () => {
    const submittedState = submitGenerate(primeGenerate());
    const project = getActiveProject(submittedState);
    const queueItem = project.queue.items[0]!;
    const emptyState = {
      ...submittedState,
      projects: submittedState.projects.map((candidate) =>
        candidate.id === project.id ? { ...candidate, queue: { items: [] } } : candidate
      ),
    };

    const state = workbenchReducer(emptyState, {
      items: [
        { ...queueItem, status: 'completed' },
        { ...queueItem, id: 'bad-canvas', snapshot: { ...queueItem.snapshot, canvas: { document: {} } } },
        { ...queueItem, id: 'bad-backend-ids', backendItemIds: [1, Number.NaN] },
        { ...queueItem, id: 'non-array-backend-ids', backendItemIds: 7 },
        { ...queueItem, id: 'non-array-completed-ids', completedBackendItemIds: {} },
        { ...queueItem, id: 'duplicate-backend-ids', backendItemIds: [1, 1] },
      ],
      projectId: project.id,
      type: 'restoreQueueItemsFromJournal',
    });

    expect(getActiveProject(state).queue.items).toEqual([]);
  });

  it('routes queue results back to the originating project after the user switches projects', () => {
    let state = submitGenerate(primeGenerate());
    const originProject = getActiveProject(state);
    const queueItem = originProject.queue.items[0];

    expect(queueItem).toBeDefined();

    state = workbenchReducer(state, { type: 'createProject' });
    state = workbenchReducer(state, { projectId: originProject.id, type: 'switchProject' });

    const otherProjectId = state.projects.find((project) => project.id !== originProject.id)?.id;

    expect(otherProjectId).toBeDefined();

    state = workbenchReducer(state, { projectId: otherProjectId as string, type: 'switchProject' });
    state = workbenchReducer(state, {
      backendItemIds: [42],
      projectId: originProject.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, {
      images: [createImage('origin-image.png', queueItem.id)],
      projectId: originProject.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    const updatedOriginProject = getProject(state, originProject.id);
    const activeProject = getActiveProject(state);

    expect(activeProject.id).toBe(otherProjectId);
    expect(updatedOriginProject.canvas.stagingArea.pendingImageIds).toEqual(['origin-image.png']);
    expect(updatedOriginProject.queue.items[0]?.status).toBe('completed');
    expect(activeProject.canvas.stagingArea.pendingImageIds).toEqual([]);
    expect(activeProject.queue.items).toEqual([]);
  });

  it('reopens a failed Canvas run around its rerun and stages the rerun at the submitted bbox', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0]!;
    const submittedBbox = queueItem.snapshot.canvas.document.bbox;

    state = workbenchReducer(state, {
      backendItemIds: [41],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, {
      error: 'Out of memory',
      notify: false,
      projectId: project.id,
      queueItemId: queueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });
    state = workbenchReducer(state, {
      bbox: { ...submittedBbox, x: submittedBbox.x + 64 },
      type: 'setCanvasBbox',
    });
    state = workbenchReducer(state, {
      projectId: project.id,
      queueItemId: queueItem.id,
      retriedItems: [
        { itemId: 99, sourceItemId: 41 },
        { itemId: 100, sourceItemId: 7 },
      ],
      type: 'reopenQueueItemForRetry',
    });

    const reopened = getProject(state, project.id).queue.items[0];
    expect(reopened).toEqual(expect.objectContaining({ backendItemIds: [99], status: 'running' }));
    expect(reopened).not.toHaveProperty('error');

    state = workbenchReducer(state, {
      images: [createImage('rerun.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    const { stagingArea } = getProject(state, project.id).canvas;
    expect(stagingArea.pendingImageIds).toEqual(['rerun.png']);
    expect(stagingArea.pendingImages[0]).toEqual(
      expect.objectContaining({
        placement: expect.objectContaining({ x: submittedBbox.x, y: submittedBbox.y }),
        sourceBackendItemId: 99,
      })
    );
  });

  it('leaves a run finished when it is still active or its Canvas document was replaced', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0]!;
    const reopen = (current: WorkbenchState) =>
      workbenchReducer(current, {
        projectId: project.id,
        queueItemId: queueItem.id,
        retriedItems: [{ itemId: 99, sourceItemId: 41 }],
        type: 'reopenQueueItemForRetry',
      });

    state = workbenchReducer(state, {
      backendItemIds: [41],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    expect(reopen(state)).toBe(state);

    state = workbenchReducer(state, {
      notify: false,
      projectId: project.id,
      queueItemId: queueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });
    state = workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });
    expect(reopen(state)).toBe(state);
  });

  it("appends a workflow candidate to its explicit project and selects its resolved staging slot when auto-switch is 'latest'", () => {
    let state = createInitialWorkbenchState();
    const originProjectId = state.activeProjectId;
    const firstCandidate = createStagingCandidate('first.png', 'layer-workflow:first', {
      height: 100,
      opacity: 1,
      width: 200,
      x: 5,
      y: 7,
    });
    const appendedCandidate = createStagingCandidate('second.png', 'layer-workflow:second', {
      height: 240,
      opacity: 0.45,
      width: 320,
      x: 11,
      y: 17,
    });

    state = workbenchReducer(state, { mode: 'latest', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, { type: 'createProject' });
    const activeProjectId = state.activeProjectId;
    state = workbenchReducer(state, {
      candidate: firstCandidate,
      projectId: originProjectId,
      type: 'appendCanvasStagingCandidate',
    });
    state = workbenchReducer(state, {
      candidate: appendedCandidate,
      projectId: originProjectId,
      type: 'appendCanvasStagingCandidate',
    });

    const originProject = getProject(state, originProjectId);
    const slots = getCanvasStagingSlots(originProject.canvas, originProject.queue.items);

    expect(state.activeProjectId).toBe(activeProjectId);
    expect(originProject.canvas.stagingArea.pendingImages).toEqual([firstCandidate, appendedCandidate]);
    expect(originProject.canvas.stagingArea.pendingImageIds).toEqual(['first.png', 'second.png']);
    expect(originProject.canvas.stagingArea.isVisible).toBe(true);
    expect(originProject.canvas.stagingArea.selectedImageIndex).toBe(1);
    expect(slots[originProject.canvas.stagingArea.selectedImageIndex]).toMatchObject({
      candidate: { imageName: 'second.png', placement: appendedCandidate.placement },
      kind: 'candidate',
    });
    expect(getProject(state, activeProjectId).canvas.stagingArea.pendingImages).toEqual([]);
  });

  it('accepts a staged candidate in the addressed project after another project becomes active', () => {
    let state = withEmptyCanvas(createInitialWorkbenchState());
    const originProjectId = state.activeProjectId;
    const candidate = createStagingCandidate('origin-result.png', 'layer-workflow:origin', {
      height: 64,
      opacity: 0.75,
      width: 96,
      x: 13,
      y: 17,
    });
    state = workbenchReducer(state, {
      candidate,
      projectId: originProjectId,
      type: 'appendCanvasStagingCandidate',
    });
    state = workbenchReducer(state, { type: 'createProject' });
    const activeProjectId = state.activeProjectId;

    state = commitSelectedStagedImage(state, originProjectId);

    expect(state.activeProjectId).toBe(activeProjectId);
    expect(getRasterLayerImageName(getDocumentLeaves(getProject(state, originProjectId).canvas.document)[0])).toBe(
      'origin-result.png'
    );
    expect(getDocumentLeaves(getProject(state, activeProjectId).canvas.document)).not.toContainEqual(
      expect.objectContaining({
        source: expect.objectContaining({ image: expect.objectContaining({ imageName: 'origin-result.png' }) }),
      })
    );
  });

  it('keeps submitted Generate snapshots immutable after later settings changes', () => {
    let state = submitGenerate(primeGenerate(undefined, { positivePrompt: 'first prompt', seedMode: 'random' }));
    const firstQueueItem = getActiveProject(state).queue.items[0];

    expect(firstQueueItem).toBeDefined();

    state = primeGenerate(state, { positivePrompt: 'second prompt', seed: 999 });
    state = submitGenerate(state);

    const [secondQueueItem, unchangedFirstQueueItem] = getActiveProject(state).queue.items;
    const firstValues = unchangedFirstQueueItem?.snapshot.recall?.generateValues as GenerateWidgetValues;
    const secondValues = secondQueueItem?.snapshot.recall?.generateValues as GenerateWidgetValues;

    expect(firstValues.positivePrompt).toBe('first prompt');
    expect(firstValues.seedMode).toBe('random');
    expect(typeof firstValues.seed).toBe('number');
    expect(secondValues.positivePrompt).toBe('second prompt');
    expect(secondValues.seed).toBe(999);
  });

  it('discards selected and all staged canvas candidates without touching accepted document layers', () => {
    let state = submitGenerate(
      primeGenerate(withInvocationDestination(withEmptyCanvas(createInitialWorkbenchState()), 'canvas'))
    );
    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-1.png', queueItem.id), createImage('candidate-2.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, { type: 'discardSelectedStagedImage' });

    let project = getActiveProject(state);

    expect(getDocumentLeaves(project.canvas.document)).toEqual([]);
    expect(project.canvas.stagingArea.pendingImageIds).toEqual(['candidate-1.png']);
    expect(project.canvas.stagingArea.selectedImageIndex).toBe(0);
    expect(project.canvas.stagingArea.isVisible).toBe(true);

    state = workbenchReducer(state, { type: 'discardAllStagedImages' });
    project = getActiveProject(state);

    expect(project.canvas.stagingArea.pendingImages).toEqual([]);
    expect(project.canvas.stagingArea.pendingImageIds).toEqual([]);
    expect(project.canvas.stagingArea.isVisible).toBe(false);
  });

  it('normalizes ordinary queue results to the queued bbox origin and native image size', () => {
    let state = submitGenerate(primeGenerate());
    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, { bbox: { height: 256, width: 256, x: 40, y: 24 }, type: 'setCanvasBbox' });
    state = workbenchReducer(state, {
      images: [createImage('candidate-1.png', queueItem.id), createImage('candidate-2.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(1);
    state = workbenchReducer(state, { direction: 1, type: 'cycleStagedImage' });
    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(0);
    state = workbenchReducer(state, { direction: -1, type: 'cycleStagedImage' });

    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(1);
    expect(getActiveProject(state).canvas.stagingArea.pendingImages[1]?.placement).toEqual({
      height: 768,
      opacity: 1,
      width: 512,
      x: 0,
      y: 0,
    });

    state = commitSelectedStagedImage(state);

    let project = getActiveProject(state);
    const acceptedLayer = project.canvas.document.stacks.raster[0] as CanvasLayer | undefined;

    expect(getRasterLayerImageName(acceptedLayer)).toBe('candidate-2.png');
    expect(getRasterLayerPlacement(acceptedLayer)).toEqual({ height: 768, opacity: 1, width: 512, x: 0, y: 0 });

    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    expect(getRasterLayerImageName(project.canvas.document.stacks.raster[0] as CanvasLayer)).toBe('candidate-2.png');
  });

  it('appends later canvas results to the active staging session instead of replacing candidates', () => {
    let state = submitGenerate(primeGenerate());
    const firstQueueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, { mode: 'latest', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, {
      images: [createImage('candidate-1.png', firstQueueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: firstQueueItem.id,
      type: 'routeQueueItemResults',
    });
    state = submitGenerate(primeGenerate(state, { positivePrompt: 'second prompt' }));

    const secondQueueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-2.png', secondQueueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: secondQueueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getActiveProject(state).canvas.stagingArea.pendingImageIds).toEqual(['candidate-1.png', 'candidate-2.png']);
    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(1);
  });

  it('keeps staged canvas candidates isolated per project when switching projects', () => {
    let state = submitGenerate(primeGenerate());
    const originProject = getActiveProject(state);
    const queueItem = originProject.queue.items[0];

    state = workbenchReducer(state, { type: 'createProject' });
    state = workbenchReducer(state, { projectId: originProject.id, type: 'switchProject' });

    const otherProjectId = state.projects.find((project) => project.id !== originProject.id)?.id;

    expect(otherProjectId).toBeDefined();

    state = workbenchReducer(state, {
      images: [createImage('origin-candidate.png', queueItem.id)],
      projectId: originProject.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { projectId: otherProjectId as string, type: 'switchProject' });

    expect(getActiveProject(state).canvas.stagingArea.pendingImages).toEqual([]);

    state = workbenchReducer(state, { projectId: originProject.id, type: 'switchProject' });

    expect(getActiveProject(state).canvas.stagingArea.pendingImageIds).toEqual(['origin-candidate.png']);
  });

  it('marks cancellable running queue items cancelled for backend cancellation', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [42],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, { queueItemId: queueItem.id, type: 'cancelQueueItem' });

    expect(getActiveProject(state).queue.items[0]).toMatchObject({ cancellationPending: true, status: 'cancelled' });
  });

  it('marks a local queue item cancelled only when all backend items were cancelled', () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 2 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11, 12],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, {
      backendItemId: 11,
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendCancelled',
    });
    state = workbenchReducer(state, {
      backendItemId: 12,
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendCancelled',
    });

    expect(getActiveProject(state).queue.items[0]).toMatchObject({
      cancelledBackendItemIds: [11, 12],
      status: 'cancelled',
    });
  });

  it('clears completed and failed queue items while preserving active and cancelled items', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('completed-result.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = submitGenerate(primeGenerate(state));
    const activeQueueItem = getActiveProject(state).queue.items[0];
    state = workbenchReducer(state, { queueItemId: activeQueueItem.id, type: 'cancelQueueItem' });
    state = submitGenerate(primeGenerate(state));
    const failedQueueItem = getActiveProject(state).queue.items[0];
    state = workbenchReducer(state, {
      error: 'failed',
      projectId: project.id,
      queueItemId: failedQueueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });

    state = workbenchReducer(state, { type: 'clearCompletedQueueItems' });

    expect(getActiveProject(state).queue.items.map((item) => item.status)).toEqual(['cancelled']);
  });

  it('keeps cancellation terminal when backend item ids arrive late', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, { queueItemId: queueItem.id, type: 'cancelQueueItem' });
    state = workbenchReducer(state, {
      backendItemIds: [42],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, {
      images: [createImage('late-result.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, {
      error: 'backend cancellation completed',
      projectId: project.id,
      queueItemId: queueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });

    const cancelledItem = getActiveProject(state).queue.items[0];

    expect(cancelledItem?.status).toBe('cancelled');
    expect(cancelledItem?.backendItemIds).toEqual([42]);
    expect(getActiveProject(state).canvas.stagingArea.pendingImages).toEqual([]);
    expect(state.notifications.map((notification) => notification.titleKey ?? notification.title)).toEqual([
      'notifications.run.cancelRequested',
      'notifications.run.queued',
    ]);
  });

  it('records notifications for errors and successful operations', () => {
    let state = submitGenerate(primeGenerate());

    expect(state.notifications[0]?.titleKey).toBe('notifications.run.queued');

    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = commitSelectedStagedImage(state);
    state = workbenchReducer(state, { message: 'boom', type: 'recordError' });

    expect(state.notifications.map((notification) => notification.titleKey ?? notification.title)).toEqual([
      'Error',
      'notifications.run.completed',
      'notifications.run.queued',
    ]);
    expect(state.notifications.every((notification) => !notification.isRead)).toBe(true);

    state = workbenchReducer(state, { type: 'markAllNotificationsRead' });

    expect(state.notifications.every((notification) => notification.isRead)).toBe(true);

    state = workbenchReducer(state, { type: 'clearNotifications' });

    expect(state.notifications).toEqual([]);

    expect(state.notifications).toEqual([]);
  });

  it('says which run failed and where, stays until dismissed, and offers to show and retry the failed item', () => {
    let state = submitGenerate(primeGenerate(undefined, { positivePrompt: 'A lighthouse in a storm' }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0]!;

    state = workbenchReducer(state, {
      error: 'CUDA out of memory',
      failedBackendItemIds: [41],
      projectId: project.id,
      queueItemId: queueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });

    expect(state.notifications[0]).toMatchObject({
      actions: [
        { itemId: 41, kind: 'reveal-queue-item' },
        { itemIds: [41], kind: 'retry-queue-item' },
      ],
      category: 'run-outcome',
      kind: 'error',
      message: '“A lighthouse in a storm”: CUDA out of memory',
      persistent: true,
      projectId: project.id,
      titleKey: 'notifications.runOutcome.failed',
      titleValues: { project: project.name },
    });
  });

  it('coalesces back-to-back identical notifications instead of stacking them', () => {
    const state = createInitialWorkbenchState();

    const once = reduceWorkbench(state, { message: 'Bitmap persistence failed', type: 'recordError' });
    const twice = reduceWorkbench(once, { message: 'Bitmap persistence failed', type: 'recordError' });

    expect(twice.notifications).toHaveLength(once.notifications.length);
    expect(twice.notifications[0]?.occurrenceCount).toBe(2);
    expect(twice.notifications[0]?.id).toBe(once.notifications[0]?.id);
  });

  describe('enqueue notifications', () => {
    it('records an enqueue notification when submitResolvedInvocationSnapshot adds a queue item', () => {
      const state = primeGenerate();

      const next = workbenchReducer(state, {
        backendSupportsCancellation: true,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        projectId: state.activeProjectId,
        type: 'submitResolvedInvocationSnapshot',
      });

      expect(getActiveProject(next).queue.items.length).toBeGreaterThan(0);
      expect(next.notifications[0]?.titleKey).toBe('notifications.run.queued');
      expect(next.notifications[0]?.category).toBe('enqueue');
      expect(next.notifications[0]?.kind).toBe('success');
    });
  });

  it('applies workflow edits to the project graph with undo and auto-source', () => {
    let state = createInitialWorkbenchState();

    expect(getActiveProject(state).invocation.sourceId).toBe('generate');

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
      type: 'applyWorkflowAction',
    });

    const project = getActiveProject(state);

    expect(getActiveProjectGraph(project).nodes).toHaveLength(1);
    expect(project.workflowHistories[project.workflows.activeWorkflowId]?.past.at(-1)?.label).toBe('Add workflow node');
    expect(project.undoRedo.past.some((entry) => entry.label === 'Add workflow node')).toBe(false);
    expect(project.invocation.sourceId).toBe('workflow');
    expect(project.invocation.destination).toBe('gallery');

    state = workbenchReducer(state, { type: 'undoWorkflowChange' });

    expect(getActiveProjectGraph(getActiveProject(state)).nodes).toHaveLength(0);
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });
  });

  it("adds a workflow beside the others and keeps each workflow's history to itself", () => {
    let state = createInitialWorkbenchState();
    const originalGraphId = getActiveProjectGraph(getActiveProject(state)).id;

    state = workbenchReducer(state, {
      action: { patch: { name: 'First' }, type: 'setMetadata' },
      type: 'applyWorkflowAction',
    });
    state = workbenchReducer(state, {
      document: { ...getActiveProjectGraph(getActiveProject(state)), id: 'second-graph', name: 'Second' },
      label: 'Test add',
      type: 'addProjectWorkflow',
    });

    const project = getActiveProject(state);

    expect(project.workflows.activeWorkflowId).toBe('second-graph');
    expect(project.workflows.entries.map((entry) => entry.document.id)).toEqual([originalGraphId, 'second-graph']);
    expect(project.invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });

    state = workbenchReducer(state, { type: 'undoWorkflowChange' });
    expect(getActiveProjectGraph(getActiveProject(state)).id).toBe('second-graph');
    expect(getActiveProject(state).workflows.entries[0]?.document.name).toBe('First');

    state = workbenchReducer(state, { type: 'selectProjectWorkflow', workflowId: originalGraphId });
    state = workbenchReducer(state, { type: 'undoWorkflowChange' });
    expect(getActiveProjectGraph(getActiveProject(state)).name).toBe('Untitled Workflow');
    expect(getActiveProject(state).workflows.entries[1]?.document.name).toBe('Second');
  });

  it('does not queue Upscale while its required settings are incomplete', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });

    expect(getActiveProject(state).invocation.sourceId).toBe('upscale');

    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      route: { destination: 'canvas', destinationLocked: false, sourceId: 'upscale', sourceLocked: true },
      projectId: state.activeProjectId,
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getActiveProject(state).queue.items).toEqual([]);
  });

  describe('expanded positive prompts on the compiled submission', () => {
    const submitResolvedWithPrompts = (positivePrompt: string, positivePrompts?: string[]) =>
      submitResolvedTo(primeGenerate(undefined, { positivePrompt }), positivePrompts);
    const submitResolvedTo = (primed: WorkbenchState, positivePrompts?: string[]) =>
      workbenchReducer(primed, {
        backendSupportsCancellation: true,
        expandedFrom: preparedExpansionSource(primed, 'generate'),
        positivePrompts,
        projectId: primed.activeProjectId,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        type: 'submitResolvedInvocationSnapshot',
      });

    it('records every prompt of a multi-prompt expansion', () => {
      const state = submitResolvedWithPrompts('a {red|green} cat', ['a red cat', 'a green cat']);
      const queueItem = getActiveProject(state).queue.items[0];

      expect(queueItem?.snapshot.backendSubmission).toMatchObject({
        positivePrompts: ['a red cat', 'a green cat'],
      });
      // Placeholder sizing multiplies the iterations by the prompt count.
      expect(queueItem?.snapshot.presentation.batchCount).toBe(2);
    });
  });

  describe('expanded Video prompts on the compiled submission', () => {
    const wanModel: ModelConfig = {
      base: 'wan',
      file_size: 1,
      format: 'diffusers',
      hash: 'wan-t2v-hash',
      key: 'wan-t2v_a14b-diffusers',
      name: 'Wan 2.2 t2v_a14b',
      path: 'wan-t2v_a14b-diffusers',
      source: 'wan-t2v_a14b-diffusers',
      source_type: 'path',
      type: 'main',
      variant: 't2v_a14b',
    };
    const submitVideo = (
      values: Record<string, unknown>,
      positivePrompts?: string[],
      positivePromptsSampleSeed: number | null = null
    ) => {
      let state = workbenchReducer(createInitialWorkbenchState(), {
        region: 'left',
        type: 'toggleRegionWidget',
        widgetId: 'video',
      });
      state = workbenchReducer(state, {
        type: 'patchWidgetValues',
        values: { model: wanModel, positivePrompt: 'a {red|green} fox', ...values },
        widgetId: 'video',
      });

      return workbenchReducer(state, {
        backendSupportsCancellation: true,
        expandedFrom: preparedExpansionSource(state, 'video'),
        models: [wanModel],
        positivePrompts,
        positivePromptsSampleSeed,
        projectId: state.activeProjectId,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'video', sourceLocked: false },
        type: 'submitResolvedInvocationSnapshot',
      });
    };

    it('queues one video per expanded prompt', () => {
      const queueItem = getActiveProject(submitVideo({}, ['a red fox', 'a green fox'])).queue.items[0];

      expect(queueItem?.snapshot.backendSubmission).toMatchObject({
        kind: 'generate',
        positivePrompts: ['a red fox', 'a green fox'],
      });
      expect(queueItem?.snapshot.presentation.batchCount).toBe(2);
    });
  });

  describe('workflow seed modes on the compiled submission', () => {
    const SEED_MAX = 4_294_967_295;
    const seedTemplate = {
      category: 'noise',
      classification: 'stable',
      description: '',
      inputs: {
        seed: {
          default: 0,
          description: '',
          exclusiveMaximum: null,
          exclusiveMinimum: null,
          fieldKind: 'input',
          input: 'any',
          maximum: SEED_MAX,
          minimum: 0,
          multipleOf: null,
          name: 'seed',
          options: null,
          required: false,
          title: 'Seed',
          type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
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
      outputs: {},
      outputType: 'noise_output',
      tags: [],
      title: 'Noise',
      type: 'noise',
      useCache: true,
      version: '1.0.0',
    };
    const primeWorkflow = (seed: number, seedMode: 'random' | 'fixed' | 'increment' | 'decrement', batchCount = 3) => {
      workflowTemplatesMock.snapshot = { error: null, status: 'loaded', templates: { noise: seedTemplate } };

      let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'automate', type: 'applyPreset' });

      state = workbenchReducer(state, { type: 'patchWidgetValues', values: { batchCount }, widgetId: 'workflow' });
      state = workbenchReducer(state, {
        action: {
          node: {
            data: {
              inputs: { seed: { label: '', name: 'seed', seedMode, value: seed } },
              isIntermediate: true,
              isOpen: true,
              label: '',
              nodePack: 'invokeai',
              notes: '',
              type: 'noise',
              useCache: true,
              version: '1.0.0',
            },
            id: 'noise-1',
            position: { x: 0, y: 0 },
            type: 'invocation',
          },
          type: 'addNode',
        },
        type: 'applyWorkflowAction',
      });

      return state;
    };
    const submitWorkflow = (state: WorkbenchState) =>
      workbenchReducer(state, {
        backendSupportsCancellation: true,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'workflow', sourceLocked: false },
        projectId: state.activeProjectId,
        type: 'submitResolvedInvocationSnapshot',
      });
    const readSubmission = (state: WorkbenchState, index = 0) =>
      getActiveProject(state).queue.items[index]?.snapshot.backendSubmission;
    const readNodeSeed = (state: WorkbenchState) => {
      const node = getActiveProjectGraph(getActiveProject(state)).nodes[0];

      return node?.type === 'invocation' ? node.data.inputs.seed?.value : undefined;
    };

    it('validates and compiles with captured templates after the template store changes', () => {
      const state = primeWorkflow(42, 'increment');
      const runtime = captureInvocationRuntimeInputs('workflow');
      workflowTemplatesMock.snapshot = { error: 'reloading', status: 'error', templates: {} };
      const next = workbenchReducer(state, {
        backendSupportsCancellation: true,
        projectId: state.activeProjectId,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'workflow', sourceLocked: false },
        runtime,
        type: 'submitResolvedInvocationSnapshot',
      });
      expect(readSubmission(next)).toMatchObject({
        kind: 'workflow',
        batchCount: 3,
        graph: { nodes: { 'noise-1': { seed: 42 } } },
        seeds: [{ fieldName: 'seed', nodeId: 'noise-1', seed: 42, seedStep: 1 }],
      });
      expect(readNodeSeed(next)).toBe(45);
      expect(getActiveProject(next).queue.items).toHaveLength(1);

      const rejected = workbenchReducer(state, {
        backendSupportsCancellation: true,
        projectId: state.activeProjectId,
        route: getActiveProject(state).invocation,
        runtime: { ...runtime, templates: { error: null, status: 'loading', templates: {} } },
        type: 'submitResolvedInvocationSnapshot',
      });
      expect(getActiveProject(rejected).queue.items).toEqual([]);
      expect(readNodeSeed(rejected)).toBe(42);
    });

    it('does not enqueue or advance seeds when a workflow generator changed during preparation', () => {
      let state = primeWorkflow(42, 'increment');
      const field = (name: string, typeName: string) => ({
        ...seedTemplate.inputs.seed,
        default: undefined,
        maximum: null,
        minimum: null,
        name,
        title: name,
        type: { batch: false, cardinality: 'SINGLE' as const, name: typeName },
      });
      const template = (type: string, inputs: Record<string, unknown>) => ({ ...seedTemplate, inputs, type });
      workflowTemplatesMock.snapshot = {
        error: null,
        status: 'loaded',
        templates: {
          noise: seedTemplate,
          string: template('string', { value: field('value', 'StringField') }),
          string_batch: template('string_batch', {
            batch_group_id: { ...field('batch_group_id', 'EnumField'), default: 'None', options: ['None'] },
            strings: {
              ...field('strings', 'StringField'),
              default: [],
              type: { batch: true, cardinality: 'COLLECTION', name: 'StringField' },
            },
          }),
          string_generator: template('string_generator', { generator: field('generator', 'StringGeneratorField') }),
        },
      };
      const generator = {
        input: 'a {red|green} cat',
        maxPrompts: 4,
        type: 'string_generator_dynamic_prompts_combinatorial' as const,
      };
      for (const [id, type, inputs] of [
        ['generator-1', 'string_generator', { generator }],
        ['batch-1', 'string_batch', { batch_group_id: 'None' }],
        ['string-1', 'string', { value: 'static' }],
      ] as const) {
        state = workbenchReducer(state, {
          action: {
            node: {
              data: {
                inputs: Object.fromEntries(
                  Object.entries(inputs).map(([name, value]) => [name, { label: '', name, value }])
                ),
                isIntermediate: true,
                isOpen: true,
                label: '',
                nodePack: 'invokeai',
                notes: '',
                type,
                useCache: true,
                version: '1.0.0',
              },
              id,
              position: { x: 0, y: 0 },
              type: 'invocation',
            },
            type: 'addNode',
          },
          type: 'applyWorkflowAction',
        });
      }
      for (const [source, sourceHandle, target, targetHandle] of [
        ['generator-1', 'strings', 'batch-1', 'strings'],
        ['batch-1', 'value', 'string-1', 'value'],
      ] as const) {
        state = workbenchReducer(state, {
          action: {
            edge: { id: `${source}->${target}`, source, sourceHandle, target, targetHandle, type: 'default' },
            type: 'addEdge',
          },
          type: 'applyWorkflowAction',
        });
      }
      const submission = {
        backendSupportsCancellation: true,
        projectId: state.activeProjectId,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'workflow', sourceLocked: false },
        runtime: captureInvocationRuntimeInputs('workflow'),
        type: 'submitResolvedInvocationSnapshot',
        workflowGenerators: {
          'generator-1': { items: ['a red cat', 'a green cat'], key: getWorkflowGeneratorKey(generator) },
        },
        workflowId: getActiveProject(state).workflows.activeWorkflowId,
      } satisfies WorkbenchAction;
      const valid = workbenchReducer(state, submission);
      expect(readSubmission(valid)).toMatchObject({
        batchData: [[{ fieldName: 'value', items: ['a red cat', 'a green cat'], nodeId: 'string-1' }]],
        kind: 'workflow',
      });
      expect(getActiveProject(valid).queue.items).toHaveLength(1);
      expect(readNodeSeed(valid)).toBe(48);

      state = workbenchReducer(state, {
        action: {
          fieldName: 'generator',
          nodeId: 'generator-1',
          type: 'setFieldValue',
          value: { ...generator, input: 'a {black|white} dog' },
        },
        type: 'applyWorkflowAction',
        workflowId: submission.workflowId,
      });
      const rejected = workbenchReducer(state, submission);
      expect(getActiveProject(rejected).queue.items).toEqual([]);
      expect(readNodeSeed(rejected)).toBe(42);
    });

    it('carries batch-node groups on the submission and sizes the placeholders for every session', () => {
      const floatField = {
        ...(seedTemplate.inputs.seed as NonNullable<(typeof seedTemplate.inputs)['seed']>),
        maximum: null,
        minimum: null,
        name: 'value',
        title: 'Value',
        type: { batch: false, cardinality: 'SINGLE' as const, name: 'FloatField' },
      };
      const batchTemplates = {
        float: {
          ...seedTemplate,
          inputs: { value: floatField },
          outputType: 'float_output',
          title: 'Float',
          type: 'float',
        },
        float_batch: {
          ...seedTemplate,
          inputs: {
            batch_group_id: {
              ...floatField,
              default: 'None',
              name: 'batch_group_id',
              options: ['None', 'Group 1'],
              title: 'Batch Group',
              type: { batch: false, cardinality: 'SINGLE' as const, name: 'EnumField' },
            },
            floats: {
              ...floatField,
              default: [],
              name: 'floats',
              required: true,
              title: 'Floats',
              type: { batch: true, cardinality: 'COLLECTION' as const, name: 'FloatField' },
            },
          },
          outputType: 'float_output',
          title: 'Float Batch',
          type: 'float_batch',
        },
        noise: seedTemplate,
      };
      let state = primeWorkflow(42, 'fixed');

      workflowTemplatesMock.snapshot = { error: null, status: 'loaded', templates: batchTemplates };

      const invocationNode = (id: string, type: string, inputs: Record<string, unknown>) => ({
        data: {
          inputs: Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, { label: '', name, value }])),
          isIntermediate: true,
          isOpen: true,
          label: '',
          nodePack: 'invokeai',
          notes: '',
          type,
          useCache: true,
          version: '1.0.0',
        },
        id,
        position: { x: 0, y: 0 },
        type: 'invocation' as const,
      });

      state = workbenchReducer(state, {
        action: { node: invocationNode('float-1', 'float', { value: 0.5 }), type: 'addNode' },
        type: 'applyWorkflowAction',
      });
      state = workbenchReducer(state, {
        action: {
          edge: {
            id: 'batch-edge',
            source: 'batch-1',
            sourceHandle: 'value',
            target: 'float-1',
            targetHandle: 'value',
            type: 'default',
          },
          node: invocationNode('batch-1', 'float_batch', { batch_group_id: 'None', floats: [1.5, 2.5] }),
          type: 'addNodeAndEdge',
        },
        type: 'applyWorkflowAction',
      });
      state = submitWorkflow(state);

      expect(readSubmission(state)).toMatchObject({
        batchCount: 3,
        batchData: [[{ fieldName: 'value', items: [1.5, 2.5], nodeId: 'float-1' }]],
        kind: 'workflow',
      });
      expect(readSubmission(state)).toMatchObject({ graph: { nodes: { 'float-1': { value: 0.5 } } } });
      expect(Object.keys((readSubmission(state) as { graph: { nodes: object } }).graph.nodes)).not.toContain('batch-1');
      expect(getActiveProject(state).queue.items[0]?.snapshot.presentation.batchCount).toBe(6);
    });

    it('zips a stepping seed into the batch, advances the node, and continues on the next submission', () => {
      let state = submitWorkflow(primeWorkflow(42, 'increment'));

      expect(readSubmission(state)).toMatchObject({
        batchCount: 3,
        kind: 'workflow',
        seeds: [{ fieldName: 'seed', nodeId: 'noise-1', seed: 42, seedStep: 1 }],
        workflow: {
          author: '',
          contact: '',
          description: '',
          edges: [],
          exposedFields: [],
          form: { elements: expect.any(Object), rootElementId: expect.any(String) },
          meta: { category: 'user', version: '3.0.0' },
          name: 'Untitled Workflow',
          nodes: [{ id: 'noise-1' }],
          notes: '',
          tags: '',
          version: '1.0.0',
        },
      });
      expect(readSubmission(state)).not.toHaveProperty('workflow.id');
      expect(readSubmission(state)).toMatchObject({ graph: { nodes: { 'noise-1': { seed: 42 } } } });
      expect(getActiveProject(state).queue.items[0]?.snapshot.presentation.batchCount).toBe(3);
      expect(readNodeSeed(state)).toBe(45);

      state = submitWorkflow(state);

      expect(readSubmission(state)).toMatchObject({ seeds: [{ seed: 45 }] });
      expect(readSubmission(state, 1)).toMatchObject({ seeds: [{ seed: 42 }] });
      expect(readNodeSeed(state)).toBe(48);
    });

    describe('attribution to the originating project workflow', () => {
      const workflowIdOf = (state: WorkbenchState) => getActiveProject(state).workflows.activeWorkflowId;
      const seedOf = (state: WorkbenchState, workflowId: string) => {
        const entry = getActiveProject(state).workflows.entries.find(
          (candidate) => candidate.document.id === workflowId
        );
        const node = entry?.document.nodes[0];

        return node?.type === 'invocation' ? node.data.inputs.seed?.value : undefined;
      };
      const submitFrom = (state: WorkbenchState, workflowId: string) =>
        workbenchReducer(state, {
          backendSupportsCancellation: true,
          projectId: state.activeProjectId,
          route: { destination: 'gallery', destinationLocked: false, sourceId: 'workflow', sourceLocked: false },
          type: 'submitResolvedInvocationSnapshot',
          workflowId,
        });

      it('submits and advances the captured workflow even after another one became active', () => {
        let state = primeWorkflow(42, 'increment', 1);
        const original = workflowIdOf(state);

        state = workbenchReducer(state, {
          document: { ...getActiveProjectGraph(getActiveProject(state)), id: 'other', name: 'Other', nodes: [] },
          label: 'Add',
          type: 'addProjectWorkflow',
        });
        expect(workflowIdOf(state)).toBe('other');

        state = submitFrom(state, original);

        const item = getActiveProject(state).queue.items[0];

        expect(item?.snapshot.backendSubmission).toMatchObject({
          graph: { nodes: { 'noise-1': { seed: 42 } } },
          kind: 'workflow',
          projectWorkflowId: original,
        });
        expect(seedOf(state, original)).toBe(43);
        expect(workflowIdOf(state)).toBe('other');
        expect(getActiveProjectGraph(getActiveProject(state)).nodes).toEqual([]);
      });

      it('records the newest submitted successful run on the originating workflow and nowhere else', () => {
        vi.useFakeTimers({ now: new Date('2026-06-10T00:00:00.000Z') });

        try {
          let state = primeWorkflow(42, 'increment', 1);
          const original = workflowIdOf(state);

          state = workbenchReducer(state, {
            copyId: 'copy',
            copyName: 'Copy',
            type: 'duplicateProjectWorkflow',
            workflowId: original,
          });
          state = submitFrom(state, original);
          vi.advanceTimersByTime(1000);
          state = submitFrom(state, original);

          const [second, first] = getActiveProject(state).queue.items;

          expect(second?.snapshot.submittedAt).not.toBe(first?.snapshot.submittedAt);

          state = workbenchReducer(state, {
            images: [createImage('second.png', second!.id)],
            projectId: state.activeProjectId,
            queueItemId: second!.id,
            type: 'routeQueueItemResults',
          });
          state = workbenchReducer(state, {
            images: [createImage('first.png', first!.id)],
            projectId: state.activeProjectId,
            queueItemId: first!.id,
            type: 'routeQueueItemResults',
          });

          const entries = getActiveProject(state).workflows.entries;

          expect(entries.find((entry) => entry.document.id === original)?.lastRun?.imageName).toBe('second.png');
          expect(entries.find((entry) => entry.document.id === 'copy')?.lastRun).toBeUndefined();
        } finally {
          vi.useRealTimers();
        }
      });
    });
  });

  describe('seed modes on the compiled submission', () => {
    const SEED_MAX = 4_294_967_295;
    const readSeed = (state: WorkbenchState) => getProjectWidgetValues(getActiveProject(state), 'generate').seed;
    const readSampleSeed = (state: WorkbenchState) =>
      getProjectWidgetValues(getActiveProject(state), 'generate').dynamicPromptsSampleSeed;
    const readSubmission = (state: WorkbenchState) =>
      getActiveProject(state).queue.items[0]?.snapshot.backendSubmission;
    const submitWithPrompts = (
      state: WorkbenchState,
      positivePrompts?: string[],
      positivePromptsSampleSeed = getProjectWidgetValues(getActiveProject(state), 'generate')
        .dynamicPromptsCombinatorial
        ? null
        : (readSampleSeed(state) as number)
    ) =>
      workbenchReducer(state, {
        backendSupportsCancellation: true,
        expandedFrom: preparedExpansionSource(state, 'generate'),
        positivePrompts,
        positivePromptsSampleSeed,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        projectId: state.activeProjectId,
        type: 'submitResolvedInvocationSnapshot',
      });

    it('increment queues consecutive seeds and leaves the editable seed after the batch', () => {
      let state = submitGenerate(primeGenerate(undefined, { batchCount: 3, seed: 42, seedMode: 'increment' }));

      expect(readSubmission(state)).toMatchObject({ batchCount: 3, seed: 42, seedStep: 1 });
      expect(readSeed(state)).toBe(45);

      state = submitGenerate(state);

      expect(readSubmission(state)).toMatchObject({ seed: 45, seedStep: 1 });
      expect(readSeed(state)).toBe(48);
      expect(getActiveProject(state).queue.items[1]?.snapshot.backendSubmission).toMatchObject({ seed: 42 });
    });

    it('advances by every seed a prompt set consumes', () => {
      const perIteration = submitWithPrompts(
        primeGenerate(undefined, {
          batchCount: 2,
          dynamicPromptsSeedBehaviour: 'per-iteration',
          positivePrompt: 'a {red|green} cat',
          seed: 42,
          seedMode: 'increment',
        }),
        ['a red cat', 'a green cat']
      );
      const perImage = submitWithPrompts(
        primeGenerate(undefined, {
          batchCount: 2,
          dynamicPromptsSeedBehaviour: 'per-image',
          positivePrompt: 'a {red|green} cat',
          seed: 42,
          seedMode: 'increment',
        }),
        ['a red cat', 'a green cat']
      );

      expect(readSeed(perIteration)).toBe(44);
      expect(readSeed(perImage)).toBe(46);
    });

    describe('dynamic prompts sample seed', () => {
      const RANDOM_SAMPLE = {
        dynamicPromptsCombinatorial: false,
        dynamicPromptsResample: true,
        dynamicPromptsSampleSeed: 7,
        positivePrompt: 'a {red|green} cat',
      } satisfies Partial<GenerateWidgetValues>;

      it('draws a new sample after queueing the previewed one, alongside the image seed advance', () => {
        const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);

        try {
          const state = submitWithPrompts(
            primeGenerate(undefined, { ...RANDOM_SAMPLE, batchCount: 1, seed: 42, seedMode: 'increment' }),
            ['a green cat']
          );

          expect(readSubmission(state)).toMatchObject({ positivePrompts: ['a green cat'], seed: 42 });
          expect(readSampleSeed(state)).toBe(Math.floor(0.5 * SEED_MAX));
          expect(readSeed(state)).toBe(43);
        } finally {
          random.mockRestore();
        }
      });
    });

    it('advances Upscale and Video seeds through the same boundary', () => {
      const upscaleModels = [
        createUpscaleModel('main', 'main', 'sd-1'),
        createUpscaleModel('spandrel', 'spandrel_image_to_image', 'any'),
        createUpscaleModel('tile', 'controlnet', 'sd-1', 'Tile ControlNet'),
      ];
      let state = workbenchReducer(createInitialWorkbenchState(), {
        type: 'patchWidgetValues',
        values: {
          ...createDefaultUpscaleWidgetValues(upscaleModels),
          batchCount: 2,
          inputImage: { height: 64, image_name: 'input.png', width: 64 },
          seed: 10,
          seedMode: 'increment',
        },
        widgetId: 'upscale',
      });

      state = workbenchReducer(state, {
        backendSupportsCancellation: true,
        models: upscaleModels,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'upscale', sourceLocked: false },
        projectId: state.activeProjectId,
        type: 'submitResolvedInvocationSnapshot',
      });

      expect(readSubmission(state)).toMatchObject({ batchCount: 2, seed: 10, seedStep: 1 });
      expect(getProjectWidgetValues(getActiveProject(state), 'upscale').seed).toBe(12);
      expect(getProjectWidgetValues(getActiveProject(state), 'generate').seed).not.toBe(12);

      const wanModel: ModelConfig = {
        base: 'wan',
        file_size: 1,
        format: 'diffusers',
        hash: 'wan-t2v-hash',
        key: 'wan-t2v_a14b-diffusers',
        name: 'Wan 2.2 t2v_a14b',
        path: 'wan-t2v_a14b-diffusers',
        source: 'wan-t2v_a14b-diffusers',
        source_type: 'path',
        type: 'main',
        variant: 't2v_a14b',
      };
      state = workbenchReducer(createInitialWorkbenchState(), {
        region: 'left',
        type: 'toggleRegionWidget',
        widgetId: 'video',
      });
      state = workbenchReducer(state, {
        type: 'patchWidgetValues',
        values: { batchCount: 3, model: wanModel, positivePrompt: 'a fox running', seed: 7, seedMode: 'decrement' },
        widgetId: 'video',
      });
      state = workbenchReducer(state, { sourceId: 'video', type: 'setInvocationSource' });
      state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
      state = workbenchReducer(state, {
        backendSupportsCancellation: true,
        models: [wanModel],
        projectId: state.activeProjectId,
        route: getActiveProject(state).invocation,
        type: 'submitResolvedInvocationSnapshot',
      });

      expect(readSubmission(state)).toMatchObject({ batchCount: 3, seed: 7, seedStep: -1 });
      expect(getProjectWidgetValues(getActiveProject(state), 'video').seed).toBe(4);
    });

    describe('from the canvas, which compiles outside the reducer', () => {
      const submitCanvas = (state: WorkbenchState, values: GenerateWidgetValues, expansion?: ExpandedPositivePrompts) =>
        workbenchReducer(state, {
          backendSupportsCancellation: true,
          canvas: structuredClone(getActiveProject(state).canvas),
          destination: 'canvas',
          generate: {
            negativePromptNodeId: 'negative_prompt',
            positivePromptNodeId: 'positive_prompt',
            seedNodeId: 'seed',
            values,
          },
          graph: {
            backendGraph: { edges: [], id: 'canvas-backend-graph', nodes: {} },
            edges: [],
            id: 'canvas-graph',
            label: 'Canvas',
            nodes: [],
            updatedAt: '2026-06-09T00:00:00.000Z',
            version: 1,
          },
          ...expansion,
          projectId: state.activeProjectId,
          type: 'submitCanvasInvocationSnapshot',
        });

      it('advances the settings the snapshot was compiled from', () => {
        const values = createGenerateValues({ batchCount: 2, seed: 10, seedMode: 'increment' });
        const state = submitCanvas(primeGenerate(undefined, values), values);

        expect(readSubmission(state)).toMatchObject({ seed: 10, seedStep: 1 });
        expect(readSeed(state)).toBe(12);
      });
    });
  });

  describe('prompt templates on the compiled submission', () => {
    const template = {
      id: 't1',
      name: 'Cinematic',
      negativePrompt: '{prompt}, lowres',
      positivePrompt: '{prompt}, cinematic',
    };

    it('submits the merged prompts while the widget keeps the authored text', () => {
      const state = submitGenerate(
        primeGenerate(undefined, {
          negativePrompt: 'blurry',
          negativePromptEnabled: true,
          positivePrompt: 'a cat',
          promptTemplate: template,
        })
      );
      const queueItem = getActiveProject(state).queue.items[0];

      expect(queueItem?.snapshot.backendSubmission).toMatchObject({
        negativePrompt: 'blurry, lowres',
        positivePrompt: 'a cat, cinematic',
      });
      expect(queueItem?.snapshot.recall?.generateValues?.positivePrompt).toBe('a cat');
    });
  });

  it('records submitted Generate prompt pairs in project prompt history', () => {
    let state = primeGenerate(undefined, { negativePrompt: ' blurry ', positivePrompt: ' a cat ' });

    state = submitGenerate(state);

    expect(getActiveProject(state).promptHistory).toEqual([{ negativePrompt: 'blurry', positivePrompt: 'a cat' }]);
  });

  it('keeps the Video prompt independent of the Generate prompt in both directions', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      sourceId: 'generate',
      type: 'patchProjectPromptDraft',
      values: { negativePrompt: 'blurry', negativePromptEnabled: true, positivePrompt: 'a still portrait' },
    });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { negativePrompt: 'static camera', negativePromptEnabled: false, positivePrompt: 'a fox running' },
      widgetId: 'video',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'generate')).toMatchObject({
      negativePrompt: 'blurry',
      negativePromptEnabled: true,
      positivePrompt: 'a still portrait',
    });
    expect(getProjectWidgetValues(getActiveProject(state), 'video')).toMatchObject({
      negativePrompt: 'static camera',
      negativePromptEnabled: false,
      positivePrompt: 'a fox running',
    });

    const beforeVideo = getProjectWidgetValues(getActiveProject(state), 'video');

    state = workbenchReducer(state, {
      sourceId: 'generate',
      type: 'patchProjectPromptDraft',
      values: { positivePrompt: 'a still landscape' },
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'video')).toBe(beforeVideo);
  });

  it('deduplicates prompt history by prompt pair and moves the newest submission to the top', () => {
    let state = primeGenerate(undefined, { negativePrompt: 'low quality', positivePrompt: 'a cat' });

    state = submitGenerate(state);
    state = workbenchReducer(state, {
      type: 'setGenerateSettings',
      values: createGenerateValues({ negativePrompt: 'blurry', positivePrompt: 'a dog' }),
    });
    state = submitGenerate(state);
    state = workbenchReducer(state, {
      type: 'setGenerateSettings',
      values: createGenerateValues({ negativePrompt: 'low quality', positivePrompt: 'a cat' }),
    });
    state = submitGenerate(state);

    expect(getActiveProject(state).promptHistory).toEqual([
      { negativePrompt: 'low quality', positivePrompt: 'a cat' },
      { negativePrompt: 'blurry', positivePrompt: 'a dog' },
    ]);
  });

  it('hydrates older projects with empty prompt history', () => {
    const initial = createInitialWorkbenchState();
    const legacyState = {
      ...initial,
      projects: initial.projects.map(({ promptHistory: _promptHistory, ...project }) => project),
    } as unknown as WorkbenchState;

    const hydrated = workbenchReducer(initial, { state: legacyState, type: 'hydrateWorkbench' });

    expect(getActiveProject(hydrated).promptHistory).toEqual([]);
  });

  it('routes Gallery destination results to Gallery without staging them on Canvas', () => {
    let state = primeGenerate();

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('gallery-image.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    const updatedProject = getActiveProject(state);
    const galleryValues = getProjectWidgetValues(updatedProject, 'gallery');
    const expectedImage = {
      ...createImage('gallery-image.png', queueItem.id),
      boardId: 'none',
      imageCategory: 'general' as const,
      starred: false,
    };

    expect(updatedProject.canvas.stagingArea.pendingImages).toEqual([]);
    expect(galleryValues.recentImages).toEqual([expectedImage]);
    expect(galleryValues.selectedImage).toEqual(legacyGeneratedImageToGalleryItem(expectedImage));
    expect(galleryValues.selectedImageName).toBe('image:gallery-image.png');
  });

  it('preserves the browsed window when a result lands on the viewed board', () => {
    let state = primeGenerate();

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { page: 11, type: 'setGalleryPage' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    expect(getProjectWidgetValues(project, 'gallery').galleryPage).toBe(11);

    state = workbenchReducer(state, {
      images: [createImage('fresh.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').galleryPage).toBe(11);
  });

  it('retains only the newest 60 images after 1,000 routed Gallery results', () => {
    let state = primeGenerate();

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    for (let index = 0; index < 1_000; index += 1) {
      state = workbenchReducer(state, {
        images: [createImage(`gallery-image-${index}.png`, queueItem.id)],
        projectId: project.id,
        queueItemId: queueItem.id,
        type: 'routeQueueItemResults',
      });
    }

    const galleryValues = getProjectWidgetValues(getActiveProject(state), 'gallery');
    const recentImageNames = (galleryValues.recentImages as GeneratedImageContract[]).map((image) => image.imageName);

    expect(GALLERY_RECENT_IMAGE_LIMIT).toBe(60);
    expect(recentImageNames).toEqual(
      Array.from({ length: GALLERY_RECENT_IMAGE_LIMIT }, (_, index) => `gallery-image-${999 - index}.png`)
    );
    expect(galleryValues.selectedImageName).toBe('image:gallery-image-999.png');
    expect(galleryValues.selectedImageNames).toEqual(['image:gallery-image-999.png']);
  });

  it('validates, deduplicates, and truncates persisted recent images during hydration', () => {
    const initial = createInitialWorkbenchState();
    const legacyImages = Array.from({ length: 1_000 }, (_, index) =>
      createImage(`persisted-image-${999 - index}.png`, 'legacy-queue-item')
    );
    const selectedImage = legacyImages.at(-1)!;
    const legacyRecentImages: unknown[] = [
      legacyImages[0],
      null,
      legacyImages[0],
      { imageName: 42 },
      ...legacyImages.slice(1),
    ];
    const persisted: WorkbenchState = {
      ...initial,
      projects: initial.projects.map((project) => {
        const galleryInstance = project.widgetInstances.gallery!;

        return {
          ...project,
          widgetInstances: {
            ...project.widgetInstances,
            gallery: {
              ...galleryInstance,
              state: {
                ...galleryInstance.state,
                values: {
                  ...galleryInstance.state.values,
                  recentImages: legacyRecentImages,
                  selectedImage,
                  selectedImageName: selectedImage.imageName,
                },
              },
            },
          },
        };
      }),
    };

    const hydrated = workbenchReducer(initial, { state: persisted, type: 'hydrateWorkbench' });
    const galleryValues = getProjectWidgetValues(getActiveProject(hydrated), 'gallery');
    const recentImageNames = (galleryValues.recentImages as GeneratedImageContract[]).map((image) => image.imageName);

    expect(recentImageNames).toEqual(
      Array.from({ length: GALLERY_RECENT_IMAGE_LIMIT }, (_, index) => `persisted-image-${999 - index}.png`)
    );
    expect(galleryValues.selectedImage).toEqual(selectedImage);
    expect(galleryValues.selectedImageName).toBe(selectedImage.imageName);
  });

  it('routes partial Gallery destination results without completing the local queue item', () => {
    let state = primeGenerate(undefined, { batchCount: 2 });

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [createImage('gallery-image-1.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });

    const updatedQueueItem = getActiveProject(state).queue.items[0];
    const galleryValues = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(updatedQueueItem.status).toBe('pending');
    expect(updatedQueueItem.completedBackendItemIds).toEqual([11]);
    expect(updatedQueueItem.resultImages).toEqual([createImage('gallery-image-1.png', queueItem.id)]);
    expect((galleryValues.recentImages as GeneratedImageContract[]).map((image) => image.imageName)).toEqual([
      'gallery-image-1.png',
    ]);
  });

  it.each(['latest', 'progress'] as const)(
    'turns Auto-switch Off in the same step when the user selects a saved Gallery image under %s',
    (mode) => {
      let state = workbenchReducer(createInitialWorkbenchState(), {
        settings: { viewerAutoSwitch: mode },
        type: 'setActiveProjectSettings',
      });

      state = workbenchReducer(state, { item: createGalleryImageItem('selected.png'), type: 'selectGalleryItem' });

      expect(getActiveProject(state).settings.viewerAutoSwitch).toBe('off');
      expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe('image:selected.png');
    }
  );

  describe('a comparison waiting for its second image', () => {
    const galleryValues = (state: WorkbenchState) => getProjectWidgetValues(getActiveProject(state), 'gallery');
    const arm = (name = 'base.png') => {
      const base = createGalleryImageItem(name);
      let state = workbenchReducer(createInitialWorkbenchState(), { item: base, type: 'selectGalleryItem' });

      state = workbenchReducer(state, { image: base, type: 'setGalleryCompareImage' });
      return state;
    };

    it('compares the next image picked against it', () => {
      const state = workbenchReducer(arm(), { item: createGalleryImageItem('other.png'), type: 'selectGalleryItem' });

      expect(galleryValues(state)).toMatchObject({
        compareImage: { name: 'base.png' },
        selectedImageName: 'image:other.png',
      });
    });
  });

  describe('deleting the shown image of an open comparison', () => {
    const galleryValues = (state: WorkbenchState) => getProjectWidgetValues(getActiveProject(state), 'gallery');
    const compareWithReference = () => {
      const state = workbenchReducer(createInitialWorkbenchState(), {
        item: createGalleryImageItem('candidate-1.png'),
        type: 'selectGalleryItem',
      });

      return workbenchReducer(state, {
        image: createGalleryImageItem('reference.png'),
        type: 'setGalleryCompareImage',
      });
    };

    it('keeps the comparison for the next candidate the deletion selects', () => {
      let state = workbenchReducer(compareWithReference(), {
        itemKeys: ['image:candidate-1.png'],
        keepComparisonForSuccessor: true,
        type: 'removeGalleryItems',
      });

      expect(galleryValues(state).compareImage).toMatchObject({ name: 'reference.png' });
      state = workbenchReducer(state, { item: createGalleryImageItem('candidate-2.png'), type: 'selectGalleryItem' });
      expect(galleryValues(state)).toMatchObject({
        compareImage: { name: 'reference.png' },
        selectedImageName: 'image:candidate-2.png',
      });
    });
  });

  describe('deleting an image shown in another project', () => {
    const galleryValues = (state: WorkbenchState, projectId: string) =>
      getProjectWidgetValues(getProject(state, projectId), 'gallery');
    const twoComparisons = () => {
      let state = createInitialWorkbenchState();
      const deletingProjectId = state.activeProjectId;

      state = workbenchReducer(state, { item: createGalleryImageItem('x.png'), type: 'selectGalleryItem' });
      state = workbenchReducer(state, {
        image: createGalleryImageItem('reference.png'),
        type: 'setGalleryCompareImage',
      });
      state = workbenchReducer(state, { type: 'createProject' });
      const otherProjectId = state.activeProjectId;

      state = workbenchReducer(state, { item: createGalleryImageItem('x.png'), type: 'selectGalleryItem' });
      state = workbenchReducer(state, { image: createGalleryImageItem('y.png'), type: 'setGalleryCompareImage' });
      state = workbenchReducer(state, { projectId: deletingProjectId, type: 'switchProject' });

      return { deletingProjectId, otherProjectId, state };
    };

    it('keeps the comparison for the successor only in the deleting project', () => {
      const { deletingProjectId, otherProjectId, state: compared } = twoComparisons();
      let state = workbenchReducer(compared, {
        itemKeys: ['image:x.png'],
        keepComparisonForSuccessor: true,
        type: 'removeGalleryItems',
      });

      expect(galleryValues(state, deletingProjectId).compareImage).toMatchObject({ name: 'reference.png' });
      expect(galleryValues(state, otherProjectId)).toMatchObject({
        compareImage: null,
        selectedImage: null,
        selectedImageName: null,
        selectedImageNames: [],
      });

      state = workbenchReducer(state, {
        item: createGalleryImageItem('later.png'),
        projectId: otherProjectId,
        type: 'selectGalleryItem',
      });
      expect(galleryValues(state, otherProjectId).compareImage).toBeNull();
    });
  });

  it('preserves a manually selected image when later Gallery results arrive', () => {
    let state = primeGenerate();
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    const selectedImage = createGalleryImageItem('selected.png');
    state = workbenchReducer(state, { item: selectedImage, type: 'selectGalleryItem' });
    state = workbenchReducer(state, {
      images: [createImage('completed.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe('image:selected.png');
  });

  it.each(['latest', 'progress'] as const)('selects each finished result under %s', (mode) => {
    let state = primeGenerate();
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { settings: { viewerAutoSwitch: mode }, type: 'setActiveProjectSettings' });
    state = submitGenerate(state);
    const first = getActiveProject(state).queue.items[0]!;
    state = submitGenerate(state);
    const second = getActiveProject(state).queue.items[0]!;
    const projectId = getActiveProject(state).id;

    for (const [name, item] of [
      ['first.png', first],
      ['second.png', second],
    ] as const) {
      state = workbenchReducer(state, {
        images: [createImage(name, item.id)],
        projectId,
        queueItemId: item.id,
        type: 'routeQueueItemResults',
      });
      expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe(`image:${name}`);
    }
  });

  it('selects a finished Gallery video once, so a later result keeps the preview', () => {
    let state = primeGenerate();
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);
    const videoItem = getActiveProject(state).queue.items[0];
    const projectId = getActiveProject(state).id;

    state = workbenchReducer(state, {
      backendItemId: 1,
      images: [],
      projectId,
      queueItemId: videoItem.id,
      type: 'routeQueueItemPartialResults',
      videos: [createVideo('clip.mp4', videoItem.id, 'board-1')],
    });
    const selected = getProjectWidgetValues(getActiveProject(state), 'gallery');
    expect(selected.selectedImageName).toBe('video:clip.mp4');
    expect(selected.selectedImage).toMatchObject({
      boardId: 'board-1',
      durationSeconds: 5,
      fullUrl: '/api/v1/videos/i/clip.mp4/full',
      kind: 'video',
    });

    state = submitGenerate(state);
    const imageItem = getActiveProject(state).queue.items[0];
    state = workbenchReducer(state, {
      images: [createImage('later.png', imageItem.id)],
      projectId,
      queueItemId: imageItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, {
      images: [],
      projectId,
      queueItemId: videoItem.id,
      type: 'routeQueueItemResults',
      videos: [createVideo('clip.mp4', videoItem.id)],
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe('image:later.png');
  });

  it('stamps a host page into the navigation query already on a multi-selection', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'board-deep', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, {
      item: createGalleryImageItem('deep.png'),
      preserveNavigationQuery: false,
      selectionPage: 30,
      type: 'selectGalleryItem',
    });
    state = workbenchReducer(state, { boardId: 'board-elsewhere', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, { searchTerm: 'sunset', type: 'setGallerySearchTerm' });
    state = workbenchReducer(state, {
      itemKeys: ['image:failed.png', 'image:successor.png'],
      preserveNavigationQuery: true,
      primaryItem: createGalleryImageItem('successor.png'),
      selectionPage: 30,
      type: 'setGalleryMultiSelection',
    });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');
    const query = values.selectedImageQuery as { boardId: string; page: number; searchTerm: string };

    expect(values.selectedImagePage).toBe(30);
    expect(query.page).toBe(30);
    expect(query.boardId).toBe('board-deep');
    expect(query.searchTerm).toBe('');
    expect(values.galleryPage).toBe(0);
  });

  it('ties a selected semantic result page to the active ranking', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { semanticImageQuery: { kind: 'text', query: 'sunset' } },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, {
      item: createGalleryImageItem('ranked.png'),
      selectionPage: 3,
      type: 'selectGalleryItem',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageQuery).toMatchObject({
      page: 3,
      semanticKey: 'text:sunset',
    });
  });

  it('returns the search scope to the selected board whatever clears the search, so the next search starts there', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), {
      type: 'patchWidgetValues',
      values: { searchScope: 'all', searchTerm: 'cat' },
      widgetId: 'gallery',
    });
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').searchScope).toBe('all');

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { searchTerm: '', semanticImageQuery: null, semanticSearchText: null, starredOnly: false },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, { type: 'patchWidgetValues', values: { searchTerm: 'dog' }, widgetId: 'gallery' });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery')).toMatchObject({
      searchScope: 'board',
      searchTerm: 'dog',
    });
  });

  it('carries a similarity search to another board, which it then ranks within, from the first page', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'board-a', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { galleryPage: 4, semanticImageQuery: { kind: 'text', query: 'sunset' }, semanticSearchText: 'sunset' },
      widgetId: 'gallery',
    });

    state = workbenchReducer(state, { boardId: 'board-b', type: 'selectGalleryBoard' });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.semanticImageQuery).toEqual({ kind: 'text', query: 'sunset' });
    expect(values.semanticSearchText).toBe('sunset');
    expect(values.selectedBoardId).toBe('board-b');
    expect(values.galleryPage).toBe(0);
  });

  it('sends gallery submissions to the project inbox, not the board being browsed, unless following it', () => {
    let state = createInitialWorkbenchState();

    listGalleryBoardsInCache('browsed', 'inbox');
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { projectBoardId: 'inbox' },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, { boardId: 'browsed', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = primeGenerate(state);
    state = submitGenerate(state);

    expect(getActiveProject(state).queue.items[0]?.snapshot.galleryBoardId).toBe('inbox');
  });

  it('folds a stream of workflow field edits into one undo step and starts another after a pause', () => {
    vi.useFakeTimers({ now: new Date('2026-06-10T00:00:00.000Z') });

    try {
      let state = createInitialWorkbenchState();
      const setValue = (fieldName: string, value: number) =>
        workbenchReducer(state, {
          action: { fieldName, nodeId: 'node-1', type: 'setFieldValue', value },
          type: 'applyWorkflowAction',
        });
      const fieldValue = (fieldName: string) => {
        const node = getActiveProjectGraph(getActiveProject(state)).nodes[0];

        return node?.type === 'invocation' ? node.data.inputs[fieldName]?.value : undefined;
      };
      const past = () =>
        getActiveProject(state).workflowHistories[getActiveProject(state).workflows.activeWorkflowId]?.past ?? [];

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
        type: 'applyWorkflowAction',
      });
      const stepsAfterAdd = past().length;

      // Keystrokes a second apart keep one edit going: the window slides with each keystroke.
      state = setValue('a', 1);
      vi.setSystemTime(new Date('2026-06-10T00:00:01.000Z'));
      state = setValue('a', 12);
      vi.setSystemTime(new Date('2026-06-10T00:00:02.000Z'));
      state = setValue('a', 123);

      expect(past()).toHaveLength(stepsAfterAdd + 1);
      expect(past().at(-1)?.label).toBe('Edit workflow field value');

      vi.setSystemTime(new Date('2026-06-10T00:00:05.000Z'));
      state = setValue('a', 3);
      state = setValue('b', 7);

      expect(past()).toHaveLength(stepsAfterAdd + 3);

      state = workbenchReducer(state, { type: 'undoWorkflowChange' });
      expect(fieldValue('b')).toBeUndefined();
      expect(fieldValue('a')).toBe(3);

      state = workbenchReducer(state, { type: 'undoWorkflowChange' });
      expect(fieldValue('a')).toBe(123);

      state = workbenchReducer(state, { type: 'undoWorkflowChange' });
      expect(fieldValue('a')).toBeUndefined();

      state = workbenchReducer(state, { type: 'redoWorkflowChange' });
      expect(fieldValue('a')).toBe(123);

      state = setValue('a', 4);
      expect(past()).toHaveLength(stepsAfterAdd + 2);
      expect(getActiveProject(state).undoRedo.future).toHaveLength(0);

      state = setValue('b', 8);
      state = workbenchReducer(state, { type: 'undoWorkflowChange' });
      state = setValue('a', 5);

      expect(past()).toHaveLength(stepsAfterAdd + 3);
      state = workbenchReducer(state, { type: 'undoWorkflowChange' });
      expect(fieldValue('a')).toBe(4);

      state = workbenchReducer(state, {
        action: { nodeId: 'node-1', position: { x: 10, y: 10 }, type: 'setNodePosition' },
        type: 'applyWorkflowAction',
      });
      expect(past()).toHaveLength(stepsAfterAdd + 2);
    } finally {
      vi.useRealTimers();
    }
  });
  describe('gallery semantic search mode', () => {
    const galleryValues = (state: WorkbenchState) => getProjectWidgetValues(getActiveProject(state), 'gallery');

    it('entering carries the text across and ranks it at once; leaving carries it back', () => {
      let state = createInitialWorkbenchState();

      state = workbenchReducer(state, { searchTerm: ' sunset ', type: 'setGallerySearchTerm' });
      state = workbenchReducer(state, { page: 3, type: 'setGalleryPage' });
      state = workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' });

      expect(galleryValues(state)).toMatchObject({
        galleryPage: 0,
        searchTerm: '',
        semanticImageQuery: { kind: 'text', query: 'sunset' },
        semanticSearchText: ' sunset ',
      });

      state = workbenchReducer(state, { page: 2, type: 'setGalleryPage' });
      state = workbenchReducer(state, { enabled: false, type: 'setGallerySemanticSearchMode' });

      expect(galleryValues(state)).toMatchObject({
        galleryPage: 0,
        searchTerm: ' sunset ',
        semanticImageQuery: null,
        semanticSearchText: null,
      });
    });

    it('commits only the text the field still holds', () => {
      let state = createInitialWorkbenchState();

      state = workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' });
      state = workbenchReducer(state, { text: 'sun', type: 'setGallerySemanticSearchText' });
      state = workbenchReducer(state, { page: 2, type: 'setGalleryPage' });

      expect(galleryValues(state).semanticSearchText).toBe('sun');
      expect(workbenchReducer(state, { text: 'su', type: 'commitGallerySemanticSearch' })).toBe(state);

      state = workbenchReducer(state, { text: 'sun', type: 'commitGallerySemanticSearch' });

      expect(galleryValues(state)).toMatchObject({
        galleryPage: 0,
        semanticImageQuery: { kind: 'text', query: 'sun' },
        semanticSearchText: 'sun',
      });

      state = workbenchReducer(state, { page: 4, type: 'setGalleryPage' });
      state = workbenchReducer(state, { text: 'sun ', type: 'setGallerySemanticSearchText' });

      const paged = state;

      state = workbenchReducer(state, { text: 'sun ', type: 'commitGallerySemanticSearch' });

      expect(state).toBe(paged);

      state = workbenchReducer(state, { text: '  ', type: 'setGallerySemanticSearchText' });
      state = workbenchReducer(state, { text: '  ', type: 'commitGallerySemanticSearch' });

      expect(galleryValues(state)).toMatchObject({
        galleryPage: 0,
        semanticImageQuery: null,
        semanticSearchText: '  ',
      });
    });
  });
});

describe('workbench account and project settings', () => {
  it('drops legacy preferences carried inside persisted accounts on hydrate', () => {
    const initial = createInitialWorkbenchState();
    const legacy = {
      ...initial,
      account: { activeLayoutPresetId: 'gallery', preferences: { themeId: 'osakaJade' } },
    } as unknown as WorkbenchState;

    const state = workbenchReducer(initial, { state: legacy, type: 'hydrateWorkbench' });

    expect(state.account).toEqual({
      activeLayoutPresetId: 'compose',
      customLayoutPresets: [],
      layoutPresetMetadataOverrides: {},
      layoutPresetOrder: ['compose', 'edit', 'video', 'automate'],
      layoutPresetOverrides: {},
      layoutPresetRouteOverrides: {},
      workspaceLayoutsAdopted: true,
    });
  });

  it('normalizes persisted preset routes and tolerates legacy custom presets without one', () => {
    let seeded = createInitialWorkbenchState();
    seeded = addCustomWorkspace(seeded, { id: 'custom-route-fixture', label: 'Route fixture' });
    const fixture = seeded.account.customLayoutPresets?.[0];
    expect(fixture).toBeDefined();
    if (!fixture) {
      return;
    }

    const { defaultRoute: _removedRoute, ...legacyFixture } = fixture;
    const hydrated = {
      ...seeded,
      account: {
        activeLayoutPresetId: 'compose',
        customLayoutPresets: [
          legacyFixture,
          {
            ...fixture,
            defaultRoute: { destination: 'gallery', sourceId: 'workflow' },
            id: 'custom-valid-route',
          },
          {
            ...fixture,
            defaultRoute: { destination: 'nowhere', sourceId: 'unknown' },
            id: 'custom-invalid-route',
          },
        ],
        layoutPresetRouteOverrides: {
          broken: { destination: 'nowhere', sourceId: 'unknown' },
          canvas: { destination: 'gallery', sourceId: 'upscale' },
        },
      },
    } as unknown as WorkbenchState;

    const state = workbenchReducer(seeded, { state: hydrated, type: 'hydrateWorkbench' });

    expect(state.account.customLayoutPresets?.map((preset) => [preset.id, preset.defaultRoute])).toEqual([
      ['custom-route-fixture', undefined],
      ['custom-valid-route', { destination: 'gallery', sourceId: 'workflow' }],
      ['custom-invalid-route', undefined],
    ]);
    expect(state.account.layoutPresetRouteOverrides).toEqual({
      edit: { destination: 'gallery', sourceId: 'upscale' },
    });
  });

  it('updates project settings on the active project only', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      settings: { useCpuNoise: false },
      type: 'setActiveProjectSettings',
    });

    expect(getActiveProject(state).settings).toEqual({ useCpuNoise: false, viewerAutoSwitch: 'progress' });

    state = workbenchReducer(state, { settings: { useCpuNoise: undefined }, type: 'setActiveProjectSettings' });

    expect(getActiveProject(state).settings).toEqual({ viewerAutoSwitch: 'progress' });
  });
});

describe('workbench backend connection recovery', () => {
  it('tracks backend connection status', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      status: 'disconnected',
      type: 'setBackendConnectionStatus',
      error: 'server down',
    });

    expect(state.backendConnection.status).toBe('disconnected');
    expect(state.backendConnection.error).toBe('server down');
    expect(state.backendConnection.lastDisconnectedAt).toBeDefined();

    state = workbenchReducer(state, { status: 'connected', type: 'setBackendConnectionStatus' });

    expect(state.backendConnection.status).toBe('connected');
    expect(state.backendConnection.error).toBeUndefined();
    expect(state.backendConnection.lastConnectedAt).toBeDefined();
  });

  it('prunes qualified items across projects without treating a same-name video as an image input', () => {
    const image = createGalleryImageItem('shared');
    const video = createGalleryVideoItem('shared');
    const recentImage = galleryItemToRecentImage(image);
    let state = createInitialWorkbenchState();
    const firstProjectId = state.activeProjectId;

    state = workbenchReducer(state, { type: 'createProject' });
    const secondProjectId = state.activeProjectId;
    state = workbenchReducer(state, {
      projectId: firstProjectId,
      type: 'patchWidgetValues',
      values: {
        compareImage: image,
        recentImages: [recentImage],
        selectedImage: image,
        selectedImageName: 'image:shared',
        selectedImageNames: ['image:shared'],
      },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, {
      projectId: secondProjectId,
      type: 'patchWidgetValues',
      values: {
        recentImages: [recentImage],
        selectedImage: video,
        selectedImageName: 'video:shared',
        selectedImageNames: ['video:shared'],
      },
      widgetId: 'gallery',
    });
    for (const projectId of [firstProjectId, secondProjectId]) {
      state = workbenchReducer(state, {
        projectId,
        type: 'patchWidgetValues',
        values: { inputImage: { height: image.height, image_name: image.name, width: image.width } },
        widgetId: 'upscale',
      });
    }

    state = workbenchReducer(state, {
      itemKeys: ['video:shared'],
      type: 'removeGalleryItems',
    });

    expect(getProjectWidgetValues(getProject(state, firstProjectId), 'gallery')).toMatchObject({
      compareImage: image,
      recentImages: [recentImage],
      selectedImage: image,
      selectedImageName: 'image:shared',
      selectedImageNames: ['image:shared'],
    });
    expect(getProjectWidgetValues(getProject(state, secondProjectId), 'gallery')).toMatchObject({
      recentImages: [recentImage],
      selectedImage: null,
      selectedImageName: null,
      selectedImageNames: [],
    });
    expect(getProjectWidgetValues(getProject(state, firstProjectId), 'upscale').inputImage).not.toBeNull();
    expect(getProjectWidgetValues(getProject(state, secondProjectId), 'upscale').inputImage).not.toBeNull();

    state = workbenchReducer(state, {
      itemKeys: ['image:shared'],
      type: 'removeGalleryItems',
    });

    for (const projectId of [firstProjectId, secondProjectId]) {
      expect(getProjectWidgetValues(getProject(state, projectId), 'gallery').recentImages).toEqual([]);
      expect(getProjectWidgetValues(getProject(state, projectId), 'upscale').inputImage).toBeNull();
    }
  });

  it('reconciles authoritative board outcomes by deleting confirmed media and moving every survivor', () => {
    const boardId = 'deleted-board';
    const deletedImage = createGalleryImageItem('deleted-image', boardId);
    const failedImage = createGalleryImageItem('failed-image', boardId);
    const deletedVideo = createGalleryVideoItem('deleted-video', boardId);
    const failedVideo = createGalleryVideoItem('failed-video', boardId);
    let state = createInitialWorkbenchState();

    while (state.projects.length < 4) {
      state = workbenchReducer(state, { type: 'createProject' });
    }

    const fixtures = [
      {
        item: deletedImage,
        recentImages: [galleryItemToRecentImage(deletedImage)],
      },
      {
        item: failedImage,
        recentImages: [galleryItemToRecentImage(failedImage)],
      },
      { item: deletedVideo, recentImages: [] },
      { item: failedVideo, recentImages: [] },
    ] as const;

    for (const [index, project] of state.projects.entries()) {
      const fixture = fixtures[index]!;
      state = workbenchReducer(state, {
        projectId: project.id,
        type: 'patchWidgetValues',
        values: {
          projectBoardId: index === 0 ? boardId : null,
          recentImages: fixture.recentImages,
          selectedBoardId: boardId,
          selectedImage: fixture.item,
          selectedImageName: `${fixture.item.kind}:${fixture.item.name}`,
          selectedImageNames: [`${fixture.item.kind}:${fixture.item.name}`],
        },
        widgetId: 'gallery',
      });
    }

    state = workbenchReducer(state, {
      outcome: {
        boardId,
        deletedBoardImageNames: [],
        deletedBoardVideoNames: [],
        deletedImageNames: [deletedImage.name],
        deletedVideoNames: [deletedVideo.name],
        failedImageNames: [failedImage.name],
        failedVideoNames: [failedVideo.name],
      },
      type: 'reconcileDeletedGalleryBoard',
    });

    const [deletedImageProject, failedImageProject, deletedVideoProject, failedVideoProject] = state.projects.map(
      (project) => getProjectWidgetValues(project, 'gallery')
    );

    expect(deletedImageProject).toMatchObject({
      projectBoardId: null,
      recentImages: [],
      selectedBoardId: 'none',
      selectedImage: null,
      selectedImageName: null,
      selectedImageNames: [],
    });
    expect(failedImageProject?.selectedImage).toMatchObject({ boardId: 'none', kind: 'image', name: failedImage.name });
    expect(failedImageProject?.recentImages).toEqual([{ ...galleryItemToRecentImage(failedImage), boardId: 'none' }]);
    expect(deletedVideoProject).toMatchObject({
      selectedImage: null,
      selectedImageName: null,
      selectedImageNames: [],
    });
    expect(failedVideoProject?.selectedImage).toMatchObject({ boardId: 'none', kind: 'video', name: failedVideo.name });

    for (const values of [failedImageProject, deletedVideoProject, failedVideoProject]) {
      expect(values?.selectedBoardId).toBe('none');
    }
  });

  it('removes deleted backend images and upscale references from every project', () => {
    let state = createInitialWorkbenchState();
    const firstProjectId = state.activeProjectId;
    const image = createImage('shared.png', 'backend-gallery');

    state = workbenchReducer(state, { type: 'createProject' });
    const secondProjectId = state.activeProjectId;
    state = workbenchReducer(state, {
      projectId: firstProjectId,
      type: 'patchWidgetValues',
      values: { recentImages: [image], selectedImage: image, selectedImageName: image.imageName },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, {
      projectId: secondProjectId,
      type: 'patchWidgetValues',
      values: { recentImages: [image], selectedImage: image, selectedImageName: image.imageName },
      widgetId: 'gallery',
    });
    for (const projectId of [firstProjectId, secondProjectId]) {
      state = workbenchReducer(state, {
        projectId,
        type: 'patchWidgetValues',
        values: { inputImage: { height: image.height, image_name: image.imageName, width: image.width } },
        widgetId: 'upscale',
      });
    }

    state = workbenchReducer(state, {
      itemKeys: [`image:${image.imageName}`],
      type: 'removeGalleryItems',
    });

    expect(getProjectWidgetValues(getProject(state, firstProjectId), 'gallery').recentImages).toEqual([]);
    expect(getProjectWidgetValues(getProject(state, secondProjectId), 'gallery').recentImages).toEqual([]);
    expect(getProjectWidgetValues(getProject(state, firstProjectId), 'upscale').inputImage).toBeNull();
    expect(getProjectWidgetValues(getProject(state, secondProjectId), 'upscale').inputImage).toBeNull();
  });

  it('does not hydrate stale persisted backend connection state', () => {
    const initial = createInitialWorkbenchState();
    const persisted = {
      ...initial,
      backendConnection: { lastConnectedAt: '2026-06-10T00:00:00.000Z', status: 'connected' },
    } as WorkbenchState;

    const state = workbenchReducer(initial, { state: persisted, type: 'hydrateWorkbench' });

    expect(state.backendConnection).toEqual({ status: 'connecting' });
  });

  it('does not hydrate stale notifications that would toast again after reload', () => {
    const initial = createInitialWorkbenchState();
    const persisted = workbenchReducer(initial, {
      kind: 'success',
      message: 'Old success toast',
      title: 'Invocation completed',
      type: 'recordNotice',
    });

    const state = workbenchReducer(initial, { state: persisted, type: 'hydrateWorkbench' });

    expect(persisted.notifications).toHaveLength(1);
    expect(state.notifications).toEqual([]);
  });
});

describe('nextLayerName', () => {
  it('fills the lowest gap so names do not collide after a deletion', () => {
    // The lowest free slot, not the count-derived "Layer 3", which would collide.
    expect(nextLayerName(['Layer 1', 'Layer 3'])).toBe('Layer 2');
  });
});

describe('workbenchReducer canvas v2 layer reducers', () => {
  it('routes a canvas mutation to its bound project after the active project switches', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('origin-layer')]);
    const originProjectId = state.activeProjectId;

    state = workbenchReducer(state, { type: 'createProject' });
    const activeProjectId = state.activeProjectId;
    state = workbenchReducer(state, {
      mutation: { id: 'origin-layer', patch: { opacity: 0.25 }, type: 'updateCanvasLayer' },
      projectId: originProjectId,
      type: 'applyCanvasProjectMutation',
    } as WorkbenchAction);

    expect(getDocumentLeaves(getProject(state, originProjectId).canvas.document)[0]?.opacity).toBe(0.25);
    expect(getDocumentLeaves(getProject(state, activeProjectId).canvas.document)[0]?.opacity).not.toBe(0.25);
  });

  it('seeds a new project canvas with a single empty inpaint mask, unselected', () => {
    const state = createInitialWorkbenchState();
    const { document } = getActiveProject(state).canvas;

    expect(getDocumentLeaves(document)).toHaveLength(1);
    const mask = getDocumentLeaves(document)[0];
    expect(mask?.type).toBe('inpaint_mask');
    expect(mask && 'mask' in mask ? mask.mask.bitmap : 'missing').toBeNull();
    expect(document.selectedLayerId).toBeNull();
  });

  it('adds a layer at its anchor and selects it', () => {
    let state = withEmptyCanvas(createInitialWorkbenchState());

    state = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('a'),
      type: 'addCanvasLayer',
    });
    state = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('b'),
      type: 'addCanvasLayer',
    });

    expect(getLayerIds(state)).toEqual(['b', 'a']);
    expect(getCanvas(state).document.selectedLayerId).toBe('b');

    state = workbenchReducer(state, {
      anchor: { ...stackTopAnchor(state.activeProjectId), beforeId: 'a' },
      layer: createRasterLayer('c'),
      type: 'addCanvasLayer',
    });

    expect(getLayerIds(state)).toEqual(['b', 'c', 'a']);

    const foreign = workbenchReducer(state, {
      anchor: stackTopAnchor('other-project'),
      layer: createRasterLayer('d'),
      type: 'addCanvasLayer',
    });
    const wrongStack = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId, 'control'),
      layer: createRasterLayer('d'),
      type: 'addCanvasLayer',
    });
    expect(foreign).toBe(state);
    expect(wrongStack).toBe(state);
    expect(getCanvas(state).document.selectedLayerId).toBe('c');
  });

  it('commits a staged image at its anchor and rolls it back by identity', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    state = workbenchReducer(state, { id: 'b', type: 'setCanvasSelectedLayer' });
    const candidate: Project['canvas']['stagingArea']['pendingImages'][number] = {
      height: 64,
      imageName: 'staged.png',
      imageUrl: 'url',
      placement: { height: 64, opacity: 1, width: 64, x: 0, y: 0 },
      queuedAt: 'now',
      sourceQueueItemId: 'queue-1',
      thumbnailUrl: 'thumb',
      width: 64,
    };
    const stagingArea: Project['canvas']['stagingArea'] = {
      ...getCanvas(state).stagingArea,
      isVisible: true,
      pendingImageIds: ['queue-1'],
      pendingImages: [candidate],
      selectedImageIndex: 0,
      sourceQueueItemId: 'queue-1',
    };
    state = {
      ...state,
      projects: state.projects.map((project) =>
        project.id === state.activeProjectId ? { ...project, canvas: { ...project.canvas, stagingArea } } : project
      ),
    };
    const projectId = state.activeProjectId;
    const layer: CanvasRasterLayerContractV2 = { ...createRasterLayer('accepted', 'staged.png'), name: 'Accepted' };
    const event = {
      createdAt: '2026-07-16T00:00:00.000Z',
      id: 'event-accepted',
      summary: 'Accepted staged.png into a new raster layer',
      type: 'canvas-layer-accepted' as const,
    };
    const commit = (anchor: CanvasNodeInsertionAnchor) =>
      workbenchReducer(state, {
        anchor,
        candidateFingerprint: getCanvasStagingCandidateFingerprint(candidate),
        continueStaging: false,
        event,
        layer,
        selectedImageIndex: 0,
        type: 'commitStagedImage',
      });

    expect(commit(stackTopAnchor('other-project'))).toBe(state);
    expect(commit(stackTopAnchor(projectId, 'control'))).toBe(state);

    const committed = commit({ ...stackTopAnchor(projectId), afterId: 'a' });
    expect(getLayerIds(committed)).toEqual(['a', 'accepted', 'b']);
    expect(getCanvas(committed).document.selectedLayerId).toBe('accepted');
    expect(getCanvas(committed).stagingArea.pendingImages).toEqual([]);

    const rollback = (rolledBackLayer: CanvasRasterLayerContractV2) =>
      workbenchReducer(committed, {
        continueStaging: false,
        event,
        layer: rolledBackLayer,
        selectedLayerId: 'b',
        stagingArea,
        type: 'rollbackStagedImageCommit',
      });

    expect(rollback({ ...layer })).toBe(committed);
    const rolledBack = rollback(layer);
    expect(getLayerIds(rolledBack)).toEqual(['a', 'b']);
    expect(getCanvas(rolledBack).document.selectedLayerId).toBe('b');
    expect(getCanvas(rolledBack).stagingArea).toBe(stagingArea);
  });

  it('removes layers and repairs selection to the nearest remaining layer (below, then above)', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);

    state = workbenchReducer(state, { id: 'b', type: 'setCanvasSelectedLayer' });
    state = workbenchReducer(state, { ids: ['b'], type: 'removeCanvasLayers' });

    // 'b' sat between 'a' and 'c'; the nearest survivor below it is 'c'.
    expect(getLayerIds(state)).toEqual(['a', 'c']);
    expect(getCanvas(state).document.selectedLayerId).toBe('c');

    state = workbenchReducer(state, { id: 'c', type: 'setCanvasSelectedLayer' });
    state = workbenchReducer(state, { ids: ['c'], type: 'removeCanvasLayers' });

    expect(getCanvas(state).document.selectedLayerId).toBe('a');

    state = workbenchReducer(state, { ids: ['a'], type: 'removeCanvasLayers' });

    expect(getLayerIds(state)).toEqual([]);
    expect(getCanvas(state).document.selectedLayerId).toBeNull();
  });

  it('atomically restores deleted layers into their original non-contiguous order', () => {
    const layers = [createRasterLayer('a'), createRasterLayer('b'), createRasterLayer('c')];
    const initial = withCanvasLayers(createInitialWorkbenchState(), [layers[1]!]);
    const projectId = initial.activeProjectId;
    const removed = workbenchReducer(initial, {
      add: [
        { anchor: { ...stackTopAnchor(projectId), beforeId: 'b' }, nodes: [layers[0]!] },
        { anchor: { ...stackTopAnchor(projectId), afterId: 'b' }, nodes: [layers[2]!] },
      ],
      enabledUpdates: [],
      selectedLayerId: 'c',
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getLayerIds(removed)).toEqual(['a', 'b', 'c']);
    expect(getCanvas(removed).document.selectedLayerId).toBe('c');
  });

  it('atomically replaces non-contiguous layers with one result and restores them', () => {
    const layers = [createRasterLayer('a'), createRasterLayer('b'), createRasterLayer('c')];
    const result = createRasterLayer('result');
    const initial = workbenchReducer(withCanvasLayers(createInitialWorkbenchState(), layers), {
      id: 'c',
      type: 'setCanvasSelectedLayer',
    });
    const merged = workbenchReducer(initial, {
      add: [{ anchor: { ...stackTopAnchor(initial.activeProjectId), beforeId: 'a' }, nodes: [result] }],
      enabledUpdates: [],
      removeIds: ['a', 'c'],
      selectedLayerId: 'result',
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getLayerIds(merged)).toEqual(['result', 'b']);
    expect(getCanvas(merged).document.selectedLayerId).toBe('result');

    const restored = workbenchReducer(merged, {
      add: [
        { anchor: { ...stackTopAnchor(initial.activeProjectId), beforeId: 'b' }, nodes: [layers[0]!] },
        { anchor: { ...stackTopAnchor(initial.activeProjectId), afterId: 'b' }, nodes: [layers[2]!] },
      ],
      enabledUpdates: [],
      removeIds: ['result'],
      selectedLayerId: 'c',
      type: 'applyCanvasLayerStackMutation',
    });
    expect(getDocumentLeaves(getCanvas(restored).document)).toEqual(layers);
    expect(getCanvas(restored).document.selectedLayerId).toBe('c');
  });

  it('applies and reverses a layer stack mutation atomically while preserving unrelated layer identity', () => {
    const upper = createRasterLayer('upper');
    const below = createRasterLayer('below');
    const unrelated = { ...createRasterLayer('unrelated'), isEnabled: false };
    const result = createRasterLayer('result');
    const initial = workbenchReducer(withCanvasLayers(createInitialWorkbenchState(), [unrelated, upper, below]), {
      id: unrelated.id,
      type: 'setCanvasSelectedLayer',
    });
    const unrelatedBefore = getDocumentLeaves(getCanvas(initial).document)[0];

    const applied = workbenchReducer(initial, {
      add: [{ anchor: { ...stackTopAnchor(initial.activeProjectId), beforeId: upper.id }, nodes: [result] }],
      enabledUpdates: [
        { id: upper.id, isEnabled: false },
        { id: below.id, isEnabled: false },
      ],
      selectedLayerId: result.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(applied).toBeDefined();
    expect(getDocumentLeaves(getCanvas(applied).document).map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['unrelated', false],
      ['result', true],
      ['upper', false],
      ['below', false],
    ]);
    expect(getCanvas(applied).document.selectedLayerId).toBe(result.id);
    expect(getDocumentLeaves(getCanvas(applied).document)[0]).toBe(unrelatedBefore);

    const reverted = workbenchReducer(applied, {
      enabledUpdates: [
        { id: upper.id, isEnabled: true },
        { id: below.id, isEnabled: true },
      ],
      removeIds: [result.id],
      selectedLayerId: unrelated.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getDocumentLeaves(getCanvas(reverted).document).map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['unrelated', false],
      ['upper', true],
      ['below', true],
    ]);
    expect(getCanvas(reverted).document.selectedLayerId).toBe(unrelated.id);
    expect(getDocumentLeaves(getCanvas(reverted).document)[0]).toBe(unrelatedBefore);
  });

  it('rejects an invalid layer stack mutation without applying any valid fields', () => {
    const initial = workbenchReducer(
      withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]),
      { id: 'b', type: 'setCanvasSelectedLayer' }
    );
    const invalidActions = [
      {
        add: [{ anchor: stackTopAnchor(initial.activeProjectId), nodes: [createRasterLayer('a')] }],
        enabledUpdates: [{ id: 'b', isEnabled: false }],
        selectedLayerId: 'b',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        add: [{ anchor: stackTopAnchor('other-project'), nodes: [createRasterLayer('c')] }],
        enabledUpdates: [{ id: 'b', isEnabled: false }],
        selectedLayerId: 'c',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        add: [{ anchor: stackTopAnchor(initial.activeProjectId, 'control'), nodes: [createRasterLayer('c')] }],
        enabledUpdates: [{ id: 'b', isEnabled: false }],
        selectedLayerId: 'c',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        enabledUpdates: [{ id: 'a', isEnabled: false }],
        removeIds: ['missing'],
        selectedLayerId: 'b',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        enabledUpdates: [{ id: 'missing', isEnabled: false }],
        selectedLayerId: 'b',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        enabledUpdates: [],
        selectedLayerId: 'missing',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        enabledUpdates: [{ id: 'a', isEnabled: false }],
        removeIds: ['a'],
        selectedLayerId: 'b',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        add: [{ anchor: stackTopAnchor(initial.activeProjectId), nodes: [createRasterLayer('a')] }],
        enabledUpdates: [],
        removeIds: ['a'],
        selectedLayerId: 'b',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        add: [
          { anchor: stackTopAnchor(initial.activeProjectId), nodes: [createRasterLayer('c'), createRasterLayer('c')] },
        ],
        enabledUpdates: [{ id: 'b', isEnabled: false }],
        selectedLayerId: 'c',
        type: 'applyCanvasLayerStackMutation',
      },
    ] satisfies CanvasProjectMutation[];

    for (const action of invalidActions) {
      expect(workbenchReducer(initial, action)).toBe(initial);
    }
  });

  it('reorders layers only when the id set matches, and preserves layer identity', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);
    const originalLayerA = getDocumentLeaves(getCanvas(state).document).find((layer) => layer.id === 'a');

    const reordered = workbenchReducer(state, {
      orders: [{ orderedIds: ['c', 'a', 'b'], parentId: null, stack: 'raster' }],
      type: 'reorderCanvasSiblings',
    });

    expect(getLayerIds(reordered)).toEqual(['c', 'a', 'b']);
    expect(getDocumentLeaves(getCanvas(reordered).document).find((layer) => layer.id === 'a')).toBe(originalLayerA);

    const ignoredMissing = workbenchReducer(state, {
      orders: [{ orderedIds: ['c', 'a'], parentId: null, stack: 'raster' }],
      type: 'reorderCanvasSiblings',
    });
    const ignoredUnknown = workbenchReducer(state, {
      orders: [{ orderedIds: ['c', 'a', 'z'], parentId: null, stack: 'raster' }],
      type: 'reorderCanvasSiblings',
    });
    const ignoredWrongStack = workbenchReducer(state, {
      orders: [{ orderedIds: ['c', 'a', 'b'], parentId: null, stack: 'control' }],
      type: 'reorderCanvasSiblings',
    });

    expect(getLayerIds(ignoredMissing)).toEqual(['a', 'b', 'c']);
    expect(getLayerIds(ignoredUnknown)).toEqual(['a', 'b', 'c']);
    expect(getLayerIds(ignoredWrongStack)).toEqual(['a', 'b', 'c']);
  });

  it('replaces a leaf, moving it to the top of its new stack when the type changes, and no-ops when the id is missing', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);
    const beforeDocument = getCanvas(state).document;
    const replacement: CanvasControlLayerContract = {
      adapter: {
        beginEndStepPct: [0.2, 0.8],
        controlMode: 'more_control',
        kind: 'control_lora',
        model: 'control-model',
        weight: 0.65,
      },
      blendMode: 'multiply',
      filter: { settings: { radius: 3 }, type: 'canny' },
      id: 'b',
      isEnabled: false,
      isLocked: true,
      name: 'Complete replacement',
      opacity: 0.4,
      source: {
        bitmap: { contentHash: 'hash', height: 23, imageName: 'replacement.png', width: 17 },
        offset: { x: -4, y: 9 },
        type: 'paint',
      },
      transform: { rotation: 0.75, scaleX: -2, scaleY: 3, x: 12, y: -8 },
      type: 'control',
      withTransparencyEffect: true,
    };

    const replaced = workbenchReducer(state, {
      layer: replacement,
      layerId: 'b',
      type: 'replaceCanvasLayer',
    });
    const afterLayers = getDocumentLeaves(getCanvas(replaced).document);

    expect(afterLayers.map((layer) => layer.id)).toEqual(['b', 'a', 'c']);
    expect(afterLayers[0]).toBe(replacement);
    expect(getCanvas(replaced).document.stacks.control).toEqual([replacement]);
    expect(afterLayers[1]).toBe(getDocumentLeaves(beforeDocument)[0]);
    expect(afterLayers[2]).toBe(getDocumentLeaves(beforeDocument)[2]);

    const missing = workbenchReducer(replaced, {
      layer: replacement,
      layerId: 'missing',
      type: 'replaceCanvasLayer',
    });
    expect(missing).toBe(replaced);
    expect(getCanvas(missing).document).toBe(getCanvas(replaced).document);
  });

  it('applies per-type config patches and ignores mismatched layer types', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createControlLayer('ctrl')]);

    state = workbenchReducer(state, {
      config: { adapter: { weight: 0.25 }, layerType: 'control', withTransparencyEffect: true },
      id: 'ctrl',
      type: 'updateCanvasLayerConfig',
    });

    const controlLayer = getDocumentLeaves(getCanvas(state).document)[0];

    expect(controlLayer?.type).toBe('control');

    if (controlLayer?.type === 'control') {
      expect(controlLayer.adapter.weight).toBe(0.25);
      expect(controlLayer.adapter.kind).toBe('controlnet');
      expect(controlLayer.withTransparencyEffect).toBe(true);
    }

    const unchanged = workbenchReducer(state, {
      config: { isTransparencyLocked: true, layerType: 'raster' },
      id: 'ctrl',
      type: 'updateCanvasLayerConfig',
    });

    expect(getDocumentLeaves(getCanvas(unchanged).document)[0]).toBe(controlLayer);
  });

  it('merges a layer down into the raster sibling below and refuses when no raster sibling is below', () => {
    const mergedSource = { bitmap: { height: 8, imageName: 'merged.png', width: 8 }, type: 'paint' } as const;
    const withControl = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('top'),
      createControlLayer('bottom'),
    ]);
    expect(
      workbenchReducer(withControl, { source: mergedSource, type: 'mergeCanvasLayersDown', upperLayerId: 'top' })
    ).toBe(withControl);

    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('top'),
      createRasterLayer('bottom'),
    ]);
    state = workbenchReducer(state, { id: 'bottom', type: 'setCanvasSelectedLayer' });
    state = workbenchReducer(state, { source: mergedSource, type: 'mergeCanvasLayersDown', upperLayerId: 'top' });

    const layers = getDocumentLeaves(getCanvas(state).document);

    expect(getLayerIds(state)).toEqual(['bottom']);
    expect(layers[0]?.type).toBe('raster');
    expect(layers[0]?.id).toBe('bottom');
    expect(layers[0]?.type === 'raster' && layers[0].source).toEqual(mergedSource);
    expect(getCanvas(state).document.selectedLayerId).toBe('bottom');
  });

  it('resizes the document, translating layer transforms and clamping the bbox in-bounds', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);

    state = workbenchReducer(state, {
      height: 256,
      offsetX: 20,
      offsetY: 10,
      type: 'resizeCanvasDocument',
      width: 300,
    });

    const document = getCanvas(state).document;

    expect(document.width).toBe(300);
    expect(document.height).toBe(256);
    expect(getDocumentLeaves(document)[0]?.transform.x).toBe(20);
    expect(getDocumentLeaves(document)[0]?.transform.y).toBe(10);
    expect(document.bbox.x + document.bbox.width).toBeLessThanOrEqual(300);
    expect(document.bbox.y + document.bbox.height).toBeLessThanOrEqual(256);
  });

  it('refuses an offset resize while results are pending, since their pinned rects would not move with the leaves', () => {
    let state = submitGenerate(
      primeGenerate(withInvocationDestination(withEmptyCanvas(createInitialWorkbenchState()), 'canvas'))
    );
    const offset = { height: 2048, offsetX: 64, offsetY: 32, type: 'resizeCanvasDocument', width: 2048 } as const;
    const inFlight = getCanvas(state);

    expect(getCanvas(workbenchReducer(state, offset))).toBe(inFlight);

    const queueItem = getActiveProject(state).queue.items[0]!;
    state = workbenchReducer(state, {
      images: [createImage('staged.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, {
      notify: false,
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      status: 'completed',
      type: 'setQueueItemStatus',
    });
    const staged = getCanvas(state);
    expect(getCanvas(workbenchReducer(state, offset))).toBe(staged);
    expect(getCanvas(workbenchReducer(state, { ...offset, offsetX: 0, offsetY: 0 })).document.width).toBe(2048);
  });

  it('replaces the whole document with a deep copy', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = {
      ...createEmptyCanvasDocument(),
      stacks: stacksFrom([createRasterLayer('fresh')]),
      selectedLayerId: 'fresh',
    };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getLayerIds(state)).toEqual(['fresh']);
    expect(getCanvas(state).document).not.toBe(replacement);
    expect(getDocumentLeaves(getCanvas(state).document)[0]).not.toBe(getDocumentLeaves(replacement)[0]);
  });

  it('clears the staging area on replaceCanvasDocument (staged candidates belong to the outgoing document)', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const staged: Project['canvas']['stagingArea']['pendingImages'][number] = {
      height: 64,
      imageName: 'staged-1',
      imageUrl: 'url',
      placement: { height: 64, opacity: 1, width: 64, x: 0, y: 0 },
      queuedAt: 'now',
      sourceQueueItemId: 'queue-1',
      thumbnailUrl: 'thumb',
      width: 64,
    };
    state = {
      ...state,
      projects: state.projects.map((project) =>
        project.id === state.activeProjectId
          ? {
              ...project,
              canvas: {
                ...project.canvas,
                stagingArea: {
                  ...project.canvas.stagingArea,
                  isVisible: true,
                  pendingImageIds: ['queue-1'],
                  pendingImages: [staged],
                  selectedImageIndex: 0,
                  sourceQueueItemId: 'queue-1',
                },
              },
            }
          : project
      ),
    };

    state = workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });

    const { stagingArea } = getCanvas(state);
    expect(stagingArea.pendingImages).toEqual([]);
    expect(stagingArea.pendingImageIds).toEqual([]);
    expect(stagingArea.isVisible).toBe(false);
    expect(stagingArea.sourceQueueItemId).toBeUndefined();
  });

  it('repairs a dangling selectedLayerId on replaceCanvasDocument (falls back to the top layer)', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = {
      ...createEmptyCanvasDocument(),
      stacks: stacksFrom([createRasterLayer('top'), createRasterLayer('bottom')]),
      selectedLayerId: 'ghost',
    };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getCanvas(state).document.selectedLayerId).toBe('top');
  });

  it('saves, restores, and deletes canvas snapshots', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);

    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'First', type: 'saveCanvasSnapshot' });

    expect(getCanvas(state).snapshots).toHaveLength(1);
    expect(getDocumentLeaves(getCanvas(state).snapshots[0]?.document).map((layer) => layer.id)).toEqual(['a']);

    state = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('b'),
      type: 'addCanvasLayer',
    });
    expect(getLayerIds(state)).toEqual(['b', 'a']);

    state = workbenchReducer(state, { snapshotId: 'snap-1', type: 'restoreCanvasSnapshot' });
    expect(getLayerIds(state)).toEqual(['a']);

    state = workbenchReducer(state, { snapshotId: 'snap-1', type: 'deleteCanvasSnapshot' });
    expect(getCanvas(state).snapshots).toEqual([]);
  });

  it('keeps the newest restore points, and restores a cleared canvas with the results it had staged', () => {
    let state = submitGenerate(
      primeGenerate(withInvocationDestination(withEmptyCanvas(createInitialWorkbenchState()), 'canvas'))
    );
    const queueItem = getActiveProject(state).queue.items[0];
    state = workbenchReducer(state, {
      images: [createImage('kept.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('a'),
      type: 'addCanvasLayer',
    });
    for (let index = 0; index < MAX_CANVAS_SNAPSHOTS; index += 1) {
      state = workbenchReducer(state, {
        createdAt: 'now',
        id: `old-${index}`,
        name: 'Old',
        type: 'saveCanvasSnapshot',
      });
    }
    state = workbenchReducer(state, {
      createdAt: 'now',
      id: 'before-clear',
      includeStaging: true,
      kind: 'clear-canvas',
      name: 'Before clearing',
      type: 'saveCanvasSnapshot',
    });
    expect(getCanvas(state).snapshots.map((snapshot) => snapshot.id)).toEqual([
      ...Array.from({ length: MAX_CANVAS_SNAPSHOTS - 1 }, (_, index) => `old-${index + 1}`),
      'before-clear',
    ]);
    expect(
      getCanvas(state)
        .snapshots.at(-1)
        ?.stagedImages?.map((image) => image.imageName)
    ).toEqual(['kept.png']);

    state = workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });
    expect(getCanvas(state).stagingArea.pendingImages).toEqual([]);
    state = workbenchReducer(state, { snapshotId: 'before-clear', type: 'restoreCanvasSnapshot' });
    expect(getLayerIds(state)).toEqual(['a']);
    expect(getCanvas(state).stagingArea.pendingImageIds).toEqual(['kept.png']);
  });

  it('bumps documentRevision on wholesale swaps (restore/replace) but not on ordinary edits', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const initialRevision = getCanvas(state).documentRevision;

    state = workbenchReducer(state, { id: 'a', patch: { opacity: 0.3 }, type: 'updateCanvasLayer' });
    state = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('b'),
      type: 'addCanvasLayer',
    });
    state = workbenchReducer(state, { bbox: { height: 32, width: 32, x: 0, y: 0 }, type: 'setCanvasBbox' });
    state = workbenchReducer(state, { height: 256, type: 'resizeCanvasDocument', width: 256 });
    expect(getCanvas(state).documentRevision).toBe(initialRevision);

    state = workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 1);

    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'First', type: 'saveCanvasSnapshot' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 1);
    state = workbenchReducer(state, { snapshotId: 'snap-1', type: 'restoreCanvasSnapshot' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 2);

    state = workbenchReducer(state, { snapshotId: 'missing', type: 'restoreCanvasSnapshot' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 2);
  });

  it('never records project undo entries for canvas layer edits', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);

    state = workbenchReducer(state, { id: 'a', patch: { opacity: 0.3 }, type: 'updateCanvasLayer' });
    state = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('b'),
      type: 'addCanvasLayer',
    });
    state = workbenchReducer(state, { ids: ['a'], type: 'removeCanvasLayers' });

    expect(getActiveProject(state).undoRedo.past).toEqual([]);
  });
});

describe('workbenchReducer canvas staging auto-switch + canvas submission', () => {
  const stageResults = (state: WorkbenchState, imageNames: string[]): WorkbenchState => {
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    return workbenchReducer(state, {
      images: imageNames.map((name) => createImage(name, queueItem.id)),
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
  };

  it("'latest' selects the newest pending candidate as results arrive", () => {
    let state = submitGenerate(primeGenerate());

    state = workbenchReducer(state, { mode: 'latest', type: 'setCanvasStagingAutoSwitch' });
    state = stageResults(state, ['first.png', 'second.png']);

    expect(getCanvas(state).stagingArea.autoSwitchMode).toBe('latest');
    expect(getCanvas(state).stagingArea.selectedImageIndex).toBe(1);
  });

  it("'off' keeps the current candidate selected as results arrive", () => {
    let state = submitGenerate(primeGenerate());

    state = workbenchReducer(state, { mode: 'latest', type: 'setCanvasStagingAutoSwitch' });
    state = stageResults(state, ['first.png', 'second.png']);
    state = workbenchReducer(state, { mode: 'off', type: 'setCanvasStagingAutoSwitch' });
    state = stageResults(state, ['first.png', 'second.png', 'third.png']);

    expect(getCanvas(state).stagingArea.autoSwitchMode).toBe('off');
    expect(getCanvas(state).stagingArea.selectedImageIndex).toBe(1);
  });

  const selectedStagingSlot = (state: WorkbenchState) => {
    const project = getActiveProject(state);

    return getCanvasStagingSlots(project.canvas, project.queue.items)[project.canvas.stagingArea.selectedImageIndex];
  };

  it("'progress' keeps each finished result on screen until the next image's first frame", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    const firstFrame = (itemIndex: number) =>
      workbenchReducer(state, {
        itemIndex,
        projectId: project.id,
        queueItemId: queueItem.id,
        type: 'noteQueueItemFirstProgressFrame',
      });

    state = workbenchReducer(state, { mode: 'progress', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = firstFrame(1);
    expect(selectedStagingSlot(state)).toMatchObject({ itemIndex: 1, kind: 'placeholder' });

    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [createImage('candidate-1.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });
    expect(selectedStagingSlot(state)).toMatchObject({
      candidate: { imageName: 'candidate-1.png' },
      kind: 'candidate',
    });

    state = firstFrame(2);
    expect(selectedStagingSlot(state)).toMatchObject({ itemIndex: 2, kind: 'placeholder' });
    expect(firstFrame(2)).toBe(state);
  });

  describe('picking a slot steers Auto-switch as Preview does', () => {
    const startBatch = (mode: 'latest' | 'off' | 'progress') => {
      let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
      const project = getActiveProject(state);
      const queueItem = project.queue.items[0]!;
      state = workbenchReducer(state, { mode, type: 'setCanvasStagingAutoSwitch' });
      state = workbenchReducer(state, {
        backendItemIds: [11, 12, 13],
        projectId: project.id,
        queueItemId: queueItem.id,
        type: 'markQueueItemBackendSubmitted',
      });
      const land = (current: WorkbenchState, backendItemId: number, imageName: string) =>
        workbenchReducer(current, {
          backendItemId,
          images: [createImage(imageName, queueItem.id)],
          projectId: project.id,
          queueItemId: queueItem.id,
          type: 'routeQueueItemPartialResults',
        });
      const firstFrame = (current: WorkbenchState, itemIndex: number) =>
        workbenchReducer(current, {
          itemIndex,
          projectId: project.id,
          queueItemId: queueItem.id,
          type: 'noteQueueItemFirstProgressFrame',
        });
      return { firstFrame, land, state };
    };

    it('stepping onto a finished result pauses Auto-switch as clicking it does', () => {
      const { land, state: started } = startBatch('latest');
      let state = land(land(started, 11, 'first.png'), 12, 'second.png');

      state = workbenchReducer(state, { direction: -1, type: 'cycleStagedImage' });

      expect(getCanvas(state).stagingArea.autoSwitchMode).toBe('off');
      expect(selectedStagingSlot(state)).toMatchObject({ candidate: { imageName: 'first.png' } });
    });
  });

  it('keeps each open project its own follow state, and drops it with the project', () => {
    const startRun = (current: WorkbenchState) => {
      let state = submitGenerate(primeGenerate(withInvocationDestination(current, 'canvas'), { batchCount: 2 }));
      const project = getActiveProject(state);
      const queueItem = project.queue.items[0]!;
      state = workbenchReducer(state, { mode: 'progress', type: 'setCanvasStagingAutoSwitch' });
      state = workbenchReducer(state, {
        backendItemIds: [11, 12],
        projectId: project.id,
        queueItemId: queueItem.id,
        type: 'markQueueItemBackendSubmitted',
      });
      return { projectId: project.id, queueItemId: queueItem.id, state };
    };
    const selectedIn = (state: WorkbenchState, projectId: string) => {
      const project = state.projects.find((candidate) => candidate.id === projectId)!;
      return getCanvasStagingSlots(project.canvas, project.queue.items)[project.canvas.stagingArea.selectedImageIndex];
    };
    const a = startRun(createInitialWorkbenchState());
    let state = workbenchReducer(a.state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, { type: 'createProject' });
    const b = startRun(state);
    state = b.state;

    state = workbenchReducer(state, {
      itemIndex: 1,
      projectId: a.projectId,
      queueItemId: a.queueItemId,
      type: 'noteQueueItemFirstProgressFrame',
    });
    expect(state.canvasStagingFollow?.[a.projectId]?.framedSlotIds).toEqual([`${a.queueItemId}:0`]);
    expect(selectedIn(state, a.projectId)).toMatchObject({ itemIndex: 2, kind: 'placeholder' });
    expect(state.canvasStagingFollow?.[b.projectId]).toBeUndefined();

    state = workbenchReducer(state, { imageIndex: 0, type: 'setStagedImageIndex' });
    expect(state.canvasStagingFollow?.[b.projectId]?.pinnedSlotId).toBe(`${b.queueItemId}:0`);
    expect(state.canvasStagingFollow?.[a.projectId]?.pinnedSlotId).toBe(`${a.queueItemId}:1`);

    state = workbenchReducer(state, { projectId: a.projectId, type: 'closeProject' });
    expect(state.canvasStagingFollow?.[a.projectId]).toBeUndefined();
    expect(state.canvasStagingFollow?.[b.projectId]?.pinnedSlotId).toBe(`${b.queueItemId}:0`);
  });

  const createCanvasGenerateSnapshot = () => ({
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
    values: createGenerateValues({ positivePrompt: 'canvas prompt', seed: 101, seedMode: 'fixed' }),
  });

  const submitCanvasGeneration = (state: WorkbenchState): { queueItemId: string; state: WorkbenchState } => {
    const graph: GraphContract = {
      edges: [],
      id: 'canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };
    const next = workbenchReducer(state, {
      backendSupportsCancellation: true,
      canvas: structuredClone(getCanvas(state)),
      destination: 'canvas',
      generate: createCanvasGenerateSnapshot(),
      graph,
      projectId: state.activeProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });

    return { queueItemId: getActiveProject(next).queue.items[0]!.id, state: next };
  };

  it('routes canvas results into staging when the session (documentRevision) still matches', () => {
    const { queueItemId, state: submitted } = submitCanvasGeneration(createInitialWorkbenchState());

    const state = workbenchReducer(submitted, {
      images: [createImage('fresh.png', queueItemId)],
      projectId: getActiveProject(submitted).id,
      queueItemId,
      type: 'routeQueueItemResults',
    });

    expect(getCanvas(state).stagingArea.pendingImageIds).toEqual(['fresh.png']);
    expect(getCanvas(state).stagingArea.isVisible).toBe(true);
  });

  it('drops mid-flight canvas results after a new-canvas swap so cleared staging is not resurrected (F2)', () => {
    const { queueItemId, state: submitted } = submitCanvasGeneration(createInitialWorkbenchState());

    const swapped = workbenchReducer(submitted, {
      document: createEmptyCanvasDocument(),
      type: 'replaceCanvasDocument',
    });

    const state = workbenchReducer(swapped, {
      images: [createImage('stale.png', queueItemId)],
      projectId: getActiveProject(swapped).id,
      queueItemId,
      type: 'routeQueueItemResults',
    });

    const { stagingArea } = getCanvas(state);
    expect(stagingArea.pendingImages).toEqual([]);
    expect(stagingArea.pendingImageIds).toEqual([]);
    expect(stagingArea.isVisible).toBe(false);
    expect(getActiveProject(state).queue.items[0]?.status).toBe('completed');
  });

  it('enqueues a Canvas expansion with detached authored metadata and the actual sample seed for recall', () => {
    const initial = createInitialWorkbenchState();
    const canvas = structuredClone(getCanvas(initial));
    canvas.document.bbox = { height: 48, width: 32, x: 17, y: 23 };
    const recordedCanvas = {
      controlModels: {},
      document: canvas.document,
      settings: { strength: 0.75 },
      version: 1,
    };
    const graph: GraphContract = {
      backendGraph: {
        edges: [],
        id: 'canvas-backend-graph',
        nodes: { metadata: { canvas_v3_metadata: recordedCanvas, id: 'metadata', type: 'core_metadata' } },
      },
      edges: [],
      id: 'canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };

    const values = createGenerateValues({
      dynamicPromptsCombinatorial: false,
      dynamicPromptsSampleSeed: 99,
      negativePrompt: 'avoid blur',
      positivePrompt: 'inpaint {red|green} cat',
      seed: 42,
      seedMode: 'fixed',
    });
    const state = workbenchReducer(initial, {
      backendSupportsCancellation: true,
      canvas,
      destination: 'canvas',
      generate: {
        negativePromptNodeId: 'negative_prompt',
        positivePromptNodeId: 'positive_prompt',
        seedNodeId: 'seed',
        values,
      },
      graph,
      positivePrompts: ['inpaint green cat'],
      positivePromptsSampleSeed: 7,
      projectId: initial.activeProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });

    const queueItem = getActiveProject(state).queue.items[0];

    expect(queueItem?.snapshot.sourceId).toBe('canvas');
    expect(queueItem?.snapshot.destination).toBe('canvas');
    expect(queueItem?.snapshot.graph).toEqual({ id: 'canvas-graph', label: 'Canvas' });
    expect(queueItem?.snapshot.recall?.canvasMetadata).toEqual({ canvas_v3_metadata: recordedCanvas });
    expect(queueItem?.snapshot.recall?.canvasMetadata?.canvas_v3_metadata).not.toBe(recordedCanvas);
    expect(queueItem?.snapshot.recall?.generateValues).toMatchObject({
      dynamicPromptsSampleSeed: 7,
      negativePrompt: 'avoid blur',
      positivePrompt: 'inpaint {red|green} cat',
      seed: 42,
      seedMode: 'fixed',
    });
    expect(queueItem?.snapshot).not.toHaveProperty('generate');
    expect(queueItem?.snapshot).not.toHaveProperty('widgetStates');
    expect(queueItem?.snapshot).not.toHaveProperty('widgetInstances');
    expect(queueItem?.snapshot.backendSubmission).toMatchObject({
      batchCount: 1,
      kind: 'generate',
      negativePrompt: 'avoid blur',
      negativePromptNodeId: 'negative_prompt',
      positivePrompt: 'inpaint {red|green} cat',
      positivePromptNodeId: 'positive_prompt',
      positivePrompts: ['inpaint green cat'],
      seed: 42,
      seedNodeId: 'seed',
      seedStep: 0,
    });
    expect(queueItem?.snapshot.resultNodeIds).toEqual(['canvas_output']);
    expect(getActiveProject(state).invocation.sourceId).toBe('canvas');

    canvas.document.bbox.x = 999;
    recordedCanvas.settings.strength = 0.1;
    values.positivePrompt = 'later prompt';
    values.dynamicPromptsSampleSeed = 101;

    expect(queueItem?.snapshot.recall?.canvasMetadata?.canvas_v3_metadata).toMatchObject({
      document: { bbox: { height: 48, width: 32, x: 17, y: 23 } },
      settings: { strength: 0.75 },
      version: 1,
    });
    expect(queueItem?.snapshot.recall?.generateValues).toMatchObject({
      dynamicPromptsSampleSeed: 7,
      positivePrompt: 'inpaint {red|green} cat',
    });
  });

  it('places generated candidates at the frozen queue bbox after the live bbox changes', () => {
    const graph: GraphContract = {
      edges: [],
      id: 'frozen-placement-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };
    let state = createInitialWorkbenchState();
    const frozenCanvas = structuredClone(getCanvas(state));
    frozenCanvas.document.bbox = { height: 48, width: 32, x: 17, y: 23 };
    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      canvas: frozenCanvas,
      destination: 'canvas',
      generate: createCanvasGenerateSnapshot(),
      graph,
      projectId: state.activeProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });
    const queueItemId = getActiveProject(state).queue.items[0]!.id;
    state = workbenchReducer(state, {
      bbox: { height: 64, width: 64, x: 101, y: 202 },
      type: 'setCanvasBbox',
    });

    state = workbenchReducer(state, {
      images: [createImage('frozen-placement.png', queueItemId)],
      projectId: state.activeProjectId,
      queueItemId,
      type: 'routeQueueItemResults',
    });

    expect(getCanvas(state).stagingArea.pendingImages[0]?.placement).toMatchObject({ x: 17, y: 23 });
  });

  it('submitCanvasInvocationSnapshot targets the project it names, not the active one', () => {
    const graph: GraphContract = {
      edges: [],
      id: 'canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };

    let state = createInitialWorkbenchState();
    const originatingProjectId = state.activeProjectId;
    state = workbenchReducer(state, { type: 'createProject' });
    const otherProjectId = state.activeProjectId;

    expect(otherProjectId).not.toBe(originatingProjectId);

    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      canvas: structuredClone(getProject(state, originatingProjectId).canvas),
      destination: 'canvas',
      generate: createCanvasGenerateSnapshot(),
      graph,
      projectId: originatingProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });

    expect(getProject(state, originatingProjectId).queue.items).toHaveLength(1);
    expect(getProject(state, originatingProjectId).queue.items[0]?.snapshot.graph.id).toBe('canvas-graph');
    expect(getProject(state, otherProjectId).queue.items).toHaveLength(0);
  });
});

describe('auto invocation route switching', () => {
  const getRoute = (state: WorkbenchState) => getActiveProject(state).invocation;

  it('switches to generate with a gallery destination on a high-confidence settings patch', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    state = workbenchReducer(state, { type: 'patchGenerateSettings', values: { steps: 25 } });

    expect(getRoute(state)).toMatchObject({ destination: 'gallery', sourceId: 'generate' });
  });

  it('switches to upscale on widget-value edits and shared prompt draft edits', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { inputImage: { height: 512, image_name: 'input.png', width: 768 } },
      widgetId: 'upscale',
    });

    expect(getRoute(state)).toMatchObject({ destination: 'gallery', sourceId: 'upscale' });

    let promptState = createInitialWorkbenchState();

    promptState = workbenchReducer(promptState, {
      sourceId: 'upscale',
      type: 'patchProjectPromptDraft',
      values: { positivePrompt: 'crisp detail' },
    });

    expect(getRoute(promptState)).toMatchObject({ destination: 'gallery', sourceId: 'upscale' });
  });

  it('switches to canvas with a canvas destination on content mutations but not selection', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });

    const selected = workbenchReducer(state, { id: 'a', type: 'setCanvasSelectedLayer' });

    expect(getRoute(selected)).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });

    const edited = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('b'),
      type: 'addCanvasLayer',
    });

    expect(getRoute(edited)).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });
  });

  it('respects the source and destination locks', () => {
    let sourceLockedState = createInitialWorkbenchState();

    sourceLockedState = workbenchReducer(sourceLockedState, {
      destination: 'canvas',
      type: 'setInvocationDestination',
    });
    sourceLockedState = workbenchReducer(sourceLockedState, { type: 'toggleSourceLock' });
    sourceLockedState = workbenchReducer(sourceLockedState, {
      anchor: stackTopAnchor(sourceLockedState.activeProjectId),
      layer: createRasterLayer('a'),
      type: 'addCanvasLayer',
    });

    expect(getRoute(sourceLockedState)).toMatchObject({ destination: 'canvas', sourceId: 'generate' });

    let destinationLockedState = createInitialWorkbenchState();

    destinationLockedState = workbenchReducer(destinationLockedState, {
      destination: 'canvas',
      type: 'setInvocationDestination',
    });
    destinationLockedState = workbenchReducer(destinationLockedState, { type: 'toggleDestinationLock' });
    destinationLockedState = workbenchReducer(destinationLockedState, {
      type: 'patchWidgetValues',
      values: { inputImage: { height: 512, image_name: 'input.png', width: 768 } },
      widgetId: 'upscale',
    });

    expect(getRoute(destinationLockedState)).toMatchObject({ destination: 'canvas', sourceId: 'upscale' });
  });
});

describe('canvas schema adoption', () => {
  const typedControl = (id: string): CanvasControlLayerContract => ({
    ...createControlLayer(id),
    controlType: 'depth',
    detection: { settings: {}, type: 'depth_anything_depth_estimation' },
    source: { image: { height: 64, imageName: 'photo.png', width: 64 }, type: 'image' },
  });
  const withV4Canvas = (layers: readonly CanvasLayerContract[]): WorkbenchState =>
    workbenchReducer(createInitialWorkbenchState(), {
      document: { ...createEmptyCanvasDocument(), stacks: stacksFrom([...layers]), version: 4 },
      type: 'replaceCanvasDocument',
    });
  const versions = (state: WorkbenchState) => [getCanvas(state).version, getCanvas(state).document.version];

  it('moves a v4 canvas to schema 6 when a control layer gains a type, and keeps 6 after it loses it', () => {
    let state = withV4Canvas([createControlLayer('ctrl')]);
    expect(versions(state)).toEqual([4, 4]);

    state = workbenchReducer(state, { layer: typedControl('ctrl'), layerId: 'ctrl', type: 'replaceCanvasLayer' });
    expect(versions(state)).toEqual([6, 6]);

    state = workbenchReducer(state, { layer: createControlLayer('ctrl'), layerId: 'ctrl', type: 'replaceCanvasLayer' });
    expect(versions(state)).toEqual([6, 6]);

    state = workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });
    expect(getCanvas(state).version).toBe(6);
  });

  it('adopts 4, not 5, for custom-font typography alone', () => {
    const text: CanvasLayerContract = {
      ...createRasterLayer('text'),
      source: {
        align: 'left',
        color: '#fff',
        content: 'hello',
        fontFamily: 'Example',
        fontRef: { contentHash: 'a'.repeat(64), family: 'Example', id: 'font-1', label: 'Example Regular' },
        fontSize: 24,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text',
      },
    };
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('text')]);
    expect(versions(state)).toEqual([3, 3]);

    state = workbenchReducer(state, { layer: text, layerId: 'text', type: 'replaceCanvasLayer' });
    expect(versions(state)).toEqual([4, 4]);
  });
});

describe('loading stored layouts that dock retired widgets', () => {
  it('drops the status-bar Diagnostics copy and collapses the bar it fronted', () => {
    const initial = createInitialWorkbenchState().projects[0]!;
    const compose = storedOn('compose');
    const loaded = asStoredBeforeSplit(
      {
        ...initial,
        widgetInstances: {
          ...initial.widgetInstances,
          'diagnostics:bottom': { ...initial.widgetInstances.diagnostics!, id: 'diagnostics:bottom' },
        },
      },
      {
        ...compose,
        widgetRegions: {
          ...compose.widgetRegions,
          bottom: {
            ...compose.widgetRegions.bottom,
            activeInstanceId: 'diagnostics:bottom',
            alignEndInstanceIds: ['diagnostics:bottom', 'notifications'],
            instanceIds: ['server-status', 'queue-status', 'diagnostics:bottom', 'notifications'],
            isCollapsed: false,
          },
        },
      }
    );

    expect(loaded.legacyLayout?.widgetRegions.bottom).toMatchObject({
      alignEndInstanceIds: ['notifications'],
      instanceIds: ['server-status', 'queue-status', 'notifications'],
      isCollapsed: true,
    });
    expect(loaded.widgetInstances).not.toHaveProperty('diagnostics:bottom');
  });

  it('drops the retired Project panel from a saved project and a stored custom workspace', () => {
    const initial = createInitialWorkbenchState();
    const projectInstance: Project['widgetInstances'][string] = {
      createdAt: 'then',
      id: 'project',
      state: { id: 'project', label: 'Project', values: {}, version: 1 },
      typeId: 'project',
    };
    const compose = storedOn('compose');
    const withLeftProject = {
      ...compose.widgetRegions,
      left: {
        ...compose.widgetRegions.left,
        activeInstanceId: 'project',
        alignEndInstanceIds: ['project'],
        instanceIds: ['generate', 'project', 'upscale'],
        isCollapsed: false,
      },
    };
    const leftWithoutProject = {
      activeInstanceId: 'generate',
      alignEndInstanceIds: [],
      instanceIds: ['generate', 'upscale'],
      isCollapsed: false,
    };
    const loaded = asStoredBeforeSplit(
      {
        ...initial.projects[0]!,
        widgetInstances: { ...initial.projects[0]!.widgetInstances, project: projectInstance },
      },
      { ...compose, widgetRegions: withLeftProject }
    );

    expect(loaded.legacyLayout?.widgetRegions.left).toMatchObject(leftWithoutProject);
    expect(loaded.widgetInstances).not.toHaveProperty('project');
    // A project on today's shape keeps no retired instance either.
    expect(
      normalizeWorkbenchProject({
        ...initial.projects[0]!,
        widgetInstances: { ...initial.projects[0]!.widgetInstances, project: projectInstance },
      }).widgetInstances
    ).not.toHaveProperty('project');

    const preset = layoutPresets[0]!.snapshot;
    const state = workbenchReducer(initial, {
      state: {
        ...initial,
        account: {
          ...initial.account,
          customLayoutPresets: [
            {
              id: 'custom-project',
              label: 'Project',
              snapshot: {
                ...preset,
                widgetInstances: { ...preset.widgetInstances, project: { id: 'project', typeId: 'project' } },
                widgetRegions: withLeftProject,
              },
            },
          ],
        },
      },
      type: 'hydrateWorkbench',
    });
    const storedPreset = state.account.customLayoutPresets?.[0]?.snapshot;

    expect(storedPreset?.widgetRegions.left).toMatchObject(leftWithoutProject);
    expect(storedPreset?.widgetInstances).not.toHaveProperty('project');
  });
});
