import type { GalleryImageItem, GalleryVideoItem, GeneratedImageContract } from '@features/gallery';
import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/api';
import type {
  CanvasControlLayerContract,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
} from '@workbench/canvas-engine/contracts';
import type { GraphContract } from '@workbench/graphContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import {
  GALLERY_RECENT_IMAGE_LIMIT,
  legacyGeneratedImageToGalleryItem,
  registerImageCluster,
} from '@features/gallery/contracts';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { MAX_PROMPT_HISTORY } from '@features/generation/settings';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { getDocumentLeaves } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { createEmptyCanvasDocument } from './canvasMigration';
import { getCanvasStagingCandidateFingerprint, getCanvasStagingSlots } from './canvasStagingView';
import { layoutPresets } from './layoutPresets';
import { resolveSavedLayoutPreset } from './layoutPresetSnapshots';
import { PROJECT_EVENT_LIMIT } from './projectEvents';
import { DEFAULT_PROJECT_SETTINGS } from './settings/store';
import { getProjectWidgetValues } from './widgetState';
import {
  clampPanelSize,
  createDraftProject,
  getPanelCollapseThreshold,
  shouldSnapPanelShut,
  normalizeWorkbenchAccount,
  normalizeWorkbenchProject,
} from './workbenchState';
import {
  createInitialWorkbenchState,
  nextLayerName,
  type WorkbenchAction,
  workbenchReducer as reduceWorkbench,
} from './workbenchState.testing';

// Seed capabilities to match app boot; submission fails closed without them.
seedArchitectureCapabilities();

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
  'rollbackStagedImageCommit',
  'resizeCanvasDocument',
  'restoreCanvasSnapshot',
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
  dynamicPromptsMaxPrompts: 100,
  dynamicPromptsSampleSeed: 0,
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
  promptTemplate: null,
  promptTemplateViewMode: false,
  mistralEncoderModel: null,
  qwen3EncoderModel: null,
  qwenVLEncoderModel: null,
  qwen3VLEncoderModel: null,
  wanT5EncoderModel: null,
  wanLowNoiseModel: null,
  ideogram4UnconditionalModel: null,
  wanGuidanceScaleLowNoise: null,
  ideogram4SamplerPreset: 'V4_QUALITY_48',
  ideogram4Steps: null,
  ideogram4GuidanceScale: null,
  ideogram4Mu: null,
  ideogram4ColorPalette: [],
  krea2RebalanceEnabled: false,
  krea2RebalanceMultiplier: 4,
  krea2RebalanceWeights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0',
  krea2SeedVarianceEnabled: false,
  krea2SeedVarianceStrength: 0.1,
  krea2SeedVarianceRandomizePercent: 50,
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

/** Reconstructs a v1-style `{x,y,width,height,opacity}` placement from a v2 raster layer's transform, mirroring `CanvasWidgetView`'s rendering math. */
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

const createInpaintMaskLayer = (id: string): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
    name: id,
    opacity: 1,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'inpaint_mask',
  }) as CanvasLayerContract;

/** Adds layers top-to-bottom in array order (each `addCanvasLayer` inserts at index 0). */
// Start without the default inpaint mask so these tests isolate layer mechanics.
const withEmptyCanvas = (state: WorkbenchState): WorkbenchState =>
  workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });

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
  workbenchReducer(state, { backendSupportsCancellation: true, type: 'submitInvocationSnapshot' });

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
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getQueuedRandDevice(state)).toBe('xpu');
  });

  it('passes the XPU runtime snapshot through the upscale submission boundary', () => {
    generationDeviceMock.options = [{ device: 'xpu:0', name: 'Intel Arc' }];
    const models = [
      createUpscaleModel('main', 'main', 'sd-1'),
      createUpscaleModel('spandrel', 'spandrel_image_to_image', 'any'),
      createUpscaleModel('tile', 'controlnet', 'sd-1', 'Tile ControlNet'),
    ];
    const values = {
      ...createDefaultUpscaleWidgetValues(models),
      inputImage: { height: 64, image_name: 'input.png', width: 64 },
      seed: 1,
      seedMode: 'fixed',
    };
    let state = workbenchReducer(createInitialWorkbenchState(), {
      settings: { useCpuNoise: false },
      type: 'setActiveProjectSettings',
    });

    state = workbenchReducer(state, { type: 'patchWidgetValues', values, widgetId: 'upscale' });
    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      models,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'upscale', sourceLocked: false },
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getQueuedRandDevice(state)).toBe('xpu');
  });
});

describe('workbench hydration invariants', () => {
  it('seeds a draft when a projectless session hydrates', () => {
    const initial = createInitialWorkbenchState();
    // Model the projectless cache written after closing the last tab.
    const emptySession: WorkbenchState = { ...initial, activeProjectId: '', projects: [] };

    const hydrated = workbenchReducer(initial, { state: emptySession, type: 'hydrateWorkbench' });

    expect(hydrated.projects).toHaveLength(1);
    expect(hydrated.activeProjectId).toBe(hydrated.projects[0]?.id);
    expect(getActiveProject(hydrated).widgetRegions.left.instanceIds.length).toBeGreaterThan(0);
  });

  it('builds that draft from the cached account, not the shipped defaults', () => {
    const initial = createInitialWorkbenchState();
    const project = getActiveProject(initial);
    const customizedDefault = {
      ...resolveSavedLayoutPreset(initial.account, initial.account.activeLayoutPresetId).snapshot,
      widgetRegions: {
        ...project.widgetRegions,
        left: { ...project.widgetRegions.left, instanceIds: ['generate'] },
      },
    };
    const emptySession: WorkbenchState = {
      ...initial,
      account: {
        ...initial.account,
        layoutPresetOverrides: { [initial.account.activeLayoutPresetId]: customizedDefault },
      },
      activeProjectId: '',
      projects: [],
    };

    const hydrated = workbenchReducer(initial, { state: emptySession, type: 'hydrateWorkbench' });

    expect(getActiveProject(hydrated).widgetRegions.left.instanceIds).toEqual(['generate']);
  });

  it('repairs an active project id that names no hydrated project', () => {
    const initial = createInitialWorkbenchState();
    const danglingActiveId: WorkbenchState = { ...initial, activeProjectId: 'project-that-was-refused' };

    const hydrated = workbenchReducer(initial, { state: danglingActiveId, type: 'hydrateWorkbench' });

    expect(hydrated.activeProjectId).toBe(hydrated.projects[0]?.id);
  });

  it('leaves a populated session alone', () => {
    const initial = createInitialWorkbenchState();

    const hydrated = workbenchReducer(initial, { state: initial, type: 'hydrateWorkbench' });

    expect(hydrated.projects.map((project) => project.id)).toEqual(initial.projects.map((project) => project.id));
    expect(hydrated.activeProjectId).toBe(initial.activeProjectId);
  });
});

describe('workbench widget region defaults', () => {
  it('starts new projects from the curated Compose widget defaults', () => {
    const state = createInitialWorkbenchState();
    const project = getActiveProject(state);

    expect(project.widgetRegions.left.instanceIds).toEqual(['generate', 'upscale']);
    expect(project.widgetRegions.right.instanceIds).toEqual(['gallery', 'image-map', 'queue']);
    expect(project.widgetRegions.bottom.instanceIds).toEqual([
      'server-status',
      'queue-status',
      'gallery:bottom',
      'notifications',
      'autosave-status',
    ]);
  });

  it('hydrates the old default right rail to the curated Compose defaults while preserving customized rails', () => {
    const initial = createInitialWorkbenchState();
    const legacyDefault = {
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          right: { ...project.widgetRegions.right, instanceIds: ['queue', 'gallery', 'layers'] },
        },
      })),
    } satisfies WorkbenchState;
    const customized = {
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          right: { ...project.widgetRegions.right, instanceIds: ['gallery', 'layers'] },
        },
      })),
    } satisfies WorkbenchState;

    const hydratedLegacyDefault = workbenchReducer(initial, { state: legacyDefault, type: 'hydrateWorkbench' });
    const hydratedCustomized = workbenchReducer(initial, { state: customized, type: 'hydrateWorkbench' });

    expect(getActiveProject(hydratedLegacyDefault).widgetRegions.right.instanceIds).toEqual([
      'gallery',
      'image-map',
      'queue',
    ]);
    expect(getActiveProject(hydratedCustomized).widgetRegions.right.instanceIds).toEqual(['gallery', 'layers']);
  });

  it('hydrates a pre-image-map right rail to the curated rail rather than splicing Image Map in', () => {
    // Untouched legacy rails must adopt the exact curated preset to avoid false drift.
    const initial = createInitialWorkbenchState();
    const withRightIds = (instanceIds: Project['widgetRegions']['right']['instanceIds']): WorkbenchState => ({
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          right: { ...project.widgetRegions.right, instanceIds },
        },
      })),
    });
    const preImageMapDefault = withRightIds(['gallery', 'preview', 'queue', 'layers', 'diagnostics', 'project']);
    const customized = withRightIds(['preview', 'gallery', 'queue']);

    const hydratedDefault = workbenchReducer(initial, { state: preImageMapDefault, type: 'hydrateWorkbench' });
    const hydratedCustomized = workbenchReducer(initial, { state: customized, type: 'hydrateWorkbench' });

    expect(getActiveProject(hydratedDefault).widgetRegions.right.instanceIds).toEqual([
      'gallery',
      'image-map',
      'queue',
    ]);
    // A rail the user actually arranged is still theirs.
    expect(getActiveProject(hydratedCustomized).widgetRegions.right.instanceIds).toEqual([
      'preview',
      'gallery',
      'queue',
    ]);
    // Edit folds the editors into Layers and Video keeps a still-free rail.
    for (const preset of layoutPresets) {
      if (preset.id !== 'edit' && preset.id !== 'video') {
        expect(preset.snapshot.widgetRegions.right.instanceIds).toContain('image-map');
      }
    }
  });

  it('adopts the shipped Edit rail for untouched legacy rails and leaves customized rails alone', () => {
    const initial = createInitialWorkbenchState();
    const withRight = (instanceIds: Project['widgetRegions']['right']['instanceIds']): WorkbenchState => ({
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          right: { ...project.widgetRegions.right, activeInstanceId: 'gallery', instanceIds },
        },
      })),
    });
    const legacyEditRail = withRight(['layers', 'preview', 'gallery', 'image-map', 'queue']);
    const layersOnlyRail = withRight(['layers']);
    const custom = withRight(['image-map', 'layers']);

    const hydratedLegacy = getActiveProject(
      workbenchReducer(initial, { state: legacyEditRail, type: 'hydrateWorkbench' })
    );
    expect(hydratedLegacy.widgetRegions.right).toMatchObject({
      activeInstanceId: 'layers',
      instanceIds: ['layers', 'preview'],
    });

    const hydratedLayersOnly = getActiveProject(
      workbenchReducer(initial, { state: layersOnlyRail, type: 'hydrateWorkbench' })
    );
    expect(hydratedLayersOnly.widgetRegions.right).toMatchObject({
      activeInstanceId: 'layers',
      instanceIds: ['layers', 'preview'],
    });

    const hydratedCustom = getActiveProject(workbenchReducer(initial, { state: custom, type: 'hydrateWorkbench' }));
    expect(hydratedCustom.widgetRegions.right.instanceIds).toEqual(['image-map', 'layers']);
    expect(hydratedCustom.widgetRegions.right.activeInstanceId).toBe('gallery');
  });

  it('adds Upscale to untouched legacy left rails while preserving customized rails', () => {
    const initial = createInitialWorkbenchState();
    const withoutUpscaleInstance = Object.fromEntries(
      Object.entries(getActiveProject(initial).widgetInstances).filter(([id]) => id !== 'upscale')
    ) as Project['widgetInstances'];
    const withLeftIds = (instanceIds: Project['widgetRegions']['left']['instanceIds']): WorkbenchState => ({
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetInstances: withoutUpscaleInstance,
        widgetRegions: { ...project.widgetRegions, left: { ...project.widgetRegions.left, instanceIds } },
      })),
    });

    const migrated = workbenchReducer(initial, {
      state: withLeftIds(['generate', 'workflow']),
      type: 'hydrateWorkbench',
    });
    const customized = workbenchReducer(initial, {
      state: withLeftIds(['generate', 'gallery']),
      type: 'hydrateWorkbench',
    });

    expect(getActiveProject(migrated).widgetRegions.left.instanceIds).toEqual(['generate', 'workflow', 'upscale']);
    expect(getActiveProject(migrated).widgetInstances.upscale?.typeId).toBe('upscale');
    expect(getActiveProject(customized).widgetRegions.left.instanceIds).toEqual(['generate', 'gallery']);
  });

  it('adds queue-status to untouched legacy bottom rails while preserving customized rails', () => {
    const initial = createInitialWorkbenchState();
    const withoutQueueStatusInstance = Object.fromEntries(
      Object.entries(getActiveProject(initial).widgetInstances).filter(([id]) => id !== 'queue-status')
    ) as Project['widgetInstances'];
    const withBottomIds = (instanceIds: Project['widgetRegions']['bottom']['instanceIds']): WorkbenchState => ({
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetInstances: withoutQueueStatusInstance,
        widgetRegions: { ...project.widgetRegions, bottom: { ...project.widgetRegions.bottom, instanceIds } },
      })),
    });

    const migrated = workbenchReducer(initial, {
      state: withBottomIds(['server-status', 'gallery:bottom', 'notifications', 'autosave-status']),
      type: 'hydrateWorkbench',
    });
    const customized = workbenchReducer(initial, {
      state: withBottomIds(['server-status', 'notifications']),
      type: 'hydrateWorkbench',
    });
    const alreadyPresent = workbenchReducer(initial, {
      state: withBottomIds(['queue-status', 'server-status', 'gallery:bottom', 'notifications', 'autosave-status']),
      type: 'hydrateWorkbench',
    });

    expect(getActiveProject(migrated).widgetRegions.bottom.instanceIds).toEqual([
      'server-status',
      'queue-status',
      'gallery:bottom',
      'notifications',
      'autosave-status',
    ]);
    expect(getActiveProject(migrated).widgetInstances['queue-status']?.typeId).toBe('queue-status');
    expect(getActiveProject(customized).widgetRegions.bottom.instanceIds).toEqual(['server-status', 'notifications']);
    expect(getActiveProject(alreadyPresent).widgetRegions.bottom.instanceIds).toEqual([
      'queue-status',
      'server-status',
      'gallery:bottom',
      'notifications',
      'autosave-status',
    ]);
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

describe('workbench panel resize bounds', () => {
  it('clamps a side panel into its raised bounds', () => {
    expect(clampPanelSize('left', 900)).toBe(720);
    expect(clampPanelSize('left', 700)).toBe(700);
    expect(clampPanelSize('left', 100)).toBe(350);
  });

  it('snaps shut past the overshoot and reopens with hysteresis', () => {
    expect(shouldSnapPanelShut('left', 350, false)).toBe(false);
    expect(shouldSnapPanelShut('left', 271, false)).toBe(false);
    expect(shouldSnapPanelShut('left', 270, false)).toBe(true);
    // Once shut, dragging back only reopens past the halfway band.
    expect(shouldSnapPanelShut('left', 290, true)).toBe(true);
    expect(shouldSnapPanelShut('left', 311, true)).toBe(false);
  });

  it('measures the overshoot against whichever floor the region has', () => {
    expect(shouldSnapPanelShut('bottom', 96, false)).toBe(false);
    expect(shouldSnapPanelShut('bottom', 17, false)).toBe(false);
    expect(shouldSnapPanelShut('bottom', 16, false)).toBe(true);
  });

  it('exposes the collapse threshold as a size below the floor', () => {
    expect(getPanelCollapseThreshold('left')).toBe(270);
    expect(getPanelCollapseThreshold('bottom')).toBe(16);
  });

  it('reopens a collapsed region at the size it was collapsed from', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { region: 'left', sizePx: 500, type: 'setRegionWidgetSize' });

    const activeInstanceId = getActiveProject(state).widgetRegions.left.activeInstanceId;

    // What a drag past the floor does: hide the panel, leave its width alone.
    state = workbenchReducer(state, { isCollapsed: true, region: 'left', type: 'setRegionWidgetCollapsed' });
    expect(getActiveProject(state).widgetRegions.left).toMatchObject({ isCollapsed: true, sizePx: 500 });

    state = workbenchReducer(state, { region: 'left', type: 'selectRegionWidget', widgetId: activeInstanceId });
    expect(getActiveProject(state).widgetRegions.left).toMatchObject({ isCollapsed: false, sizePx: 500 });
  });
});

describe('workbench widget region opening', () => {
  it('enables and selects a center widget without toggling it back off', () => {
    let state = createInitialWorkbenchState();
    const activeProject = getActiveProject(state);
    const instanceIds: Project['widgetRegions']['center']['instanceIds'] = ['canvas'];

    state = {
      ...state,
      projects: state.projects.map((project) =>
        project.id === activeProject.id
          ? {
              ...project,
              widgetRegions: {
                ...project.widgetRegions,
                center: {
                  ...project.widgetRegions.center,
                  activeInstanceId: 'canvas',
                  instanceIds,
                },
              },
            }
          : project
      ),
    };

    state = workbenchReducer(state, { region: 'center', type: 'openRegionWidget', widgetId: 'preview' });

    expect(getActiveProject(state).widgetRegions.center.activeInstanceId).toBe('preview');
    expect(getActiveProject(state).widgetRegions.center.instanceIds).toEqual(['canvas', 'preview']);

    state = workbenchReducer(state, { region: 'center', type: 'openRegionWidget', widgetId: 'preview' });

    expect(getActiveProject(state).widgetRegions.center.activeInstanceId).toBe('preview');
    expect(getActiveProject(state).widgetRegions.center.instanceIds).toEqual(['canvas', 'preview']);
    expect(getActiveProject(state).widgetRegions.center.isCollapsed).toBe(false);
  });

  it('opens and uncollapses the target panel region', () => {
    let state = createInitialWorkbenchState();
    const activeProject = getActiveProject(state);
    const instanceIds: Project['widgetRegions']['bottom']['instanceIds'] = ['diagnostics'];

    state = {
      ...state,
      projects: state.projects.map((project) =>
        project.id === activeProject.id
          ? {
              ...project,
              layout: {
                ...project.layout,
                panels: { ...project.layout.panels, isBottomOpen: false },
              },
              widgetRegions: {
                ...project.widgetRegions,
                bottom: {
                  ...project.widgetRegions.bottom,
                  activeInstanceId: 'diagnostics',
                  instanceIds,
                  isCollapsed: true,
                },
              },
            }
          : project
      ),
    };

    state = workbenchReducer(state, { region: 'bottom', type: 'openRegionWidget', widgetId: 'queue' });

    expect(getActiveProject(state).layout.panels.isBottomOpen).toBe(true);
    expect(getActiveProject(state).widgetRegions.bottom.activeInstanceId).toBe('queue');
    expect(getActiveProject(state).widgetRegions.bottom.instanceIds).toEqual(['diagnostics', 'queue']);
    expect(getActiveProject(state).widgetRegions.bottom.isCollapsed).toBe(false);
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
      graphHistory: [{ document: project.projectGraph, id: 'legacy-snapshot' }],
      queue: { items: [{}] },
    } as unknown as Project;

    const normalized = normalizeWorkbenchProject(legacyProject);

    expect(normalized.events).toEqual([]);
    expect('graphHistory' in normalized).toBe(false);
    expect(normalized.queue.items).toEqual([]);
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

    // A fresh project owns its default from the start, so Generate's count never reaches it, reload after reload.
    expect(workflowValuesOf(fresh).batchCount).toBe(1);
    expect(workflowValuesOf(normalizeWorkbenchProject(fresh)).batchCount).toBe(1);
    // A project saved before the widget owned a count keeps the runs it effectively had.
    expect(workflowValuesOf(normalizeWorkbenchProject(withoutWorkflowCount(fresh))).batchCount).toBe(4);
    // A count the workflow already owns is never overwritten by Generate's.
    expect(workflowValuesOf(normalizeWorkbenchProject(owned)).batchCount).toBe(2);
  });

  it('preserves and caps session events during live normalization', () => {
    const project = createInitialWorkbenchState().projects[0]!;
    project.events = Array.from({ length: PROJECT_EVENT_LIMIT + 1 }, (_, index) => ({
      createdAt: 'now',
      id: `event-${index}`,
      summary: `Event ${index}`,
      type: 'project-created',
    }));

    const normalized = normalizeWorkbenchProject(project, { isArriving: false });

    expect(normalized.events).toEqual(project.events.slice(0, PROJECT_EVENT_LIMIT));
  });

  it('preserves the live queue during same-session normalization', () => {
    const project = getActiveProject(submitGenerate(primeGenerate()));

    const normalized = normalizeWorkbenchProject(project, { isArriving: false });

    expect(normalized.queue).toBe(project.queue);
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
    // Foreign project rankings cannot resolve in this realm; their rank pages must not become board positions.
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

  it('keeps a session-scoped search this realm can still resolve', () => {
    // Live reopen and conflict forks retain resolvable rankings; adoption alone does not imply a foreign session.
    const clusterId = registerImageCluster(['image:a.png', 'image:b.png'], 'beaches');
    const values = galleryValuesOf(
      galleryProject({
        galleryPage: 3,
        paginationMode: 'paginated',
        semanticImageQuery: { clusterId, kind: 'cluster', label: 'beaches' },
      })
    );

    expect(values.semanticImageQuery).toEqual({ clusterId, kind: 'cluster', label: 'beaches' });
    expect(values.galleryPage).toBe(3);
  });

  it('drops an infinite window anchor on adoption, and keeps a paginated page', () => {
    // Infinite anchors are session-only; paginated positions persist, and absent paginationMode means infinite.
    expect(galleryValuesOf(galleryProject({ galleryPage: 5, paginationMode: 'infinite' })).galleryPage).toBe(0);
    expect(galleryValuesOf(galleryProject({ galleryPage: 5 })).galleryPage).toBe(0);
    expect(galleryValuesOf(galleryProject({ galleryPage: 5, paginationMode: 'paginated' })).galleryPage).toBe(5);
  });

  it('drops the anchor and an unresolvable search together on adoption', () => {
    const values = galleryValuesOf(
      galleryProject({
        galleryPage: 5,
        paginationMode: 'infinite',
        semanticImageQuery: { clusterId: 'evicted', kind: 'cluster', label: 'beaches' },
      })
    );

    expect(values.semanticImageQuery).toBeNull();
    expect(values.galleryPage).toBe(0);
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

  it('preserves both projects when a retarget target is already open', () => {
    const source = createDraftProject([]);
    const target = {
      ...createDraftProject([source]),
      id: `${source.id}-copy`,
      name: 'independently edited target',
      settings: { ...source.settings, useCpuNoise: !source.settings.useCpuNoise },
    };
    const state = { ...createInitialWorkbenchState(), activeProjectId: source.id, projects: [source, target] };

    const next = workbenchReducer(state, {
      boardId: 'board-copy',
      name: target.name,
      project: target,
      projectId: source.id,
      sourceName: source.name,
      targetProjectId: target.id,
      type: 'retargetProject',
    });

    expect(next).toBe(state);
    expect(next.projects).toEqual([source, target]);
  });

  it('keeps a search the new realm can rebuild, and the page it was read on', () => {
    const values = galleryValuesOf(
      galleryProject({
        galleryPage: 3,
        paginationMode: 'paginated',
        semanticImageQuery: { kind: 'text', query: 'sunset' },
      })
    );

    expect(values.semanticImageQuery).toEqual({ kind: 'text', query: 'sunset' });
    expect(values.galleryPage).toBe(3);
  });
});

describe('workbench widget alignment', () => {
  it('moves a bottom widget between the strip clusters and back', () => {
    const initial = createInitialWorkbenchState();
    const instanceId = getActiveProject(initial).widgetRegions.bottom.instanceIds[0]!;

    const alignedEnd = workbenchReducer(initial, {
      align: 'end',
      instanceId,
      region: 'bottom',
      type: 'setWidgetInstanceAlignment',
    });

    expect(getActiveProject(alignedEnd).widgetRegions.bottom.alignEndInstanceIds).toEqual([
      ...(getActiveProject(initial).widgetRegions.bottom.alignEndInstanceIds ?? []),
      instanceId,
    ]);
    // Placement itself is untouched: alignment is a render split, not a move.
    expect(getActiveProject(alignedEnd).widgetRegions.bottom.instanceIds).toEqual(
      getActiveProject(initial).widgetRegions.bottom.instanceIds
    );

    const alignedStart = workbenchReducer(alignedEnd, {
      align: 'start',
      instanceId,
      region: 'bottom',
      type: 'setWidgetInstanceAlignment',
    });

    expect(getActiveProject(alignedStart).widgetRegions.bottom.alignEndInstanceIds).toEqual(
      getActiveProject(initial).widgetRegions.bottom.alignEndInstanceIds ?? []
    );
  });

  it('aligning an already-aligned widget is a no-op', () => {
    const initial = createInitialWorkbenchState();
    const instanceId = getActiveProject(initial).widgetRegions.bottom.instanceIds[0]!;
    const once = workbenchReducer(initial, {
      align: 'end',
      instanceId,
      region: 'bottom',
      type: 'setWidgetInstanceAlignment',
    });
    const twice = workbenchReducer(once, {
      align: 'end',
      instanceId,
      region: 'bottom',
      type: 'setWidgetInstanceAlignment',
    });

    expect(twice).toBe(once);
  });
});

describe('workbench widget state updates', () => {
  it('patches a missing undefined widget value key as a real state change', () => {
    const initial = createInitialWorkbenchState();
    const next = workbenchReducer(initial, {
      type: 'patchWidgetValues',
      values: { optionalValue: undefined },
      widgetId: 'diagnostics',
    });
    const values = getProjectWidgetValues(getActiveProject(next), 'diagnostics');

    expect(next).not.toBe(initial);
    expect(Object.prototype.hasOwnProperty.call(values, 'optionalValue')).toBe(true);
  });

  it('can patch a widget instance in a non-active project', () => {
    let state = createInitialWorkbenchState();
    const firstProjectId = state.activeProjectId;

    state = workbenchReducer(state, { type: 'createProject' });
    const secondProjectId = state.activeProjectId;

    expect(secondProjectId).not.toBe(firstProjectId);

    state = workbenchReducer(state, {
      instanceId: 'generate',
      projectId: firstProjectId,
      type: 'patchWidgetInstanceValues',
      values: { projectScoped: true },
    });

    expect(getProject(state, firstProjectId).widgetInstances.generate?.state.values.projectScoped).toBe(true);
    expect(getProject(state, secondProjectId).widgetInstances.generate?.state.values.projectScoped).toBeUndefined();
  });

  it('clones replacement widget instance values before storing them', () => {
    let state = createInitialWorkbenchState();
    const values: Record<string, unknown> = { mutable: 'before' };

    state = workbenchReducer(state, { instanceId: 'generate', type: 'setWidgetInstanceValues', values });
    values.mutable = 'after';

    expect(getActiveProject(state).widgetInstances.generate?.state.values.mutable).toBe('before');
  });

  it('clones patched widget instance values before storing them', () => {
    let state = createInitialWorkbenchState();
    const values: Record<string, unknown> = { nested: { mutable: 'before' } };

    state = workbenchReducer(state, { instanceId: 'generate', type: 'patchWidgetInstanceValues', values });
    (values.nested as { mutable: string }).mutable = 'after';

    expect(getActiveProject(state).widgetInstances.generate?.state.values.nested).toEqual({ mutable: 'before' });
  });
});

describe('workbench layout presets', () => {
  it('starts initial and newly created projects on the Compose default route', () => {
    const initial = createInitialWorkbenchState();
    const withNewProject = workbenchReducer(initial, { type: 'createProject' });

    expect(getProject(initial, initial.activeProjectId).invocation).toEqual({
      destination: 'gallery',
      destinationLocked: false,
      sourceId: 'generate',
      sourceLocked: false,
    });
    expect(getProject(withNewProject, withNewProject.activeProjectId).invocation).toEqual({
      destination: 'gallery',
      destinationLocked: false,
      sourceId: 'generate',
      sourceLocked: false,
    });
  });

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
            layout: { ...compose.snapshot.layout, centerViewId: 'gallery' },
          },
        },
        layoutPresetRouteOverrides: {
          compose: { destination: 'canvas', sourceId: 'upscale' },
        },
      },
    };

    const withNewProject = workbenchReducer(customized, { type: 'createProject' });
    const project = getProject(withNewProject, withNewProject.activeProjectId);

    expect(project.invocation).toEqual({
      destination: 'canvas',
      destinationLocked: false,
      sourceId: 'upscale',
      sourceLocked: false,
    });
    expect(project.layout).toMatchObject({ centerViewId: 'gallery', presetId: 'compose' });
  });

  it('repairs a stale saved Compose route when its source is absent from the saved layout', () => {
    const initial = createInitialWorkbenchState();
    const compose = layoutPresets[0]!;
    const left = compose.snapshot.widgetRegions.left;
    const { upscale: _removedUpscale, ...widgetInstances } = compose.snapshot.widgetInstances;
    const customized: WorkbenchState = {
      ...initial,
      account: {
        ...initial.account,
        layoutPresetOverrides: {
          compose: {
            ...compose.snapshot,
            widgetInstances,
            widgetRegions: {
              ...compose.snapshot.widgetRegions,
              left: { ...left, instanceIds: left.instanceIds.filter((id) => id !== 'upscale') },
            },
          },
        },
        layoutPresetRouteOverrides: {
          compose: { destination: 'gallery', sourceId: 'upscale' },
        },
      },
    };

    const withNewProject = workbenchReducer(customized, { type: 'createProject' });

    expect(getProject(withNewProject, withNewProject.activeProjectId).invocation).toEqual({
      destination: 'canvas',
      destinationLocked: false,
      sourceId: 'generate',
      sourceLocked: false,
    });
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

  it('applies an account override for a built-in preset route', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      defaultRoute: { destination: 'canvas', sourceId: 'upscale' },
      presetId: 'compose',
      type: 'setLayoutPresetRoute',
    });
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });

    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'canvas', sourceId: 'upscale' });
  });

  it('applies preset routing with the edit preference off in the same undo entry as the layout', () => {
    const initial = createInitialWorkbenchState();
    const previousInvocation = getActiveProject(initial).invocation;
    const state = reduceWorkbench(
      initial,
      { presetId: 'edit', type: 'applyPreset' },
      { autoSwitchInvocationRoute: false }
    );
    const project = getActiveProject(state);

    expect(project.invocation).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });
    expect(project.undoRedo.past).toHaveLength(1);
    expect(project.undoRedo.past[0]?.project.invocation).toEqual(previousInvocation);
  });

  it('preserves each locked route field while applying the other preset default', () => {
    let sourceLocked = createInitialWorkbenchState();
    sourceLocked = workbenchReducer(sourceLocked, { sourceId: 'upscale', type: 'setInvocationSource' });
    sourceLocked = workbenchReducer(sourceLocked, { type: 'toggleSourceLock' });
    sourceLocked = workbenchReducer(sourceLocked, { presetId: 'automate', type: 'applyPreset' });

    expect(getActiveProject(sourceLocked).invocation).toMatchObject({
      destination: 'gallery',
      sourceId: 'upscale',
      sourceLocked: true,
    });

    let destinationLocked = createInitialWorkbenchState();
    destinationLocked = workbenchReducer(destinationLocked, {
      destination: 'canvas',
      type: 'setInvocationDestination',
    });
    destinationLocked = workbenchReducer(destinationLocked, { type: 'toggleDestinationLock' });
    destinationLocked = workbenchReducer(destinationLocked, { presetId: 'automate', type: 'applyPreset' });

    expect(getActiveProject(destinationLocked).invocation).toMatchObject({
      destination: 'canvas',
      destinationLocked: true,
      sourceId: 'workflow',
    });
  });

  it('falls back safely when a custom preset no longer contains its default source', () => {
    let seeded = createInitialWorkbenchState();
    seeded = workbenchReducer(seeded, {
      label: 'Generate only',
      presetId: 'custom-generate-only',
      type: 'addLayoutPreset',
    });
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

    let liveRouteStillPresent = workbenchReducer(seeded, {
      destination: 'canvas',
      type: 'setInvocationDestination',
    });
    liveRouteStillPresent = workbenchReducer(liveRouteStillPresent, {
      presetId: generateOnlyPreset.id,
      type: 'applyPreset',
    });
    expect(getActiveProject(liveRouteStillPresent).invocation).toMatchObject({
      destination: 'canvas',
      sourceId: 'generate',
    });

    let liveRouteMissing = workbenchReducer(seeded, { sourceId: 'canvas', type: 'setInvocationSource' });
    liveRouteMissing = workbenchReducer(liveRouteMissing, {
      destination: 'canvas',
      type: 'setInvocationDestination',
    });
    liveRouteMissing = workbenchReducer(liveRouteMissing, {
      presetId: generateOnlyPreset.id,
      type: 'applyPreset',
    });
    expect(getActiveProject(liveRouteMissing).invocation).toMatchObject({
      destination: 'gallery',
      sourceId: 'generate',
    });
  });

  it('applies the Compose preset as a full widget-region layout', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });

    const project = getActiveProject(state);

    expect(project.layout.panels).toEqual({ isBottomOpen: false, isLeftOpen: true, isRightOpen: true });
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
      instanceIds: ['server-status', 'queue-status', 'gallery:bottom', 'notifications', 'autosave-status'],
      isCollapsed: true,
      sizePx: 180,
    });
  });

  it('adds a custom preset from the active project layout and applies it later', () => {
    let state = createInitialWorkbenchState();
    const projectId = state.activeProjectId;

    state = workbenchReducer(state, { region: 'right', sizePx: 384, type: 'setRegionWidgetSize' });
    state = workbenchReducer(state, { region: 'right', type: 'selectRegionWidget', widgetId: 'queue' });
    state = workbenchReducer(state, { region: 'center', type: 'selectRegionWidget', widgetId: 'preview' });
    state = workbenchReducer(state, {
      label: 'Queue review',
      presetId: 'custom-layout-1',
      type: 'addLayoutPreset',
    });

    state = workbenchReducer(state, { presetId: 'canvas', type: 'applyPreset' });
    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'applyPreset' });

    const project = getProject(state, projectId);

    expect(state.account.customLayoutPresets).toHaveLength(1);
    expect(state.account.customLayoutPresets?.[0]).toMatchObject({ id: 'custom-layout-1', label: 'Queue review' });
    expect(project.widgetRegions.right).toMatchObject({ activeInstanceId: 'queue', sizePx: 384 });
    expect(project.widgetRegions.center.activeInstanceId).toBe('preview');
  });

  it('persists one account-wide order through reordering and custom preset deletion', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      label: 'Custom',
      presetId: 'custom-layout-1',
      type: 'addLayoutPreset',
    });

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

  it('rejects reserved, empty, and duplicate custom preset ids during hydration', () => {
    const initial = createInitialWorkbenchState();
    const customPreset = (id: string, label: string) => ({ id, label, snapshot: layoutPresets[0]!.snapshot });
    const state = workbenchReducer(initial, {
      state: {
        ...initial,
        account: {
          ...initial.account,
          activeLayoutPresetId: 'missing',
          customLayoutPresets: [
            customPreset('   ', 'Empty'),
            customPreset('compose', 'Built in'),
            customPreset(' canvas ', 'Legacy alias'),
            customPreset('custom-duplicate', 'First'),
            customPreset('custom-duplicate', 'Second'),
            customPreset(' custom-valid ', 'Valid'),
          ],
        },
      },
      type: 'hydrateWorkbench',
    });

    expect(state.account.activeLayoutPresetId).toBe('compose');
    expect(state.account.customLayoutPresets?.map(({ id, label }) => ({ id, label }))).toEqual([
      { id: 'custom-duplicate', label: 'First' },
      { id: 'custom-valid', label: 'Valid' },
    ]);
  });

  it('does not create a custom preset in a built-in or legacy id namespace', () => {
    const initial = createInitialWorkbenchState();
    const builtInCollision = workbenchReducer(initial, {
      label: 'Shadow Compose',
      presetId: 'compose',
      type: 'addLayoutPreset',
    });
    const legacyCollision = workbenchReducer(initial, {
      label: 'Shadow Edit',
      presetId: 'canvas',
      type: 'addLayoutPreset',
    });

    expect(builtInCollision).toBe(initial);
    expect(legacyCollision).toBe(initial);
  });

  it('keeps only built-in preset overrides during hydration', () => {
    const initial = createInitialWorkbenchState();
    const snapshot = layoutPresets[0]!.snapshot;
    const route = { destination: 'canvas' as const, sourceId: 'canvas' as const };
    const state = workbenchReducer(initial, {
      state: {
        ...initial,
        account: {
          ...initial.account,
          layoutPresetMetadataOverrides: {
            canvas: { label: 'Editing' },
            unknown: { label: 'Unknown' },
          },
          layoutPresetOverrides: { canvas: snapshot, unknown: snapshot },
          layoutPresetRouteOverrides: { canvas: route, unknown: route },
        },
      } as unknown as WorkbenchState,
      type: 'hydrateWorkbench',
    });

    expect(state.account.layoutPresetMetadataOverrides).toEqual({ edit: { label: 'Editing' } });
    expect(state.account.layoutPresetOverrides).toEqual({ edit: snapshot });
    expect(state.account.layoutPresetRouteOverrides).toEqual({ edit: route });
  });

  it('preserves a saved preset with an empty side region during hydration', () => {
    const initial = createInitialWorkbenchState();
    const snapshot = layoutPresets[0]!.snapshot;
    const emptyRightSnapshot = {
      ...snapshot,
      widgetRegions: {
        ...snapshot.widgetRegions,
        right: { ...snapshot.widgetRegions.right, instanceIds: [] },
      },
    };
    const state = workbenchReducer(initial, {
      state: {
        ...initial,
        account: {
          ...initial.account,
          layoutPresetOverrides: { compose: emptyRightSnapshot },
        },
      },
      type: 'hydrateWorkbench',
    });

    expect(state.account.layoutPresetOverrides).toEqual({ compose: emptyRightSnapshot });
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

  it('captures the live source and destination without routing locks when creating a custom preset', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });
    state = workbenchReducer(state, { destination: 'canvas', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { type: 'toggleRoutingLock' });
    state = workbenchReducer(state, {
      label: 'Upscale review',
      presetId: 'custom-layout-route',
      type: 'addLayoutPreset',
    });

    expect(state.account.customLayoutPresets?.[0]?.defaultRoute).toEqual({
      destination: 'canvas',
      sourceId: 'upscale',
    });
    expect(state.account.customLayoutPresets?.[0]?.defaultRoute).not.toHaveProperty('sourceLocked');
    expect(state.account.customLayoutPresets?.[0]?.defaultRoute).not.toHaveProperty('destinationLocked');
  });

  it('edits built-in and custom default routes without changing saved layouts', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      defaultRoute: { destination: 'gallery', sourceId: 'upscale' },
      presetId: 'compose',
      type: 'setLayoutPresetRoute',
    });
    state = workbenchReducer(state, {
      label: 'Custom route',
      presetId: 'custom-layout-route',
      type: 'addLayoutPreset',
    });
    const customSnapshot = state.account.customLayoutPresets?.[0]?.snapshot;
    state = workbenchReducer(state, {
      defaultRoute: { destination: 'canvas', sourceId: 'workflow' },
      presetId: 'custom-layout-route',
      type: 'setLayoutPresetRoute',
    });

    expect(state.account.layoutPresetRouteOverrides).toEqual({
      compose: { destination: 'gallery', sourceId: 'upscale' },
    });
    expect(state.account.layoutPresetOverrides).toBeUndefined();
    expect(state.account.customLayoutPresets?.[0]?.defaultRoute).toEqual({
      destination: 'canvas',
      sourceId: 'workflow',
    });
    expect(state.account.customLayoutPresets?.[0]?.snapshot).toBe(customSnapshot);
  });

  it('allows a source-less custom preset to omit its route while built-ins fall back to shipped routing', () => {
    let state = createInitialWorkbenchState();
    state = workbenchReducer(state, {
      label: 'No invoke surface',
      presetId: 'custom-no-route',
      type: 'addLayoutPreset',
    });
    state = workbenchReducer(state, {
      defaultRoute: { destination: 'canvas', sourceId: 'upscale' },
      presetId: 'compose',
      type: 'setLayoutPresetRoute',
    });
    state = workbenchReducer(state, {
      defaultRoute: null,
      presetId: 'custom-no-route',
      type: 'setLayoutPresetRoute',
    });
    state = workbenchReducer(state, {
      defaultRoute: null,
      presetId: 'compose',
      type: 'setLayoutPresetRoute',
    });

    expect(state.account.customLayoutPresets?.[0]).not.toHaveProperty('defaultRoute');
    expect(state.account.layoutPresetRouteOverrides?.compose).toBeUndefined();
  });

  it('does not retain a built-in override that matches its shipped route', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      defaultRoute: { destination: 'canvas', sourceId: 'upscale' },
      presetId: 'compose',
      type: 'setLayoutPresetRoute',
    });
    state = workbenchReducer(state, {
      defaultRoute: { destination: 'gallery', sourceId: 'generate' },
      presetId: 'compose',
      type: 'setLayoutPresetRoute',
    });

    expect(state.account.layoutPresetRouteOverrides?.compose).toBeUndefined();
  });

  it('renames and deletes a custom layout preset', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      label: 'Original',
      presetId: 'custom-layout-1',
      type: 'addLayoutPreset',
    });
    state = workbenchReducer(state, {
      label: 'Renamed',
      presetId: 'custom-layout-1',
      type: 'renameLayoutPreset',
    });
    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'deleteLayoutPreset' });

    expect(state.account.customLayoutPresets).toEqual([]);
  });

  it('edits and restores a built-in preset name and icon for the account', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { label: 'Writing', presetId: 'compose', type: 'renameLayoutPreset' });
    state = workbenchReducer(state, { iconId: 'star', presetId: 'compose', type: 'setLayoutPresetIcon' });

    expect(state.account.layoutPresetMetadataOverrides?.compose).toEqual({ iconId: 'star', label: 'Writing' });
    expect(resolveSavedLayoutPreset(state.account, 'compose')).toMatchObject({ iconId: 'star', label: 'Writing' });

    state = workbenchReducer(state, { label: 'Compose', presetId: 'compose', type: 'renameLayoutPreset' });

    expect(state.account.layoutPresetMetadataOverrides?.compose).toEqual({ iconId: 'star' });

    const layoutPresetOrder = state.account.layoutPresetOrder;
    state = workbenchReducer(state, { presetId: 'compose', type: 'restoreLayoutPresetDefault' });

    expect(state.account.layoutPresetMetadataOverrides?.compose).toBeUndefined();
    expect(resolveSavedLayoutPreset(state.account, 'compose')).toMatchObject({ iconId: 'type', label: 'Compose' });
    expect(state.account.layoutPresetOrder).toBe(layoutPresetOrder);
  });

  it('moves every project off a deleted custom layout preset', () => {
    let state = createInitialWorkbenchState();
    const firstProjectId = state.activeProjectId;

    state = workbenchReducer(state, {
      label: 'Shared custom layout',
      presetId: 'custom-layout-1',
      type: 'addLayoutPreset',
    });
    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'applyPreset' });
    state = workbenchReducer(state, { type: 'createProject' });
    const secondProjectId = state.activeProjectId;
    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'applyPreset' });

    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'deleteLayoutPreset' });

    expect(getProject(state, firstProjectId).layout.presetId).toBe('compose');
    expect(getProject(state, secondProjectId).layout.presetId).toBe('compose');
    expect(state.account.activeLayoutPresetId).toBe('compose');
  });

  it('preserves widget state across a Compose to Edit to Compose round trip', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });

    state = workbenchReducer(state, {
      type: 'patchGenerateSettings',
      values: { positivePrompt: 'a lighthouse at dusk', steps: 42 },
    });
    state = workbenchReducer(state, {
      instanceId: 'gallery',
      type: 'patchWidgetInstanceValues',
      values: { scrollOffset: 1280 },
    });

    state = workbenchReducer(state, { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'applyPreset' });

    const generateValues = getProjectWidgetValues(getActiveProject(state), 'generate');

    expect(generateValues.positivePrompt).toBe('a lighthouse at dusk');
    expect(generateValues.steps).toBe(42);
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').scrollOffset).toBe(1280);
  });

  it('saves the live arrangement onto a built-in preset and restores its shipped default', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'compose', type: 'applyPreset' });

    state = workbenchReducer(state, { region: 'right', sizePx: 360, type: 'setRegionWidgetSize' });
    state = workbenchReducer(state, { presetId: 'compose', type: 'saveLayoutPreset' });
    state = workbenchReducer(state, {
      defaultRoute: { destination: 'canvas', sourceId: 'upscale' },
      presetId: 'compose',
      type: 'setLayoutPresetRoute',
    });

    expect(state.account.layoutPresetOverrides?.compose?.widgetRegions.right.sizePx).toBe(360);
    expect(state.account.layoutPresetRouteOverrides?.compose).toEqual({ destination: 'canvas', sourceId: 'upscale' });

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });

    // Reverting now lands on the saved edit, not on the shipped arrangement.
    state = workbenchReducer(state, { region: 'right', sizePx: 500, type: 'setRegionWidgetSize' });
    state = workbenchReducer(state, { type: 'resetActiveLayout' });

    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(360);
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });

    state = workbenchReducer(state, { presetId: 'compose', type: 'restoreLayoutPresetDefault' });
    state = workbenchReducer(state, { type: 'resetActiveLayout' });

    expect(state.account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(state.account.layoutPresetRouteOverrides?.compose).toBeUndefined();
    expect(getActiveProject(state).widgetRegions.right.sizePx).toBe(450);
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });
  });

  it('resolves retired built-in preset ids onto the three shipped presets', () => {
    const initial = createInitialWorkbenchState();
    const hydrated = workbenchReducer(initial, {
      state: {
        ...initial,
        account: { activeLayoutPresetId: 'workflow' },
        projects: initial.projects.map((project) => ({
          ...project,
          layout: { ...project.layout, presetId: 'canvas' },
        })),
      } as unknown as WorkbenchState,
      type: 'hydrateWorkbench',
    });

    expect(hydrated.account.activeLayoutPresetId).toBe('automate');
    expect(getActiveProject(hydrated).layout.presetId).toBe('edit');
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

  it('marks a newly submitted queue item as proven local-only', () => {
    const state = submitGenerate(primeGenerate());

    expect(getActiveProject(state).queue.items[0]?.localRecoveryState).toBe('local-only');
  });

  it('never downgrades durable local recovery provenance', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0]!;
    state = workbenchReducer(state, {
      projectId: project.id,
      queueItemId: queueItem.id,
      state: 'durable',
      type: 'setQueueItemLocalRecoveryState',
    });
    state = workbenchReducer(state, {
      projectId: project.id,
      queueItemId: queueItem.id,
      state: 'local-only',
      type: 'setQueueItemLocalRecoveryState',
    });

    expect(getActiveProject(state).queue.items[0]?.localRecoveryState).toBe('durable');
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

  it('restores an active run when optional recall data is malformed', () => {
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
        {
          ...queueItem,
          snapshot: { ...queueItem.snapshot, recall: { generateValues: { model: 'not-a-model-config' } } },
        },
      ],
      projectId: project.id,
      type: 'restoreQueueItemsFromJournal',
    });

    expect(getActiveProject(state).queue.items).toEqual([
      expect.objectContaining({
        id: queueItem.id,
        snapshot: expect.not.objectContaining({ recall: expect.anything() }),
      }),
    ]);
  });

  it('does not notify gallery total subscribers for unchanged or non-finite totals', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { totalImages: 3, type: 'setGalleryPageInfo' });

    const unchanged = workbenchReducer(state, { totalImages: 3, type: 'setGalleryPageInfo' });
    const nonFinite = workbenchReducer(state, { totalImages: Number.NaN, type: 'setGalleryPageInfo' });

    expect(unchanged).toBe(state);
    expect(nonFinite).toBe(state);
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

  it("keeps the selected workflow candidate when auto-switch is 'off'", () => {
    let state = createInitialWorkbenchState();
    const projectId = state.activeProjectId;
    const placement = { height: 64, opacity: 1, width: 64, x: 0, y: 0 };
    const firstCandidate = createStagingCandidate('first.png', 'layer-workflow:first', placement);
    const secondCandidate = createStagingCandidate('second.png', 'layer-workflow:second', placement);

    state = workbenchReducer(state, { candidate: firstCandidate, projectId, type: 'appendCanvasStagingCandidate' });
    state = workbenchReducer(state, { candidate: secondCandidate, projectId, type: 'appendCanvasStagingCandidate' });

    expect(getCanvas(state).stagingArea.autoSwitchMode).toBe('off');
    expect(getCanvas(state).stagingArea.selectedImageIndex).toBe(0);
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

  it("selects the exact appended workflow candidate in 'latest' mode when candidate keys are duplicated", () => {
    let state = createInitialWorkbenchState();
    const projectId = state.activeProjectId;
    const firstCandidate = createStagingCandidate('duplicate.png', 'layer-workflow:duplicate', {
      height: 64,
      opacity: 1,
      width: 64,
      x: 10,
      y: 0,
    });
    const latestCandidate = {
      ...firstCandidate,
      placement: { ...firstCandidate.placement, x: 90 },
    };

    state = workbenchReducer(state, { mode: 'latest', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, { candidate: firstCandidate, projectId, type: 'appendCanvasStagingCandidate' });
    state = workbenchReducer(state, { candidate: latestCandidate, projectId, type: 'appendCanvasStagingCandidate' });

    const project = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(project.canvas, project.queue.items)[
      project.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ candidate: { placement: { x: 90 } }, kind: 'candidate' });
  });

  it('accepts a workflow candidate at its own placement, scale, and opacity', () => {
    let state = withEmptyCanvas(createInitialWorkbenchState());
    const projectId = state.activeProjectId;
    const candidate = createStagingCandidate('dimension-changing-result.png', 'layer-workflow:request-1', {
      height: 200,
      opacity: 0.35,
      width: 300,
      x: 31,
      y: 47,
    });

    state = workbenchReducer(state, { candidate, projectId, type: 'appendCanvasStagingCandidate' });
    state = commitSelectedStagedImage(state);

    const acceptedLayer = getDocumentLeaves(getActiveProject(state).canvas.document)[0];

    expect(getRasterLayerImageName(acceptedLayer)).toBe('dimension-changing-result.png');
    expect(getRasterLayerPlacement(acceptedLayer)).toEqual(candidate.placement);
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

  it('accepts a staged candidate into a selected raster layer that project undo no longer touches', () => {
    let state = submitGenerate(
      primeGenerate(withInvocationDestination(withEmptyCanvas(createInitialWorkbenchState()), 'canvas'))
    );
    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = commitSelectedStagedImage(state);

    let project = getActiveProject(state);
    const acceptedLayerId = getDocumentLeaves(project.canvas.document)[0]?.id;

    expect(getDocumentLeaves(project.canvas.document)).toHaveLength(1);
    expect(getRasterLayerImageName(getDocumentLeaves(project.canvas.document)[0])).toBe('candidate.png');
    expect(project.canvas.document.selectedLayerId).toBe(acceptedLayerId);
    expect(project.canvas.stagingArea.pendingImages).toEqual([]);
    // Canvas acceptance uses engine history, not project undo.
    expect(project.undoRedo.past).toHaveLength(0);

    state = commitSelectedStagedImage(state);
    project = getActiveProject(state);

    expect(getDocumentLeaves(project.canvas.document)).toHaveLength(1);

    // Project undo neither snapshots nor restores canvas: the accepted layer survives.
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    expect(getDocumentLeaves(project.canvas.document)).toHaveLength(1);
    expect(getRasterLayerImageName(getDocumentLeaves(project.canvas.document)[0])).toBe('candidate.png');
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

    // Project undo/redo leaves the engine-owned canvas alone.
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    expect(getRasterLayerImageName(project.canvas.document.stacks.raster[0] as CanvasLayer)).toBe('candidate-2.png');
  });

  it('cycles pending canvas placeholder slots before results complete', () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 2 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11, 12],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });

    expect(
      getCanvasStagingSlots(getCanvas(state), getActiveProject(state).queue.items).map((slot) => slot.kind)
    ).toEqual(['placeholder', 'placeholder']);

    state = workbenchReducer(state, { direction: 1, type: 'cycleStagedImage' });

    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(1);
  });

  it('stages canvas partial results while the rest of the batch keeps placeholders', () => {
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
      images: [createImage('candidate-1.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });

    const updatedProject = getActiveProject(state);
    const slots = getCanvasStagingSlots(updatedProject.canvas, updatedProject.queue.items);

    expect(updatedProject.queue.items[0]).toMatchObject({ completedBackendItemIds: [11], status: 'running' });
    expect(updatedProject.canvas.stagingArea.pendingImageIds).toEqual(['candidate-1.png']);
    expect(slots.map((slot) => slot.kind)).toEqual(['candidate', 'placeholder']);
    expect(slots[0]).toMatchObject({ itemIndex: 1, kind: 'candidate', queueItemId: queueItem.id });
    expect(updatedProject.canvas.stagingArea.pendingImages[0]).toMatchObject({ sourceBackendItemId: 11 });
    expect(updatedProject.canvas.stagingArea.pendingImages[0]?.placement).toEqual({
      height: 768,
      opacity: 1,
      width: 512,
      x: updatedProject.canvas.document.bbox.x,
      y: updatedProject.canvas.document.bbox.y,
    });
    expect(slots[1]).toMatchObject({ itemIndex: 2, queueItemId: queueItem.id });
  });

  it('preserves the selected staged candidate when final results duplicate partial results', () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 2 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    const firstImage = createImage('candidate-1.png', queueItem.id);
    const secondImage = createImage('candidate-2.png', queueItem.id);

    state = workbenchReducer(state, {
      backendItemIds: [11, 12],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [firstImage],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });
    state = workbenchReducer(state, {
      backendItemId: 12,
      images: [secondImage],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, {
      images: [firstImage, secondImage],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getActiveProject(state).canvas.stagingArea.pendingImageIds).toEqual(['candidate-1.png', 'candidate-2.png']);
    expect(getActiveProject(state).canvas.stagingArea.pendingImages).toMatchObject([
      { sourceBackendItemId: 11 },
      { sourceBackendItemId: 12 },
    ]);
    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(1);
  });

  it('clamps the selected staging slot when backend cancellation removes the selected placeholder', () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, { imageIndex: 2, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, {
      backendItemId: 13,
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendCancelled',
    });

    expect(getCanvasStagingSlots(getCanvas(state), getActiveProject(state).queue.items)).toHaveLength(2);
    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(1);
  });

  it('accepts the selected candidate when placeholders precede it in the staging strip', () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 1 }));
    const firstProject = getActiveProject(state);
    const firstQueueItem = firstProject.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11],
      projectId: firstProject.id,
      queueItemId: firstQueueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = submitGenerate(primeGenerate(state, { positivePrompt: 'second prompt' }));

    const secondProject = getActiveProject(state);
    const secondQueueItem = secondProject.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-2.png', secondQueueItem.id)],
      projectId: secondProject.id,
      queueItemId: secondQueueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });

    expect(
      getCanvasStagingSlots(getCanvas(state), getActiveProject(state).queue.items).map((slot) => slot.kind)
    ).toEqual(['placeholder', 'candidate']);
    expect(getCanvas(state).stagingArea.selectedImageIndex).toBe(1);

    state = commitSelectedStagedImage(state);

    expect(getRasterLayerImageName(getActiveProject(state).canvas.document.stacks.raster[0] as CanvasLayer)).toBe(
      'candidate-2.png'
    );
  });

  it('discards the selected candidate when placeholders precede it in the staging strip', () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 1 }));
    const firstProject = getActiveProject(state);
    const firstQueueItem = firstProject.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11],
      projectId: firstProject.id,
      queueItemId: firstQueueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = submitGenerate(primeGenerate(state, { positivePrompt: 'second prompt' }));

    const secondProject = getActiveProject(state);
    const secondQueueItem = secondProject.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-2.png', secondQueueItem.id)],
      projectId: secondProject.id,
      queueItemId: secondQueueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, { type: 'discardSelectedStagedImage' });

    expect(getActiveProject(state).canvas.stagingArea.pendingImages).toEqual([]);
    expect(
      getCanvasStagingSlots(getCanvas(state), getActiveProject(state).queue.items).map((slot) => slot.kind)
    ).toEqual(['placeholder']);
    expect(getCanvas(state).stagingArea.isVisible).toBe(true);
  });

  it('keeps thumbnail strip visibility separate from staged result preview visibility', () => {
    let state = submitGenerate(primeGenerate());
    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-1.png', queueItem.id), createImage('candidate-2.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getActiveProject(state).canvas.stagingArea.pendingImageIds).toEqual(['candidate-1.png', 'candidate-2.png']);
    expect(getActiveProject(state).canvas.stagingArea.areThumbnailsVisible).toBe(true);
    expect(getActiveProject(state).canvas.stagingArea.isVisible).toBe(true);

    state = workbenchReducer(state, { type: 'toggleCanvasStagingThumbnailsVisibility' });
    expect(getActiveProject(state).canvas.stagingArea.areThumbnailsVisible).toBe(false);
    expect(getActiveProject(state).canvas.stagingArea.isVisible).toBe(true);

    state = workbenchReducer(state, { type: 'toggleCanvasStagingVisibility' });
    expect(getActiveProject(state).canvas.stagingArea.areThumbnailsVisible).toBe(false);
    expect(getActiveProject(state).canvas.stagingArea.isVisible).toBe(false);
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

  it('cancels queue items from inactive projects when a project id is provided', () => {
    let state = submitGenerate(primeGenerate());
    const originProject = getActiveProject(state);
    const queueItem = originProject.queue.items[0];

    state = workbenchReducer(state, { type: 'createProject' });
    expect(getActiveProject(state).id).not.toBe(originProject.id);

    state = workbenchReducer(state, {
      projectId: originProject.id,
      queueItemId: queueItem.id,
      type: 'cancelQueueItem',
    });

    expect(getProject(state, originProject.id).queue.items[0]?.status).toBe('cancelled');
  });

  it('cancels all active cancellable queue items from the queue actions menu', () => {
    let state = submitGenerate(primeGenerate());

    state = workbenchReducer(state, { type: 'createProject' });
    state = submitGenerate(primeGenerate(state));
    state = workbenchReducer(state, { type: 'cancelAllQueueItems' });

    expect(
      state.projects.flatMap((project) =>
        project.queue.items.map((item) => ({ cancellationPending: item.cancellationPending, status: item.status }))
      )
    ).toEqual([
      { cancellationPending: true, status: 'cancelled' },
      { cancellationPending: true, status: 'cancelled' },
    ]);
  });

  it('cancels active queue items except the current one', () => {
    let state = submitGenerate(primeGenerate());
    const firstQueueItemId = getActiveProject(state).queue.items[0].id;

    state = submitGenerate(primeGenerate(state));
    state = workbenchReducer(state, {
      currentQueueItemId: firstQueueItemId,
      type: 'cancelAllQueueItemsExceptCurrent',
    });

    expect(
      getActiveProject(state).queue.items.map((item) => ({
        cancellationPending: item.cancellationPending,
        id: item.id,
        status: item.status,
      }))
    ).toEqual([
      { cancellationPending: true, id: getActiveProject(state).queue.items[0].id, status: 'cancelled' },
      { cancellationPending: undefined, id: firstQueueItemId, status: 'pending' },
    ]);
  });

  it('records backend item cancellation without cancelling the whole local batch', () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, {
      backendItemId: 12,
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendCancelled',
    });

    expect(getActiveProject(state).queue.items[0]).toMatchObject({
      cancelledBackendItemIds: [12],
      status: 'running',
    });
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

  it('scopes cancel all queue items to a project when requested', () => {
    let state = submitGenerate(primeGenerate());
    const originProject = getActiveProject(state);

    state = workbenchReducer(state, { type: 'createProject' });
    state = submitGenerate(primeGenerate(state));
    state = workbenchReducer(state, { projectId: originProject.id, type: 'cancelAllQueueItems' });

    expect(getProject(state, originProject.id).queue.items.map((item) => item.status)).toEqual(['cancelled']);
    expect(getActiveProject(state).queue.items.map((item) => item.status)).toEqual(['pending']);
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

  it('keeps the selected staged candidate when completed queue history is cleared', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('completed-result.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, {
      candidate: createStagingCandidate('workflow-result.png', 'layer-workflow:result', {
        height: 64,
        opacity: 1,
        width: 64,
        x: 0,
        y: 0,
      }),
      projectId: project.id,
      type: 'appendCanvasStagingCandidate',
    });
    state = workbenchReducer(state, { type: 'clearCompletedQueueItems' });

    const updatedProject = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(updatedProject.canvas, updatedProject.queue.items)[
      updatedProject.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ candidate: { imageName: 'completed-result.png' }, kind: 'candidate' });
  });

  it('can mark stale reconciled queue items failed without creating notifications', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    const notificationCount = state.notifications.length;

    state = workbenchReducer(state, {
      error: 'This run is no longer on the backend queue.',
      notify: false,
      projectId: project.id,
      queueItemId: queueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });

    expect(getActiveProject(state).queue.items[0]?.status).toBe('failed');
    expect(getActiveProject(state).queue.items[0]?.error).toBe('This run is no longer on the backend queue.');
    expect(state.notifications).toHaveLength(notificationCount);
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
    expect(state.notifications.map((notification) => notification.title)).toEqual([
      'Invocation cancellation requested',
      'Invocation queued',
    ]);
  });

  it('does not report cancellation for completed queue items', () => {
    let state = submitGenerate(primeGenerate());
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('completed-result.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { queueItemId: queueItem.id, type: 'cancelQueueItem' });

    expect(getActiveProject(state).queue.items[0]?.status).toBe('completed');
    expect(state.notifications.map((notification) => notification.title)).toEqual([
      'Invocation completed',
      'Invocation queued',
    ]);
  });

  it('records notifications for errors and successful operations', () => {
    let state = submitGenerate(primeGenerate());

    expect(state.notifications[0]?.title).toBe('Invocation queued');

    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = commitSelectedStagedImage(state);
    state = workbenchReducer(state, { message: 'boom', type: 'recordError' });

    expect(state.notifications.map((notification) => notification.title)).toEqual([
      'Error',
      'Invocation completed',
      'Invocation queued',
    ]);
    expect(state.notifications.every((notification) => !notification.isRead)).toBe(true);

    state = workbenchReducer(state, { type: 'markAllNotificationsRead' });

    expect(state.notifications.every((notification) => notification.isRead)).toBe(true);

    state = workbenchReducer(state, { type: 'clearNotifications' });

    expect(state.notifications).toEqual([]);

    expect(state.notifications).toEqual([]);
  });

  it('coalesces back-to-back identical notifications instead of stacking them', () => {
    const state = createInitialWorkbenchState();

    const once = reduceWorkbench(state, { message: 'Bitmap persistence failed', type: 'recordError' });
    const twice = reduceWorkbench(once, { message: 'Bitmap persistence failed', type: 'recordError' });

    expect(twice.notifications).toHaveLength(once.notifications.length);
    expect(twice.notifications[0]?.occurrenceCount).toBe(2);
    expect(twice.notifications[0]?.id).toBe(once.notifications[0]?.id);
  });

  it('includes the error context detail in the recordError notification message', () => {
    const state = createInitialWorkbenchState();

    const next = reduceWorkbench(state, {
      context: { error: 'network unreachable', layerId: 'layer-1' },
      message: 'Bitmap persistence failed',
      type: 'recordError',
    });

    expect(next.notifications[0]?.message).toBe('Bitmap persistence failed: network unreachable');
  });

  it('does not coalesce notifications with a different message', () => {
    const state = createInitialWorkbenchState();

    const once = reduceWorkbench(state, { message: 'Bitmap persistence failed', type: 'recordError' });
    const twice = reduceWorkbench(once, { message: 'Something else failed', type: 'recordError' });

    expect(twice.notifications).toHaveLength(once.notifications.length + 1);
    expect(twice.notifications[0]?.id).not.toBe(once.notifications[0]?.id);
  });

  it('does not coalesce repeat non-error notifications (each queued invocation gets its own toast)', () => {
    const state = primeGenerate();

    const once = workbenchReducer(state, {
      backendSupportsCancellation: true,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
      type: 'submitResolvedInvocationSnapshot',
    });
    const twice = workbenchReducer(once, {
      backendSupportsCancellation: true,
      route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
      type: 'submitResolvedInvocationSnapshot',
    });

    const enqueueNotifications = twice.notifications.filter((notification) => notification.category === 'enqueue');

    expect(enqueueNotifications).toHaveLength(2);
    expect(enqueueNotifications[0]?.id).not.toBe(enqueueNotifications[1]?.id);
    expect(enqueueNotifications.every((notification) => notification.occurrenceCount === undefined)).toBe(true);
  });

  it('resets isRead to false when a coalesced error repeats after being read', () => {
    let state = createInitialWorkbenchState();

    state = reduceWorkbench(state, { message: 'Bitmap persistence failed', type: 'recordError' });
    state = reduceWorkbench(state, { type: 'markAllNotificationsRead' });

    expect(state.notifications[0]?.isRead).toBe(true);

    state = reduceWorkbench(state, { message: 'Bitmap persistence failed', type: 'recordError' });

    expect(state.notifications[0]?.occurrenceCount).toBe(2);
    expect(state.notifications[0]?.isRead).toBe(false);
  });

  describe('enqueue notifications', () => {
    it('records an enqueue notification when submitResolvedInvocationSnapshot adds a queue item', () => {
      const state = primeGenerate();

      const next = workbenchReducer(state, {
        backendSupportsCancellation: true,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        type: 'submitResolvedInvocationSnapshot',
      });

      expect(getActiveProject(next).queue.items.length).toBeGreaterThan(0);
      expect(next.notifications[0]?.title).toBe('Invocation queued');
      expect(next.notifications[0]?.category).toBe('enqueue');
      expect(next.notifications[0]?.kind).toBe('success');
    });

    it('does not record a notification when the submit adds no queue item', () => {
      let state = createInitialWorkbenchState();

      state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });

      const notificationCount = state.notifications.length;

      const next = workbenchReducer(state, {
        backendSupportsCancellation: true,
        route: { destination: 'canvas', destinationLocked: false, sourceId: 'upscale', sourceLocked: true },
        type: 'submitResolvedInvocationSnapshot',
      });

      expect(getActiveProject(next).queue.items).toEqual([]);
      expect(next.notifications).toHaveLength(notificationCount);
    });

    it('notifies for a background-project canvas enqueue', () => {
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

      // Project A (originating) stays active; the canvas submission below
      // targets project B, simulating an async canvas enqueue that lands
      // after the user has switched away.
      state = workbenchReducer(state, { projectId: originatingProjectId, type: 'switchProject' });

      expect(state.activeProjectId).toBe(originatingProjectId);

      state = workbenchReducer(state, {
        backendSupportsCancellation: true,
        canvas: structuredClone(getProject(state, otherProjectId).canvas),
        destination: 'canvas',
        generate: {
          negativePromptNodeId: 'negative_prompt',
          positivePromptNodeId: 'positive_prompt',
          seedNodeId: 'seed',
          values: createGenerateValues({ positivePrompt: 'canvas prompt', seed: 101, seedMode: 'fixed' }),
        },
        graph,
        projectId: otherProjectId,
        type: 'submitCanvasInvocationSnapshot',
      });

      expect(getProject(state, otherProjectId).queue.items).toHaveLength(1);
      expect(state.notifications[0]?.title).toBe('Invocation queued');
      expect(state.notifications[0]?.projectId).toBe(otherProjectId);
    });
  });

  it('accepts the project graph source but does not queue an empty project graph', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });

    expect(getActiveProject(state).invocation.sourceId).toBe('workflow');

    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      route: { destination: 'canvas', destinationLocked: false, sourceId: 'workflow', sourceLocked: true },
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getActiveProject(state).queue.items).toEqual([]);
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
      type: 'applyProjectGraphAction',
    });

    const project = getActiveProject(state);

    expect(project.projectGraph.nodes).toHaveLength(1);
    expect(project.undoRedo.past.at(-1)?.label).toBe('Add workflow node');
    expect(project.invocation.sourceId).toBe('workflow');
    expect(project.invocation.destination).toBe('gallery');

    state = workbenchReducer(state, { type: 'undoProjectChange' });

    expect(getActiveProject(state).projectGraph.nodes).toHaveLength(0);
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });
  });

  it('replaceProjectGraph preserves the previous document in session undo', () => {
    let state = createInitialWorkbenchState();
    const originalGraphId = getActiveProject(state).projectGraph.id;

    state = workbenchReducer(state, {
      document: { ...getActiveProject(state).projectGraph, id: 'replacement-graph', name: 'Replacement' },
      label: 'Test replace',
      type: 'replaceProjectGraph',
    });

    const project = getActiveProject(state);

    expect(project.projectGraph.id).toBe('replacement-graph');
    expect(project.invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });
    expect(project.undoRedo.past.at(-1)?.project.projectGraph.id).toBe(originalGraphId);

    state = workbenchReducer(state, { type: 'undoProjectChange' });

    expect(getActiveProject(state).projectGraph.id).toBe(originalGraphId);
    expect(getActiveProject(state).invocation).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });
  });

  it('does not queue Upscale while its required settings are incomplete', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });

    expect(getActiveProject(state).invocation.sourceId).toBe('upscale');

    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      route: { destination: 'canvas', destinationLocked: false, sourceId: 'upscale', sourceLocked: true },
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getActiveProject(state).queue.items).toEqual([]);
  });

  it('does not queue Generate snapshots with non-finite numeric settings', () => {
    let state = primeGenerate(undefined, { seed: Number.NaN });

    state = submitGenerate(state);

    expect(getActiveProject(state).queue.items).toEqual([]);
  });

  describe('expanded positive prompts on the compiled submission', () => {
    const submitResolvedWithPrompts = (positivePrompt: string, positivePrompts?: string[]) =>
      workbenchReducer(primeGenerate(undefined, { positivePrompt }), {
        backendSupportsCancellation: true,
        positivePrompts,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        type: 'submitResolvedInvocationSnapshot',
      });

    // A single expansion is still concrete and must replace authored dynamic syntax.
    it('records a single expanded prompt rather than falling back to the literal', () => {
      const state = submitResolvedWithPrompts('a {red} cat', ['a red cat']);

      expect(getActiveProject(state).queue.items[0]?.snapshot.backendSubmission).toMatchObject({
        positivePrompt: 'a {red} cat',
        positivePrompts: ['a red cat'],
      });
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

    it('leaves a prompt with no dynamic syntax alone', () => {
      const state = submitResolvedWithPrompts('a plain cat');

      expect(getActiveProject(state).queue.items[0]?.snapshot.backendSubmission).not.toHaveProperty('positivePrompts');
    });

    // The template is what introduced the `{…}` here, so the gate has to read the
    // merged prompt or the expansion would be discarded and the literal submitted.
    it('keeps an expansion whose dynamic syntax came from the prompt template', () => {
      const state = workbenchReducer(
        primeGenerate(undefined, {
          positivePrompt: 'a cat',
          promptTemplate: {
            id: 't1',
            name: 'Tinted',
            negativePrompt: '',
            positivePrompt: '{prompt}, {red|green} tint',
          },
        }),
        {
          backendSupportsCancellation: true,
          positivePrompts: ['a cat, red tint', 'a cat, green tint'],
          route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
          type: 'submitResolvedInvocationSnapshot',
        }
      );

      expect(getActiveProject(state).queue.items[0]?.snapshot.backendSubmission).toMatchObject({
        positivePrompts: ['a cat, red tint', 'a cat, green tint'],
      });
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

      // The Automate preset mounts the Workflow widget, which the route requires.
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
        type: 'applyProjectGraphAction',
      });

      return state;
    };
    const submitWorkflow = (state: WorkbenchState) =>
      workbenchReducer(state, {
        backendSupportsCancellation: true,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'workflow', sourceLocked: false },
        type: 'submitResolvedInvocationSnapshot',
      });
    const readSubmission = (state: WorkbenchState, index = 0) =>
      getActiveProject(state).queue.items[index]?.snapshot.backendSubmission;
    const readNodeSeed = (state: WorkbenchState) => {
      const node = getActiveProject(state).projectGraph.nodes[0];

      return node?.type === 'invocation' ? node.data.inputs.seed?.value : undefined;
    };

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
        type: 'applyProjectGraphAction',
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
        type: 'applyProjectGraphAction',
      });
      state = submitWorkflow(state);

      expect(readSubmission(state)).toMatchObject({
        batchCount: 3,
        batchData: [[{ fieldName: 'value', items: [1.5, 2.5], nodeId: 'float-1' }]],
        kind: 'workflow',
      });
      // Batch nodes never reach the backend graph; the fed input keeps its static value for the server to overwrite.
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

    it('notifies when the queued workflow omits metadata the backend would reject', () => {
      // Multiple workflow_return nodes fail WorkflowWithoutID validation with 422; omit invalid workflow metadata
      // rather than reject generation.
      const returnTemplate = {
        category: 'workflow',
        classification: 'stable',
        description: '',
        inputs: {},
        nodePack: 'invokeai',
        outputType: 'workflow_return_output',
        outputs: {},
        tags: [],
        title: 'Workflow Return',
        type: 'workflow_return',
        useCache: true,
        version: '1.0.0',
      };
      let state = primeWorkflow(42, 'fixed');

      workflowTemplatesMock.snapshot = {
        error: null,
        status: 'loaded',
        templates: { noise: seedTemplate, workflow_return: returnTemplate },
      };

      for (const id of ['return-1', 'return-2']) {
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
                type: 'workflow_return',
                useCache: true,
                version: '1.0.0',
              },
              id,
              position: { x: 0, y: 0 },
              type: 'invocation',
            },
            type: 'addNode',
          },
          type: 'applyProjectGraphAction',
        });
      }

      const nextState = submitWorkflow(state);
      const submission = readSubmission(nextState);

      // `kind` guards the assertion below: a graph that fails to compile also
      // yields an object with no `workflow` key, which would pass for the
      // wrong reason.
      expect(submission).toMatchObject({ kind: 'workflow' });
      expect(submission).not.toHaveProperty('workflow');
      expect(
        nextState.notifications.find((notification) => notification.title === 'Workflow metadata omitted')
      ).toMatchObject({
        kind: 'info',
        message: 'Workflow metadata was omitted because the workflow contains multiple workflow_return nodes.',
        messageKey: 'workflowLibrary.workflowMetadataOmittedBody',
        title: 'Workflow metadata omitted',
        titleKey: 'workflowLibrary.workflowMetadataOmitted',
      });

      // The omitted-metadata notice is additional information about the run, not
      // a replacement for it: the run was queued, so the queued notification has
      // to be recorded too. `category: 'enqueue'` is also what lets the toast
      // policy honour the user's "notify when queued" preference.
      expect(nextState.notifications.filter((notification) => notification.category === 'enqueue')).toHaveLength(1);
      const repeatedState = submitWorkflow(nextState);

      expect(
        repeatedState.notifications.filter((notification) => notification.title === 'Workflow metadata omitted')
      ).toHaveLength(2);
    });

    it('holds a fixed seed as a graph constant and repeats the graph for every run', () => {
      // Generate's own iteration count no longer leaks into workflow runs.
      const state = submitWorkflow(
        workbenchReducer(primeWorkflow(42, 'fixed'), {
          type: 'patchWidgetValues',
          values: { batchCount: 5 },
          widgetId: 'generate',
        })
      );
      const submission = readSubmission(state);

      expect(submission).toMatchObject({ batchCount: 3, kind: 'workflow' });
      expect(submission).not.toHaveProperty('seeds');
      expect(submission?.kind === 'workflow' && submission.graph.nodes['noise-1']?.seed).toBe(42);
      expect(readNodeSeed(state)).toBe(42);
    });

    it('draws a random start for the batch and preserves the entered seed', () => {
      const random = vi.spyOn(Math, 'random').mockReturnValue(0.25);

      try {
        const state = submitWorkflow(primeWorkflow(42, 'random', 2));
        const start = Math.floor(0.25 * SEED_MAX);

        expect(readSubmission(state)).toMatchObject({
          graph: { nodes: { 'noise-1': { seed: start } } },
          seeds: [{ seed: start, seedStep: 1 }],
        });
        expect(readNodeSeed(state)).toBe(42);
      } finally {
        random.mockRestore();
      }
    });

    it('does not move the seed when the route is not ready to submit', () => {
      const state = primeWorkflow(42, 'increment');

      workflowTemplatesMock.snapshot = { error: null, status: 'loading', templates: {} };

      const next = submitWorkflow(state);

      expect(getActiveProject(next).queue.items).toEqual([]);
      expect(readNodeSeed(next)).toBe(42);
    });
  });

  describe('seed modes on the compiled submission', () => {
    const SEED_MAX = 4_294_967_295;
    const readSeed = (state: WorkbenchState) => getProjectWidgetValues(getActiveProject(state), 'generate').seed;
    const readSubmission = (state: WorkbenchState) =>
      getActiveProject(state).queue.items[0]?.snapshot.backendSubmission;
    const submitWithPrompts = (state: WorkbenchState, positivePrompts?: string[]) =>
      workbenchReducer(state, {
        backendSupportsCancellation: true,
        positivePrompts,
        route: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
        type: 'submitResolvedInvocationSnapshot',
      });

    it('increment queues consecutive seeds and leaves the editable seed after the batch', () => {
      let state = submitGenerate(primeGenerate(undefined, { batchCount: 3, seed: 42, seedMode: 'increment' }));

      expect(readSubmission(state)).toMatchObject({ batchCount: 3, seed: 42, seedStep: 1 });
      expect(readSeed(state)).toBe(45);

      // Queued back to back, the next submission continues where the last one ended.
      state = submitGenerate(state);

      expect(readSubmission(state)).toMatchObject({ seed: 45, seedStep: 1 });
      expect(readSeed(state)).toBe(48);
      expect(getActiveProject(state).queue.items[1]?.snapshot.backendSubmission).toMatchObject({ seed: 42 });
    });

    it('decrement counts down and wraps below zero onto the top of the range', () => {
      const state = submitGenerate(primeGenerate(undefined, { batchCount: 2, seed: 1, seedMode: 'decrement' }));

      expect(readSubmission(state)).toMatchObject({ seed: 1, seedStep: -1 });
      expect(readSeed(state)).toBe(SEED_MAX);
    });

    it('fixed holds the seed for the batch and does not move it', () => {
      const state = submitGenerate(primeGenerate(undefined, { batchCount: 3, seed: 42, seedMode: 'fixed' }));

      expect(readSubmission(state)).toMatchObject({ seed: 42, seedStep: 0 });
      expect(readSeed(state)).toBe(42);
    });

    it('random draws the start seed for the batch and preserves the entered one', () => {
      const random = vi.spyOn(Math, 'random').mockReturnValue(0.25);

      try {
        const state = submitGenerate(primeGenerate(undefined, { batchCount: 2, seed: 42, seedMode: 'random' }));

        expect(readSubmission(state)).toMatchObject({ seed: Math.floor(0.25 * SEED_MAX), seedStep: 1 });
        expect(readSeed(state)).toBe(42);
      } finally {
        random.mockRestore();
      }
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

    it('does not consume seeds when the submission is rejected', () => {
      const state = primeGenerate(undefined, { seed: 42, seedMode: 'increment', steps: Number.NaN });
      const next = submitGenerate(state);

      expect(getActiveProject(next).queue.items).toEqual([]);
      expect(readSeed(next)).toBe(42);
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
        type: 'submitResolvedInvocationSnapshot',
      });

      expect(readSubmission(state)).toMatchObject({ batchCount: 2, seed: 10, seedStep: 1 });
      expect(getProjectWidgetValues(getActiveProject(state), 'upscale').seed).toBe(12);
      // Generate's own seed is not the one that moved.
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
        type: 'submitInvocationSnapshot',
      });

      expect(readSubmission(state)).toMatchObject({ batchCount: 3, seed: 7, seedStep: -1 });
      expect(getProjectWidgetValues(getActiveProject(state), 'video').seed).toBe(4);
    });

    describe('from the canvas, which compiles outside the reducer', () => {
      const submitCanvas = (state: WorkbenchState, values: GenerateWidgetValues) =>
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
          projectId: state.activeProjectId,
          type: 'submitCanvasInvocationSnapshot',
        });

      it('advances the settings the snapshot was compiled from', () => {
        const values = createGenerateValues({ batchCount: 2, seed: 10, seedMode: 'increment' });
        const state = submitCanvas(primeGenerate(undefined, values), values);

        expect(readSubmission(state)).toMatchObject({ seed: 10, seedStep: 1 });
        expect(readSeed(state)).toBe(12);
      });

      it('leaves settings the user changed while the canvas was compiling', () => {
        const compiled = createGenerateValues({ batchCount: 2, seed: 10, seedMode: 'increment' });
        const seedEdited = submitCanvas(primeGenerate(undefined, { ...compiled, seed: 500 }), compiled);
        const modeEdited = submitCanvas(primeGenerate(undefined, { ...compiled, seedMode: 'fixed' }), compiled);

        expect(readSubmission(seedEdited)).toMatchObject({ seed: 10, seedStep: 1 });
        expect(readSeed(seedEdited)).toBe(500);
        expect(readSubmission(modeEdited)).toMatchObject({ seed: 10, seedStep: 1 });
        expect(readSeed(modeEdited)).toBe(10);
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

    // Switching the negative field off must not let a template put one back.
    it('still sends an empty negative prompt when the field is disabled', () => {
      const state = submitGenerate(
        primeGenerate(undefined, {
          negativePrompt: 'blurry',
          negativePromptEnabled: false,
          positivePrompt: 'a cat',
          promptTemplate: template,
        })
      );

      expect(getActiveProject(state).queue.items[0]?.snapshot.backendSubmission).toMatchObject({
        negativePrompt: '',
      });
    });

    // History restores textarea drafts, so it must retain authored rather than expanded text.
    it('records the authored prompt in prompt history', () => {
      const state = submitGenerate(
        primeGenerate(undefined, {
          negativePrompt: 'blurry',
          negativePromptEnabled: true,
          positivePrompt: 'a cat',
          promptTemplate: template,
        })
      );

      expect(getActiveProject(state).promptHistory).toEqual([{ negativePrompt: 'blurry', positivePrompt: 'a cat' }]);
    });
  });

  it('records submitted Generate prompt pairs in project prompt history', () => {
    let state = primeGenerate(undefined, { negativePrompt: ' blurry ', positivePrompt: ' a cat ' });

    state = submitGenerate(state);

    expect(getActiveProject(state).promptHistory).toEqual([{ negativePrompt: 'blurry', positivePrompt: 'a cat' }]);
  });

  it('does not record empty Generate prompt pairs', () => {
    let state = primeGenerate(undefined, { negativePrompt: ' ', positivePrompt: ' ' });

    state = submitGenerate(state);

    expect(getActiveProject(state).promptHistory).toEqual([]);
  });

  it('stores disabled negative prompts as null in project prompt history', () => {
    let state = primeGenerate(undefined, {
      negativePrompt: 'ignored negative prompt',
      negativePromptEnabled: false,
      positivePrompt: 'a cat',
    });

    state = submitGenerate(state);

    expect(getActiveProject(state).promptHistory).toEqual([{ negativePrompt: null, positivePrompt: 'a cat' }]);
  });

  it('patches Generate settings without replacing unchanged nested values', () => {
    const loraModel = { base: 'sdxl', key: 'lora-1', name: 'LoRA', type: 'lora' } as const;
    const state = primeGenerate(undefined, {
      loras: [{ isEnabled: true, model: loraModel, weight: 0.75 }],
      positivePrompt: 'before',
    });
    const beforeValues = getProjectWidgetValues(getActiveProject(state), 'generate') as unknown as GenerateWidgetValues;
    const nextState = workbenchReducer(state, {
      type: 'patchGenerateSettings',
      values: { positivePrompt: 'after' },
    });
    const afterValues = getProjectWidgetValues(
      getActiveProject(nextState),
      'generate'
    ) as unknown as GenerateWidgetValues;

    expect(afterValues.positivePrompt).toBe('after');
    expect(afterValues.model).toBe(beforeValues.model);
    expect(afterValues.loras).toBe(beforeValues.loras);
  });

  it('shares prompt content through Generate without changing Upscale-local textarea sizing', () => {
    const state = createInitialWorkbenchState();
    const beforeUpscale = getProjectWidgetValues(getActiveProject(state), 'upscale');
    const nextState = workbenchReducer(state, {
      sourceId: 'upscale',
      type: 'patchProjectPromptDraft',
      values: { negativePrompt: 'blur', negativePromptEnabled: false, positivePrompt: 'fine detail' },
    });

    expect(getProjectWidgetValues(getActiveProject(nextState), 'generate')).toMatchObject({
      negativePrompt: 'blur',
      negativePromptEnabled: false,
      positivePrompt: 'fine detail',
    });
    expect(getProjectWidgetValues(getActiveProject(nextState), 'upscale')).toBe(beforeUpscale);
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

    // Video writes its own widget values and leaves the shared draft alone...
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

    // ...and a later Generate edit does not reach back into Video.
    const beforeVideo = getProjectWidgetValues(getActiveProject(state), 'video');

    state = workbenchReducer(state, {
      sourceId: 'generate',
      type: 'patchProjectPromptDraft',
      values: { positivePrompt: 'a still landscape' },
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'video')).toBe(beforeVideo);
  });

  it('submits the Video prompt, not the Generate prompt, on a video invocation', () => {
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
    let state = createInitialWorkbenchState();

    // Video is no longer placed by the non-video defaults; add it first.
    state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId: 'video' });
    state = workbenchReducer(state, {
      sourceId: 'generate',
      type: 'patchProjectPromptDraft',
      values: { negativePrompt: 'blurry', positivePrompt: 'a still portrait' },
    });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { model: wanModel, negativePrompt: 'static camera', positivePrompt: 'a fox running' },
      widgetId: 'video',
    });
    state = workbenchReducer(state, { sourceId: 'video', type: 'setInvocationSource' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      models: [wanModel],
      type: 'submitInvocationSnapshot',
    });

    const submission = getActiveProject(state).queue.items[0]?.snapshot.recall?.videoValues;

    expect(submission).toMatchObject({ negativePrompt: 'static camera', positivePrompt: 'a fox running' });
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

  it('caps project prompt history', () => {
    let state = createInitialWorkbenchState();

    for (let i = 0; i < MAX_PROMPT_HISTORY + 1; i += 1) {
      state = primeGenerate(state, { positivePrompt: `prompt ${i}` });
      state = submitGenerate(state);
    }

    const history = getActiveProject(state).promptHistory;

    expect(history).toHaveLength(MAX_PROMPT_HISTORY);
    expect(history[0]?.positivePrompt).toBe(`prompt ${MAX_PROMPT_HISTORY}`);
    expect(history.at(-1)?.positivePrompt).toBe('prompt 1');
  });

  it('supports explicit prompt history remove and clear actions', () => {
    let state = primeGenerate(undefined, { negativePrompt: 'low quality', positivePrompt: 'a cat' });

    state = submitGenerate(state);
    state = workbenchReducer(state, {
      prompt: { negativePrompt: 'blurry', positivePrompt: 'a dog' },
      type: 'addPromptToHistory',
    });
    state = workbenchReducer(state, {
      prompt: { negativePrompt: 'low quality', positivePrompt: 'a cat' },
      type: 'removePromptFromHistory',
    });

    expect(getActiveProject(state).promptHistory).toEqual([{ negativePrompt: 'blurry', positivePrompt: 'a dog' }]);

    state = workbenchReducer(state, { type: 'clearPromptHistory' });

    expect(getActiveProject(state).promptHistory).toEqual([]);
  });

  it('does not roll back prompt history during project undo and redo', () => {
    let state = primeGenerate(undefined, { positivePrompt: 'history prompt' });

    state = submitGenerate(state);
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    state = workbenchReducer(state, { type: 'redoProjectChange' });

    expect(getActiveProject(state).promptHistory).toEqual([{ negativePrompt: null, positivePrompt: 'history prompt' }]);
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

  it('normalizes the gallery page to a non-negative integer', () => {
    let state = primeGenerate();

    state = workbenchReducer(state, { page: 2.9, type: 'setGalleryPage' });
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').galleryPage).toBe(2);

    state = workbenchReducer(state, { page: -3, type: 'setGalleryPage' });
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').galleryPage).toBe(0);

    state = workbenchReducer(state, { page: Number.NaN, type: 'setGalleryPage' });
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').galleryPage).toBe(0);
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

  it('leaves the window anchor alone when the result lands on a board that is not being viewed', () => {
    // The anchor describes the viewed board's listing; a result arriving in
    // some other board says nothing about where the user is looking.
    let state = primeGenerate();

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { boardId: 'board-elsewhere', type: 'selectGalleryBoard' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, { boardId: 'board-viewed', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, { page: 11, type: 'setGalleryPage' });
    state = workbenchReducer(state, {
      images: [createImage('elsewhere.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').galleryPage).toBe(11);
  });

  it('preserves the destination board on freshly routed Gallery results', () => {
    let state = primeGenerate();

    state = workbenchReducer(state, { boardId: 'board-1', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    const image = createImage('board-image.png', queueItem.id);
    state = workbenchReducer(state, {
      images: [image],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImage).toMatchObject({
      boardId: 'board-1',
      kind: 'image',
      name: image.imageName,
    });
  });

  it('keeps the board a result image was saved to instead of the run destination board', () => {
    let state = primeGenerate();

    state = workbenchReducer(state, { boardId: 'board-1', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    const image = { ...createImage('node-board-image.png', queueItem.id), boardId: 'board-b' };
    state = workbenchReducer(state, {
      images: [image],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    const galleryValues = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect((galleryValues.recentImages as Array<{ boardId: string }>)[0]?.boardId).toBe('board-b');
    expect(galleryValues.selectedImage).toMatchObject({ boardId: 'board-b', name: image.imageName });
  });

  it('appends Gallery destination results for local fallback while backend owns boards', () => {
    let state = primeGenerate();

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const firstQueueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('gallery-image-1.png', firstQueueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: firstQueueItem.id,
      type: 'routeQueueItemResults',
    });
    state = submitGenerate(state);

    const secondQueueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('gallery-image-2.png', secondQueueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: secondQueueItem.id,
      type: 'routeQueueItemResults',
    });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect((values.recentImages as GeneratedImageContract[]).map((image) => image.imageName)).toEqual([
      'gallery-image-2.png',
      'gallery-image-1.png',
    ]);
    expect(values.imageBoards).toBeUndefined();
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

  it('omits widget state from immutable queue snapshots', () => {
    let state = primeGenerate();
    const recentImages = Array.from({ length: GALLERY_RECENT_IMAGE_LIMIT }, (_, index) =>
      createImage(`recent-${index}.png`, 'previous-queue-item')
    );

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { recentImages },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const snapshot = getActiveProject(state).queue.items[0]!.snapshot as unknown as Record<string, unknown>;

    expect(snapshot).not.toHaveProperty('widgetStates');
    expect(snapshot).not.toHaveProperty('widgetInstances');
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

  it('preserves the last partial Gallery result when the final aggregate repeats the batch', () => {
    let state = primeGenerate(undefined, { batchCount: 3 });

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    const images = ['gallery-image-1.png', 'gallery-image-2.png', 'gallery-image-3.png'].map((name) =>
      createImage(name, queueItem.id)
    );

    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });

    for (const [index, image] of images.entries()) {
      state = workbenchReducer(state, {
        backendItemId: 11 + index,
        images: [image],
        projectId: project.id,
        queueItemId: queueItem.id,
        type: 'routeQueueItemPartialResults',
      });
    }

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe(
      'image:gallery-image-3.png'
    );

    state = workbenchReducer(state, {
      images,
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe(
      'image:gallery-image-3.png'
    );
  });

  it('does not reorder existing results when an older batch final aggregate arrives late', () => {
    let state = primeGenerate(undefined, { batchCount: 2 });

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const projectId = getActiveProject(state).id;
    const olderQueueItem = getActiveProject(state).queue.items[0];
    const olderImage1 = createImage('older-1.png', olderQueueItem.id);
    const olderImage2 = createImage('older-2.png', olderQueueItem.id);

    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [olderImage1],
      projectId,
      queueItemId: olderQueueItem.id,
      type: 'routeQueueItemPartialResults',
    });
    state = submitGenerate(state);

    const newerQueueItem = getActiveProject(state).queue.items[0];
    const newerImage = createImage('newer-1.png', newerQueueItem.id);

    state = workbenchReducer(state, {
      backendItemId: 21,
      images: [newerImage],
      projectId,
      queueItemId: newerQueueItem.id,
      type: 'routeQueueItemPartialResults',
    });
    state = workbenchReducer(state, {
      images: [olderImage1, olderImage2],
      projectId,
      queueItemId: olderQueueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(
      (getProjectWidgetValues(getActiveProject(state), 'gallery').recentImages as GeneratedImageContract[]).map(
        (image) => image.imageName
      )
    ).toEqual(['older-2.png', 'newer-1.png', 'older-1.png']);
  });

  it('selects the last image from a final-only Gallery batch', () => {
    let state = primeGenerate(undefined, { batchCount: 3 });

    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      images: ['gallery-image-1.png', 'gallery-image-2.png', 'gallery-image-3.png'].map((name) =>
        createImage(name, queueItem.id)
      ),
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe(
      'image:gallery-image-3.png'
    );
  });

  it('pauses live-follow when the user selects a saved Gallery image', () => {
    let state = createInitialWorkbenchState();

    expect(getActiveProject(state).settings.showProgressImagesInViewer).toBe(true);

    state = workbenchReducer(state, {
      item: createGalleryImageItem('selected.png'),
      type: 'selectGalleryItem',
    });

    expect(getActiveProject(state).settings.showProgressImagesInViewer).toBe(false);
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe('image:selected.png');
  });

  it('clears the Gallery selection while keeping the compare image and navigation stamp', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { item: createGalleryImageItem('selected.png'), type: 'selectGalleryItem' });
    state = workbenchReducer(state, {
      image: createGalleryImageItem('compare.png'),
      type: 'setGalleryCompareImage',
    });
    state = workbenchReducer(state, { type: 'clearGallerySelection' });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.selectedImage).toBeNull();
    expect(values.selectedImageName).toBeNull();
    expect(values.selectedImageNames).toEqual([]);
    expect(values.compareImage).toMatchObject({ name: 'compare.png' });
    expect(values.selectedImageQuery).toBeDefined();
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

  it('selects a result whose generation was submitted after the manual selection', () => {
    vi.useFakeTimers({ now: new Date('2026-06-10T00:00:00.000Z') });

    try {
      let state = primeGenerate();
      state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
      state = submitGenerate(state);
      const earlierItem = getActiveProject(state).queue.items[0];

      vi.setSystemTime(new Date('2026-06-10T00:00:01.000Z'));
      state = workbenchReducer(state, { item: createGalleryImageItem('selected.png'), type: 'selectGalleryItem' });
      expect(getActiveProject(state).settings.showProgressImagesInViewer).toBe(false);

      // Invoking again is the counter-signal: the user wants to see what they just asked for.
      vi.setSystemTime(new Date('2026-06-10T00:00:02.000Z'));
      state = submitGenerate(state);
      const laterItem = getActiveProject(state).queue.items[0];
      expect(laterItem.id).not.toBe(earlierItem.id);
      expect(getActiveProject(state).settings.showProgressImagesInViewer).toBe(true);

      // The batch already running when the user picked stays out of the way…
      state = workbenchReducer(state, {
        images: [createImage('earlier.png', earlierItem.id)],
        projectId: getActiveProject(state).id,
        queueItemId: earlierItem.id,
        type: 'routeQueueItemResults',
      });
      expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe('image:selected.png');

      // …while the one submitted after the pick takes the preview when it lands.
      state = workbenchReducer(state, {
        images: [createImage('later.png', laterItem.id)],
        projectId: getActiveProject(state).id,
        queueItemId: laterItem.id,
        type: 'routeQueueItemResults',
      });
      expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe('image:later.png');
    } finally {
      vi.useRealTimers();
    }
  });

  it('leaves an explicit live-follow opt-out alone when submitting', () => {
    let state = primeGenerate();
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, {
      settings: { showProgressImagesInViewer: false },
      type: 'setActiveProjectSettings',
    });

    state = submitGenerate(state);

    expect(getActiveProject(state).settings.showProgressImagesInViewer).toBe(false);
  });

  it('lifts the selection pause when live-follow is toggled explicitly', () => {
    let state = createInitialWorkbenchState();
    state = workbenchReducer(state, { item: createGalleryImageItem('selected.png'), type: 'selectGalleryItem' });
    expect(typeof getProjectWidgetValues(getActiveProject(state), 'gallery').liveFollowPausedAt).toBe('string');

    state = workbenchReducer(state, {
      settings: { showProgressImagesInViewer: true },
      type: 'setActiveProjectSettings',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').liveFollowPausedAt).toBeUndefined();
  });

  it('stamps an explicit page into the navigation query already on a multi-selection', () => {
    // Host navigation uses the selection's query and page, which may differ from the gallery's current
    // board/search.
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

  it('pauses live-follow for saved Gallery multi-selection and comparison intents', () => {
    const primaryImage = createGalleryImageItem('primary.png');
    const compareImage = createGalleryImageItem('compare.png');
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      itemKeys: ['image:primary.png', 'image:compare.png'],
      primaryItem: primaryImage,
      type: 'setGalleryMultiSelection',
    });

    expect(getActiveProject(state).settings.showProgressImagesInViewer).toBe(false);

    state = workbenchReducer(state, {
      settings: { showProgressImagesInViewer: true },
      type: 'setActiveProjectSettings',
    });
    state = workbenchReducer(state, { image: compareImage, type: 'setGalleryCompareImage' });

    expect(getActiveProject(state).settings.showProgressImagesInViewer).toBe(false);
  });

  it('resets the page and stamps the selection query when the starred-only filter changes', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { page: 4, type: 'setGalleryPage' });
    state = workbenchReducer(state, { starredOnly: true, type: 'setGalleryStarredOnly' });

    let values = getProjectWidgetValues(getActiveProject(state), 'gallery');
    expect(values.starredOnly).toBe(true);
    expect(values.galleryPage).toBe(0);

    state = workbenchReducer(state, { item: createGalleryImageItem('starred.png'), type: 'selectGalleryItem' });
    values = getProjectWidgetValues(getActiveProject(state), 'gallery');
    expect(values.selectedImageQuery).toMatchObject({ starredOnly: true });

    state = workbenchReducer(state, { page: 2, type: 'setGalleryPage' });
    state = workbenchReducer(state, { starredOnly: false, type: 'setGalleryStarredOnly' });
    state = workbenchReducer(state, {
      itemKeys: ['image:a.png', 'image:b.png'],
      primaryItem: createGalleryImageItem('b.png'),
      type: 'setGalleryMultiSelection',
    });
    values = getProjectWidgetValues(getActiveProject(state), 'gallery');
    expect(values.galleryPage).toBe(0);
    expect(values.selectedImageQuery).toMatchObject({ starredOnly: false });
  });

  it('stamps a landing generation against the unfiltered listing', () => {
    // Following a fresh, unstarred result must leave the starred-only listing.
    let state = primeGenerate();
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { starredOnly: true, type: 'setGalleryStarredOnly' });
    state = submitGenerate(state);

    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];
    state = workbenchReducer(state, {
      images: [createImage('completed.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageQuery).toMatchObject({
      searchTerm: '',
      starredOnly: false,
    });
  });

  it('exits a similarity search when the view moves to another board', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'board-a', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { galleryPage: 4, semanticImageQuery: { kind: 'text', query: 'sunset' } },
      widgetId: 'gallery',
    });

    state = workbenchReducer(state, { boardId: 'board-b', type: 'selectGalleryBoard' });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.semanticImageQuery).toBeNull();
    expect(values.selectedBoardId).toBe('board-b');
    expect(values.galleryPage).toBe(0);
  });

  it('keeps a similarity search when the board already shown is picked again', () => {
    // Re-selecting the current board must preserve persisted rankings and avoid autosaving their loss.
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'board-a', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { semanticImageQuery: { kind: 'text', query: 'sunset' } },
      widgetId: 'gallery',
    });

    state = workbenchReducer(state, { boardId: 'board-a', type: 'selectGalleryBoard' });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').semanticImageQuery).toEqual({
      kind: 'text',
      query: 'sunset',
    });
  });

  it('leaves the page stamped on the selection alone when a search is dismissed by a board move', () => {
    // This selection predates search; its board page remains valid and must not be reset with ranking pages.
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'board-a', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, {
      item: createGalleryImageItem('deep.png'),
      selectionPage: 7,
      type: 'selectGalleryItem',
    });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { semanticImageQuery: { kind: 'text', query: 'sunset' } },
      widgetId: 'gallery',
    });

    state = workbenchReducer(state, { boardId: 'board-b', type: 'selectGalleryBoard' });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.semanticImageQuery).toBeNull();
    expect(values.selectedImagePage).toBe(7);
    expect((values.selectedImageQuery as { boardId: string; page: number }).page).toBe(7);
    expect((values.selectedImageQuery as { boardId: string; page: number }).boardId).toBe('board-a');
  });

  it('exits a similarity search when the Images/Assets tab is switched, but not when it is re-clicked', () => {
    // Absent galleryView already means Images, so selecting Images is not a switch.
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { semanticImageQuery: { kind: 'text', query: 'sunset' } },
      widgetId: 'gallery',
    });

    state = workbenchReducer(state, { galleryView: 'images', type: 'setGalleryView' });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').semanticImageQuery).toEqual({
      kind: 'text',
      query: 'sunset',
    });

    state = workbenchReducer(state, { galleryView: 'assets', type: 'setGalleryView' });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.semanticImageQuery).toBeNull();
    expect(values.galleryView).toBe('assets');
  });

  it('exits a similarity search when the board being viewed is deleted', () => {
    // Board deletion bypasses selectGalleryBoard and must independently clear the old ranking.
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'doomed-board', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { semanticImageQuery: { kind: 'text', query: 'sunset' } },
      widgetId: 'gallery',
    });

    state = workbenchReducer(state, {
      outcome: {
        boardId: 'doomed-board',
        deletedBoardImageNames: [],
        deletedBoardVideoNames: [],
        deletedImageNames: [],
        deletedVideoNames: [],
        failedImageNames: [],
        failedVideoNames: [],
      },
      type: 'reconcileDeletedGalleryBoard',
    });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.selectedBoardId).toBe('none');
    expect(values.semanticImageQuery).toBeNull();
  });

  it('stores selected backend board id for gallery submissions', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'backend-board-id', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = primeGenerate(state);
    state = submitGenerate(state);

    const queueItem = getActiveProject(state).queue.items[0];

    expect(queueItem.snapshot.galleryBoardId).toBe('backend-board-id');
  });

  it('sends gallery submissions to the project board until a board is picked', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      boardId: 'project-board',
      projectId: getActiveProject(state).id,
      type: 'setGalleryProjectBoardId',
    });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = primeGenerate(state);
    state = submitGenerate(state);

    expect(getActiveProject(state).queue.items[0]?.snapshot.galleryBoardId).toBe('project-board');
  });

  it('folds a stream of workflow field edits into one undo step and starts another after a pause', () => {
    vi.useFakeTimers({ now: new Date('2026-06-10T00:00:00.000Z') });

    try {
      let state = createInitialWorkbenchState();
      const setValue = (fieldName: string, value: number) =>
        workbenchReducer(state, {
          action: { fieldName, nodeId: 'node-1', type: 'setFieldValue', value },
          type: 'applyProjectGraphAction',
        });
      const fieldValue = (fieldName: string) => {
        const node = getActiveProject(state).projectGraph.nodes[0];

        return node?.type === 'invocation' ? node.data.inputs[fieldName]?.value : undefined;
      };
      const past = () => getActiveProject(state).undoRedo.past;

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
      const stepsAfterAdd = past().length;

      // Keystrokes a second apart keep one edit going: the window slides with each keystroke.
      state = setValue('a', 1);
      vi.setSystemTime(new Date('2026-06-10T00:00:01.000Z'));
      state = setValue('a', 12);
      vi.setSystemTime(new Date('2026-06-10T00:00:02.000Z'));
      state = setValue('a', 123);

      expect(past()).toHaveLength(stepsAfterAdd + 1);
      expect(past().at(-1)?.label).toBe('Edit workflow field value');

      // A pause, or another field, opens a new step.
      vi.setSystemTime(new Date('2026-06-10T00:00:05.000Z'));
      state = setValue('a', 3);
      state = setValue('b', 7);

      expect(past()).toHaveLength(stepsAfterAdd + 3);

      state = workbenchReducer(state, { type: 'undoProjectChange' });
      expect(fieldValue('b')).toBeUndefined();
      expect(fieldValue('a')).toBe(3);

      state = workbenchReducer(state, { type: 'undoProjectChange' });
      expect(fieldValue('a')).toBe(123);

      state = workbenchReducer(state, { type: 'undoProjectChange' });
      expect(fieldValue('a')).toBeUndefined();

      // Redo brings the whole burst back at once, and a fresh edit after it starts its own step.
      state = workbenchReducer(state, { type: 'redoProjectChange' });
      expect(fieldValue('a')).toBe(123);

      state = setValue('a', 4);
      expect(past()).toHaveLength(stepsAfterAdd + 2);
      expect(getActiveProject(state).undoRedo.future).toHaveLength(0);

      // An undo inside the window ends the burst: the step the user undid back to stays its own step.
      state = setValue('b', 8);
      state = workbenchReducer(state, { type: 'undoProjectChange' });
      state = setValue('a', 5);

      expect(past()).toHaveLength(stepsAfterAdd + 3);
      state = workbenchReducer(state, { type: 'undoProjectChange' });
      expect(fieldValue('a')).toBe(4);

      // Node moves are not history: dragging a node around leaves the steps alone.
      state = workbenchReducer(state, {
        action: { nodeId: 'node-1', position: { x: 10, y: 10 }, type: 'setNodePosition' },
        type: 'applyProjectGraphAction',
      });
      expect(past()).toHaveLength(stepsAfterAdd + 2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('stores full selected gallery image data for Preview widget', () => {
    let state = createInitialWorkbenchState();
    const image = createGalleryImageItem('backend-selected.png');

    state = workbenchReducer(state, { item: image, type: 'selectGalleryItem' });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe(
      'image:backend-selected.png'
    );
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImage).toEqual(image);
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

    it('entering with an empty field ranks nothing, and re-entering changes nothing', () => {
      let state = createInitialWorkbenchState();

      state = workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' });

      expect(galleryValues(state)).toMatchObject({ semanticImageQuery: null, semanticSearchText: '' });
      expect(workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' })).toBe(state);

      const initial = createInitialWorkbenchState();

      expect(workbenchReducer(initial, { enabled: false, type: 'setGallerySemanticSearchMode' })).toBe(initial);
    });

    it('commits only the text the field still holds', () => {
      // Commits arrive on a timer. One scheduled against text that has since
      // been edited, or a field that has since been cleared, must not apply.
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

      // Blank text is no ranking; a whitespace edit is not a new ranking and
      // so leaves the page where the user has paged to.
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

    it('ignores text edits and late commits once semantic mode is left', () => {
      let state = createInitialWorkbenchState();

      expect(workbenchReducer(state, { text: 'sun', type: 'setGallerySemanticSearchText' })).toBe(state);

      state = workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' });
      state = workbenchReducer(state, { text: 'sun', type: 'setGallerySemanticSearchText' });
      state = workbenchReducer(state, { enabled: false, type: 'setGallerySemanticSearchMode' });

      expect(workbenchReducer(state, { text: 'sun', type: 'commitGallerySemanticSearch' })).toBe(state);
      expect(galleryValues(state)).toMatchObject({
        searchTerm: 'sun',
        semanticImageQuery: null,
        semanticSearchText: null,
      });
    });

    it('clears text, ranking and mode together; clearing an empty field changes nothing', () => {
      let state = createInitialWorkbenchState();

      state = workbenchReducer(state, { searchTerm: 'sunset', type: 'setGallerySearchTerm' });
      state = workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' });
      state = workbenchReducer(state, { page: 2, type: 'setGalleryPage' });
      state = workbenchReducer(state, { type: 'clearGallerySearch' });

      expect(galleryValues(state)).toMatchObject({
        galleryPage: 0,
        searchTerm: '',
        semanticImageQuery: null,
        semanticSearchText: null,
      });
      expect(workbenchReducer(state, { type: 'clearGallerySearch' })).toBe(state);

      state = workbenchReducer(state, { searchTerm: 'plain', type: 'setGallerySearchTerm' });
      state = workbenchReducer(state, { type: 'clearGallerySearch' });

      expect(galleryValues(state).searchTerm).toBe('');
    });

    it('leaves semantic mode with the ranking on a board move or a tab switch, not on a re-click', () => {
      let state = createInitialWorkbenchState();

      state = workbenchReducer(state, { boardId: 'board-a', type: 'selectGalleryBoard' });
      state = workbenchReducer(state, { searchTerm: 'sunset', type: 'setGallerySearchTerm' });
      state = workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' });
      state = workbenchReducer(state, { boardId: 'board-a', type: 'selectGalleryBoard' });

      expect(galleryValues(state)).toMatchObject({
        semanticImageQuery: { kind: 'text', query: 'sunset' },
        semanticSearchText: 'sunset',
      });

      state = workbenchReducer(state, { boardId: 'board-b', type: 'selectGalleryBoard' });

      expect(galleryValues(state)).toMatchObject({
        searchTerm: '',
        semanticImageQuery: null,
        semanticSearchText: null,
      });

      state = workbenchReducer(state, { enabled: true, type: 'setGallerySemanticSearchMode' });
      state = workbenchReducer(state, { text: 'beach', type: 'setGallerySemanticSearchText' });
      state = workbenchReducer(state, { galleryView: 'images', type: 'setGalleryView' });

      expect(galleryValues(state).semanticSearchText).toBe('beach');

      state = workbenchReducer(state, { galleryView: 'assets', type: 'setGalleryView' });

      expect(galleryValues(state)).toMatchObject({ semanticImageQuery: null, semanticSearchText: null });
    });
  });
});

describe('workbench account and project settings', () => {
  it('starts with the default layout preset and legacy-matching project settings', () => {
    const state = createInitialWorkbenchState();

    expect(state.account).toEqual({ activeLayoutPresetId: 'compose' });
    expect(getActiveProject(state).settings).toEqual(DEFAULT_PROJECT_SETTINGS);
  });

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
    });
  });

  it('heals hydrated accounts that are missing a layout preset', () => {
    const initial = createInitialWorkbenchState();
    const legacy = { ...initial, account: {} } as unknown as WorkbenchState;

    const state = workbenchReducer(initial, { state: legacy, type: 'hydrateWorkbench' });

    expect(state.account.activeLayoutPresetId).toBe('compose');
  });

  it('normalizes persisted preset routes and tolerates legacy custom presets without one', () => {
    let seeded = createInitialWorkbenchState();
    seeded = workbenchReducer(seeded, {
      label: 'Route fixture',
      presetId: 'custom-route-fixture',
      type: 'addLayoutPreset',
    });
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
      settings: { antialiasProgressImages: true, useCpuNoise: false },
      type: 'setActiveProjectSettings',
    });

    expect(getActiveProject(state).settings).toEqual({
      antialiasProgressImages: true,
      showProgressImagesInViewer: true,
      useCpuNoise: false,
    });
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

  it('writes canonical mixed selections with ordered qualified keys and clears comparison for video', () => {
    const image = createGalleryImageItem('shared');
    const video = createGalleryVideoItem('shared');
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { image, type: 'setGalleryCompareImage' });
    state = workbenchReducer(state, { item: video, type: 'selectGalleryItem' });

    let values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.selectedImage).toBe(video);
    expect(values.selectedImageName).toBe('video:shared');
    expect(values.selectedImageNames).toEqual(['video:shared']);
    expect(values.compareImage).toBeNull();

    state = workbenchReducer(state, { item: image, nextPrimaryItem: null, type: 'toggleGalleryItemInSelection' });
    values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.selectedImage).toBe(image);
    expect(values.selectedImageName).toBe('image:shared');
    expect(values.selectedImageNames).toEqual(['video:shared', 'image:shared']);

    state = workbenchReducer(state, {
      itemKeys: ['image:shared', 'video:shared'],
      primaryItem: video,
      type: 'setGalleryMultiSelection',
    });
    values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.selectedImage).toBe(video);
    expect(values.selectedImageName).toBe('video:shared');
    expect(values.selectedImageNames).toEqual(['image:shared', 'video:shared']);
    expect(values.compareImage).toBeNull();
  });

  it('atomically promotes the remaining video when the image primary is toggled off', () => {
    const image = createGalleryImageItem('shared');
    const video = createGalleryVideoItem('shared');
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      itemKeys: ['video:shared', 'image:shared'],
      primaryItem: image,
      type: 'setGalleryMultiSelection',
    });
    state = workbenchReducer(state, { image, type: 'setGalleryCompareImage' });
    state = workbenchReducer(state, {
      item: image,
      nextPrimaryItem: video,
      type: 'toggleGalleryItemInSelection',
    });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery')).toMatchObject({
      compareImage: null,
      selectedImage: video,
      selectedImageName: 'video:shared',
      selectedImageNames: ['video:shared'],
    });
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

  it('moves image and video board relationships to Uncategorized when board contents are retained', () => {
    const boardId = 'deleted-board';
    const image = createGalleryImageItem('retained-image', boardId);
    const video = createGalleryVideoItem('retained-video', boardId);
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: {
        recentImages: [galleryItemToRecentImage(image)],
        selectedBoardId: boardId,
        selectedImage: video,
        selectedImageName: `video:${video.name}`,
        selectedImageNames: [`image:${image.name}`, `video:${video.name}`],
      },
      widgetId: 'gallery',
    });
    state = workbenchReducer(state, {
      outcome: {
        boardId,
        deletedBoardImageNames: [image.name],
        deletedBoardVideoNames: [video.name],
        deletedImageNames: [],
        deletedVideoNames: [],
        failedImageNames: [],
        failedVideoNames: [],
      },
      type: 'reconcileDeletedGalleryBoard',
    });

    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values).toMatchObject({
      recentImages: [{ ...galleryItemToRecentImage(image), boardId: 'none' }],
      selectedBoardId: 'none',
      selectedImage: { ...video, boardId: 'none' },
      selectedImageName: `video:${video.name}`,
      selectedImageNames: [`image:${image.name}`, `video:${video.name}`],
    });
  });

  it('reconciles moved and starred metadata across every project recent-image overlay', () => {
    let state = createInitialWorkbenchState();
    const image = createImage('patched.png', 'backend-gallery');

    state = workbenchReducer(state, { type: 'createProject' });
    for (const project of state.projects) {
      state = workbenchReducer(state, {
        projectId: project.id,
        type: 'patchWidgetValues',
        values: { compareImage: image, recentImages: [image], selectedImage: image },
        widgetId: 'gallery',
      });
    }
    state = workbenchReducer(state, {
      changes: { boardId: 'board-2', starred: true },
      itemKeys: [`image:${image.imageName}`],
      type: 'patchGalleryItems',
    });

    for (const project of state.projects) {
      const values = getProjectWidgetValues(project, 'gallery');
      const patchedImage = { ...image, boardId: 'board-2', starred: true };

      expect(values.recentImages).toEqual([patchedImage]);
      expect(values.selectedImage).toEqual(legacyGeneratedImageToGalleryItem(patchedImage));
      expect(values.compareImage).toEqual(legacyGeneratedImageToGalleryItem(patchedImage));
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

  it('preserves live backend connection state when persistence hydrates late', () => {
    const initial = createInitialWorkbenchState();
    const connected = workbenchReducer(initial, { status: 'connected', type: 'setBackendConnectionStatus' });
    const persisted = {
      ...initial,
      backendConnection: { status: 'connecting' },
    } as WorkbenchState;

    const state = workbenchReducer(connected, { state: persisted, type: 'hydrateWorkbench' });

    expect(state.backendConnection.status).toBe('connected');
    expect(state.backendConnection.lastConnectedAt).toBe(connected.backendConnection.lastConnectedAt);
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
  it('starts at Layer 1 with no existing layers', () => {
    expect(nextLayerName([])).toBe('Layer 1');
  });

  it('picks the next free number above a contiguous run', () => {
    expect(nextLayerName(['Layer 1', 'Layer 2'])).toBe('Layer 3');
  });

  it('fills the lowest gap so names do not collide after a deletion', () => {
    // Deleting "Layer 2" from [Layer 1, Layer 2, Layer 3] must not re-mint the
    // count-derived "Layer 3" (which would collide); the lowest free slot is 2.
    expect(nextLayerName(['Layer 1', 'Layer 3'])).toBe('Layer 2');
  });

  it('ignores custom names and non-matching patterns', () => {
    expect(nextLayerName(['Backdrop', 'Layer 10 copy', 'Layer 1'])).toBe('Layer 2');
    expect(nextLayerName(['Sketch'])).toBe('Layer 1');
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

  it('seeds a new project canvas with a single empty inpaint mask, selected', () => {
    const state = createInitialWorkbenchState();
    const { document } = getActiveProject(state).canvas;

    expect(getDocumentLeaves(document)).toHaveLength(1);
    const mask = getDocumentLeaves(document)[0];
    expect(mask?.type).toBe('inpaint_mask');
    // Empty: no bitmap (no strokes) — so it never flips generation-mode detection.
    expect(mask && 'mask' in mask ? mask.mask.bitmap : 'missing').toBeNull();
    expect(document.selectedLayerId).toBe(mask?.id);
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

    // Nothing remains below 'c', so selection falls back to 'a' above it.
    expect(getCanvas(state).document.selectedLayerId).toBe('a');

    state = workbenchReducer(state, { ids: ['a'], type: 'removeCanvasLayers' });

    expect(getLayerIds(state)).toEqual([]);
    expect(getCanvas(state).document.selectedLayerId).toBeNull();
  });

  it('hides overlay layers without disabling them, so generation is untouched', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createInpaintMaskLayer('m'), createRasterLayer('r')]);

    state = workbenchReducer(state, {
      type: 'setCanvasLayersHidden',
      updates: [
        { id: 'm', isHidden: true },
        { id: 'r', isHidden: true },
      ],
    });

    const layers = getDocumentLeaves(getCanvas(state).document);
    const mask = layers.find((layer) => layer.id === 'm')!;
    const raster = layers.find((layer) => layer.id === 'r')!;

    expect(mask).toMatchObject({ isEnabled: true, isHidden: true });
    // Raster layers have no display axis; the update is skipped rather than
    // silently giving them a meaningless field.
    expect(raster).not.toHaveProperty('isHidden');
    expect(raster.isEnabled).toBe(true);
  });

  it('leaves the document untouched when a hide update changes nothing', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [createInpaintMaskLayer('m')]);
    const before = getCanvas(state).document;

    const next = workbenchReducer(state, {
      type: 'setCanvasLayersHidden',
      updates: [{ id: 'm', isHidden: false }],
    });

    expect(getCanvas(next).document).toBe(before);
  });

  it('sets many layers visibility in one bulk action, preserving unlisted layers by identity', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);
    const before = getDocumentLeaves(getCanvas(state).document);

    // Hide 'a' and 'c' in one dispatch; 'b' is unlisted and must keep its object identity.
    state = workbenchReducer(state, {
      type: 'setCanvasLayersEnabled',
      updates: [
        { id: 'a', isEnabled: false },
        { id: 'c', isEnabled: false },
      ],
    });
    const after = getDocumentLeaves(getCanvas(state).document);

    expect(after.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['a', false],
      ['b', true],
      ['c', false],
    ]);
    // 'b' unchanged ⇒ same reference; 'a'/'c' replaced.
    expect(after[1]).toBe(before[1]);
    expect(after[0]).not.toBe(before[0]);
  });

  it('applies stack-wide lock updates without replacing unrelated layers', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);
    const before = getDocumentLeaves(getCanvas(state).document);

    state = workbenchReducer(state, {
      enabledUpdates: [],
      lockedUpdates: [
        { id: 'a', isLocked: true },
        { id: 'c', isLocked: true },
      ],
      selectedLayerId: 'a',
      type: 'applyCanvasLayerStackMutation',
    });

    const after = getDocumentLeaves(getCanvas(state).document);
    expect(after.map((layer) => [layer.id, layer.isLocked])).toEqual([
      ['a', true],
      ['b', false],
      ['c', true],
    ]);
    expect(after[1]).toBe(before[1]);
  });

  it('preserves the current selection when applying or undoing lock-only stack mutations', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);
    state = workbenchReducer(state, { id: 'a', type: 'setCanvasSelectedLayer' });
    state = workbenchReducer(state, {
      enabledUpdates: [],
      lockedUpdates: [{ id: 'a', isLocked: true }],
      type: 'applyCanvasLayerStackMutation',
    });
    state = workbenchReducer(state, { id: 'c', type: 'setCanvasSelectedLayer' });
    state = workbenchReducer(state, {
      enabledUpdates: [],
      lockedUpdates: [{ id: 'a', isLocked: false }],
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getCanvas(state).document.selectedLayerId).toBe('c');
    expect(getDocumentLeaves(getCanvas(state).document)[0]?.isLocked).toBe(false);
  });

  it('repairs an omitted stack-mutation selection to the nearest same-stack neighbour', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createControlLayer('c'),
      createRasterLayer('d'),
    ]);
    state = workbenchReducer(state, { id: 'd', type: 'setCanvasSelectedLayer' });

    state = workbenchReducer(state, {
      enabledUpdates: [],
      removeIds: ['d'],
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getLayerIds(state)).toEqual(['c', 'a', 'b']);
    expect(getCanvas(state).document.selectedLayerId).toBe('b');
  });

  it('leaves an empty selection empty when a stack mutation omits the selection', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    state = workbenchReducer(state, { id: null, type: 'setCanvasSelectedLayer' });
    expect(getCanvas(state).document.selectedLayerId).toBeNull();

    state = workbenchReducer(state, {
      enabledUpdates: [],
      lockedUpdates: [{ id: 'a', isLocked: true }],
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getDocumentLeaves(getCanvas(state).document)[0]?.isLocked).toBe(true);
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

  it('sets multiple layer positions atomically and rejects invalid coordinates', () => {
    const initial = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    const moved = workbenchReducer(initial, {
      type: 'setCanvasLayerPositions',
      updates: [
        { id: 'a', x: 3, y: -2 },
        { id: 'b', x: 13, y: 18 },
      ],
    });
    expect(getDocumentLeaves(getCanvas(moved).document).map((layer) => layer.transform)).toMatchObject([
      { x: 3, y: -2 },
      { x: 13, y: 18 },
    ]);
    expect(
      workbenchReducer(initial, {
        type: 'setCanvasLayerPositions',
        updates: [{ id: 'a', x: Number.NaN, y: 0 }],
      })
    ).toBe(initial);
  });

  it('returns the same document when a bulk visibility action changes nothing', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    const before = getCanvas(state).document;

    const next = workbenchReducer(state, {
      type: 'setCanvasLayersEnabled',
      updates: [{ id: 'a', isEnabled: true }],
    });

    // 'a' is already enabled ⇒ no change ⇒ document identity preserved (no selector churn).
    expect(getCanvas(next).document).toBe(before);
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

  it('inserts a batch in order and selects the exact requested layer', () => {
    const existing = createRasterLayer('existing');
    const layerA = createRasterLayer('a');
    const layerB = createRasterLayer('b');
    const initial = withCanvasLayers(createInitialWorkbenchState(), [existing]);

    const next = workbenchReducer(initial, {
      add: [{ anchor: stackTopAnchor(initial.activeProjectId), nodes: [layerA, layerB] }],
      enabledUpdates: [],
      selectedLayerId: layerB.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getDocumentLeaves(getCanvas(next).document)).toEqual([layerA, layerB, existing]);
    expect(getCanvas(next).document.selectedLayerId).toBe(layerB.id);
  });

  it('targets a non-active project without changing the active project', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('first-existing')]);
    const firstProjectId = state.activeProjectId;
    state = workbenchReducer(state, { type: 'createProject' });
    const secondProjectId = state.activeProjectId;
    const secondProjectBefore = getProject(state, secondProjectId);
    const layerA = createRasterLayer('a');
    const layerB = createRasterLayer('b');

    const next = workbenchReducer(state, {
      add: [{ anchor: stackTopAnchor(firstProjectId), nodes: [layerA, layerB] }],
      enabledUpdates: [],
      projectId: firstProjectId,
      selectedLayerId: layerB.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getDocumentLeaves(getProject(next, firstProjectId).canvas.document).slice(0, 2)).toEqual([layerA, layerB]);
    expect(getProject(next, firstProjectId).canvas.document.selectedLayerId).toBe(layerB.id);
    expect(getProject(next, secondProjectId)).toBe(secondProjectBefore);
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

  it('preserves the layers array for a selection-only layer stack mutation', () => {
    const initial = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    const before = getDocumentLeaves(getCanvas(initial).document);

    const next = workbenchReducer(initial, {
      enabledUpdates: [],
      selectedLayerId: 'b',
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getDocumentLeaves(getCanvas(next).document)).toBe(before);
    expect(getCanvas(next).document.selectedLayerId).toBe('b');
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
    // Untouched layer objects are reused, not cloned.
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

  it('reorders several stacks atomically and refuses a duplicated stack or an unchanged order', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createControlLayer('x'),
      createRasterLayer('b'),
      createControlLayer('y'),
    ]);

    const reordered = workbenchReducer(state, {
      orders: [
        { orderedIds: ['b', 'a'], parentId: null, stack: 'raster' },
        { orderedIds: ['y', 'x'], parentId: null, stack: 'control' },
      ],
      type: 'reorderCanvasSiblings',
    });
    expect(getLayerIds(reordered)).toEqual(['y', 'x', 'b', 'a']);

    const refused = workbenchReducer(state, {
      orders: [
        { orderedIds: ['b', 'a'], parentId: null, stack: 'raster' },
        { orderedIds: ['a', 'b'], parentId: null, stack: 'raster' },
      ],
      type: 'reorderCanvasSiblings',
    });
    expect(refused).toBe(state);

    const partiallyInvalid = workbenchReducer(state, {
      orders: [
        { orderedIds: ['b', 'a'], parentId: null, stack: 'raster' },
        { orderedIds: ['y'], parentId: null, stack: 'control' },
      ],
      type: 'reorderCanvasSiblings',
    });
    expect(partiallyInvalid).toBe(state);

    const unchanged = workbenchReducer(state, {
      orders: [{ orderedIds: ['a', 'b'], parentId: null, stack: 'raster' }],
      type: 'reorderCanvasSiblings',
    });
    expect(unchanged).toBe(state);
  });

  it('updates base props with a field-wise transform merge and leaves other layers untouched', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    const originalLayerB = getDocumentLeaves(getCanvas(state).document).find((layer) => layer.id === 'b');

    const updated = workbenchReducer(state, {
      id: 'a',
      patch: { name: 'Renamed', opacity: 0.5, transform: { x: 12 } },
      type: 'updateCanvasLayer',
    });

    const layerA = getDocumentLeaves(getCanvas(updated).document).find((layer) => layer.id === 'a');

    expect(layerA?.name).toBe('Renamed');
    expect(layerA?.opacity).toBe(0.5);
    expect(layerA?.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 0 });
    expect(getDocumentLeaves(getCanvas(updated).document).find((layer) => layer.id === 'b')).toBe(originalLayerB);
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

  it('swaps a raster/control layer source but ignores mask-only layer types', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const newSource = { image: { height: 10, imageName: 'swapped.png', width: 10 }, type: 'image' } as const;

    state = workbenchReducer(state, { id: 'a', source: newSource, type: 'updateCanvasLayerSource' });

    expect(getRasterLayerImageName(getDocumentLeaves(getCanvas(state).document)[0])).toBe('swapped.png');
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

    // A raster-shaped config against a control layer is a no-op.
    const unchanged = workbenchReducer(state, {
      config: { isTransparencyLocked: true, layerType: 'raster' },
      id: 'ctrl',
      type: 'updateCanvasLayerConfig',
    });

    expect(getDocumentLeaves(getCanvas(unchanged).document)[0]).toBe(controlLayer);
  });

  it('persists filter settings on raster layers', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('raster')]);

    state = workbenchReducer(state, {
      config: { filter: { settings: { radius: 4 }, type: 'content_shuffle' }, layerType: 'raster' },
      id: 'raster',
      type: 'updateCanvasLayerConfig',
    });

    const raster = getDocumentLeaves(getCanvas(state).document)[0];
    expect(raster?.type).toBe('raster');
    if (raster?.type === 'raster') {
      expect(raster.filter).toEqual({ settings: { radius: 4 }, type: 'content_shuffle' });
    }
  });

  it('applies an inpaint-mask config patch: mask bitmap + offset, fill, noise, denoise-limit', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createInpaintMaskLayer('m')]);

    state = workbenchReducer(state, {
      config: {
        denoise: { isEnabled: true, limit: 0.8 },
        layerType: 'inpaint_mask',
        mask: { bitmap: { contentHash: 'h', height: 20, imageName: 'mask.png', width: 30 }, offset: { x: 4, y: 5 } },
        noise: { isEnabled: true, level: 0.25 },
      },
      id: 'm',
      type: 'updateCanvasLayerConfig',
    });

    const layer = getDocumentLeaves(getCanvas(state).document)[0];
    expect(layer?.type).toBe('inpaint_mask');
    if (layer?.type === 'inpaint_mask') {
      // Bitmap + content offset persisted; the fill is preserved (merged, not replaced).
      expect(layer.mask.bitmap).toMatchObject({ imageName: 'mask.png', width: 30, height: 20 });
      expect(layer.mask.offset).toEqual({ x: 4, y: 5 });
      expect(layer.mask.fill).toEqual({ color: '#e07575', style: 'diagonal' });
      expect(layer.noise).toEqual({ isEnabled: true, level: 0.25 });
      expect(layer.denoise).toEqual({ isEnabled: true, limit: 0.8 });
    }

    // A fill-only patch replaces the fill while keeping the bitmap.
    const recolored = workbenchReducer(state, {
      config: { layerType: 'inpaint_mask', mask: { fill: { color: '#00ff00', style: 'grid' } } },
      id: 'm',
      type: 'updateCanvasLayerConfig',
    });
    const after = getDocumentLeaves(getCanvas(recolored).document)[0];
    if (after?.type === 'inpaint_mask') {
      expect(after.mask.fill).toEqual({ color: '#00ff00', style: 'grid' });
      expect(after.mask.bitmap).toMatchObject({ imageName: 'mask.png' });
    }
  });

  it('removes optional config fields when a patch explicitly clears them', () => {
    const configuredMask: CanvasInpaintMaskLayerContract = {
      blendMode: 'normal',
      denoise: { isEnabled: true, limit: 0.8 },
      id: 'm',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
      name: 'm',
      noise: { isEnabled: true, level: 0.25 },
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    };
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      configuredMask,
      { ...createControlLayer('ctrl'), filter: { settings: {}, type: 'canny' } },
    ]);

    state = workbenchReducer(state, {
      config: { denoise: null, layerType: 'inpaint_mask', noise: null },
      id: 'm',
      type: 'updateCanvasLayerConfig',
    });
    state = workbenchReducer(state, {
      config: { filter: undefined, layerType: 'control' },
      id: 'ctrl',
      type: 'updateCanvasLayerConfig',
    });

    const mask = getDocumentLeaves(getCanvas(state).document)[0];
    const control = getDocumentLeaves(getCanvas(state).document)[1];

    expect(mask?.type).toBe('inpaint_mask');
    if (mask?.type === 'inpaint_mask') {
      expect(mask.noise).toBeUndefined();
      expect(mask.denoise).toBeUndefined();
      expect(Object.hasOwn(mask, 'noise')).toBe(false);
    }
    expect(control?.type).toBe('control');
    if (control?.type === 'control') {
      expect(control.filter).toBeUndefined();
    }
  });

  it('converts a layer into its new stack, preserving its id', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    const converted = createControlLayer('ignored-id');

    state = workbenchReducer(state, { id: 'b', layer: converted, targetType: 'control', type: 'convertCanvasLayer' });

    expect(getLayerIds(state)).toEqual(['b', 'a']);
    expect(getCanvas(state).document.stacks.raster.map((node) => node.id)).toEqual(['a']);
    expect(getDocumentLeaves(getCanvas(state).document)[0]?.type).toBe('control');
    expect(getDocumentLeaves(getCanvas(state).document)[0]?.id).toBe('b');
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

  it('clamps and rounds the bbox, and validates the selected layer id', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);

    state = workbenchReducer(state, { bbox: { height: -4, width: 12.6, x: 3.2, y: 9.8 }, type: 'setCanvasBbox' });

    expect(getCanvas(state).document.bbox).toEqual({ height: 1, width: 13, x: 3, y: 10 });

    // 'a' was selected on insert; selecting a non-existent id is ignored, not applied.
    expect(getCanvas(state).document.selectedLayerId).toBe('a');
    state = workbenchReducer(state, { id: 'missing', type: 'setCanvasSelectedLayer' });
    expect(getCanvas(state).document.selectedLayerId).toBe('a');

    state = workbenchReducer(state, { id: null, type: 'setCanvasSelectedLayer' });
    expect(getCanvas(state).document.selectedLayerId).toBeNull();
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

  it('canonicalizes whole-pixel geometry across document replacement and snapshots', () => {
    let state = createInitialWorkbenchState();
    const fractional = {
      ...createEmptyCanvasDocument(),
      bbox: { height: 99.6, width: 100.4, x: 1.2, y: -2.6 },
      height: 511.6,
      width: 512.4,
    };
    const expected = { bbox: { height: 100, width: 100, x: 1, y: -3 }, height: 512, width: 512 };

    state = workbenchReducer(state, { document: fractional, type: 'replaceCanvasDocument' });
    expect(getCanvas(state).document).toMatchObject(expected);

    Object.assign(getCanvas(state).document, fractional);
    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'Fractional', type: 'saveCanvasSnapshot' });
    expect(getCanvas(state).snapshots[0]?.document).toMatchObject(expected);

    Object.assign(getCanvas(state).snapshots[0]!.document, fractional);
    state = workbenchReducer(state, { snapshotId: 'snap-1', type: 'restoreCanvasSnapshot' });
    expect(getCanvas(state).document).toMatchObject(expected);
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
      selectedLayerId: 'ghost', // names no layer in the incoming document
    };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getCanvas(state).document.selectedLayerId).toBe('top');
  });

  it('nulls a dangling selectedLayerId when the replacement document has no layers', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = { ...createEmptyCanvasDocument(), stacks: stacksFrom([]), selectedLayerId: 'ghost' };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getCanvas(state).document.selectedLayerId).toBeNull();
  });

  it('preserves a valid selectedLayerId on replaceCanvasDocument', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = {
      ...createEmptyCanvasDocument(),
      stacks: stacksFrom([createRasterLayer('top'), createRasterLayer('bottom')]),
      selectedLayerId: 'bottom',
    };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getCanvas(state).document.selectedLayerId).toBe('bottom');
  });

  it('repairs a dangling selectedLayerId on restoreCanvasSnapshot (defensive against a corrupt snapshot)', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'First', type: 'saveCanvasSnapshot' });

    // Corrupt the stored snapshot so its selectedLayerId names no present layer.
    const snapshot = getCanvas(state).snapshots.find((entry) => entry.id === 'snap-1');
    expect(snapshot).toBeDefined();
    snapshot!.document.selectedLayerId = 'ghost';

    state = workbenchReducer(state, { snapshotId: 'snap-1', type: 'restoreCanvasSnapshot' });

    const restored = getCanvas(state).document;
    expect(restored.selectedLayerId).not.toBe('ghost');
    expect(restored.selectedLayerId).toBe(getDocumentLeaves(restored)[0]?.id);
  });

  it('normalizes a control adapter when restoring a saved canvas snapshot', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createControlLayer('ctrl')]);
    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'First', type: 'saveCanvasSnapshot' });
    const snapshotLayer = getDocumentLeaves(getCanvas(state).snapshots[0]!.document)[0];
    if (snapshotLayer?.type !== 'control') {
      throw new Error('expected a control layer');
    }
    snapshotLayer.adapter = {
      beginEndStepPct: [Number.NaN, 2],
      controlMode: null,
      kind: 'z_image_control',
      model: null,
      weight: Number.POSITIVE_INFINITY,
    };

    state = workbenchReducer(state, { snapshotId: 'snap-1', type: 'restoreCanvasSnapshot' });

    const restored = getDocumentLeaves(getCanvas(state).document)[0];
    expect(restored?.type === 'control' ? restored.adapter : null).toEqual({
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'z_image_control',
      model: null,
      weight: 0.75,
    });
  });

  it('saves, restores, and deletes canvas snapshots', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);

    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'First', type: 'saveCanvasSnapshot' });

    expect(getCanvas(state).snapshots).toHaveLength(1);
    expect(getDocumentLeaves(getCanvas(state).snapshots[0]?.document).map((layer) => layer.id)).toEqual(['a']);

    // Mutate the live document, then restore the snapshot back over it.
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

  it('bumps documentRevision on wholesale swaps (restore/replace) but not on ordinary edits', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const initialRevision = getCanvas(state).documentRevision;

    // Ordinary incremental edits never bump the revision.
    state = workbenchReducer(state, { id: 'a', patch: { opacity: 0.3 }, type: 'updateCanvasLayer' });
    state = workbenchReducer(state, {
      anchor: stackTopAnchor(state.activeProjectId),
      layer: createRasterLayer('b'),
      type: 'addCanvasLayer',
    });
    state = workbenchReducer(state, { bbox: { height: 32, width: 32, x: 0, y: 0 }, type: 'setCanvasBbox' });
    state = workbenchReducer(state, { height: 256, type: 'resizeCanvasDocument', width: 256 });
    expect(getCanvas(state).documentRevision).toBe(initialRevision);

    // replaceCanvasDocument is a wholesale swap: bump.
    state = workbenchReducer(state, { document: createEmptyCanvasDocument(), type: 'replaceCanvasDocument' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 1);

    // Snapshot restore is a wholesale swap even when layer ids and dimensions match.
    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'First', type: 'saveCanvasSnapshot' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 1);
    state = workbenchReducer(state, { snapshotId: 'snap-1', type: 'restoreCanvasSnapshot' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 2);

    // Restoring a nonexistent snapshot is a no-op: no bump.
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

  it("'off' keeps the selected candidate when a workflow result is inserted before it", () => {
    let state = submitGenerate(primeGenerate());

    state = stageResults(state, ['generated.png']);
    state = workbenchReducer(state, {
      candidate: createStagingCandidate('workflow.png', 'layer-workflow:result', {
        height: 64,
        opacity: 1,
        width: 64,
        x: 0,
        y: 0,
      }),
      projectId: state.activeProjectId,
      type: 'appendCanvasStagingCandidate',
    });

    const project = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(project.canvas, project.queue.items)[
      project.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ candidate: { imageName: 'generated.png' }, kind: 'candidate' });
  });

  it("'off' keeps the selected logical slot when its placeholder becomes a partial result", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, { mode: 'progress', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, { mode: 'off', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [createImage('candidate-1.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });

    const updatedProject = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(updatedProject.canvas, updatedProject.queue.items)[
      updatedProject.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ itemIndex: 1, kind: 'candidate' });
  });

  it("'off' keeps a later placeholder selected when an earlier partial result arrives", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [createImage('candidate-1.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });

    const updatedProject = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(updatedProject.canvas, updatedProject.queue.items)[
      updatedProject.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ itemIndex: 2, kind: 'placeholder' });
  });

  it("'off' keeps a later placeholder selected when an earlier partial has no visible images", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });

    const updatedProject = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(updatedProject.canvas, updatedProject.queue.items)[
      updatedProject.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ itemIndex: 2, kind: 'placeholder' });
  });

  it("'off' keeps a candidate selected when an earlier final result has no visible images", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 1 }));
    const firstProject = getActiveProject(state);
    const firstQueueItem = firstProject.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11],
      projectId: firstProject.id,
      queueItemId: firstQueueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = submitGenerate(primeGenerate(state, { batchCount: 2, positivePrompt: 'second prompt' }));

    const secondProject = getActiveProject(state);
    const secondQueueItem = secondProject.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-1.png', secondQueueItem.id), createImage('candidate-2.png', secondQueueItem.id)],
      projectId: secondProject.id,
      queueItemId: secondQueueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, {
      images: [],
      projectId: firstProject.id,
      queueItemId: firstQueueItem.id,
      type: 'routeQueueItemResults',
    });

    const updatedProject = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(updatedProject.canvas, updatedProject.queue.items)[
      updatedProject.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ candidate: { imageName: 'candidate-1.png' }, kind: 'candidate' });
  });

  it("'off' keeps a candidate selected when an earlier placeholder is cancelled", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 1 }));
    const firstProject = getActiveProject(state);
    const firstQueueItem = firstProject.queue.items[0];

    state = workbenchReducer(state, {
      backendItemIds: [11],
      projectId: firstProject.id,
      queueItemId: firstQueueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = submitGenerate(primeGenerate(state, { batchCount: 2, positivePrompt: 'second prompt' }));

    const secondProject = getActiveProject(state);
    const secondQueueItem = secondProject.queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-1.png', secondQueueItem.id), createImage('candidate-2.png', secondQueueItem.id)],
      projectId: secondProject.id,
      queueItemId: secondQueueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { imageIndex: 1, type: 'setStagedImageIndex' });
    state = workbenchReducer(state, {
      backendItemId: 11,
      projectId: firstProject.id,
      queueItemId: firstQueueItem.id,
      type: 'markQueueItemBackendCancelled',
    });

    const updatedProject = getActiveProject(state);
    const selectedSlot = getCanvasStagingSlots(updatedProject.canvas, updatedProject.queue.items)[
      updatedProject.canvas.stagingArea.selectedImageIndex
    ];

    expect(selectedSlot).toMatchObject({ candidate: { imageName: 'candidate-1.png' }, kind: 'candidate' });
  });

  it("'progress' selects the active in-progress placeholder", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, { mode: 'progress', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });

    expect(getCanvas(state).stagingArea.autoSwitchMode).toBe('progress');
    expect(getCanvas(state).stagingArea.selectedImageIndex).toBe(0);
  });

  it("'progress' advances selection to the next in-progress placeholder as partials land", () => {
    let state = submitGenerate(primeGenerate(undefined, { batchCount: 3 }));
    const project = getActiveProject(state);
    const queueItem = project.queue.items[0];

    state = workbenchReducer(state, { mode: 'progress', type: 'setCanvasStagingAutoSwitch' });
    state = workbenchReducer(state, {
      backendItemIds: [11, 12, 13],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'markQueueItemBackendSubmitted',
    });
    state = workbenchReducer(state, {
      backendItemId: 11,
      images: [createImage('candidate-1.png', queueItem.id)],
      projectId: project.id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemPartialResults',
    });

    expect(
      getCanvasStagingSlots(getCanvas(state), getActiveProject(state).queue.items).map((slot) => slot.kind)
    ).toEqual(['candidate', 'placeholder', 'placeholder']);
    expect(getCanvas(state).stagingArea.selectedImageIndex).toBe(1);
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

    // The stale generation's results arrive against the brand-new empty canvas.
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
    // The item still completes — only its staged candidates are dropped.
    expect(getActiveProject(state).queue.items[0]?.status).toBe('completed');
  });

  it('submitCanvasInvocationSnapshot enqueues a pre-compiled graph bound for the canvas', () => {
    const graph: GraphContract = {
      backendGraph: { edges: [], id: 'canvas-backend-graph', nodes: {} },
      edges: [],
      id: 'canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };

    const initial = createInitialWorkbenchState();
    const state = workbenchReducer(initial, {
      backendSupportsCancellation: true,
      canvas: structuredClone(getCanvas(initial)),
      destination: 'canvas',
      generate: {
        negativePromptNodeId: 'negative_prompt',
        positivePromptNodeId: 'positive_prompt',
        seedNodeId: 'seed',
        values: createGenerateValues({
          negativePrompt: 'avoid blur',
          positivePrompt: 'inpaint prompt',
          seed: 42,
          seedMode: 'fixed',
        }),
      },
      graph,
      projectId: initial.activeProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });

    const queueItem = getActiveProject(state).queue.items[0];

    expect(queueItem?.snapshot.sourceId).toBe('canvas');
    expect(queueItem?.snapshot.destination).toBe('canvas');
    expect(queueItem?.snapshot.graph).toEqual({ id: 'canvas-graph', label: 'Canvas' });
    expect(queueItem?.snapshot.recall?.generateValues).toMatchObject({
      negativePrompt: 'avoid blur',
      positivePrompt: 'inpaint prompt',
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
      positivePrompt: 'inpaint prompt',
      positivePromptNodeId: 'positive_prompt',
      seed: 42,
      seedNodeId: 'seed',
      seedStep: 0,
    });
    expect(queueItem?.snapshot.resultNodeIds).toEqual(['canvas_output']);
    expect(getActiveProject(state).invocation.sourceId).toBe('canvas');
  });

  it('queues the frozen canvas supplied by an immutable canvas invocation instead of cloning live state', () => {
    const graph: GraphContract = {
      edges: [],
      id: 'frozen-canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };
    const initial = createInitialWorkbenchState();
    const liveCanvas = getCanvas(initial);
    const frozenCanvas = structuredClone(liveCanvas);
    frozenCanvas.document.bbox = { height: 48, width: 32, x: 17, y: 23 };

    const state = workbenchReducer(initial, {
      backendSupportsCancellation: true,
      canvas: frozenCanvas,
      destination: 'canvas',
      generate: createCanvasGenerateSnapshot(),
      graph,
      projectId: initial.activeProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });

    expect(getActiveProject(state).queue.items[0]?.snapshot.canvas.document.bbox).toEqual(frozenCanvas.document.bbox);
    expect(getActiveProject(state).queue.items[0]?.snapshot.canvas.document.bbox).not.toEqual(liveCanvas.document.bbox);
    expect(getActiveProject(state).queue.items[0]?.snapshot.canvas).toEqual({
      document: {
        bbox: frozenCanvas.document.bbox,
        height: frozenCanvas.document.height,
        width: frozenCanvas.document.width,
      },
      documentRevision: frozenCanvas.documentRevision,
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

  it('submitCanvasInvocationSnapshot honors a Gallery destination instead of hardcoding canvas', () => {
    const graph: GraphContract = {
      edges: [],
      id: 'canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };

    const initial = createInitialWorkbenchState();
    const state = workbenchReducer(initial, {
      backendSupportsCancellation: true,
      canvas: structuredClone(getCanvas(initial)),
      destination: 'gallery',
      generate: createCanvasGenerateSnapshot(),
      graph,
      projectId: initial.activeProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });

    const queueItem = getActiveProject(state).queue.items[0];

    // Gallery-destined Canvas results must bypass canvas staging.
    expect(queueItem?.snapshot.sourceId).toBe('canvas');
    expect(queueItem?.snapshot.destination).toBe('gallery');
    expect(getActiveProject(state).invocation.destination).toBe('gallery');
    expect(getActiveProject(state).canvas.stagingArea.pendingImageIds).toHaveLength(0);
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

  it('submitCanvasInvocationSnapshot with a stale/unknown projectId is a no-op', () => {
    const graph: GraphContract = {
      edges: [],
      id: 'canvas-graph',
      label: 'Canvas',
      nodes: [],
      updatedAt: '2026-06-09T00:00:00.000Z',
      version: 1,
    };

    const initial = createInitialWorkbenchState();
    const state = workbenchReducer(initial, {
      backendSupportsCancellation: true,
      canvas: structuredClone(getCanvas(initial)),
      destination: 'canvas',
      generate: createCanvasGenerateSnapshot(),
      graph,
      projectId: 'not-a-real-project',
      type: 'submitCanvasInvocationSnapshot',
    });

    expect(state).toBe(initial);
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

  it('ignores noise-only generate patches and system-originated patches', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });

    const noiseOnly = workbenchReducer(state, {
      type: 'patchGenerateSettings',
      values: { positivePromptHeightPx: 120 },
    });

    expect(getRoute(noiseOnly).sourceId).toBe('workflow');

    const systemOriginated = workbenchReducer(state, {
      origin: 'system',
      type: 'patchGenerateSettings',
      values: { steps: 25 },
    });

    expect(getRoute(systemOriginated).sourceId).toBe('workflow');
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

  it('ignores system-originated and non-graph-bearing widget-value patches', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      origin: 'system',
      type: 'patchWidgetValues',
      values: { inputImage: { height: 512, image_name: 'input.png', width: 768 } },
      widgetId: 'upscale',
    });
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { selectedImageName: 'result.png' },
      widgetId: 'gallery',
    });

    expect(getRoute(state).sourceId).toBe('generate');
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

  it('switches to Canvas for structural and visibility edits', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    state = workbenchReducer(state, {
      add: [{ anchor: { ...stackTopAnchor(state.activeProjectId), afterId: 'a' }, nodes: [createRasterLayer('b')] }],
      enabledUpdates: [],
      selectedLayerId: 'b',
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getRoute(state)).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    state = workbenchReducer(state, {
      type: 'setCanvasLayersEnabled',
      updates: [{ id: 'a', isEnabled: false }],
    });

    expect(getRoute(state)).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });
  });

  it('routes confirmed paint intents but ignores system Canvas mutations', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    state = reduceWorkbench(state, {
      intent: { kind: 'paint' },
      projectId: state.activeProjectId,
      type: 'commitCanvasEdit',
    });

    expect(getRoute(state)).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    state = reduceWorkbench(state, {
      mutation: { bbox: { height: 768, width: 768, x: 0, y: 0 }, type: 'setCanvasBbox' },
      origin: 'system',
      projectId: state.activeProjectId,
      type: 'applyCanvasProjectMutation',
    });

    expect(getRoute(state).sourceId).toBe('workflow');
  });

  it('uses the explicit source on shared prompt edits', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });
    state = workbenchReducer(state, {
      sourceId: 'generate',
      type: 'patchProjectPromptDraft',
      values: { positivePrompt: 'generate prompt' },
    });

    expect(getRoute(state)).toMatchObject({ destination: 'gallery', sourceId: 'generate' });

    state = workbenchReducer(state, { sourceId: 'canvas', type: 'setInvocationSource' });
    state = workbenchReducer(state, {
      sourceId: 'generate',
      type: 'patchProjectPromptDraft',
      values: { positivePrompt: 'canvas parameter prompt' },
    });

    expect(getRoute(state)).toMatchObject({ destination: 'gallery', sourceId: 'canvas' });
  });

  it('never lets generate edits steal the route from an active canvas source', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { destination: 'canvas', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { sourceId: 'canvas', type: 'setInvocationSource' });
    // Generate edits express Canvas intent while serving as its parameter panel.
    state = workbenchReducer(state, { type: 'patchGenerateSettings', values: { steps: 25 } });
    state = workbenchReducer(state, { type: 'setGenerateSettings', values: createGenerateValues() });

    expect(getRoute(state)).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });
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

  it('toggles both routing locks together and normalizes a mixed lock state', () => {
    let unlockedState = createInitialWorkbenchState();

    unlockedState = workbenchReducer(unlockedState, { type: 'toggleRoutingLock' });
    expect(getRoute(unlockedState)).toMatchObject({ destinationLocked: true, sourceLocked: true });

    unlockedState = workbenchReducer(unlockedState, { type: 'toggleRoutingLock' });
    expect(getRoute(unlockedState)).toMatchObject({ destinationLocked: false, sourceLocked: false });

    let mixedState = createInitialWorkbenchState();
    mixedState = workbenchReducer(mixedState, { type: 'toggleSourceLock' });
    expect(getRoute(mixedState)).toMatchObject({ destinationLocked: false, sourceLocked: true });

    mixedState = workbenchReducer(mixedState, { type: 'toggleRoutingLock' });
    expect(getRoute(mixedState)).toMatchObject({ destinationLocked: false, sourceLocked: false });
  });

  it('never remaps the destination on a manual source change or a same-source edit', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });

    // The dropdown is the strongest intent signal: destination stays put.
    expect(getRoute(state)).toMatchObject({ destination: 'gallery', sourceId: 'workflow' });

    state = workbenchReducer(state, { type: 'patchGenerateSettings', values: { steps: 25 } });
    state = workbenchReducer(state, { destination: 'canvas', type: 'setInvocationDestination' });
    state = workbenchReducer(state, { type: 'patchGenerateSettings', values: { steps: 30 } });

    // Editing the surface already routed as the source keeps the manual destination.
    expect(getRoute(state)).toMatchObject({ destination: 'canvas', sourceId: 'generate' });
  });
});
