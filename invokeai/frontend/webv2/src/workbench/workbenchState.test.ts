import type { GeneratedImageContract } from '@features/gallery';
import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type {
  CanvasControlLayerContract,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
} from '@workbench/canvas-engine/contracts';
import type { GraphContract } from '@workbench/graphContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { MAX_PROMPT_HISTORY } from '@features/generation/settings';
import { describe, expect, it, vi } from 'vitest';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { createEmptyCanvasDocumentV2 } from './canvasMigration';
import { getCanvasStagingCandidateFingerprint, getCanvasStagingSlots } from './canvasStagingView';
import { DEFAULT_PROJECT_SETTINGS } from './settings/store';
import { getProjectWidgetValues } from './widgetState';
import { GRAPH_HISTORY_BYTE_BUDGET, normalizeGraphHistory } from './workbenchState';
import {
  createInitialWorkbenchState,
  nextLayerName,
  type WorkbenchAction,
  workbenchReducer as reduceWorkbench,
} from './workbenchState.testing';

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
  'duplicateCanvasLayer',
  'mergeCanvasLayersDown',
  'removeCanvasLayers',
  'reorderCanvasLayers',
  'replaceCanvasDocument',
  'replaceCanvasLayer',
  'rollbackStagedImageCommit',
  'resizeCanvasDocument',
  'restoreCanvasSnapshot',
  'saveCanvasSnapshot',
  'setCanvasBbox',
  'setCanvasLayersEnabled',
  'setCanvasSelectedLayer',
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
    name: `Layer ${project.canvas.document.layers.length + 1}`,
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
      candidateFingerprint: getCanvasStagingCandidateFingerprint(candidate),
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
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  clipEmbedModel: null,
  clipGEmbedModel: null,
  clipLEmbedModel: null,
  clipSkip: 0,
  colorCompensation: false,
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
  qwen3EncoderModel: null,
  qwenVLEncoderModel: null,
  referenceImages: [],
  scheduler: 'euler_a',
  seamlessXAxis: false,
  seamlessYAxis: false,
  seed: 123,
  shouldRandomizeSeed: false,
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

type CanvasLayer = Project['canvas']['document']['layers'][number];

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
// A new project's canvas now seeds one empty inpaint mask (see
// `createNewCanvasStateV2`). These layer-reducer/staging tests exercise layer
// mechanics where that default mask is incidental, so they start from an empty
// canvas document to keep their expectations focused on the layers under test.
const withEmptyCanvas = (state: WorkbenchState): WorkbenchState =>
  workbenchReducer(state, { document: createEmptyCanvasDocumentV2(), type: 'replaceCanvasDocument' });

const withCanvasLayers = (state: WorkbenchState, layers: Project['canvas']['document']['layers']): WorkbenchState =>
  [...layers]
    .reverse()
    .reduce((next, layer) => workbenchReducer(next, { layer, type: 'addCanvasLayer' }), withEmptyCanvas(state));

const getCanvas = (state: WorkbenchState) => getActiveProject(state).canvas;

const getLayerIds = (state: WorkbenchState): string[] => getCanvas(state).document.layers.map((layer) => layer.id);

const primeGenerate = (state = createInitialWorkbenchState(), overrides: Partial<GenerateWidgetValues> = {}) =>
  workbenchReducer(state, { type: 'setGenerateSettings', values: createGenerateValues(overrides) });

const submitGenerate = (state: WorkbenchState) =>
  workbenchReducer(state, { backendSupportsCancellation: true, type: 'submitInvocationSnapshot' });

describe('workbench widget region defaults', () => {
  it('enables Diagnostics in the right side panel rail', () => {
    const state = createInitialWorkbenchState();

    expect(getActiveProject(state).widgetRegions.right.instanceIds).toContain('diagnostics');
  });

  it('hydrates the old default right rail with Diagnostics while preserving customized rails', () => {
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
      'preview',
      'queue',
      'layers',
      'diagnostics',
      'project',
    ]);
    expect(getActiveProject(hydratedCustomized).widgetRegions.right.instanceIds).toEqual(['gallery', 'layers']);
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
  it('applies the Default preset as a full widget-region layout', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { presetId: 'canvas-default', type: 'applyPreset' });

    const project = getActiveProject(state);

    expect(project.layout.panels).toEqual({ isBottomOpen: false, isLeftOpen: true, isRightOpen: true });
    expect(project.widgetRegions.left).toMatchObject({
      activeInstanceId: 'generate',
      instanceIds: ['generate', 'workflow', 'upscale'],
      isCollapsed: false,
      sizePx: 450,
    });
    expect(project.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: ['preview', 'canvas', 'gallery:center', 'workflow:center'],
      isCollapsed: false,
      sizePx: 0,
    });
    expect(project.widgetRegions.right).toMatchObject({
      activeInstanceId: 'gallery',
      instanceIds: ['gallery', 'preview', 'queue', 'layers', 'diagnostics', 'project'],
      isCollapsed: false,
      sizePx: 450,
    });
    expect(project.widgetRegions.bottom).toMatchObject({
      activeInstanceId: 'gallery:bottom',
      instanceIds: [
        'server-status',
        'diagnostics:bottom',
        'gallery:bottom',
        'notifications',
        'autosave-status',
        'version-status',
        'workflow:bottom',
      ],
      isCollapsed: true,
      sizePx: 180,
    });
  });

  it('adds a custom preset from the active project layout and applies it later', () => {
    let state = createInitialWorkbenchState();
    const projectId = state.activeProjectId;

    state = workbenchReducer(state, { region: 'right', sizePx: 336, type: 'setRegionWidgetSize' });
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
    expect(project.widgetRegions.right).toMatchObject({ activeInstanceId: 'queue', sizePx: 336 });
    expect(project.widgetRegions.center.activeInstanceId).toBe('preview');
  });

  it('renames and deletes only custom layout presets', () => {
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
    state = workbenchReducer(state, { presetId: 'canvas-default', type: 'renameLayoutPreset', label: 'Nope' });
    state = workbenchReducer(state, { presetId: 'custom-layout-1', type: 'deleteLayoutPreset' });

    expect(state.account.customLayoutPresets).toEqual([]);
  });
});

describe('workbenchReducer Phase 5 generation flow', () => {
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

  it('appends a workflow candidate to its explicit project and selects its resolved staging slot', () => {
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

    const acceptedLayer = getActiveProject(state).canvas.document.layers[0];

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
    expect(getRasterLayerImageName(getProject(state, originProjectId).canvas.document.layers[0])).toBe(
      'origin-result.png'
    );
    expect(getProject(state, activeProjectId).canvas.document.layers).not.toContainEqual(
      expect.objectContaining({
        source: expect.objectContaining({ image: expect.objectContaining({ imageName: 'origin-result.png' }) }),
      })
    );
  });

  it('keeps submitted Generate snapshots immutable after later settings changes', () => {
    let state = submitGenerate(primeGenerate(undefined, { positivePrompt: 'first prompt', shouldRandomizeSeed: true }));
    const firstQueueItem = getActiveProject(state).queue.items[0];

    expect(firstQueueItem).toBeDefined();

    state = primeGenerate(state, { positivePrompt: 'second prompt', seed: 999 });
    state = submitGenerate(state);

    const [secondQueueItem, unchangedFirstQueueItem] = getActiveProject(state).queue.items;
    const firstValues = unchangedFirstQueueItem?.snapshot.widgetStates.generate
      .values as unknown as GenerateWidgetValues;
    const secondValues = secondQueueItem?.snapshot.widgetStates.generate.values as unknown as GenerateWidgetValues;

    expect(firstValues.positivePrompt).toBe('first prompt');
    expect(firstValues.shouldRandomizeSeed).toBe(true);
    expect(typeof firstValues.seed).toBe('number');
    expect(secondValues.positivePrompt).toBe('second prompt');
    expect(secondValues.seed).toBe(999);
  });

  it('accepts a staged candidate into a selected raster layer that project undo no longer touches', () => {
    let state = submitGenerate(primeGenerate(withEmptyCanvas(createInitialWorkbenchState())));
    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = commitSelectedStagedImage(state);

    let project = getActiveProject(state);
    const acceptedLayerId = project.canvas.document.layers[0]?.id;

    expect(project.canvas.document.layers).toHaveLength(1);
    expect(getRasterLayerImageName(project.canvas.document.layers[0])).toBe('candidate.png');
    expect(project.canvas.document.selectedLayerId).toBe(acceptedLayerId);
    expect(project.canvas.stagingArea.pendingImages).toEqual([]);
    // Deliberate semantic change (P0.2): canvas is engine-owned, so accepting a
    // staged image does not create a project-level undo entry.
    expect(project.undoRedo.past).toHaveLength(0);

    state = commitSelectedStagedImage(state);
    project = getActiveProject(state);

    expect(project.canvas.document.layers).toHaveLength(1);

    // Project undo neither snapshots nor restores canvas: the accepted layer survives.
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    expect(project.canvas.document.layers).toHaveLength(1);
    expect(getRasterLayerImageName(project.canvas.document.layers[0])).toBe('candidate.png');
  });

  it('discards selected and all staged canvas candidates without touching accepted document layers', () => {
    let state = submitGenerate(primeGenerate(withEmptyCanvas(createInitialWorkbenchState())));
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

    expect(project.canvas.document.layers).toEqual([]);
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
    const acceptedLayer = project.canvas.document.layers[0];

    expect(getRasterLayerImageName(acceptedLayer)).toBe('candidate-2.png');
    expect(getRasterLayerPlacement(acceptedLayer)).toEqual({ height: 768, opacity: 1, width: 512, x: 0, y: 0 });

    // Project undo/redo leaves the engine-owned canvas alone.
    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    expect(getRasterLayerImageName(project.canvas.document.layers[0])).toBe('candidate-2.png');
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

    expect(
      getCanvasStagingSlots(getCanvas(state), getActiveProject(state).queue.items).map((slot) => slot.kind)
    ).toEqual(['placeholder', 'candidate']);
    expect(getCanvas(state).stagingArea.selectedImageIndex).toBe(1);

    state = commitSelectedStagedImage(state);

    expect(getRasterLayerImageName(getActiveProject(state).canvas.document.layers[0])).toBe('candidate-2.png');
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

    expect(getActiveProject(state).queue.items[0]?.status).toBe('cancelled');
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

    expect(state.projects.flatMap((project) => project.queue.items.map((item) => item.status))).toEqual([
      'cancelled',
      'cancelled',
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

    expect(getActiveProject(state).queue.items.map((item) => ({ id: item.id, status: item.status }))).toEqual([
      { id: getActiveProject(state).queue.items[0].id, status: 'cancelled' },
      { id: firstQueueItemId, status: 'pending' },
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

    // A meaningful graph edit lands in the document, creates an undo entry,
    // and steers the unlocked invocation source to the project graph.
    expect(project.projectGraph.nodes).toHaveLength(1);
    expect(project.undoRedo.past.at(-1)?.label).toBe('Add workflow node');
    expect(project.invocation.sourceId).toBe('workflow');

    state = workbenchReducer(state, { type: 'undoProjectChange' });

    expect(getActiveProject(state).projectGraph.nodes).toHaveLength(0);
  });

  it('replaceProjectGraph snapshots the previous document into graph history', () => {
    let state = createInitialWorkbenchState();
    const originalGraphId = getActiveProject(state).projectGraph.id;

    state = workbenchReducer(state, {
      document: { ...getActiveProject(state).projectGraph, id: 'replacement-graph', name: 'Replacement' },
      label: 'Test replace',
      type: 'replaceProjectGraph',
    });

    const project = getActiveProject(state);

    expect(project.projectGraph.id).toBe('replacement-graph');
    expect(project.graphHistory[0]?.document?.id).toBe(originalGraphId);
    expect(project.graphHistory[0]?.retainedBytes).toBeGreaterThan(0);
    expect(project.graphHistory[0]?.document).toBe(project.undoRedo.past.at(-1)?.project.projectGraph);

    state = workbenchReducer(state, {
      snapshotId: project.graphHistory[0]?.id ?? '',
      type: 'restoreProjectGraphSnapshot',
    });

    expect(getActiveProject(state).projectGraph.id).toBe(originalGraphId);
  });

  it('keeps the newest graph-history entries within the count limit and recomputes retained bytes', () => {
    const entries = Array.from({ length: 50 }, (_, index) => ({
      createdAt: `2026-07-19T00:00:${String(index).padStart(2, '0')}.000Z`,
      document: { edges: [], nodes: [], version: 1 as const },
      id: `snapshot-${index}`,
      label: `Snapshot ${index}`,
      retainedBytes: 0,
    }));
    const normalized = normalizeGraphHistory(entries);

    expect(normalized).toHaveLength(40);
    expect(normalized.map((entry) => entry.id)).toEqual(entries.slice(0, 40).map((entry) => entry.id));
    expect(normalized.every((entry) => (entry.retainedBytes ?? 0) > 0)).toBe(true);
  });

  it('measures loaded graph-history entries instead of trusting forged retained-byte metadata', () => {
    const encode = vi.spyOn(TextEncoder.prototype, 'encode').mockImplementation((input) => {
      const byteLength = String(input).includes('"id":"oversized"') ? GRAPH_HISTORY_BYTE_BUDGET + 1 : 128;

      return Object.defineProperty(new Uint8Array(0), 'byteLength', { value: byteLength });
    });

    try {
      const normalized = normalizeGraphHistory([
        {
          createdAt: '2026-07-19T00:00:00.000Z',
          document: { edges: [], nodes: [], version: 1 as const },
          id: 'oversized',
          label: 'Forged low byte count',
          retainedBytes: 0,
        },
        {
          createdAt: '2026-07-19T00:00:01.000Z',
          document: { edges: [], nodes: [], version: 1 as const },
          id: 'within-budget',
          label: 'Forged high byte count',
          retainedBytes: GRAPH_HISTORY_BYTE_BUDGET + 1,
        },
      ]);

      expect(normalized).toHaveLength(1);
      expect(normalized[0]).toMatchObject({ id: 'within-budget', retainedBytes: 128 });
    } finally {
      encode.mockRestore();
    }
  });

  it('admits newest graph-history entries without exceeding the cumulative byte budget', () => {
    const halfBudgetPlusOne = Math.floor(GRAPH_HISTORY_BYTE_BUDGET / 2) + 1;
    const encode = vi.spyOn(TextEncoder.prototype, 'encode').mockImplementation((input) => {
      const byteLength = String(input).includes('"id":"small"') ? 128 : halfBudgetPlusOne;

      return Object.defineProperty(new Uint8Array(0), 'byteLength', { value: byteLength });
    });

    try {
      const createEntry = (id: string) => ({
        createdAt: '2026-07-19T00:00:00.000Z',
        document: { edges: [], nodes: [], version: 1 as const },
        id,
        label: id,
        retainedBytes: 0,
      });
      const normalized = normalizeGraphHistory([
        createEntry('newest-large'),
        createEntry('older-large'),
        createEntry('small'),
      ]);

      expect(normalized.map((entry) => entry.id)).toEqual(['newest-large', 'small']);
      expect(normalized.reduce((total, entry) => total + (entry.retainedBytes ?? 0), 0)).toBeLessThanOrEqual(
        GRAPH_HISTORY_BYTE_BUDGET
      );
    } finally {
      encode.mockRestore();
    }
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

    expect(updatedProject.canvas.stagingArea.pendingImages).toEqual([]);
    expect(galleryValues.recentImages).toEqual([createImage('gallery-image.png', queueItem.id)]);
    expect(galleryValues.selectedImage).toEqual(createImage('gallery-image.png', queueItem.id));
    expect(galleryValues.selectedImageName).toBe('gallery-image.png');
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

  it('stores selected backend board id for gallery submissions', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { boardId: 'backend-board-id', type: 'selectGalleryBoard' });
    state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
    state = primeGenerate(state);
    state = submitGenerate(state);

    const queueItem = getActiveProject(state).queue.items[0];

    expect(queueItem.snapshot.widgetStates.gallery.values.selectedBoardId).toBe('backend-board-id');
  });

  it('stores full selected gallery image data for Preview widget', () => {
    let state = createInitialWorkbenchState();
    const image = createImage('backend-selected.png', 'backend-gallery');

    state = workbenchReducer(state, { image, type: 'selectGalleryImage' });

    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImageName).toBe('backend-selected.png');
    expect(getProjectWidgetValues(getActiveProject(state), 'gallery').selectedImage).toEqual(image);
  });
});

describe('workbench account and project settings', () => {
  it('starts with the default layout preset and legacy-matching project settings', () => {
    const state = createInitialWorkbenchState();

    expect(state.account).toEqual({ activeLayoutPresetId: 'canvas-default' });
    expect(getActiveProject(state).settings).toEqual(DEFAULT_PROJECT_SETTINGS);
  });

  it('drops legacy preferences carried inside persisted accounts on hydrate', () => {
    const initial = createInitialWorkbenchState();
    const legacy = {
      ...initial,
      account: { activeLayoutPresetId: 'gallery', preferences: { themeId: 'forest' } },
    } as unknown as WorkbenchState;

    const state = workbenchReducer(initial, { state: legacy, type: 'hydrateWorkbench' });

    expect(state.account).toEqual({ activeLayoutPresetId: 'gallery', customLayoutPresets: [] });
  });

  it('heals hydrated accounts that are missing a layout preset', () => {
    const initial = createInitialWorkbenchState();
    const legacy = { ...initial, account: {} } as unknown as WorkbenchState;

    const state = workbenchReducer(initial, { state: legacy, type: 'hydrateWorkbench' });

    expect(state.account.activeLayoutPresetId).toBe('canvas-default');
  });

  it('updates project settings on the active project only', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      settings: { antialiasProgressImages: true, preferNumericAttentionStyle: true, useCpuNoise: false },
      type: 'setActiveProjectSettings',
    });

    expect(getActiveProject(state).settings).toEqual({
      antialiasProgressImages: true,
      preferNumericAttentionStyle: true,
      showProgressDetails: false,
      showProgressImagesInViewer: true,
      showPromptSyntaxHighlighting: false,
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

  it('refreshes every project gallery when backend data may have changed', () => {
    const initial = createInitialWorkbenchState();
    const previousTokens = initial.projects.map(
      (project) => getProjectWidgetValues(project, 'gallery').galleryRefreshToken
    );
    const previousImageTokens = initial.projects.map(
      (project) => getProjectWidgetValues(project, 'gallery').galleryImagesRefreshToken
    );
    const state = workbenchReducer(initial, { type: 'refreshBackendData' });

    expect(state.projects).toHaveLength(initial.projects.length);

    for (const [index, project] of state.projects.entries()) {
      const galleryValues = getProjectWidgetValues(project, 'gallery');

      expect(galleryValues.galleryRefreshToken).toBeDefined();
      expect(galleryValues.galleryRefreshToken).not.toBe(previousTokens[index]);
      expect(galleryValues.galleryImagesRefreshToken).toBeDefined();
      expect(galleryValues.galleryImagesRefreshToken).not.toBe(previousImageTokens[index]);
    }
  });

  it('can refresh gallery images without invalidating board data', () => {
    const initial = createInitialWorkbenchState();
    const state = workbenchReducer(initial, { type: 'touchGalleryImagesRefresh' });
    const previousValues = getProjectWidgetValues(getActiveProject(initial), 'gallery');
    const values = getProjectWidgetValues(getActiveProject(state), 'gallery');

    expect(values.galleryRefreshToken).toBe(previousValues.galleryRefreshToken);
    expect(values.galleryImagesRefreshToken).toBeDefined();
    expect(values.galleryImagesRefreshToken).not.toBe(previousValues.galleryImagesRefreshToken);
  });

  it('can refresh gallery images in a non-active project without touching the active project', () => {
    let state = createInitialWorkbenchState();
    const firstProjectId = state.activeProjectId;

    state = workbenchReducer(state, { type: 'createProject' });
    const secondProjectId = state.activeProjectId;
    const firstPreviousValues = getProjectWidgetValues(getProject(state, firstProjectId), 'gallery');
    const secondPreviousValues = getProjectWidgetValues(getProject(state, secondProjectId), 'gallery');

    state = workbenchReducer(state, { projectId: firstProjectId, type: 'touchGalleryImagesRefresh' });

    const firstValues = getProjectWidgetValues(getProject(state, firstProjectId), 'gallery');
    const secondValues = getProjectWidgetValues(getProject(state, secondProjectId), 'gallery');

    expect(firstValues.galleryImagesRefreshToken).not.toBe(firstPreviousValues.galleryImagesRefreshToken);
    expect(secondValues.galleryImagesRefreshToken).toBe(secondPreviousValues.galleryImagesRefreshToken);
  });

  it('can remove gallery images from a non-active project without touching the active project', () => {
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
      imageNames: [image.imageName],
      projectId: firstProjectId,
      type: 'removeGalleryImages',
    });

    expect(getProjectWidgetValues(getProject(state, firstProjectId), 'gallery').recentImages).toEqual([]);
    expect(getProjectWidgetValues(getProject(state, secondProjectId), 'gallery').recentImages).toEqual([image]);
    expect(getProjectWidgetValues(getProject(state, firstProjectId), 'upscale').inputImage).toBeNull();
    expect(getProjectWidgetValues(getProject(state, secondProjectId), 'upscale').inputImage).toMatchObject({
      image_name: image.imageName,
    });
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

    expect(getProject(state, originProjectId).canvas.document.layers[0]?.opacity).toBe(0.25);
    expect(getProject(state, activeProjectId).canvas.document.layers[0]?.opacity).not.toBe(0.25);
  });

  it('seeds a new project canvas with a single empty inpaint mask, selected', () => {
    const state = createInitialWorkbenchState();
    const { document } = getActiveProject(state).canvas;

    expect(document.layers).toHaveLength(1);
    const mask = document.layers[0];
    expect(mask?.type).toBe('inpaint_mask');
    // Empty: no bitmap (no strokes) — so it never flips generation-mode detection.
    expect(mask && 'mask' in mask ? mask.mask.bitmap : 'missing').toBeNull();
    expect(document.selectedLayerId).toBe(mask?.id);
  });

  it('adds a layer at the top and selects it, honoring an explicit insert index', () => {
    let state = withEmptyCanvas(createInitialWorkbenchState());

    state = workbenchReducer(state, { layer: createRasterLayer('a'), type: 'addCanvasLayer' });
    state = workbenchReducer(state, { layer: createRasterLayer('b'), type: 'addCanvasLayer' });

    expect(getLayerIds(state)).toEqual(['b', 'a']);
    expect(getCanvas(state).document.selectedLayerId).toBe('b');

    state = workbenchReducer(state, { index: 1, layer: createRasterLayer('c'), type: 'addCanvasLayer' });

    expect(getLayerIds(state)).toEqual(['b', 'c', 'a']);
    expect(getCanvas(state).document.selectedLayerId).toBe('c');
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

  it('sets many layers visibility in one bulk action, preserving unlisted layers by identity', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);
    const before = getCanvas(state).document.layers;

    // Hide 'a' and 'c' in one dispatch; 'b' is unlisted and must keep its object identity.
    state = workbenchReducer(state, {
      type: 'setCanvasLayersEnabled',
      updates: [
        { id: 'a', isEnabled: false },
        { id: 'c', isEnabled: false },
      ],
    });
    const after = getCanvas(state).document.layers;

    expect(after.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['a', false],
      ['b', true],
      ['c', false],
    ]);
    // 'b' unchanged ⇒ same reference; 'a'/'c' replaced.
    expect(after[1]).toBe(before[1]);
    expect(after[0]).not.toBe(before[0]);
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
    const unrelatedBefore = getCanvas(initial).document.layers[0];

    const applied = workbenchReducer(initial, {
      add: { index: 1, layers: [result] },
      enabledUpdates: [
        { id: upper.id, isEnabled: false },
        { id: below.id, isEnabled: false },
      ],
      selectedLayerId: result.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(applied).toBeDefined();
    expect(getCanvas(applied).document.layers.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['unrelated', false],
      ['result', true],
      ['upper', false],
      ['below', false],
    ]);
    expect(getCanvas(applied).document.selectedLayerId).toBe(result.id);
    expect(getCanvas(applied).document.layers[0]).toBe(unrelatedBefore);

    const reverted = workbenchReducer(applied, {
      enabledUpdates: [
        { id: upper.id, isEnabled: true },
        { id: below.id, isEnabled: true },
      ],
      removeIds: [result.id],
      selectedLayerId: unrelated.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getCanvas(reverted).document.layers.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['unrelated', false],
      ['upper', true],
      ['below', true],
    ]);
    expect(getCanvas(reverted).document.selectedLayerId).toBe(unrelated.id);
    expect(getCanvas(reverted).document.layers[0]).toBe(unrelatedBefore);
  });

  it('inserts a batch in order and selects the exact requested layer', () => {
    const existing = createRasterLayer('existing');
    const layerA = createRasterLayer('a');
    const layerB = createRasterLayer('b');
    const initial = withCanvasLayers(createInitialWorkbenchState(), [existing]);

    const next = workbenchReducer(initial, {
      add: { index: 0, layers: [layerA, layerB] },
      enabledUpdates: [],
      selectedLayerId: layerB.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getCanvas(next).document.layers).toEqual([layerA, layerB, existing]);
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
      add: { index: 0, layers: [layerA, layerB] },
      enabledUpdates: [],
      projectId: firstProjectId,
      selectedLayerId: layerB.id,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getProject(next, firstProjectId).canvas.document.layers.slice(0, 2)).toEqual([layerA, layerB]);
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
        add: { index: 0, layers: [createRasterLayer('a')] },
        enabledUpdates: [{ id: 'b', isEnabled: false }],
        selectedLayerId: 'b',
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
        add: { index: 0, layers: [createRasterLayer('c')] },
        enabledUpdates: [],
        removeIds: ['a'],
        selectedLayerId: 'b',
        type: 'applyCanvasLayerStackMutation',
      },
      {
        add: { index: 0, layers: [createRasterLayer('c'), createRasterLayer('c')] },
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
    const before = getCanvas(initial).document.layers;

    const next = workbenchReducer(initial, {
      enabledUpdates: [],
      selectedLayerId: 'b',
      type: 'applyCanvasLayerStackMutation',
    });

    expect(getCanvas(next).document.layers).toBe(before);
    expect(getCanvas(next).document.selectedLayerId).toBe('b');
  });

  it('duplicates a layer above its source with a copy name and selects the duplicate', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);

    state = workbenchReducer(state, { newId: 'b-copy', sourceId: 'b', type: 'duplicateCanvasLayer' });

    expect(getLayerIds(state)).toEqual(['a', 'b-copy', 'b']);
    expect(getCanvas(state).document.selectedLayerId).toBe('b-copy');

    const [, duplicate, source] = getCanvas(state).document.layers;

    expect(duplicate?.name).toBe('b copy');
    expect(duplicate).not.toBe(source);
    expect(getRasterLayerImageName(duplicate)).toBe(getRasterLayerImageName(source));
  });

  it('reorders layers only when the id set matches, and preserves layer identity', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('a'),
      createRasterLayer('b'),
      createRasterLayer('c'),
    ]);
    const originalLayerA = getCanvas(state).document.layers.find((layer) => layer.id === 'a');

    const reordered = workbenchReducer(state, { orderedIds: ['c', 'a', 'b'], type: 'reorderCanvasLayers' });

    expect(getLayerIds(reordered)).toEqual(['c', 'a', 'b']);
    // Untouched layer objects are reused, not cloned.
    expect(getCanvas(reordered).document.layers.find((layer) => layer.id === 'a')).toBe(originalLayerA);

    const ignoredMissing = workbenchReducer(state, { orderedIds: ['c', 'a'], type: 'reorderCanvasLayers' });
    const ignoredUnknown = workbenchReducer(state, { orderedIds: ['c', 'a', 'z'], type: 'reorderCanvasLayers' });

    expect(getLayerIds(ignoredMissing)).toEqual(['a', 'b', 'c']);
    expect(getLayerIds(ignoredUnknown)).toEqual(['a', 'b', 'c']);
  });

  it('updates base props with a field-wise transform merge and leaves other layers untouched', () => {
    const state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    const originalLayerB = getCanvas(state).document.layers.find((layer) => layer.id === 'b');

    const updated = workbenchReducer(state, {
      id: 'a',
      patch: { name: 'Renamed', opacity: 0.5, transform: { x: 12 } },
      type: 'updateCanvasLayer',
    });

    const layerA = getCanvas(updated).document.layers.find((layer) => layer.id === 'a');

    expect(layerA?.name).toBe('Renamed');
    expect(layerA?.opacity).toBe(0.5);
    expect(layerA?.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 0 });
    expect(getCanvas(updated).document.layers.find((layer) => layer.id === 'b')).toBe(originalLayerB);
  });

  it('replaces one complete layer contract in place and no-ops when the id is missing', () => {
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
    const afterLayers = getCanvas(replaced).document.layers;

    expect(afterLayers.map((layer) => layer.id)).toEqual(['a', 'b', 'c']);
    expect(afterLayers[1]).toBe(replacement);
    expect(afterLayers[1]).toEqual(replacement);
    expect(afterLayers[0]).toBe(beforeDocument.layers[0]);
    expect(afterLayers[2]).toBe(beforeDocument.layers[2]);

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

    expect(getRasterLayerImageName(getCanvas(state).document.layers[0])).toBe('swapped.png');
  });

  it('applies per-type config patches and ignores mismatched layer types', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createControlLayer('ctrl')]);

    state = workbenchReducer(state, {
      config: { adapter: { weight: 0.25 }, layerType: 'control', withTransparencyEffect: true },
      id: 'ctrl',
      type: 'updateCanvasLayerConfig',
    });

    const controlLayer = getCanvas(state).document.layers[0];

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

    expect(getCanvas(unchanged).document.layers[0]).toBe(controlLayer);
  });

  it('persists filter settings on raster layers', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('raster')]);

    state = workbenchReducer(state, {
      config: { filter: { settings: { radius: 4 }, type: 'content_shuffle' }, layerType: 'raster' },
      id: 'raster',
      type: 'updateCanvasLayerConfig',
    });

    const raster = getCanvas(state).document.layers[0];
    expect(raster?.type).toBe('raster');
    if (raster?.type === 'raster') {
      expect(raster.filter).toEqual({ settings: { radius: 4 }, type: 'content_shuffle' });
    }
  });

  it('applies an inpaint-mask config patch: mask bitmap + offset, fill, noise, denoise-limit', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createInpaintMaskLayer('m')]);

    state = workbenchReducer(state, {
      config: {
        denoiseLimit: 0.8,
        layerType: 'inpaint_mask',
        mask: { bitmap: { contentHash: 'h', height: 20, imageName: 'mask.png', width: 30 }, offset: { x: 4, y: 5 } },
        noiseLevel: 0.25,
      },
      id: 'm',
      type: 'updateCanvasLayerConfig',
    });

    const layer = getCanvas(state).document.layers[0];
    expect(layer?.type).toBe('inpaint_mask');
    if (layer?.type === 'inpaint_mask') {
      // Bitmap + content offset persisted; the fill is preserved (merged, not replaced).
      expect(layer.mask.bitmap).toMatchObject({ imageName: 'mask.png', width: 30, height: 20 });
      expect(layer.mask.offset).toEqual({ x: 4, y: 5 });
      expect(layer.mask.fill).toEqual({ color: '#e07575', style: 'diagonal' });
      expect(layer.noiseLevel).toBe(0.25);
      expect(layer.denoiseLimit).toBe(0.8);
    }

    // A fill-only patch replaces the fill while keeping the bitmap.
    const recolored = workbenchReducer(state, {
      config: { layerType: 'inpaint_mask', mask: { fill: { color: '#00ff00', style: 'grid' } } },
      id: 'm',
      type: 'updateCanvasLayerConfig',
    });
    const after = getCanvas(recolored).document.layers[0];
    if (after?.type === 'inpaint_mask') {
      expect(after.mask.fill).toEqual({ color: '#00ff00', style: 'grid' });
      expect(after.mask.bitmap).toMatchObject({ imageName: 'mask.png' });
    }
  });

  it('removes optional config fields when a patch explicitly sets them to undefined', () => {
    const configuredMask: CanvasInpaintMaskLayerContract = {
      blendMode: 'normal',
      denoiseLimit: 0.8,
      id: 'm',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
      name: 'm',
      noiseLevel: 0.25,
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    };
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      configuredMask,
      { ...createControlLayer('ctrl'), filter: { settings: {}, type: 'canny' } },
    ]);

    state = workbenchReducer(state, {
      config: { denoiseLimit: undefined, layerType: 'inpaint_mask', noiseLevel: undefined },
      id: 'm',
      type: 'updateCanvasLayerConfig',
    });
    state = workbenchReducer(state, {
      config: { filter: undefined, layerType: 'control' },
      id: 'ctrl',
      type: 'updateCanvasLayerConfig',
    });

    const mask = getCanvas(state).document.layers[0];
    const control = getCanvas(state).document.layers[1];

    expect(mask?.type).toBe('inpaint_mask');
    if (mask?.type === 'inpaint_mask') {
      expect(mask.noiseLevel).toBeUndefined();
      expect(mask.denoiseLimit).toBeUndefined();
    }
    expect(control?.type).toBe('control');
    if (control?.type === 'control') {
      expect(control.filter).toBeUndefined();
    }
  });

  it('converts a layer in place, preserving its id and z-order', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a'), createRasterLayer('b')]);
    const converted = createControlLayer('ignored-id');

    state = workbenchReducer(state, { id: 'b', layer: converted, targetType: 'control', type: 'convertCanvasLayer' });

    expect(getLayerIds(state)).toEqual(['a', 'b']);
    expect(getCanvas(state).document.layers[1]?.type).toBe('control');
    expect(getCanvas(state).document.layers[1]?.id).toBe('b');
  });

  it('merges a layer down: the layer below becomes a raster with the merged paint source', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [
      createRasterLayer('top'),
      createControlLayer('bottom'),
    ]);

    state = workbenchReducer(state, { id: 'bottom', type: 'setCanvasSelectedLayer' });
    const mergedSource = { bitmap: { height: 8, imageName: 'merged.png', width: 8 }, type: 'paint' } as const;

    state = workbenchReducer(state, { source: mergedSource, type: 'mergeCanvasLayersDown', upperLayerId: 'top' });

    const layers = getCanvas(state).document.layers;

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
    expect(document.layers[0]?.transform.x).toBe(20);
    expect(document.layers[0]?.transform.y).toBe(10);
    expect(document.bbox.x + document.bbox.width).toBeLessThanOrEqual(300);
    expect(document.bbox.y + document.bbox.height).toBeLessThanOrEqual(256);
  });

  it('replaces the whole document with a deep copy', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = {
      ...createEmptyCanvasDocumentV2(),
      layers: [createRasterLayer('fresh')],
      selectedLayerId: 'fresh',
    };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getLayerIds(state)).toEqual(['fresh']);
    expect(getCanvas(state).document).not.toBe(replacement);
    expect(getCanvas(state).document.layers[0]).not.toBe(replacement.layers[0]);
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

    state = workbenchReducer(state, { document: createEmptyCanvasDocumentV2(), type: 'replaceCanvasDocument' });

    const { stagingArea } = getCanvas(state);
    expect(stagingArea.pendingImages).toEqual([]);
    expect(stagingArea.pendingImageIds).toEqual([]);
    expect(stagingArea.isVisible).toBe(false);
    expect(stagingArea.sourceQueueItemId).toBeUndefined();
  });

  it('repairs a dangling selectedLayerId on replaceCanvasDocument (falls back to the top layer)', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = {
      ...createEmptyCanvasDocumentV2(),
      layers: [createRasterLayer('top'), createRasterLayer('bottom')],
      selectedLayerId: 'ghost', // names no layer in the incoming document
    };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getCanvas(state).document.selectedLayerId).toBe('top');
  });

  it('nulls a dangling selectedLayerId when the replacement document has no layers', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = { ...createEmptyCanvasDocumentV2(), layers: [], selectedLayerId: 'ghost' };

    state = workbenchReducer(state, { document: replacement, type: 'replaceCanvasDocument' });

    expect(getCanvas(state).document.selectedLayerId).toBeNull();
  });

  it('preserves a valid selectedLayerId on replaceCanvasDocument', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createRasterLayer('a')]);
    const replacement = {
      ...createEmptyCanvasDocumentV2(),
      layers: [createRasterLayer('top'), createRasterLayer('bottom')],
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
    expect(restored.selectedLayerId).toBe(restored.layers[0]?.id);
  });

  it('normalizes a control adapter when restoring a saved canvas snapshot', () => {
    let state = withCanvasLayers(createInitialWorkbenchState(), [createControlLayer('ctrl')]);
    state = workbenchReducer(state, { createdAt: 'now', id: 'snap-1', name: 'First', type: 'saveCanvasSnapshot' });
    const snapshotLayer = getCanvas(state).snapshots[0]!.document.layers[0];
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

    const restored = getCanvas(state).document.layers[0];
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
    expect(getCanvas(state).snapshots[0]?.document.layers.map((layer) => layer.id)).toEqual(['a']);

    // Mutate the live document, then restore the snapshot back over it.
    state = workbenchReducer(state, { layer: createRasterLayer('b'), type: 'addCanvasLayer' });
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
    state = workbenchReducer(state, { layer: createRasterLayer('b'), type: 'addCanvasLayer' });
    state = workbenchReducer(state, { bbox: { height: 32, width: 32, x: 0, y: 0 }, type: 'setCanvasBbox' });
    state = workbenchReducer(state, { height: 256, type: 'resizeCanvasDocument', width: 256 });
    expect(getCanvas(state).documentRevision).toBe(initialRevision);

    // replaceCanvasDocument is a wholesale swap: bump.
    state = workbenchReducer(state, { document: createEmptyCanvasDocumentV2(), type: 'replaceCanvasDocument' });
    expect(getCanvas(state).documentRevision).toBe(initialRevision + 1);

    // restoreCanvasSnapshot is a wholesale swap: bump — even though the restored
    // document reuses the saved layer ids at the same dimensions.
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
    state = workbenchReducer(state, { layer: createRasterLayer('b'), type: 'addCanvasLayer' });
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
    values: createGenerateValues({ positivePrompt: 'canvas prompt', seed: 101, shouldRandomizeSeed: false }),
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

    // The user confirms a new canvas while the generation is still in flight: a
    // wholesale swap that clears staging and bumps documentRevision (new session).
    const swapped = workbenchReducer(submitted, {
      document: createEmptyCanvasDocumentV2(),
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
          shouldRandomizeSeed: false,
        }),
      },
      graph,
      projectId: initial.activeProjectId,
      type: 'submitCanvasInvocationSnapshot',
    });

    const queueItem = getActiveProject(state).queue.items[0];

    expect(queueItem?.snapshot.sourceId).toBe('canvas');
    expect(queueItem?.snapshot.destination).toBe('canvas');
    expect(queueItem?.snapshot.graph.id).toBe('canvas-graph');
    expect(queueItem?.snapshot.generate).toMatchObject({
      negativePromptNodeId: 'negative_prompt',
      positivePromptNodeId: 'positive_prompt',
      seedNodeId: 'seed',
      values: { negativePrompt: 'avoid blur', positivePrompt: 'inpaint prompt', seed: 42, shouldRandomizeSeed: false },
    });
    expect(queueItem?.snapshot.widgetStates.generate.values).toMatchObject({
      negativePrompt: 'avoid blur',
      positivePrompt: 'inpaint prompt',
      seed: 42,
      shouldRandomizeSeed: false,
    });
    expect(queueItem?.snapshot.backendSubmission).toMatchObject({
      batchCount: 1,
      kind: 'generate',
      negativePrompt: 'avoid blur',
      negativePromptNodeId: 'negative_prompt',
      positivePrompt: 'inpaint prompt',
      positivePromptNodeId: 'positive_prompt',
      seed: 42,
      seedNodeId: 'seed',
      shouldRandomizeSeed: false,
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

    // A Canvas source still runs, but the resolved Gallery destination rides
    // through so `routeQueueItemResults` keeps it out of canvas staging.
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
