import { describe, expect, it } from 'vitest';

import type { GenerateWidgetValues, MainModelConfig } from './generation/types';
import type { GeneratedImageContract, Project, WorkbenchState } from './types';

import { MAX_PROMPT_HISTORY } from './generation/promptHistory';
import { DEFAULT_PROJECT_SETTINGS } from './settings/store';
import { getProjectWidgetValues } from './widgetState';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState';

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

const getProject = (state: WorkbenchState, projectId: string): Project => {
  const project = state.projects.find((candidate) => candidate.id === projectId);

  expect(project).toBeDefined();

  return project as Project;
};

const getActiveProject = (state: WorkbenchState): Project => getProject(state, state.activeProjectId);

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

describe('workbench layout presets', () => {
  it('applies the Default preset as a full widget-region layout', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { presetId: 'canvas-default', type: 'applyPreset' });

    const project = getActiveProject(state);

    expect(project.layout.panels).toEqual({ isBottomOpen: false, isLeftOpen: true, isRightOpen: true });
    expect(project.widgetRegions.left).toMatchObject({
      activeInstanceId: 'generate',
      instanceIds: ['generate', 'workflow'],
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
      activeInstanceId: 'queue:bottom',
      instanceIds: [
        'server-status',
        'diagnostics:bottom',
        'queue:bottom',
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

  it('accepts a staged candidate into an undoable raster layer and prevents duplicate accepts', () => {
    let state = submitGenerate(primeGenerate());
    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { type: 'acceptStagedImage' });

    let project = getActiveProject(state);

    expect(project.canvas.document.layers).toHaveLength(1);
    expect(project.canvas.document.layers[0]?.imageName).toBe('candidate.png');
    expect(project.canvas.stagingArea.pendingImages).toEqual([]);
    expect(project.undoRedo.past).toHaveLength(1);

    state = workbenchReducer(state, { type: 'acceptStagedImage' });
    project = getActiveProject(state);

    expect(project.canvas.document.layers).toHaveLength(1);

    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    expect(project.canvas.document.layers).toEqual([]);
    expect(project.canvas.stagingArea.pendingImages).toEqual([]);
  });

  it('discards selected and all staged canvas candidates without touching accepted document layers', () => {
    let state = submitGenerate(primeGenerate());
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

  it('cycles staged canvas candidates and accepts the selected candidate placement into a raster layer', () => {
    let state = submitGenerate(primeGenerate());
    const queueItem = getActiveProject(state).queue.items[0];

    state = workbenchReducer(state, {
      images: [createImage('candidate-1.png', queueItem.id), createImage('candidate-2.png', queueItem.id)],
      projectId: getActiveProject(state).id,
      queueItemId: queueItem.id,
      type: 'routeQueueItemResults',
    });
    state = workbenchReducer(state, { direction: -1, type: 'cycleStagedImage' });

    expect(getActiveProject(state).canvas.stagingArea.selectedImageIndex).toBe(1);

    const selectedPlacement = getActiveProject(state).canvas.stagingArea.pendingImages[1]?.placement;

    state = workbenchReducer(state, { type: 'acceptStagedImage' });

    let project = getActiveProject(state);

    expect(project.canvas.document.layers[0]?.imageName).toBe('candidate-2.png');
    expect(project.canvas.document.layers[0]?.placement).toEqual(selectedPlacement);

    state = workbenchReducer(state, { type: 'undoProjectChange' });
    project = getActiveProject(state);

    expect(project.canvas.document.layers).toEqual([]);

    state = workbenchReducer(state, { type: 'redoProjectChange' });
    project = getActiveProject(state);

    expect(project.canvas.document.layers[0]?.imageName).toBe('candidate-2.png');
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
    state = workbenchReducer(state, { type: 'acceptStagedImage' });
    state = workbenchReducer(state, { message: 'boom', type: 'recordError' });

    expect(state.notifications.map((notification) => notification.title)).toEqual([
      'Error',
      'Canvas layer accepted',
      'Invocation completed',
      'Invocation queued',
    ]);
    expect(state.notifications.every((notification) => !notification.isRead)).toBe(true);

    state = workbenchReducer(state, { type: 'markAllNotificationsRead' });

    expect(state.notifications.every((notification) => notification.isRead)).toBe(true);

    state = workbenchReducer(state, { type: 'clearNotifications' });

    expect(state.notifications).toEqual([]);

    state = workbenchReducer(state, { type: 'clearErrorLog' });

    expect(state.errorLog).toEqual([]);
  });

  it('accepts the project graph source but does not queue an empty project graph', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'project-graph', type: 'setInvocationSource' });

    expect(getActiveProject(state).invocation.sourceId).toBe('project-graph');

    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      route: { destination: 'canvas', destinationLocked: false, sourceId: 'project-graph', sourceLocked: true },
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
    expect(project.invocation.sourceId).toBe('project-graph');

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

    state = workbenchReducer(state, {
      snapshotId: project.graphHistory[0]?.id ?? '',
      type: 'restoreProjectGraphSnapshot',
    });

    expect(getActiveProject(state).projectGraph.id).toBe(originalGraphId);
  });

  it('does not queue sources that are still unavailable', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });

    expect(getActiveProject(state).invocation.sourceId).toBe('generate');

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
