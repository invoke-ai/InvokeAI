import { describe, expect, it } from 'vitest';

import type { GenerateWidgetValues, MainModelConfig } from './generation/types';
import { DEFAULT_PROJECT_SETTINGS } from './settings/store';
import type { GeneratedImageContract, Project, WorkbenchState } from './types';
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
  clipSkip: 0,
  height: 1024,
  model,
  modelKey: model.key,
  negativePrompt: '',
  positivePrompt: 'first prompt',
  scheduler: 'euler_a',
  seamlessXAxis: false,
  seamlessYAxis: false,
  seed: 123,
  shouldRandomizeSeed: false,
  steps: 30,
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

    expect(getActiveProject(state).widgetRegions.right.enabledWidgetIds).toContain('diagnostics');
  });

  it('hydrates the old default right rail with Diagnostics while preserving customized rails', () => {
    const initial = createInitialWorkbenchState();
    const legacyDefault = {
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          right: { ...project.widgetRegions.right, enabledWidgetIds: ['queue', 'gallery', 'layers'] },
        },
      })),
    } satisfies WorkbenchState;
    const customized = {
      ...initial,
      projects: initial.projects.map((project) => ({
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          right: { ...project.widgetRegions.right, enabledWidgetIds: ['gallery', 'layers'] },
        },
      })),
    } satisfies WorkbenchState;

    const hydratedLegacyDefault = workbenchReducer(initial, { state: legacyDefault, type: 'hydrateWorkbench' });
    const hydratedCustomized = workbenchReducer(initial, { state: customized, type: 'hydrateWorkbench' });

    expect(getActiveProject(hydratedLegacyDefault).widgetRegions.right.enabledWidgetIds).toEqual([
      'queue',
      'gallery',
      'layers',
      'models',
      'diagnostics',
      'project',
    ]);
    expect(getActiveProject(hydratedCustomized).widgetRegions.right.enabledWidgetIds).toEqual(['gallery', 'layers']);
  });
});

describe('workbench widget region opening', () => {
  it('enables and selects a center widget without toggling it back off', () => {
    let state = createInitialWorkbenchState();
    const activeProject = getActiveProject(state);
    const enabledWidgetIds: Project['widgetRegions']['center']['enabledWidgetIds'] = ['canvas'];

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
                  activeWidgetId: 'canvas',
                  enabledWidgetIds,
                },
              },
            }
          : project
      ),
    };

    state = workbenchReducer(state, { region: 'center', type: 'openRegionWidget', widgetId: 'models' });

    expect(getActiveProject(state).widgetRegions.center.activeWidgetId).toBe('models');
    expect(getActiveProject(state).widgetRegions.center.enabledWidgetIds).toEqual(['canvas', 'models']);

    state = workbenchReducer(state, { region: 'center', type: 'openRegionWidget', widgetId: 'models' });

    expect(getActiveProject(state).widgetRegions.center.activeWidgetId).toBe('models');
    expect(getActiveProject(state).widgetRegions.center.enabledWidgetIds).toEqual(['canvas', 'models']);
    expect(getActiveProject(state).widgetRegions.center.isCollapsed).toBe(false);
  });

  it('opens and uncollapses the target panel region', () => {
    let state = createInitialWorkbenchState();
    const activeProject = getActiveProject(state);
    const enabledWidgetIds: Project['widgetRegions']['bottom']['enabledWidgetIds'] = ['diagnostics'];

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
                  activeWidgetId: 'diagnostics',
                  enabledWidgetIds,
                  isCollapsed: true,
                },
              },
            }
          : project
      ),
    };

    state = workbenchReducer(state, { region: 'bottom', type: 'openRegionWidget', widgetId: 'queue' });

    expect(getActiveProject(state).layout.panels.isBottomOpen).toBe(true);
    expect(getActiveProject(state).widgetRegions.bottom.activeWidgetId).toBe('queue');
    expect(getActiveProject(state).widgetRegions.bottom.enabledWidgetIds).toEqual(['diagnostics', 'queue']);
    expect(getActiveProject(state).widgetRegions.bottom.isCollapsed).toBe(false);
  });
});

describe('workbenchReducer Phase 5 generation flow', () => {
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

    expect(updatedProject.canvas.stagingArea.pendingImages).toEqual([]);
    expect(updatedProject.widgetStates.gallery.values.recentImages).toEqual([
      createImage('gallery-image.png', queueItem.id),
    ]);
    expect(updatedProject.widgetStates.gallery.values.selectedImage).toEqual(
      createImage('gallery-image.png', queueItem.id)
    );
    expect(updatedProject.widgetStates.gallery.values.selectedImageName).toBe('gallery-image.png');
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

    const values = getActiveProject(state).widgetStates.gallery.values;

    expect((values.recentImages as GeneratedImageContract[]).map((image) => image.imageName)).toEqual([
      'gallery-image-2.png',
      'gallery-image-1.png',
    ]);
    expect(values.imageBoards).toBeUndefined();
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

    expect(getActiveProject(state).widgetStates.gallery.values.selectedImageName).toBe('backend-selected.png');
    expect(getActiveProject(state).widgetStates.gallery.values.selectedImage).toEqual(image);
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

    expect(state.account).toEqual({ activeLayoutPresetId: 'gallery' });
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
    const previousTokens = initial.projects.map((project) => project.widgetStates.gallery.values.galleryRefreshToken);
    const state = workbenchReducer(initial, { type: 'refreshBackendData' });

    expect(state.projects).toHaveLength(initial.projects.length);

    for (const [index, project] of state.projects.entries()) {
      expect(project.widgetStates.gallery.values.galleryRefreshToken).toBeDefined();
      expect(project.widgetStates.gallery.values.galleryRefreshToken).not.toBe(previousTokens[index]);
    }
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
});
