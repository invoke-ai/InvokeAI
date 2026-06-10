import { describe, expect, it } from 'vitest';

import type { GenerateWidgetValues, MainModelConfig } from './generation/types';
import type { GeneratedImageContract, Project, WorkbenchState } from './types';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState';

const model: MainModelConfig = {
  base: 'sdxl',
  key: 'test-model',
  name: 'Test Model',
  type: 'main',
};

const createGenerateValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  batchCount: 1,
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  height: 1024,
  model,
  modelKey: model.key,
  negativePrompt: '',
  positivePrompt: 'first prompt',
  scheduler: 'euler_a',
  seed: 123,
  shouldRandomizeSeed: false,
  steps: 30,
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
      'diagnostics',
    ]);
    expect(getActiveProject(hydratedCustomized).widgetRegions.right.enabledWidgetIds).toEqual(['gallery', 'layers']);
  });
});

describe('workbenchReducer Phase 5 generation flow', () => {
  it('routes queue results back to the originating project after the user switches projects', () => {
    let state = submitGenerate(primeGenerate());
    const originProject = getActiveProject(state);
    const queueItem = originProject.queue.items[0];

    expect(queueItem).toBeDefined();

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
    expect(firstValues.shouldRandomizeSeed).toBe(false);
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

  it('does not queue unavailable workflow/project graph sources in Phase 5', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { sourceId: 'project-graph', type: 'setInvocationSource' });

    expect(getActiveProject(state).invocation.sourceId).toBe('generate');

    state = workbenchReducer(state, {
      backendSupportsCancellation: true,
      route: { destination: 'canvas', destinationLocked: false, sourceId: 'project-graph', sourceLocked: true },
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

describe('workbench preferences', () => {
  it('defaults to the dark theme with motion enabled', () => {
    const state = createInitialWorkbenchState();

    expect(state.account.preferences).toEqual({ reduceMotion: false, showFocusRegionHighlight: true, themeId: 'dark' });
  });

  it('updates the theme without dropping other preferences', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { preferences: { reduceMotion: true }, type: 'setPreferences' });
    state = workbenchReducer(state, { preferences: { showFocusRegionHighlight: false }, type: 'setPreferences' });
    state = workbenchReducer(state, { preferences: { themeId: 'forest' }, type: 'setPreferences' });

    expect(state.account.preferences).toEqual({
      reduceMotion: true,
      showFocusRegionHighlight: false,
      themeId: 'forest',
    });
  });

  it('preserves preferences when applying a layout preset', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { preferences: { themeId: 'mono' }, type: 'setPreferences' });
    state = workbenchReducer(state, { presetId: 'gallery', type: 'applyPreset' });

    expect(state.account.activeLayoutPresetId).toBe('gallery');
    expect(state.account.preferences.themeId).toBe('mono');
  });

  it('heals hydrated state that predates preferences', () => {
    const initial = createInitialWorkbenchState();
    const legacy = {
      ...initial,
      account: { activeLayoutPresetId: initial.account.activeLayoutPresetId },
    } as unknown as WorkbenchState;

    const state = workbenchReducer(initial, { state: legacy, type: 'hydrateWorkbench' });

    expect(state.account.preferences).toEqual({ reduceMotion: false, showFocusRegionHighlight: true, themeId: 'dark' });
  });

  it('heals hydrated state with an unsupported theme id', () => {
    const initial = createInitialWorkbenchState();
    const persisted = {
      ...initial,
      account: {
        ...initial.account,
        preferences: { reduceMotion: true, showFocusRegionHighlight: false, themeId: 'sunset' },
      },
    } as unknown as WorkbenchState;

    const state = workbenchReducer(initial, { state: persisted, type: 'hydrateWorkbench' });

    expect(state.account.preferences).toEqual({
      reduceMotion: true,
      showFocusRegionHighlight: false,
      themeId: 'dark',
    });
  });

  it('rejects unsupported theme ids when updating preferences', () => {
    const state = workbenchReducer(createInitialWorkbenchState(), {
      preferences: { themeId: 'sunset' },
      type: 'setPreferences',
    } as unknown as Parameters<typeof workbenchReducer>[1]);

    expect(state.account.preferences.themeId).toBe('dark');
  });
});
