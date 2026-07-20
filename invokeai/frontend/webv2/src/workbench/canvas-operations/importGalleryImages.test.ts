import type { GalleryImage } from '@features/gallery';
import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { uploadCanvasImage } from '@workbench/canvas-operations/backend/canvasImages';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskFromImage,
  createRegionalGuidanceFromImage,
  createRegionalGuidanceLayerWithRefImage,
  DEFAULT_INPAINT_MASK_FILL,
  nextRegionalGuidanceFillColor,
} from '@workbench/widgets/layers/layerOps';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState, type WorkbenchAction } from '@workbench/workbenchState.testing';
import { describe, expect, it, vi } from 'vitest';

import { importGalleryImagesToCanvas, type GalleryCanvasImportDestination } from './importGalleryImages';

const queriesFor = (getState: () => WorkbenchState) => ({
  getProject: (projectId: string) => getState().projects.find((project) => project.id === projectId) ?? null,
  isActiveProject: (projectId: string) => getState().activeProjectId === projectId,
});

const image = (imageName: string, width = 320, height = 160): GalleryImage => ({
  boardId: 'none',
  height,
  imageCategory: 'general',
  imageName,
  imageUrl: `https://images.example/${imageName}`,
  queuedAt: '2026-01-01T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  starred: false,
  thumbnailUrl: `https://images.example/thumb/${imageName}`,
  width,
});

const withProject = (mutate?: (project: Project) => Project): { project: Project; state: WorkbenchState } => {
  const state = createInitialWorkbenchState();
  const current = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
  const document = {
    ...current.canvas.document,
    bbox: { height: 512, width: 768, x: 31, y: 47 },
    layers: [createEmptyPaintLayer('Existing', 'previous')],
    selectedLayerId: 'previous',
  };
  const base = { ...current, canvas: { ...current.canvas, document } };
  const project = mutate ? mutate(base) : base;
  return { project, state: { ...state, projects: [project] } };
};

const setModel = (project: Project, base: GenerateWidgetValues['model']['base']): Project => {
  const generate = project.widgetInstances.generate;
  if (!generate) {
    throw new Error('Expected generate widget');
  }
  const previous = getProjectWidgetValues(project, 'generate');
  const model = { base, key: 'test-model', name: 'Test Model', type: 'main' } as GenerateWidgetValues['model'];
  const values: GenerateWidgetValues = {
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
    negativePrompt: '',
    negativePromptEnabled: true,
    negativePromptHeightPx: 56,
    positivePrompt: '',
    positivePromptHeightPx: 96,
    qwen3EncoderModel: null,
    qwenVLEncoderModel: null,
    referenceImages: [],
    scheduler: 'euler_a',
    seamlessXAxis: false,
    seamlessYAxis: false,
    seed: 1,
    shouldRandomizeSeed: true,
    steps: 30,
    t5EncoderModel: null,
    vae: null,
    vaePrecision: 'fp32',
    width: 1024,
    ...previous,
    model,
    modelKey: model.key,
  };
  return {
    ...project,
    widgetInstances: {
      ...project.widgetInstances,
      generate: {
        ...generate,
        state: { ...generate.state, values: values as unknown as Record<string, unknown> },
      },
    },
  };
};

const engine = (
  projectId: string,
  canCommitStructural: CanvasEngine['layers']['canCommitStructural'] = vi.fn(() => true),
  commitStructural: CanvasEngine['layers']['commitStructural'] = vi.fn(() => true)
): CanvasEngine => ({ layers: { canCommitStructural, commitStructural }, projectId }) as unknown as CanvasEngine;

const getForwardLayers = (action: WorkbenchAction | CanvasProjectMutation): readonly CanvasLayerContract[] => {
  const mutation = action.type === 'applyCanvasProjectMutation' ? action.mutation : action;
  if (mutation.type !== 'applyCanvasLayerStackMutation' || !mutation.add) {
    throw new Error('Expected add stack mutation');
  }
  return mutation.add.layers;
};

const expectedLayer = (
  destination: Exclude<GalleryCanvasImportDestination, 'control-resized'>,
  source: GalleryImage,
  index: number,
  bbox: Project['canvas']['document']['bbox'],
  base: string | null
): CanvasLayerContract => {
  const ref = { height: source.height, imageName: source.imageName, width: source.width };
  switch (destination) {
    case 'raster':
      return {
        blendMode: 'normal',
        id: expect.any(String) as string,
        isEnabled: true,
        isLocked: false,
        name: `Layer ${index + 1}`,
        opacity: 1,
        source: { image: ref, type: 'image' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: bbox.x, y: bbox.y },
        type: 'raster',
      };
    case 'control':
      return {
        ...createControlLayer(`Control Layer ${index + 1}`, 'expected', base),
        id: expect.any(String) as string,
        source: { image: ref, type: 'image' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: bbox.x, y: bbox.y },
      };
    case 'inpaint-mask':
      return createInpaintMaskFromImage({
        fill: DEFAULT_INPAINT_MASK_FILL,
        id: expect.any(String) as string,
        image: ref,
        name: `Inpaint Mask ${index + 1}`,
        rect: bbox,
      });
    case 'regional-guidance':
      return createRegionalGuidanceFromImage({
        fill: { color: nextRegionalGuidanceFillColor(index), style: 'solid' },
        id: expect.any(String) as string,
        image: ref,
        name: `Regional Guidance ${index + 1}`,
        rect: bbox,
      });
    case 'regional-reference': {
      const layer = createRegionalGuidanceLayerWithRefImage(`Regional Guidance ${index + 1}`, index, base);
      const reference = layer.referenceImages[0]!;
      return {
        ...layer,
        id: expect.any(String) as string,
        referenceImages: [
          { ...reference, config: { ...reference.config, image: source }, id: expect.any(String) as string },
        ],
      };
    }
  }
};

describe('importGalleryImagesToCanvas', () => {
  it.each<Exclude<GalleryCanvasImportDestination, 'control-resized'>>([
    'raster',
    'control',
    'inpaint-mask',
    'regional-guidance',
    'regional-reference',
  ])('builds the %s destination contract in input order and commits one history entry', async (destination) => {
    const { project, state } = withProject((value) => setModel(value, 'z-image'));
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const canvasEngine = engine(project.id);
    const images = [image('first.png'), image('second.png', 640, 480)];

    const result = await importGalleryImagesToCanvas({
      destination,
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: canvasEngine,
      ...queriesFor(() => state),
      images,
      project,
    });

    expect(result.status).toBe('imported');
    if (result.status !== 'imported') {
      return;
    }
    expect(result.failedImageNames).toEqual([]);
    expect(dispatch).not.toHaveBeenCalled();
    expect(canvasEngine.layers.commitStructural).toHaveBeenCalledOnce();
    const [, forward, inverse] = vi.mocked(canvasEngine.layers.commitStructural).mock.calls[0]!;
    const layers = getForwardLayers(forward);
    expect(layers).toEqual(
      images.map((source, index) => expectedLayer(destination, source, index, project.canvas.document.bbox, 'z-image'))
    );
    expect(result.layerIds).toEqual(layers.map((layer) => layer.id));
    expect(forward).toMatchObject({
      add: { index: 0 },
      enabledUpdates: [],
      selectedLayerId: layers[1]!.id,
      type: 'applyCanvasLayerStackMutation',
    });
    expect(inverse).toEqual({
      enabledUpdates: [],
      removeIds: layers.map((layer) => layer.id),
      selectedLayerId: 'previous',
      type: 'applyCanvasLayerStackMutation',
    });
  });

  it('fetches and uploads each resized control once with optimal model dimensions and durable hidden policy', async () => {
    const { project, state } = withProject((value) => setModel(value, 'flux2'));
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const fetchImage = vi.fn<typeof fetch>(() => Promise.resolve(new Response(new Blob(['pixels']), { status: 200 })));
    const uploadImage = vi.fn<typeof uploadCanvasImage>((_blob, options) =>
      Promise.resolve({
        height: options!.resizeTo!.height,
        imageName: `resized-${options!.fileName}`,
        width: options!.resizeTo!.width,
      })
    );

    const result = await importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: null,
      fetchImage,
      ...queriesFor(() => state),
      images: [image('wide.png', 1600, 900), image('square.png', 400, 400)],
      project,
      uploadImage,
    });

    expect(result.status).toBe('imported');
    expect(fetchImage.mock.calls.map(([url]) => url)).toEqual([
      'https://images.example/wide.png',
      'https://images.example/square.png',
    ]);
    expect(uploadImage).toHaveBeenCalledTimes(2);
    expect(uploadImage.mock.calls.map(([, options]) => options)).toEqual([
      expect.objectContaining({
        fileName: 'wide.png',
        imageCategory: 'other',
        isIntermediate: false,
        resizeTo: { height: 768, width: 1360 },
      }),
      expect.objectContaining({
        fileName: 'square.png',
        imageCategory: 'other',
        isIntermediate: false,
        resizeTo: { height: 1024, width: 1024 },
      }),
    ]);
    expect(dispatch).toHaveBeenCalledOnce();
    const forward = dispatch.mock.calls[0]![0];
    const layers = getForwardLayers(forward);
    expect(layers).toEqual([
      {
        ...createControlLayer('Control Layer 1', 'expected', 'flux2'),
        id: expect.any(String),
        source: { image: { height: 768, imageName: 'resized-wide.png', width: 1360 }, type: 'image' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 31, y: 47 },
      },
      {
        ...createControlLayer('Control Layer 2', 'expected', 'flux2'),
        id: expect.any(String),
        source: { image: { height: 1024, imageName: 'resized-square.png', width: 1024 }, type: 'image' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 31, y: 47 },
      },
    ]);
    expect(
      layers.map((layer) => (layer.type === 'control' && layer.source.type === 'image' ? layer.source.image : null))
    ).toEqual([
      { height: 768, imageName: 'resized-wide.png', width: 1360 },
      { height: 1024, imageName: 'resized-square.png', width: 1024 },
    ]);
    expect(forward).toMatchObject({
      mutation: {
        selectedLayerId: layers[1]!.id,
        type: 'applyCanvasLayerStackMutation',
      },
      projectId: project.id,
      type: 'applyCanvasProjectMutation',
    });
  });

  it('returns empty without committing', async () => {
    const { project, state } = withProject();
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const result = await importGalleryImagesToCanvas({
      destination: 'raster',
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: null,
      ...queriesFor(() => state),
      images: [],
      project,
    });
    expect(result).toEqual({ status: 'empty' });
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('blocks a locked matching engine before resized preprocessing', async () => {
    const { project, state } = withProject();
    const fetchImage = vi.fn<typeof fetch>();
    const uploadImage = vi.fn<typeof uploadCanvasImage>();
    const result = await importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: () => undefined,
      engine: engine(
        project.id,
        vi.fn(() => false)
      ),
      fetchImage,
      ...queriesFor(() => state),
      images: [image('a.png')],
      project,
      uploadImage,
    });
    expect(result).toEqual({ status: 'blocked' });
    expect(fetchImage).not.toHaveBeenCalled();
    expect(uploadImage).not.toHaveBeenCalled();
  });

  it('returns blocked when the matching engine locks during preprocessing', async () => {
    const { project, state } = withProject();
    const fetchImage = vi.fn<typeof fetch>(() => Promise.resolve(new Response(new Blob(['pixels']), { status: 200 })));
    const uploadImage = vi.fn<typeof uploadCanvasImage>(() =>
      Promise.resolve({ height: 512, imageName: 'resized', width: 512 })
    );
    const commitStructural = vi.fn<
      (_label: string, _forward: CanvasProjectMutation, _inverse: CanvasProjectMutation) => boolean
    >(() => false);
    const result = await importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: () => undefined,
      engine: engine(
        project.id,
        vi.fn(() => true),
        commitStructural
      ),
      fetchImage,
      ...queriesFor(() => state),
      images: [image('a.png')],
      project,
      uploadImage,
    });
    expect(result).toEqual({ status: 'blocked' });
    expect(commitStructural).toHaveBeenCalledOnce();
    const [, forward, inverse] = commitStructural.mock.calls[0]!;
    const layers = getForwardLayers(forward);
    expect(inverse).toEqual({
      enabledUpdates: [],
      removeIds: layers.map((layer) => layer.id),
      selectedLayerId: 'previous',
      type: 'applyCanvasLayerStackMutation',
    });
  });

  it('returns stale-project when the target is deleted during preprocessing', async () => {
    const { project, state } = withProject();
    let current = state;
    const uploadImage = vi.fn<typeof uploadCanvasImage>(() => {
      current = { ...current, projects: [] };
      return Promise.resolve({ height: 512, imageName: 'resized', width: 512 });
    });
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const result = await importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: null,
      fetchImage: () => Promise.resolve(new Response(new Blob(['pixels']), { status: 200 })),
      ...queriesFor(() => current),
      images: [image('a.png')],
      project,
      uploadImage,
    });
    expect(result).toEqual({ status: 'stale-project' });
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('dispatches to the captured project when the active project changes during preprocessing', async () => {
    const { project, state } = withProject();
    const other = { ...project, id: 'other-project' };
    let current = { ...state, projects: [project, other] };
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const matchingEngine = engine(project.id);
    const uploadImage = vi.fn<typeof uploadCanvasImage>(() => {
      current = { ...current, activeProjectId: other.id };
      return Promise.resolve({ height: 512, imageName: 'resized', width: 512 });
    });
    const result = await importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: matchingEngine,
      fetchImage: () => Promise.resolve(new Response(new Blob(['pixels']), { status: 200 })),
      ...queriesFor(() => current),
      images: [image('a.png')],
      project,
      uploadImage,
    });
    expect(result.status).toBe('imported');
    expect(matchingEngine.layers.commitStructural).not.toHaveBeenCalled();
    expect(dispatch).toHaveBeenCalledOnce();
    expect(dispatch.mock.calls[0]![0]).toMatchObject({ projectId: project.id });
  });

  it('uses the matching engine final guard when the target project becomes active during preprocessing', async () => {
    const { project, state } = withProject();
    const other = { ...project, id: 'other-project' };
    let current = { ...state, activeProjectId: other.id, projects: [project, other] };
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const commitStructural = vi.fn(() => false);
    const matchingEngine = engine(
      project.id,
      vi.fn(() => true),
      commitStructural
    );
    const uploadImage = vi.fn<typeof uploadCanvasImage>(() => {
      current = { ...current, activeProjectId: project.id };
      return Promise.resolve({ height: 512, imageName: 'resized', width: 512 });
    });

    const result = await importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: matchingEngine,
      fetchImage: () => Promise.resolve(new Response(new Blob(['pixels']), { status: 200 })),
      ...queriesFor(() => current),
      images: [image('a.png')],
      project,
      uploadImage,
    });

    expect(result).toEqual({ status: 'blocked' });
    expect(commitStructural).toHaveBeenCalledOnce();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('returns stale-document when the captured target document identity changes', async () => {
    const { project, state } = withProject();
    let current = state;
    const uploadImage = vi.fn<typeof uploadCanvasImage>(() => {
      current = {
        ...current,
        projects: current.projects.map((candidate) =>
          candidate.id === project.id
            ? { ...candidate, canvas: { ...candidate.canvas, document: { ...candidate.canvas.document } } }
            : candidate
        ),
      };
      return Promise.resolve({ height: 512, imageName: 'resized', width: 512 });
    });
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const result = await importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: null,
      fetchImage: () => Promise.resolve(new Response(new Blob(['pixels']), { status: 200 })),
      ...queriesFor(() => current),
      images: [image('a.png')],
      project,
      uploadImage,
    });
    expect(result).toEqual({ status: 'stale-document' });
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('single-flights imports per project while allowing other projects', async () => {
    const first = withProject();
    const secondProject = { ...first.project, id: 'second-project' };
    const secondState = { ...first.state, activeProjectId: secondProject.id, projects: [secondProject] };
    let release!: () => void;
    const pending = new Promise<void>((resolve) => {
      release = resolve;
    });
    const fetchImage = vi.fn<typeof fetch>(async () => {
      await pending;
      return new Response(new Blob(['pixels']), { status: 200 });
    });
    const uploadImage = vi.fn<typeof uploadCanvasImage>(() =>
      Promise.resolve({ height: 512, imageName: 'resized', width: 512 })
    );
    const firstImport = importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: () => undefined,
      engine: null,
      fetchImage,
      ...queriesFor(() => first.state),
      images: [image('a.png')],
      project: first.project,
      uploadImage,
    });
    const overlap = await importGalleryImagesToCanvas({
      destination: 'raster',
      applyCanvasMutation: () => undefined,
      engine: null,
      ...queriesFor(() => first.state),
      images: [image('b.png')],
      project: first.project,
    });
    const independent = await importGalleryImagesToCanvas({
      destination: 'raster',
      applyCanvasMutation: () => undefined,
      engine: null,
      ...queriesFor(() => secondState),
      images: [image('c.png')],
      project: secondProject,
    });
    expect(overlap).toEqual({ status: 'blocked' });
    expect(independent.status).toBe('imported');
    release();
    await firstImport;
  });

  it('bounds resized work at four, preserves successful order, and reports partial failures once', async () => {
    const { project, state } = withProject();
    let active = 0;
    let maximum = 0;
    const releases: Array<() => void> = [];
    const fetchImage = vi.fn<typeof fetch>(async (_input) => {
      active += 1;
      maximum = Math.max(maximum, active);
      await new Promise<void>((resolve) => {
        releases.push(resolve);
      });
      active -= 1;
      return new Response(new Blob(['pixels']), { status: 200 });
    });
    const uploadImage = vi.fn<typeof uploadCanvasImage>((_blob, options) => {
      if (options!.fileName === '2.png' || options!.fileName === '5.png') {
        return Promise.reject(new Error('resize failed'));
      }
      return Promise.resolve({ height: 512, imageName: `ok-${options!.fileName}`, width: 512 });
    });
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const importing = importGalleryImagesToCanvas({
      destination: 'control-resized',
      applyCanvasMutation: (projectId, mutation) =>
        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' }),
      engine: null,
      fetchImage,
      ...queriesFor(() => state),
      images: Array.from({ length: 7 }, (_, index) => image(`${index}.png`)),
      project,
      uploadImage,
    });
    await vi.waitFor(() => expect(fetchImage).toHaveBeenCalledTimes(4));
    expect(maximum).toBe(4);
    releases.splice(0).forEach((resolve) => resolve());
    await vi.waitFor(() => expect(fetchImage).toHaveBeenCalledTimes(7));
    releases.splice(0).forEach((resolve) => resolve());
    const result = await importing;
    expect(result.status).toBe('imported');
    if (result.status !== 'imported') {
      return;
    }
    expect(result.failedImageNames).toEqual(['2.png', '5.png']);
    expect(dispatch).toHaveBeenCalledOnce();
    expect(
      getForwardLayers(dispatch.mock.calls[0]![0]).map((layer) =>
        layer.type === 'control' && layer.source.type === 'image' ? layer.source.image.imageName : null
      )
    ).toEqual(['ok-0.png', 'ok-1.png', 'ok-3.png', 'ok-4.png', 'ok-6.png']);
  });

  it('throws one AggregateError when every resized image fails and releases the project flight', async () => {
    const { project, state } = withProject();
    const options = {
      destination: 'control-resized' as const,
      applyCanvasMutation: () => undefined,
      engine: null,
      fetchImage: () => Promise.resolve(new Response(new Blob(['pixels']), { status: 200 })),
      ...queriesFor(() => state),
      images: [image('a.png'), image('b.png')],
      project,
      uploadImage: vi.fn<typeof uploadCanvasImage>(() => Promise.reject(new Error('failed'))),
    };
    await expect(importGalleryImagesToCanvas(options)).rejects.toBeInstanceOf(AggregateError);
    await expect(importGalleryImagesToCanvas({ ...options, destination: 'raster' })).resolves.toMatchObject({
      status: 'imported',
    });
  });
});
