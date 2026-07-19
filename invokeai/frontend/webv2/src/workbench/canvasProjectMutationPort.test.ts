import type {
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { createBitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createEmptyCanvasDocumentV2 } from './canvasMigration';
import { createCanvasProjectMutationPort } from './canvasProjectMutationPort';
import { createWorkbenchStore } from './workbenchStore';

const createDeferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const drainUntil = async (predicate: () => boolean, maxTicks = 50): Promise<void> => {
  for (let tick = 0; tick < maxTicks && !predicate(); tick += 1) {
    await Promise.resolve();
  }
};

const layerBase = (id: string) => ({
  blendMode: 'normal' as const,
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
});

const createPaintLayer = (id: string): CanvasRasterLayerContractV2 => ({
  ...layerBase(id),
  source: { bitmap: null, type: 'paint' },
  type: 'raster',
});

const createMaskLayer = (id: string): CanvasInpaintMaskLayerContract => ({
  ...layerBase(id),
  mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
  type: 'inpaint_mask',
});

const getLayer = (
  store: ReturnType<typeof createWorkbenchStore>,
  projectId: string,
  layerId: string
): CanvasLayerContract | undefined =>
  store
    .getState()
    .projects.find((project) => project.id === projectId)
    ?.canvas.document.layers.find((layer) => layer.id === layerId);

const getPaintSource = (
  layer: CanvasLayerContract | undefined
): Extract<CanvasLayerSourceContract, { type: 'paint' }> | null => {
  if (!layer) {
    return null;
  }
  if (layer.type === 'raster' || layer.type === 'control') {
    return layer.source.type === 'paint' ? layer.source : null;
  }
  return { bitmap: layer.mask.bitmap, offset: layer.mask.offset, type: 'paint' };
};

const setupProjects = (originLayer: CanvasLayerContract, otherLayer: CanvasLayerContract) => {
  const store = createWorkbenchStore();
  const originProjectId = store.getState().activeProjectId;
  store.commands.canvas.apply(originProjectId, {
    document: createEmptyCanvasDocumentV2(),
    type: 'replaceCanvasDocument',
  });
  store.commands.canvas.apply(originProjectId, { layer: originLayer, type: 'addCanvasLayer' });
  store.commands.projects.create();
  const otherProjectId = store.getState().activeProjectId;
  store.commands.canvas.apply(otherProjectId, {
    document: createEmptyCanvasDocumentV2(),
    type: 'replaceCanvasDocument',
  });
  store.commands.canvas.apply(otherProjectId, { layer: otherLayer, type: 'addCanvasLayer' });
  store.commands.projects.switchTo(originProjectId);
  return { originProjectId, otherProjectId, store };
};

const beginUpload = (
  store: ReturnType<typeof createWorkbenchStore>,
  projectId: string,
  layerId: string,
  isMask = false,
  sourceReaders?: {
    authoritative?: (id: string) => CanvasLayerSourceContract | null;
    mirrored?: (id: string) => CanvasLayerSourceContract | null;
  }
) => {
  const uploaded = createDeferred<{ height: number; imageName: string; width: number }>();
  const uploadImage = vi.fn(() => uploaded.promise);
  const surface: RasterSurface = createTestStubRasterBackend().createSurface(8, 8);
  const port = createCanvasProjectMutationPort(store, projectId);
  const onError = vi.fn();
  const bitmapStore = createBitmapStore({
    debounceMs: 60_000,
    dispatch: port.dispatch,
    dispatchBitmap: isMask
      ? (id, bitmap, offset) =>
          port.dispatch({
            config: { layerType: 'inpaint_mask', mask: { bitmap, offset } },
            id,
            type: 'updateCanvasLayerConfig',
          })
      : undefined,
    encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
    getAuthoritativeLayerSource: sourceReaders?.authoritative,
    getLayerSource: sourceReaders?.mirrored ?? ((id) => getPaintSource(getLayer(store, projectId, id))),
    getLayerSurface: () => ({ offset: { x: 3, y: 4 }, surface }),
    hashBlob: () => Promise.resolve('pixels-hash'),
    onError,
    uploadImage,
  });
  bitmapStore.markLayerDirty(layerId);
  const flush = bitmapStore.flushPendingUploads();
  return { bitmapStore, flush, onError, uploadImage, uploaded };
};

describe('project-bound bitmap persistence', () => {
  it('finishes an upload in its origin project after switching to a project with a different layer id', async () => {
    const h = setupProjects(createPaintLayer('origin-layer'), createPaintLayer('other-layer'));
    const upload = beginUpload(h.store, h.originProjectId, 'origin-layer');
    await drainUntil(() => upload.uploadImage.mock.calls.length === 1);

    h.store.commands.projects.switchTo(h.otherProjectId);
    upload.uploaded.resolve({ height: 8, imageName: 'origin-upload.png', width: 8 });
    await upload.flush;

    expect(getPaintSource(getLayer(h.store, h.originProjectId, 'origin-layer'))?.bitmap?.imageName).toBe(
      'origin-upload.png'
    );
    expect(getPaintSource(getLayer(h.store, h.otherProjectId, 'other-layer'))?.bitmap).toBeNull();
    upload.bitmapStore.dispose();
  });

  it('does not mutate a colliding layer id in the newly active project', async () => {
    const h = setupProjects(createPaintLayer('shared-layer'), createPaintLayer('shared-layer'));
    const upload = beginUpload(h.store, h.originProjectId, 'shared-layer');
    await drainUntil(() => upload.uploadImage.mock.calls.length === 1);

    h.store.commands.projects.switchTo(h.otherProjectId);
    upload.uploaded.resolve({ height: 8, imageName: 'origin-shared.png', width: 8 });
    await upload.flush;

    expect(getPaintSource(getLayer(h.store, h.originProjectId, 'shared-layer'))?.bitmap?.imageName).toBe(
      'origin-shared.png'
    );
    expect(getPaintSource(getLayer(h.store, h.otherProjectId, 'shared-layer'))?.bitmap).toBeNull();
    upload.bitmapStore.dispose();
  });

  it('routes a mask upload back to the origin project after switching projects', async () => {
    const h = setupProjects(createMaskLayer('shared-mask'), createMaskLayer('shared-mask'));
    const upload = beginUpload(h.store, h.originProjectId, 'shared-mask', true);
    await drainUntil(() => upload.uploadImage.mock.calls.length === 1);

    h.store.commands.projects.switchTo(h.otherProjectId);
    upload.uploaded.resolve({ height: 8, imageName: 'origin-mask.png', width: 8 });
    await upload.flush;

    expect(getPaintSource(getLayer(h.store, h.originProjectId, 'shared-mask'))?.bitmap?.imageName).toBe(
      'origin-mask.png'
    );
    expect(getPaintSource(getLayer(h.store, h.otherProjectId, 'shared-mask'))?.bitmap).toBeNull();
    upload.bitmapStore.dispose();
  });

  it('accepts the pending upload after switching away from and reacquiring the origin project', async () => {
    const h = setupProjects(createPaintLayer('reacquired'), createPaintLayer('other'));
    const upload = beginUpload(h.store, h.originProjectId, 'reacquired');
    await drainUntil(() => upload.uploadImage.mock.calls.length === 1);

    h.store.commands.projects.switchTo(h.otherProjectId);
    h.store.commands.projects.switchTo(h.originProjectId);
    upload.uploaded.resolve({ height: 8, imageName: 'reacquired.png', width: 8 });
    await upload.flush;

    expect(getPaintSource(getLayer(h.store, h.originProjectId, 'reacquired'))?.bitmap?.imageName).toBe(
      'reacquired.png'
    );
    upload.bitmapStore.dispose();
  });

  it('treats a project-bound bitmap commit as accepted when a subscriber throws after the reducer lands it', async () => {
    const h = setupProjects(createPaintLayer('subscriber-throw'), createPaintLayer('other'));
    let mirroredSource = getPaintSource(getLayer(h.store, h.originProjectId, 'subscriber-throw'));
    const unsubscribe = h.store.subscribe(() => {
      throw new Error('subscriber failed after commit');
    });
    const unsubscribeMirror = h.store.subscribe(() => {
      mirroredSource = getPaintSource(getLayer(h.store, h.originProjectId, 'subscriber-throw'));
    });
    const upload = beginUpload(h.store, h.originProjectId, 'subscriber-throw', false, {
      authoritative: (id) => getPaintSource(getLayer(h.store, h.originProjectId, id)),
      mirrored: () => mirroredSource,
    });
    await drainUntil(() => upload.uploadImage.mock.calls.length === 1);

    upload.uploaded.resolve({ height: 8, imageName: 'landed-before-subscriber-throw.png', width: 8 });
    await expect(upload.flush).resolves.toBeUndefined();
    await expect(upload.bitmapStore.flushPendingUploads()).resolves.toBeUndefined();

    expect(getPaintSource(getLayer(h.store, h.originProjectId, 'subscriber-throw'))).toMatchObject({
      bitmap: { imageName: 'landed-before-subscriber-throw.png' },
      offset: { x: 3, y: 4 },
    });
    expect(upload.uploadImage).toHaveBeenCalledOnce();
    expect(upload.onError).not.toHaveBeenCalled();
    expect(mirroredSource?.bitmap).toBeNull();
    unsubscribeMirror();
    unsubscribe();
    upload.bitmapStore.dispose();
  });

  it('treats project deletion during upload as terminal removal', async () => {
    const h = setupProjects(createPaintLayer('deleted-project-layer'), createPaintLayer('survivor'));
    const upload = beginUpload(h.store, h.originProjectId, 'deleted-project-layer');
    await drainUntil(() => upload.uploadImage.mock.calls.length === 1);

    h.store.commands.projects.switchTo(h.otherProjectId);
    h.store.commands.projects.close(h.originProjectId);
    upload.uploaded.resolve({ height: 8, imageName: 'orphaned.png', width: 8 });
    await upload.flush;
    await upload.bitmapStore.flushPendingUploads();

    expect(h.store.getState().projects.some((project) => project.id === h.originProjectId)).toBe(false);
    expect(getPaintSource(getLayer(h.store, h.otherProjectId, 'survivor'))?.bitmap).toBeNull();
    expect(upload.uploadImage).toHaveBeenCalledOnce();
    upload.bitmapStore.dispose();
  });
});
