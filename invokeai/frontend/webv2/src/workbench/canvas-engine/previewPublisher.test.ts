import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { LayerFilterOutputDimensionError } from '@workbench/canvas-engine/filterError';
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { PreviewStateController } from './controllers/previewStateController';
import { createPreviewPublisher, type CreatePreviewPublisherDeps, type PreviewPublisher } from './previewPublisher';

const surface = (): RasterSurface => ({}) as RasterSurface;

const guardFor = (layerId: string): LayerExportGuard =>
  ({ cacheVersion: 1, documentGeneration: 1, layerId, projectId: 'p' }) as LayerExportGuard;

const documentWith = (...layerIds: string[]): CanvasDocumentContractV2 =>
  ({ layers: layerIds.map((id) => ({ id })) }) as CanvasDocumentContractV2;

/** The real state controller — it has no collaborators, so faking it would only weaken the assertions. */
let previews: PreviewStateController;
let invalidateAll: Mock<() => void>;
let invalidateLayer: Mock<(layerId: string) => void>;
let decodeBlob: Mock<CreatePreviewPublisherDeps['decodeBlob']>;
let resolveImage: Mock<(imageName: string) => Promise<Blob>>;
let getDocument: () => CanvasDocumentContractV2 | null;
let isGuardCurrent: Mock<(guard: LayerExportGuard) => boolean>;

const build = (overrides: Partial<CreatePreviewPublisherDeps> = {}): PreviewPublisher =>
  createPreviewPublisher({
    decodeBlob,
    getDocument: () => getDocument(),
    invalidateAll,
    invalidateLayer,
    isGuardCurrent,
    previews,
    resolveImage,
    ...overrides,
  });

beforeEach(() => {
  previews = new PreviewStateController();
  invalidateAll = vi.fn();
  invalidateLayer = vi.fn();
  resolveImage = vi.fn(() => Promise.resolve(new Blob(['x'])));
  decodeBlob = vi.fn(() => Promise.resolve({ decodedHeight: 8, decodedWidth: 8, surface: surface() }));
  getDocument = () => documentWith('layer-1');
  isGuardCurrent = vi.fn(() => true);
});

describe('staged previews', () => {
  it('prefetches unseen candidate bytes so first selection does not start another full-image request', async () => {
    const publisher = build();

    publisher.preloadStagedPreview('next.png');
    await vi.waitFor(() => expect(resolveImage).toHaveBeenCalledOnce());
    expect(decodeBlob).not.toHaveBeenCalled();

    publisher.setStagedPreview({ imageName: 'next.png' });
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());

    expect(resolveImage).toHaveBeenCalledOnce();
    expect(decodeBlob).toHaveBeenCalledOnce();
  });

  it('coalesces repeated in-flight decodes during rapid candidate cycling', async () => {
    const releases: Array<() => void> = [];
    decodeBlob.mockImplementation(
      () =>
        new Promise((resolve) => {
          releases.push(() => resolve({ decodedHeight: 8, decodedWidth: 8, surface: surface() }));
        })
    );
    const publisher = build();
    publisher.setStagedPreview({ imageName: 'a.png' });
    await vi.waitFor(() => expect(decodeBlob).toHaveBeenCalledTimes(1));
    publisher.setStagedPreview({ imageName: 'b.png' });
    await vi.waitFor(() => expect(decodeBlob).toHaveBeenCalledTimes(2));
    publisher.setStagedPreview({ imageName: 'a.png' });
    await Promise.resolve();

    expect(decodeBlob).toHaveBeenCalledTimes(2);
    expect(resolveImage).toHaveBeenCalledTimes(2);

    releases[0]?.();
    releases[1]?.();
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());

    expect(previews.getStaged()).toMatchObject({ height: 8, width: 8 });
  });

  it('decodes an image-name input and publishes it', async () => {
    build().setStagedPreview({ imageName: 'staged.png' });
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());
    expect(resolveImage).toHaveBeenCalledWith('staged.png', expect.any(AbortSignal));
    expect(previews.getStaged()).toMatchObject({ height: 8, width: 8 });
    expect(invalidateAll).toHaveBeenCalled();
  });

  it('copies the placement rather than aliasing the callers object', async () => {
    const placement = { height: 4, opacity: 0.5, width: 4, x: 0, y: 0 };
    build().setStagedPreview({ imageName: 'staged.png', placement });
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());
    placement.opacity = 1;
    expect(previews.getStaged()?.placement).toMatchObject({ opacity: 0.5 });
  });

  it('decodes a base64 data URL at the requested size through the raster seam', async () => {
    build().setStagedPreview({ dataUrl: `data:image/png;base64,${btoa('pixels')}`, height: 3, width: 5 });
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());
    const [blob, dimensions] = decodeBlob.mock.calls[0] ?? [];
    expect(blob?.type).toBe('image/png');
    await expect(blob?.text()).resolves.toBe('pixels');
    expect(dimensions).toEqual({ height: 3, scale: true, width: 5 });
    // The declared size wins over whatever the decode reported.
    expect(previews.getStaged()).toMatchObject({ height: 3, width: 5 });
  });

  it('discards a decode that a newer set superseded while it was in flight', async () => {
    let release: (() => void) | undefined;
    decodeBlob.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve({ decodedHeight: 1, decodedWidth: 1, surface: surface() });
        })
    );
    const publisher = build();
    publisher.setStagedPreview({ imageName: 'first.png' });
    await vi.waitFor(() => expect(decodeBlob).toHaveBeenCalledOnce());
    publisher.setStagedPreview({ imageName: 'second.png' });
    release?.();
    await vi.waitFor(() => expect(previews.getStaged()).toMatchObject({ width: 8 }));
    expect(previews.getStaged()).not.toMatchObject({ width: 1 });
  });

  it('does not publish a decode that was invalidated while in flight', async () => {
    let release: (() => void) | undefined;
    decodeBlob.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve({ decodedHeight: 8, decodedWidth: 8, surface: surface() });
        })
    );
    const publisher = build();
    publisher.setStagedPreview({ imageName: 'cooling.png' });
    await vi.waitFor(() => expect(decodeBlob).toHaveBeenCalledOnce());

    publisher.clearStagedPreview();
    release?.();
    await Promise.resolve();
    await Promise.resolve();

    expect(previews.getStaged()).toBeNull();
  });

  it('drops in-flight decode coalescing state when the cache is cleared', async () => {
    const releases: Array<() => void> = [];
    decodeBlob.mockImplementation(
      () =>
        new Promise((resolve) => {
          releases.push(() => resolve({ decodedHeight: 8, decodedWidth: 8, surface: surface() }));
        })
    );
    const publisher = build();
    publisher.setStagedPreview({ imageName: 'candidate.png' });
    await vi.waitFor(() => expect(decodeBlob).toHaveBeenCalledOnce());

    publisher.clearStagedPreview();
    publisher.clearStagedPreviewCache();
    publisher.setStagedPreview({ imageName: 'candidate.png' });
    await vi.waitFor(() => expect(decodeBlob).toHaveBeenCalledTimes(2));

    releases.forEach((release) => release());
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());
    expect(resolveImage).toHaveBeenCalledTimes(2);
  });

  it('leaves a prior preview standing when a later decode fails', async () => {
    const publisher = build();
    publisher.setStagedPreview({ imageName: 'good.png' });
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());
    decodeBlob.mockRejectedValueOnce(new Error('decode blew up'));
    publisher.setStagedPreview({ imageName: 'bad.png' });
    await vi.waitFor(() => expect(decodeBlob).toHaveBeenCalledTimes(2));
    expect(previews.getStaged()).not.toBeNull();
  });

  it('clears on a null input, and only repaints when something was showing', async () => {
    const publisher = build();
    publisher.setStagedPreview(null);
    expect(invalidateAll).not.toHaveBeenCalled();

    publisher.setStagedPreview({ imageName: 'staged.png' });
    await vi.waitFor(() => expect(previews.getStaged()).not.toBeNull());
    invalidateAll.mockClear();

    publisher.setStagedPreview(null);
    expect(previews.getStaged()).toBeNull();
    expect(invalidateAll).toHaveBeenCalledTimes(1);
  });
});

describe('setGuardedFilterPreview', () => {
  const input = { imageName: 'filtered.png', rect: { height: 8, width: 8, x: 0, y: 0 } };

  it('publishes the decoded surface against the layer and repaints just that layer', async () => {
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'))).resolves.toBe('shown');
    expect(previews.getFilter('layer-1')).toMatchObject({ rect: input.rect });
    expect(invalidateLayer).toHaveBeenCalledWith('layer-1');
  });

  it('copies the rect so a later caller mutation cannot move the published preview', async () => {
    const rect = { height: 8, width: 8, x: 0, y: 0 };
    await build().setGuardedFilterPreview('layer-1', { imageName: 'f.png', rect }, guardFor('layer-1'));
    rect.x = 99;
    expect(previews.getFilter('layer-1')?.rect).toMatchObject({ x: 0 });
  });

  it('reports missing when the layer is gone, without decoding', async () => {
    getDocument = () => documentWith('other');
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'))).resolves.toBe('missing');
    expect(decodeBlob).not.toHaveBeenCalled();
  });

  it('reports stale when the guard names a different layer', async () => {
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-2'))).resolves.toBe('stale');
    expect(decodeBlob).not.toHaveBeenCalled();
  });

  it('reports stale when the guard has been outdated by an edit', async () => {
    isGuardCurrent = vi.fn(() => false);
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'))).resolves.toBe('stale');
  });

  it('re-checks the guard after the decode, not only before it', async () => {
    // Current on the pre-decode check, outdated by the time the image lands.
    isGuardCurrent = vi.fn().mockReturnValueOnce(true).mockReturnValue(false);
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'))).resolves.toBe('stale');
    expect(previews.getFilter('layer-1')).toBeUndefined();
    expect(invalidateLayer).not.toHaveBeenCalled();
  });

  it('discards a decode that a newer request for the same layer superseded', async () => {
    decodeBlob.mockImplementationOnce(() => {
      // Stand in for a competing publish landing while this decode is in flight.
      previews.beginGuardedFilter('layer-1');
      return Promise.resolve({ decodedHeight: 8, decodedWidth: 8, surface: surface() });
    });
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'))).resolves.toBe('stale');
    expect(previews.getFilter('layer-1')).toBeUndefined();
  });

  it('throws when a typed filter returns dimensions that do not describe the rect', async () => {
    decodeBlob.mockResolvedValueOnce({ decodedHeight: 4, decodedWidth: 4, surface: surface() });
    await expect(
      build().setGuardedFilterPreview('layer-1', { ...input, filterType: 'canny' }, guardFor('layer-1'))
    ).rejects.toBeInstanceOf(LayerFilterOutputDimensionError);
    expect(previews.getFilter('layer-1')).toBeUndefined();
  });

  it('does not police dimensions when no filter type was declared', async () => {
    decodeBlob.mockResolvedValueOnce({ decodedHeight: 4, decodedWidth: 4, surface: surface() });
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'))).resolves.toBe('shown');
  });

  it('reports stale rather than throwing on a transient decode failure', async () => {
    decodeBlob.mockRejectedValueOnce(new Error('network'));
    await expect(build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'))).resolves.toBe('stale');
  });

  it.each([
    ['a pre-decode refusal', () => (getDocument = () => documentWith('other'))],
    ['a decode failure', () => decodeBlob.mockRejectedValueOnce(new Error('network'))],
  ])('releases the guarded-request slot after %s', async (_label, arrange) => {
    arrange();
    await build().setGuardedFilterPreview('layer-1', input, guardFor('layer-1'));
    // A released slot leaves only the bumped token behind, never a pending request.
    previews.clearFilter('layer-1');
    expect(previews.filterLayerIds()).toEqual(['layer-1']);
  });
});

describe('clearing filter previews', () => {
  const publishOne = async (publisher: PreviewPublisher, layerId: string) => {
    getDocument = () => documentWith(layerId);
    await publisher.setGuardedFilterPreview(
      layerId,
      { imageName: 'f.png', rect: { height: 8, width: 8, x: 0, y: 0 } },
      guardFor(layerId)
    );
  };

  it('repaints only when the layer actually had a preview', async () => {
    const publisher = build();
    publisher.clearFilterPreview('layer-1');
    expect(invalidateLayer).not.toHaveBeenCalled();

    await publishOne(publisher, 'layer-1');
    invalidateLayer.mockClear();
    publisher.clearFilterPreview('layer-1');
    expect(previews.getFilter('layer-1')).toBeUndefined();
    expect(invalidateLayer).toHaveBeenCalledWith('layer-1');
  });

  it('drops every layers preview on a wholesale clear', async () => {
    const publisher = build();
    await publishOne(publisher, 'layer-1');
    await publishOne(publisher, 'layer-2');
    publisher.clearAllFilterPreviews();
    expect(previews.getFilter('layer-1')).toBeUndefined();
    expect(previews.getFilter('layer-2')).toBeUndefined();
  });
});
