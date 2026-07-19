import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStateContractV2,
} from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import type { ComposeForGenerationOptions, GenerationCompositeHost, GenerationModeFacts } from './generationComposite';

import { createCompositeDedupeCache } from './compositeForGeneration';
import { composeForGeneration } from './generationComposite';

const rasterLayer = (id: string, size = 64): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: size, imageName: `${id}.png`, width: size }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const controlLayer = (id: string): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 0.75 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 64, imageName: `${id}.png`, width: 64 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

const regionalLayer = (id: string): CanvasLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: { height: 64, imageName: `${id}-bmp`, width: 64 }, fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  negativePrompt: null,
  opacity: 1,
  positivePrompt: 'region',
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

const inpaintMaskLayer = (id: string, noiseLevel?: number): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: { height: 64, imageName: `${id}-bmp`, width: 64 }, fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  ...(noiseLevel !== undefined ? { noiseLevel } : {}),
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const makeDoc = (layers: CanvasLayerContract[], size = 64): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: size, width: size, x: 0, y: 0 },
  height: size,
  layers,
  selectedLayerId: null,
  version: 2,
  width: size,
});

const makeCanvas = (document: CanvasDocumentContractV2): CanvasStateContractV2 => ({
  document: structuredClone(document),
  documentRevision: 0,
  snapshots: [],
  stagingArea: {
    areThumbnailsVisible: true,
    autoSwitchMode: 'off',
    isVisible: false,
    pendingImageIds: [],
    pendingImages: [],
    selectedImageIndex: 0,
  },
  version: 2,
});

/** Uniform-alpha ImageData (255 = fully opaque → bboxFullyCovered true). */
const uniformImageData = (width: number, height: number, alpha: number): ImageData => {
  const data = new Uint8ClampedArray(Math.max(1, width) * Math.max(1, height) * 4);
  for (let i = 3; i < data.length; i += 4) {
    data[i] = alpha;
  }
  return { colorSpace: 'srgb', data, height, width } as unknown as ImageData;
};

interface HostHarness {
  host: GenerationCompositeHost;
  uploadImage: ReturnType<typeof vi.fn>;
  release: ReturnType<typeof vi.fn>;
  /** Ordered log of `upload:<name>` entries (tests may push their own markers). */
  events: string[];
  /** Layer ids requested from the raster snapshot, in request order. */
  surfaceIds: string[];
}

interface HostOptions {
  alpha?: number;
  uploadImage?: (blob: Blob) => Promise<CanvasImageUploadResult>;
}

/**
 * Assembles a fake {@link GenerationCompositeHost} mirroring the widget invoke
 * harness: the raster test stub for surfaces, a counting uploader, a spy-able
 * release, and a per-call-unique hash so every distinct composite entry uploads
 * (plan-key dedupe still reuses across calls).
 */
const makeHost = (document: CanvasDocumentContractV2, options: HostOptions = {}): HostHarness => {
  const stub = createTestStubRasterBackend();
  const events: string[] = [];
  const surfaceIds: string[] = [];
  const alpha = options.alpha ?? 255;
  const release = vi.fn();
  const dedupe = createCompositeDedupeCache();

  let uploadCounter = 0;
  const uploadImage = vi.fn(
    options.uploadImage ??
      ((blob: Blob): Promise<CanvasImageUploadResult> => {
        void blob;
        uploadCounter += 1;
        events.push(`upload:composite-${uploadCounter}.png`);
        return Promise.resolve({ height: 64, imageName: `composite-${uploadCounter}.png`, width: 64 });
      })
  );

  let hashCounter = 0;
  const layerSurfaces = new Map<string, RasterSurface>();

  const host: GenerationCompositeHost = {
    captureDocumentSnapshot: () => ({ canvas: makeCanvas(document), documentGeneration: 0 }),
    captureRasterSnapshot: (documentSnapshot, layerIds) => {
      const detached = new Map<string, { rect: Rect; surface: RasterSurface }>();
      for (const layerId of layerIds) {
        surfaceIds.push(layerId);
        let surface = layerSurfaces.get(layerId);
        if (!surface) {
          surface = stub.createSurface(64, 64);
          layerSurfaces.set(layerId, surface);
        }
        detached.set(layerId, { rect: { height: 64, width: 64, x: 0, y: 0 }, surface });
      }
      return Promise.resolve({
        snapshot: {
          canvas: documentSnapshot.canvas,
          documentGeneration: documentSnapshot.documentGeneration,
          emptyLayerIds: new Set<string>(),
          layerSurfaces: detached,
          release,
        },
        status: 'ok',
      });
    },
    dedupe,
    getCompositeExecutorDeps: () => ({
      backend: {
        createSurface: (w: number, h: number): RasterSurface => stub.createSurface(w, h),
        encodeSurface: (surface: RasterSurface): Promise<Blob> => stub.encodeSurface(surface),
      },
      // Unique per call: byHash never collapses distinct entries (the stub
      // encodes size-only blobs), so upload counts stay per-entry; cross-call
      // reuse goes through the plan-key cache.
      hashBlob: () => {
        hashCounter += 1;
        return Promise.resolve(`hash-${hashCounter}`);
      },
      readImageData: (_surface, rect) => uniformImageData(rect.width, rect.height, alpha),
      uploadImage: uploadImage as (blob: Blob) => Promise<CanvasImageUploadResult>,
    }),
  };

  return { events, host, release, surfaceIds, uploadImage };
};

const compose = (host: GenerationCompositeHost, overrides: Partial<ComposeForGenerationOptions> = {}) =>
  composeForGeneration(host, {
    detectMode: () => 'img2img',
    signal: new AbortController().signal,
    ...overrides,
  });

describe('composeForGeneration', () => {
  it('returns no-document when the host has no active document', async () => {
    const harness = makeHost(makeDoc([rasterLayer('layer-a')]));
    harness.host.captureDocumentSnapshot = () => null;

    const result = await compose(harness.host);

    expect(result).toEqual({ status: 'no-document' });
    expect(harness.uploadImage).not.toHaveBeenCalled();
    expect(harness.release).not.toHaveBeenCalled();
  });

  it.each(['stale', 'aborted', 'not-ready', 'over-budget'] as const)(
    'passes the raster-capture %s status through without compositing',
    async (status) => {
      const harness = makeHost(makeDoc([rasterLayer('layer-a')]));
      harness.host.captureRasterSnapshot = vi.fn(() => Promise.resolve({ status }));
      const detectMode = vi.fn(() => 'img2img' as const);

      const result = await compose(harness.host, { detectMode });

      expect(result).toEqual({ status });
      expect(detectMode).not.toHaveBeenCalled();
      expect(harness.uploadImage).not.toHaveBeenCalled();
      expect(harness.host.dedupe.byKey.size).toBe(0);
      expect(harness.host.dedupe.byHash.size).toBe(0);
    }
  );

  it('forwards the abort signal to raster capture', async () => {
    const harness = makeHost(makeDoc([rasterLayer('layer-a')]));
    const captureRasterSnapshot = vi.fn(() => Promise.resolve({ status: 'aborted' as const }));
    harness.host.captureRasterSnapshot = captureRasterSnapshot;
    const controller = new AbortController();

    await compose(harness.host, { signal: controller.signal });

    expect(captureRasterSnapshot).toHaveBeenCalledWith(
      expect.any(Object),
      ['layer-a'],
      expect.objectContaining({ signal: controller.signal })
    );
  });

  it('resolves txt2img from the bounds pre-pass: no base upload, detectMode never consulted, controls and regionals still composited', async () => {
    // No raster layers → no content bounds → txt2img without executing the base
    // plan; control + regional composites are mode-independent and still run.
    const harness = makeHost(makeDoc([controlLayer('control-a'), regionalLayer('region-a')]));
    const detectMode = vi.fn(() => 'img2img' as const);

    const result = await compose(harness.host, { detectMode });

    expect(detectMode).not.toHaveBeenCalled();
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    expect(result.composites.mode).toBe('txt2img');
    expect(result.composites.baseImageName).toBeNull();
    expect(result.composites.maskImageName).toBeNull();
    expect(result.composites.noiseMaskImageName).toBeNull();
    expect(result.composites.controlImages).toEqual([{ imageName: 'composite-1.png', layerId: 'control-a' }]);
    expect(result.composites.regionalMaskImages).toEqual([{ imageName: 'composite-2.png', layerId: 'region-a' }]);
    expect(harness.uploadImage).toHaveBeenCalledTimes(2);
  });

  it('consults detectMode exactly once with the composite facts when content overlaps the bbox', async () => {
    const harness = makeHost(makeDoc([rasterLayer('base')]));
    const detectMode = vi.fn(() => 'img2img' as const);

    const result = await compose(harness.host, { detectMode });

    expect(detectMode).toHaveBeenCalledTimes(1);
    expect(detectMode).toHaveBeenCalledWith({
      bbox: { height: 64, width: 64, x: 0, y: 0 },
      bboxFullyCovered: true,
      contentBounds: { height: 64, width: 64, x: 0, y: 0 },
      hasActiveInpaintMask: false,
    });
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    expect(result.composites.mode).toBe('img2img');
    expect(result.composites.baseImageName).toBe('composite-1.png');
    expect(result.composites.canvas.document.layers[0]?.id).toBe('base');
    expect(result.composites.bbox).toEqual({ height: 64, width: 64, x: 0, y: 0 });
  });

  it('executes the inpaint mask before consulting detectMode and reports its coverage', async () => {
    const harness = makeHost(makeDoc([rasterLayer('base'), inpaintMaskLayer('mask')]));
    const detectMode = vi.fn((facts: GenerationModeFacts) => {
      harness.events.push('detectMode');
      return facts.hasActiveInpaintMask ? ('inpaint' as const) : ('img2img' as const);
    });

    const result = await compose(harness.host, { detectMode });

    // Base upload, then the inpaint-mask upload, then (and only then) the mode
    // strategy — the mask's coverage is one of its inputs.
    expect(harness.events).toEqual(['upload:composite-1.png', 'upload:composite-2.png', 'detectMode']);
    expect(detectMode).toHaveBeenCalledWith(expect.objectContaining({ hasActiveInpaintMask: true }));
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    expect(result.composites.mode).toBe('inpaint');
    expect(result.composites.maskImageName).toBe('composite-2.png');
  });

  it.each([
    { expectedMask: null, expectedNoise: null, expectedUploads: 2, mode: 'img2img' as const },
    { expectedMask: 'composite-2.png', expectedNoise: 'composite-3.png', expectedUploads: 3, mode: 'inpaint' as const },
    {
      expectedMask: 'composite-2.png',
      expectedNoise: 'composite-3.png',
      expectedUploads: 3,
      mode: 'outpaint' as const,
    },
  ])(
    'composites the noise mask only for inpaint/outpaint ($mode)',
    async ({ expectedMask, expectedNoise, expectedUploads, mode }) => {
      // The inpaint mask defines a noiseLevel, so the plan carries a noise-mask
      // entry — it must execute only when the resolved mode consumes it.
      const harness = makeHost(makeDoc([rasterLayer('base'), inpaintMaskLayer('mask', 0.5)]));

      const result = await compose(harness.host, { detectMode: () => mode });

      expect(result.status).toBe('ok');
      if (result.status !== 'ok') {
        return;
      }
      expect(result.composites.mode).toBe(mode);
      expect(result.composites.maskImageName).toBe(expectedMask);
      expect(result.composites.noiseMaskImageName).toBe(expectedNoise);
      expect(harness.uploadImage).toHaveBeenCalledTimes(expectedUploads);
    }
  );

  it('skips a control layer the predicate rejects, with no upload for it', async () => {
    const harness = makeHost(makeDoc([controlLayer('control-a'), controlLayer('control-b')]));

    const result = await compose(harness.host, {
      shouldCompositeControlLayer: (layer) => layer.id !== 'control-a',
    });

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    expect(result.composites.controlImages).toEqual([{ imageName: 'composite-1.png', layerId: 'control-b' }]);
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
  });

  it('rethrows a control-predicate throw after releasing the snapshot, committing nothing', async () => {
    // A raster layer first, so the base composite has already uploaded (and
    // populated the operation-scoped cache) before the predicate aborts.
    const harness = makeHost(makeDoc([rasterLayer('base'), controlLayer('control-a')]));

    await expect(
      compose(harness.host, {
        shouldCompositeControlLayer: () => {
          throw new Error('[missing_model] Control layer "control-a" is invalid.');
        },
      })
    ).rejects.toThrow('[missing_model] Control layer "control-a" is invalid.');

    expect(harness.release).toHaveBeenCalledTimes(1);
    expect(harness.host.dedupe.byKey.size).toBe(0);
    expect(harness.host.dedupe.byHash.size).toBe(0);
  });

  it('silently skips a regional mask the predicate rejects, with no upload for it', async () => {
    const harness = makeHost(makeDoc([regionalLayer('region-a'), regionalLayer('region-b')]));

    const result = await compose(harness.host, {
      shouldCompositeRegionalMask: (layer) => layer.id !== 'region-a',
    });

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    expect(result.composites.regionalMaskImages).toEqual([{ imageName: 'composite-1.png', layerId: 'region-b' }]);
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
  });

  it('calls the predicates in plan z-order (controls, then regionals)', async () => {
    const harness = makeHost(
      makeDoc([
        controlLayer('control-a'),
        regionalLayer('region-a'),
        controlLayer('control-b'),
        regionalLayer('region-b'),
      ])
    );
    const order: string[] = [];

    await compose(harness.host, {
      shouldCompositeControlLayer: (layer) => {
        order.push(`control:${layer.id}`);
        return true;
      },
      shouldCompositeRegionalMask: (layer) => {
        order.push(`regional:${layer.id}`);
        return true;
      },
    });

    expect(order).toEqual(['control:control-a', 'control:control-b', 'regional:region-a', 'regional:region-b']);
  });

  it('commits the dedupe cache on success so a second compose reuses every upload', async () => {
    const document = makeDoc([rasterLayer('base'), controlLayer('control-a')]);
    const harness = makeHost(document);

    const first = await compose(harness.host);
    expect(first.status).toBe('ok');
    const uploadsAfterFirst = harness.uploadImage.mock.calls.length;
    expect(uploadsAfterFirst).toBe(2);
    expect(harness.host.dedupe.byKey.size).toBeGreaterThan(0);

    const second = await compose(harness.host);
    expect(second.status).toBe('ok');
    expect(harness.uploadImage.mock.calls.length).toBe(uploadsAfterFirst);
    if (first.status === 'ok' && second.status === 'ok') {
      expect(second.composites.baseImageName).toBe(first.composites.baseImageName);
      expect(second.composites.controlImages).toEqual(first.composites.controlImages);
    }
  });

  it('releases the raster snapshot on success', async () => {
    const harness = makeHost(makeDoc([rasterLayer('base')]));

    await compose(harness.host);

    expect(harness.release).toHaveBeenCalledTimes(1);
  });

  it('releases the raster snapshot and commits nothing when an upload fails', async () => {
    const harness = makeHost(makeDoc([rasterLayer('base')]), {
      uploadImage: () => Promise.reject(new Error('upload exploded')),
    });

    await expect(compose(harness.host)).rejects.toThrow('upload exploded');

    expect(harness.release).toHaveBeenCalledTimes(1);
    expect(harness.host.dedupe.byKey.size).toBe(0);
    expect(harness.host.dedupe.byHash.size).toBe(0);
  });
});
