import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { Rect } from '@workbench/generation/canvas/types';
import type {
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { planComposites } from '@workbench/generation/canvas/compositePlan';
import { describe, expect, it, vi } from 'vitest';

import type { ExecuteCompositePlanDeps } from './compositeForGeneration';

import {
  createCompositeDedupeCache,
  executeCompositePlan,
  executeMaskComposite,
  toGrayscaleMaskPixels,
} from './compositeForGeneration';

const imageRef = (imageName: string, width = 64, height = 48): CanvasImageRef => ({ height, imageName, width });

const rasterLayer = (
  id: string,
  overrides: Partial<CanvasRasterLayerContractV2> = {}
): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: imageRef(id), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const makeDoc = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 100,
});

const BBOX: Rect = { height: 100, width: 100, x: 0, y: 0 };

/** Builds an ImageData-like object with a uniform alpha (255 = opaque). */
const uniformImageData = (width: number, height: number, alpha: number): ImageData => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 3; i < data.length; i += 4) {
    data[i] = alpha;
  }
  return { colorSpace: 'srgb', data, height, width } as unknown as ImageData;
};

/** An opaque scan except a single transparent pixel (a hole). */
const holeImageData = (width: number, height: number): ImageData => {
  const image = uniformImageData(width, height, 255);
  image.data[3] = 0;
  return image;
};

interface Harness {
  deps: ExecuteCompositePlanDeps;
  createdTargets: StubRasterSurface[];
  layerSurfaces: Map<string, StubRasterSurface>;
  uploadImage: ReturnType<typeof vi.fn>;
  encodeSurface: ReturnType<typeof vi.fn>;
}

const makeHarness = (readImageData?: (surface: RasterSurface, rect: Rect) => ImageData): Harness => {
  const stub = createTestStubRasterBackend();
  const createdTargets: StubRasterSurface[] = [];
  const layerSurfaces = new Map<string, StubRasterSurface>();

  const encodeSurface = vi.fn((surface: RasterSurface) => stub.encodeSurface(surface));
  const backend = {
    createSurface: (w: number, h: number): StubRasterSurface => {
      const surface = stub.createSurface(w, h);
      createdTargets.push(surface);
      return surface;
    },
    encodeSurface,
  };

  const getLayerSurface = (layerId: string): Promise<{ surface: RasterSurface; rect: Rect }> => {
    let surface = layerSurfaces.get(layerId);
    if (!surface) {
      // Layer caches are content-sized; created off the raw stub so they don't
      // pollute the composite-target capture. Origin-anchored (0,0) here.
      surface = stub.createSurface(64, 48);
      layerSurfaces.set(layerId, surface);
    }
    return Promise.resolve({ rect: { height: 48, width: 64, x: 0, y: 0 }, surface });
  };

  let counter = 0;
  const uploadImage = vi.fn((blob: Blob): Promise<CanvasImageUploadResult> => {
    void blob;
    counter += 1;
    return Promise.resolve({ height: 100, imageName: `uploaded-${counter}`, width: 100 });
  });

  const deps: ExecuteCompositePlanDeps = {
    backend,
    dedupe: createCompositeDedupeCache(),
    getLayerSurface,
    // Content-addressed hash: identical blob bytes → identical hash.
    hashBlob: (blob: Blob) => blob.text(),
    readImageData,
    uploadImage,
  };

  return { createdTargets, deps, encodeSurface, layerSurfaces, uploadImage };
};

describe('executeCompositePlan — compositing', () => {
  it('composites enabled raster layers bottom→top into a bbox-sized surface', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    const doc = makeDoc([rasterLayer('top', { opacity: 0.4 }), rasterLayer('bottom', { opacity: 0.9 })]);
    const plan = planComposites(doc, BBOX);

    await executeCompositePlan(plan, harness.deps);

    expect(harness.createdTargets).toHaveLength(1);
    const target = harness.createdTargets[0]!;
    expect(target.width).toBe(100);
    expect(target.height).toBe(100);

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(2);
    // Bottom layer (array index 1) is drawn first.
    expect(drawImages[0]!.args[0]).toBe(harness.layerSurfaces.get('bottom')!.canvas);
    expect(drawImages[1]!.args[0]).toBe(harness.layerSurfaces.get('top')!.canvas);

    // Per-layer opacity is applied.
    const alphas = target.callLog.filter((e) => e.op === 'set' && e.args[0] === 'globalAlpha').map((e) => e.args[1]);
    expect(alphas).toContain(0.4);
    expect(alphas).toContain(0.9);
  });

  it('translates the composite by the bbox origin (crops to the bbox)', async () => {
    const harness = makeHarness(() => uniformImageData(50, 50, 255));
    const doc = makeDoc([rasterLayer('a')]);
    const bbox: Rect = { height: 50, width: 50, x: 20, y: 10 };
    const plan = planComposites(doc, bbox);

    await executeCompositePlan(plan, harness.deps);

    const target = harness.createdTargets[0]!;
    // The layer sits at document origin; under the bbox translate it draws at (-20, -10).
    const setTransforms = target.callLog.filter((e) => e.op === 'setTransform');
    expect(setTransforms.some((e) => e.args[4] === -20 && e.args[5] === -10)).toBe(true);
  });
});

describe('executeCompositePlan — mode geometry', () => {
  it('reports bboxFullyCovered=true for an opaque composite', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);
    const result = await executeCompositePlan(plan, harness.deps);
    expect(result.bboxFullyCovered).toBe(true);
  });

  it('reports bboxFullyCovered=false when the composite has a transparent hole', async () => {
    const harness = makeHarness(() => holeImageData(100, 100));
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);
    const result = await executeCompositePlan(plan, harness.deps);
    expect(result.bboxFullyCovered).toBe(false);
  });

  it('computes contentBounds as the union of layer bounds in document space', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    const doc = makeDoc([
      rasterLayer('a'), // 64x48 at (0,0)
      rasterLayer('b', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 100 } }), // 64x48 at (100,100)
    ]);
    const plan = planComposites(doc, BBOX);
    const result = await executeCompositePlan(plan, harness.deps);
    expect(result.contentBounds).toEqual({ height: 148, width: 164, x: 0, y: 0 });
  });

  it('returns contentBounds=null when there is no enabled raster content', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 0));
    const plan = planComposites(makeDoc([rasterLayer('a', { isEnabled: false })]), BBOX);
    const result = await executeCompositePlan(plan, harness.deps);
    expect(result.contentBounds).toBeNull();
  });
});

describe('executeCompositePlan — upload + dedupe', () => {
  it('reserves the composite surface and coverage buffer until execution settles', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    const release = vi.fn();
    const reserve = vi.fn(() => ({ lease: { release }, status: 'ok' as const }));
    harness.deps.reserve = reserve;
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);

    await executeCompositePlan(plan, harness.deps);

    expect(reserve).toHaveBeenCalledWith(80_000);
    expect(release).toHaveBeenCalledOnce();
  });

  it('refuses a composite before allocating when its reservation is over budget', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    harness.deps.reserve = () => ({ availableBytes: 0, requestedBytes: 80_000, status: 'over-budget' });
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);

    await expect(executeCompositePlan(plan, harness.deps)).rejects.toThrow(
      'The canvas composite exceeds the available raster memory budget'
    );
    expect(harness.createdTargets).toHaveLength(0);
  });

  it('encodes and uploads once, returning the uploaded image name', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);

    const result = await executeCompositePlan(plan, harness.deps);

    expect(harness.encodeSurface).toHaveBeenCalledTimes(1);
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
    expect(result.base.imageName).toBe('uploaded-1');
    expect(result.base.reusedUpload).toBe(false);
  });

  it('re-running the same plan skips compositing, encoding and upload entirely', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);

    await executeCompositePlan(plan, harness.deps);
    harness.createdTargets.length = 0;
    const second = await executeCompositePlan(plan, harness.deps);

    // No new composite target, no new encode, no new upload.
    expect(harness.createdTargets).toHaveLength(0);
    expect(harness.encodeSurface).toHaveBeenCalledTimes(1);
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
    expect(second.base.reusedUpload).toBe(true);
    expect(second.base.imageName).toBe('uploaded-1');
  });

  it('a changed plan with identical pixels re-composites but reuses the upload by content hash', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 255));
    const planA = planComposites(makeDoc([rasterLayer('a', { opacity: 1 })]), BBOX);
    const planB = planComposites(makeDoc([rasterLayer('a', { opacity: 0.5 })]), BBOX);
    expect(planA.entries[0]!.key).not.toBe(planB.entries[0]!.key);

    await executeCompositePlan(planA, harness.deps);
    const resultB = await executeCompositePlan(planB, harness.deps);

    // The changed plan re-composites + re-encodes (same-size stub blob → same hash)...
    expect(harness.encodeSurface).toHaveBeenCalledTimes(2);
    // ...but the identical pixel hash means no second upload.
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
    expect(resultB.base.reusedUpload).toBe(true);
    expect(resultB.base.imageName).toBe('uploaded-1');
  });
});

// ---- Grayscale mask composite ---------------------------------------------

const inpaintMask = (
  id: string,
  overrides: Partial<{ denoiseLimit: number; noiseLevel: number }> = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  denoiseLimit: overrides.denoiseLimit,
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: imageRef(`${id}-bmp`, 64, 48), fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  noiseLevel: overrides.noiseLevel,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const maskEntryOf = (doc: CanvasDocumentContractV2) =>
  planComposites(doc, BBOX).entries.find((e) => e.kind === 'inpaint-mask')!;

/** A grayscale ImageData that is uniformly white or has a single dark pixel. */
const grayImageData = (width: number, height: number, hasDark: boolean): ImageData => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
    data[i + 3] = 255;
  }
  if (hasDark) {
    data[0] = 0;
  }
  return { colorSpace: 'srgb', data, height, width } as unknown as ImageData;
};

describe('toGrayscaleMaskPixels', () => {
  const pixel = (alpha: number): ImageData => {
    const data = new Uint8ClampedArray([120, 130, 140, alpha]);
    return { colorSpace: 'srgb', data, height: 1, width: 1 } as unknown as ImageData;
  };

  it('turns a masked pixel dark by the attribute value (1.0 → black)', () => {
    const img = pixel(255);
    toGrayscaleMaskPixels(img, 1);
    expect(Array.from(img.data)).toEqual([0, 0, 0, 255]);
  });

  it('turns a masked pixel mid-gray at partial strength', () => {
    const img = pixel(255);
    toGrayscaleMaskPixels(img, 0.5);
    // 255 - round(255 * 0.5) = 255 - 128 = 127
    expect(Array.from(img.data)).toEqual([127, 127, 127, 255]);
  });

  it('leaves an unmasked (transparent) pixel white', () => {
    const img = pixel(0);
    toGrayscaleMaskPixels(img, 1);
    expect(Array.from(img.data)).toEqual([255, 255, 255, 255]);
  });
});

describe('executeMaskComposite', () => {
  it('composites mask layers onto a white bbox surface and uploads the result', async () => {
    const harness = makeHarness(() => grayImageData(100, 100, true));
    const entry = maskEntryOf(makeDoc([inpaintMask('m1')]));

    const result = await executeMaskComposite(entry, harness.deps);

    // White background fill + one mask draw.
    const target = harness.createdTargets[0]!;
    const fills = target.callLog.filter((e) => e.op === 'set' && e.args[0] === 'fillStyle');
    expect(fills.map((e) => e.args[1])).toContain('white');
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
    expect(result.imageName).toBe('uploaded-1');
    expect(result.hasContent).toBe(true);
  });

  it('reports no content when the composite is fully white', async () => {
    const harness = makeHarness(() => grayImageData(100, 100, false));
    const entry = maskEntryOf(makeDoc([inpaintMask('m1')]));
    const result = await executeMaskComposite(entry, harness.deps);
    expect(result.hasContent).toBe(false);
  });

  it('dedupes a repeated mask plan key with no second upload', async () => {
    const harness = makeHarness(() => grayImageData(100, 100, true));
    const doc = makeDoc([inpaintMask('m1', { denoiseLimit: 0.5 })]);

    const first = await executeMaskComposite(maskEntryOf(doc), harness.deps);
    const second = await executeMaskComposite(maskEntryOf(doc), harness.deps);

    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
    expect(second.reusedUpload).toBe(true);
    expect(second.imageName).toBe(first.imageName);
    expect(second.hasContent).toBe(true);
  });

  it('darken-composites multiple mask layers', async () => {
    const harness = makeHarness(() => grayImageData(100, 100, true));
    const entry = maskEntryOf(makeDoc([inpaintMask('a'), inpaintMask('b')]));

    await executeMaskComposite(entry, harness.deps);

    const target = harness.createdTargets[0]!;
    const composites = target.callLog
      .filter((e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation')
      .map((e) => e.args[1]);
    expect(composites).toContain('darken');
  });
});

describe('executeCompositePlan — raster adjustments baked into generation pixels', () => {
  it('reads + writes the adjusted layer pixels (bake) and NOT for a plain layer', async () => {
    const writeImageData = vi.fn();
    const readImageData = vi.fn((_surface: RasterSurface, rect: Rect) =>
      uniformImageData(rect.width, rect.height, 255)
    );
    const harness = makeHarness(readImageData);
    harness.deps.writeImageData = writeImageData;

    // Plain layer: no adjustments → no bake write beyond the coverage scan writes (none here).
    const plainDoc = makeDoc([rasterLayer('a')]);
    await executeCompositePlan(planComposites(plainDoc, BBOX), harness.deps);
    expect(writeImageData).not.toHaveBeenCalled();

    // Adjusted layer: brightness bake → the executor reads the temp, applies the
    // LUT, and writes the adjusted pixels back before compositing.
    const adjustedDoc = makeDoc([rasterLayer('b', { adjustments: { brightness: 0.5, contrast: 0, saturation: 0 } })]);
    await executeCompositePlan(planComposites(adjustedDoc, BBOX), harness.deps);
    expect(writeImageData).toHaveBeenCalledTimes(1);
  });
});
