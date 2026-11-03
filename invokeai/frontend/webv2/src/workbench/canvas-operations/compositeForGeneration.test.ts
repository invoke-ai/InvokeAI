import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { Rect } from '@workbench/canvas-operations/generationContracts';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { applyAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { RasterCompositeOverBudgetError } from '@workbench/canvas-engine/render/rasterComposite';
import { MEASURED_NOISY_TEXT } from '@workbench/canvas-engine/render/rasterComposite.testing';
import {
  planComposites,
  planControlComposites,
  planRegionalMaskComposites,
} from '@workbench/canvas-operations/generationCompositePlan';
import { describe, expect, it, vi } from 'vitest';

import type { ExecuteCompositePlanDeps } from './compositeForGeneration';

import {
  createCompositeDedupeCache,
  executeCompositePlan,
  executeControlComposite,
  executeMaskComposite,
  executeRegionalMaskComposite,
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

const controlLayer = (id: string, overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: imageRef(id), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
  ...overrides,
});

const makeDoc = (layers: CanvasLayerContract[]): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
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

  const getLayerSurface = (layerId: string): Promise<{ surface: RasterSurface; rect: Rect; release(): void }> => {
    let surface = layerSurfaces.get(layerId);
    if (!surface) {
      // Layer caches are content-sized; created off the raw stub so they don't
      // pollute the composite-target capture. Origin-anchored (0,0) here.
      surface = stub.createSurface(64, 48);
      layerSurfaces.set(layerId, surface);
    }
    return Promise.resolve({ rect: { height: 48, width: 64, x: 0, y: 0 }, release: () => undefined, surface });
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

  it('reports whether any composited pixel is visible, cached with the plan key', async () => {
    const empty = makeHarness(() => uniformImageData(100, 100, 0));
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);
    expect(await executeCompositePlan(plan, empty.deps)).toMatchObject({
      bboxFullyCovered: false,
      bboxHasContent: false,
    });
    // The second run reuses the cached entry and still knows the frame is empty.
    expect(await executeCompositePlan(plan, empty.deps)).toMatchObject({ bboxHasContent: false });

    const faint = makeHarness(() => uniformImageData(100, 100, 1));
    expect(await executeCompositePlan(plan, faint.deps)).toMatchObject({
      bboxFullyCovered: false,
      bboxHasContent: true,
    });
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

    const refused = executeCompositePlan(plan, harness.deps);
    await expect(refused).rejects.toBeInstanceOf(RasterCompositeOverBudgetError);
    await expect(refused).rejects.toThrow('The canvas composite exceeds the available raster memory budget.');
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

describe('executeControlComposite', () => {
  it('flattens transparent control pixels over black without a CPU readback', async () => {
    const readImageData = vi.fn(() => uniformImageData(100, 100, 0));
    const harness = makeHarness(readImageData);
    const control = planControlComposites(makeDoc([controlLayer('control')]), BBOX)[0];
    expect(control).toBeDefined();

    await executeControlComposite(control!.entry, harness.deps);

    const target = harness.createdTargets[0]!;
    expect(target.callLog).toContainEqual({ args: ['globalCompositeOperation', 'destination-over'], op: 'set' });
    expect(target.callLog).toContainEqual({ args: ['fillStyle', 'black'], op: 'set' });
    expect(target.callLog).toContainEqual({ args: [0, 0, 100, 100], op: 'fillRect' });
    expect(readImageData).not.toHaveBeenCalled();
    expect(harness.encodeSurface).toHaveBeenCalledWith(target);
  });
});

describe('executeRegionalMaskComposite', () => {
  const regionalLayer: CanvasLayerContract = {
    autoNegative: false,
    blendMode: 'normal',
    id: 'region',
    isEnabled: true,
    isLocked: false,
    mask: { bitmap: imageRef('region-bmp'), fill: { color: '#00ff00', style: 'solid' } },
    name: 'Region',
    negativePrompt: null,
    opacity: 0.5,
    positivePrompt: 'a sign',
    referenceImages: [],
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'regional_guidance',
  };
  const regionalEntry = () => planRegionalMaskComposites(makeDoc([regionalLayer]), BBOX)[0]!.entry;

  /** A transparent scan with one faint pixel at each opposite corner of a box. */
  const boxImageData = (width: number, height: number): ImageData => {
    const image = uniformImageData(width, height, 0);
    image.data[(12 * width + 10) * 4 + 3] = 1;
    image.data[(40 * width + 29) * 4 + 3] = 255;
    return image;
  };

  it('measures the alpha bounds on the coverage readback it already takes', async () => {
    const readImageData = vi.fn(() => boxImageData(100, 100));
    const harness = makeHarness(readImageData);

    const result = await executeRegionalMaskComposite(regionalEntry(), harness.deps);

    expect(result).toMatchObject({ bounds: { height: 29, width: 20, x: 10, y: 12 }, imageName: 'uploaded-1' });
    expect(readImageData).toHaveBeenCalledTimes(1);
  });

  it('reuses the cached bounds with the cached upload', async () => {
    const readImageData = vi.fn(() => boxImageData(100, 100));
    const harness = makeHarness(readImageData);

    await executeRegionalMaskComposite(regionalEntry(), harness.deps);
    const second = await executeRegionalMaskComposite(regionalEntry(), harness.deps);

    expect(second).toMatchObject({ bounds: { height: 29, width: 20, x: 10, y: 12 }, reusedUpload: true });
    expect(readImageData).toHaveBeenCalledTimes(1);
    expect(harness.encodeSurface).toHaveBeenCalledTimes(1);
  });

  it('reports empty bounds for a mask with no pixels in the bbox', async () => {
    const harness = makeHarness(() => uniformImageData(100, 100, 0));

    const result = await executeRegionalMaskComposite(regionalEntry(), harness.deps);

    expect(result.bounds).toEqual({ height: 0, width: 0, x: 0, y: 0 });
  });
});

// ---- Grayscale mask composite ---------------------------------------------

const inpaintMask = (
  id: string,
  overrides: Partial<{ denoiseLimit: number; noiseLevel: number }> = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  ...(overrides.denoiseLimit === undefined ? {} : { denoise: { isEnabled: true, limit: overrides.denoiseLimit } }),
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: imageRef(`${id}-bmp`, 64, 48), fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  ...(overrides.noiseLevel === undefined ? {} : { noise: { isEnabled: true, level: overrides.noiseLevel } }),
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const maskEntryOf = (doc: CanvasDocumentContractV3) =>
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

    const adjustedDoc = makeDoc([
      rasterLayer('b', {
        adjustments: [
          { brightness: 0.5, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
        ],
      }),
    ]);
    await executeCompositePlan(planComposites(adjustedDoc, BBOX), harness.deps);
    expect(writeImageData).toHaveBeenCalledTimes(1);
  });

  const grain = {
    amount: 0.2,
    distribution: 'gaussian',
    id: 'grain',
    isEnabled: true,
    grain: 'color',
    response: 'flat',
    size: 1,
    seed: 77,
    type: 'noise',
  } as const;
  const midGray = (width: number, height: number): ImageData => {
    const image = uniformImageData(width, height, 255);
    for (let i = 0; i < image.data.length; i += 4) {
      image.data.fill(128, i, i + 3);
    }
    return image;
  };

  // A noisy stack always adjusts the layer-local region the bbox samples (plus one pixel of resampling margin) at
  // its layer-local origin, before resampling, even where a color-only stack would adjust the output instead.
  it.each([
    // The sampled region (22×12) exceeds the 20×10 output.
    { bbox: { height: 10, width: 20, x: 10, y: 5 }, origin: { x: 2, y: 1 }, scale: 1, size: [22, 12] },
    { bbox: { height: 100, width: 100, x: 20, y: 10 }, origin: { x: 12, y: 6 }, scale: 1, size: [52, 42] },
    // Downscaled 2×: output x 0–20 samples layer x 6–46, y 0–10 samples layer y 4–24.
    { bbox: { height: 10, width: 20, x: 10, y: 5 }, origin: { x: 5, y: 3 }, scale: 0.5, size: [42, 22] },
  ])('adjusts noise layer-local at $origin.x,$origin.y for bbox $bbox.x,$bbox.y at scale $scale', async (testCase) => {
    const written: ImageData[] = [];
    const harness = makeHarness((_surface, rect) => midGray(rect.width, rect.height));
    harness.deps.writeImageData = (_surface, imageData) => {
      written.push(imageData);
    };
    const reserve = vi.fn((_bytes: number) => ({ lease: { release: () => undefined }, status: 'ok' as const }));
    harness.deps.reserve = reserve;
    const doc = makeDoc([
      rasterLayer('n', {
        adjustments: [grain],
        transform: { rotation: 0, scaleX: testCase.scale, scaleY: testCase.scale, x: 7, y: 3 },
      }),
    ]);

    await executeCompositePlan(planComposites(doc, testCase.bbox), harness.deps);

    const [width, height] = testCase.size as [number, number];
    const expected = midGray(width, height);
    applyAdjustments(expected, [grain], testCase.origin);
    expect(written).toHaveLength(1);
    expect(Array.from(written[0]!.data)).toEqual(Array.from(expected.data));
    // Output and scan buffers, plus the layer-local scratch surface and its ImageData, which can exceed the output.
    const outputPixels = testCase.bbox.width * testCase.bbox.height;
    expect(reserve).toHaveBeenCalledWith((outputPixels * 2 + Math.max(outputPixels, width * height) * 2) * 4);
  });

  it('refuses a measured text scratch past its reservation and releases everything', async () => {
    const { bbox, layer, measured, plannedScratchPixels } = MEASURED_NOISY_TEXT;
    const harness = makeHarness();
    const layerSurface = createTestStubRasterBackend().createSurface(measured.width, measured.height);
    const releaseLayer = vi.fn();
    harness.deps.getLayerSurface = () =>
      Promise.resolve({ rect: measured, release: releaseLayer, surface: layerSurface });
    // Output and scan buffers, plus the planned scratch surface and its ImageData: no room for the measured excess.
    const memory = new RasterMemoryBudgetController({
      budgetBytes: (bbox.width * bbox.height * 2 + plannedScratchPixels * 2) * 4,
    });
    harness.deps.reserve = (bytes) => memory.reserveOperation(bytes, { purpose: 'invocation-composite' });

    await expect(executeCompositePlan(planComposites(makeDoc([layer]), bbox), harness.deps)).rejects.toBeInstanceOf(
      RasterCompositeOverBudgetError
    );

    expect(harness.createdTargets.map(({ height, width }) => [width, height])).toEqual([[bbox.width, bbox.height]]);
    expect(releaseLayer).toHaveBeenCalledOnce();
    expect(memory.snapshot().reservedBytes).toBe(0);
    expect(harness.uploadImage).not.toHaveBeenCalled();
  });
});
