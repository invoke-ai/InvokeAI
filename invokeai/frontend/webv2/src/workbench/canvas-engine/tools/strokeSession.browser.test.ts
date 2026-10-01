import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Rect } from '@workbench/canvas-engine/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { ADMIT_ALL, commitStroke } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { createStrokeSession } from '@workbench/canvas-engine/tools/strokeSession';
import { describe, expect, it, vi } from 'vitest';

const pointer = (x: number, y: number, pressure = 0.5): PointerInput => ({
  buttons: 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure,
  screenPoint: { x, y },
  timeStamp: 0,
});

const paint = (
  path: PointerInput[],
  batchSize: number,
  opts: {
    opacity: number;
    size: number;
    thinning: number;
    hardness?: number;
    clipMaskRect?: Rect;
    /** Batches delivered per presented frame; 1 renders every batch. */
    batchesPerFrame?: number;
  }
): { pixels: Uint8ClampedArray; rect: Rect } => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  layers.getOrCreate('L', 0, 0);
  const clipMask = opts.clipMaskRect
    ? (() => {
        const surface = backend.createSurface(opts.clipMaskRect.width, opts.clipMaskRect.height);
        surface.ctx.fillStyle = '#fff';
        surface.ctx.fillRect(0, 0, opts.clipMaskRect.width, opts.clipMaskRect.height);
        return { rect: opts.clipMaskRect, surface };
      })()
    : null;
  const frames: (() => void)[] = [];
  const ctx = {
    backend,
    createPath2D: (d?: string) => new Path2D(d),
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: vi.fn(),
    scheduleFrame: (task: () => void) => {
      frames.push(task);
      return () => {
        const index = frames.indexOf(task);
        if (index >= 0) {
          frames.splice(index, 1);
        }
      };
    },
  } as unknown as ToolContext;
  const session = createStrokeSession({
    edit: ADMIT_ALL,
    clipMask,
    color: '#3b82f6',
    composite: 'source-over',
    hardness: opts.hardness ?? 1,
    ctx,
    layerId: 'L',
    opacity: opts.opacity,
    size: opts.size,
    pressureOpacity: false,
    thinning: opts.thinning,
    tool: 'brush',
  });
  const batchesPerFrame = opts.batchesPerFrame ?? 1;
  for (let i = 0, batch = 1; i < path.length; i += batchSize, batch += 1) {
    session.addPoints(path.slice(i, i + batchSize));
    if (batch % batchesPerFrame === 0) {
      frames.splice(0).forEach((task) => task());
    }
  }
  const event = commitStroke(session)!;
  return { pixels: event.afterImageData.data, rect: event.dirtyRect };
};

/** Tolerance covers premultiplied-alpha rounding; gaps, seams and double composites exceed it substantially. */
const ROUNDING_TOLERANCE = 6;

const tapCoverage = (
  composite: 'source-over' | 'destination-out',
  options: { hardness?: number; pressure?: number; pressureOpacity?: boolean; size?: number } = {}
): { max: number; sum: number } => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  const entry = layers.getOrCreate('L', 32, 32);
  if (composite === 'destination-out') {
    entry.surface.ctx.fillStyle = '#000';
    entry.surface.ctx.fillRect(0, 0, 32, 32);
    layers.publishPixels('L');
  }
  const ctx = {
    backend,
    createPath2D: (d?: string) => new Path2D(d),
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: vi.fn(),
    scheduleFrame: (task: () => void) => {
      task();
      return () => undefined;
    },
  } as unknown as ToolContext;
  const session = createStrokeSession({
    edit: ADMIT_ALL,
    color: '#3b82f6',
    composite,
    ctx,
    hardness: options.hardness ?? 1,
    layerId: 'L',
    opacity: 1,
    pressureOpacity: options.pressureOpacity ?? false,
    size: options.size ?? 0.1,
    thinning: 0,
    tool: composite === 'source-over' ? 'brush' : 'eraser',
  });
  session.addPoints([pointer(16.5, 16.5, options.pressure)]);
  const event = commitStroke(session)!;
  let max = 0;
  let sum = 0;
  for (let index = 3; index < event.afterImageData.data.length; index += 4) {
    const coverage = Math.abs(event.afterImageData.data[index]! - event.beforeImageData.data[index]!);
    max = Math.max(max, coverage);
    sum += coverage;
  }
  return { max, sum };
};

describe('sub-pixel taps', () => {
  it.each(['source-over', 'destination-out'] as const)('keeps a 0.1px %s tap sub-pixel', (composite) => {
    const coverage = tapCoverage(composite);
    expect(coverage.max).toBeGreaterThan(0);
    expect(coverage.sum).toBeGreaterThan(0);
    expect(coverage.max).toBeLessThan(64);
    expect(coverage.sum).toBeLessThan(128);
  });

  it('honors pressure-driven opacity for a sub-pixel tap', () => {
    const light = tapCoverage('source-over', { pressure: 0, pressureOpacity: true, size: 0.9 });
    const heavy = tapCoverage('source-over', { pressure: 1, pressureOpacity: true, size: 0.9 });

    expect(light.sum).toBeGreaterThan(0);
    expect(heavy.sum).toBeGreaterThan(light.sum * 8);
  });
});

describe('hardness', () => {
  it('a feathered stroke clipped to a selection never bleeds past the mask and keeps its soft edge inside', () => {
    const result = paint([pointer(40, 64), pointer(40.01, 64)], 2, {
      clipMaskRect: { height: 128, width: 64, x: 0, y: 0 },
      hardness: 0.2,
      opacity: 1,
      size: 48,
      thinning: 0,
    });
    const alphaAt = (x: number, y: number): number => {
      if (x < result.rect.x || x >= result.rect.x + result.rect.width) {
        return 0;
      }
      return result.pixels[((y - result.rect.y) * result.rect.width + (x - result.rect.x)) * 4 + 3]!;
    };
    expect(alphaAt(40, 64)).toBeGreaterThan(200);
    // Soft edge survives the mask inside it.
    expect(alphaAt(20, 64)).toBeGreaterThan(8);
    expect(alphaAt(20, 64)).toBeLessThan(250);
    // The blur's bleed is cut hard at the mask boundary.
    for (const x of [64, 66, 72, 80]) {
      expect(alphaAt(x, 64)).toBe(0);
    }
  });

  const alphaAt = (result: { pixels: Uint8ClampedArray; rect: Rect }, x: number, y: number): number =>
    result.pixels[((y - result.rect.y) * result.rect.width + (x - result.rect.x)) * 4 + 3]!;
  const dab = (hardness: number) =>
    paint([pointer(64, 64), pointer(64.01, 64)], 2, { hardness, opacity: 1, size: 48, thinning: 0 });

  it('feathers the edge without touching the core, and hardness 1 stays crisp', () => {
    const hard = dab(1);
    const softDab = dab(0.2);
    expect(alphaAt(softDab, 64, 64)).toBeGreaterThan(235);
    expect(alphaAt(hard, 64, 64)).toBeGreaterThan(250);

    // Just inside the 24px radius: crisp is near-opaque, feathered is midway.
    const edgeHard = alphaAt(hard, 64 + 22, 64);
    const edgeSoft = alphaAt(softDab, 64 + 22, 64);
    expect(edgeHard).toBeGreaterThan(200);
    expect(edgeSoft).toBeGreaterThan(8);
    expect(edgeSoft).toBeLessThan(edgeHard - 60);

    // Just outside: the feather carries alpha past the crisp silhouette.
    expect(alphaAt(hard, 64 + 27, 64)).toBe(0);
    expect(alphaAt(softDab, 64 + 27, 64)).toBeGreaterThan(4);
  });
});

const sweep = (): PointerInput[] => {
  const points: PointerInput[] = [];
  for (let i = 0; i < 240; i++) {
    points.push(pointer(200 + i * 6 + Math.sin(i / 9) * 40, 400 + Math.cos(i / 7) * 160));
  }
  return points;
};

describe('incremental compositing produces the same coverage as recompositing everything', () => {
  // Band updates and incremental before-snapshots must preserve interior coverage. Boundary antialiasing may
  // differ by subpixels across batches; interior gaps or compounded opacity may not. Deliberately shrinking the
  // band by 30px fails every case.
  const cases: {
    batchSize: number;
    batchesPerFrame?: number;
    label: string;
    opacity: number;
    size: number;
    thinning: number;
  }[] = [
    { batchSize: 1, label: 'one sample per batch, opaque', opacity: 1, size: 220, thinning: 0 },
    {
      batchSize: 1,
      batchesPerFrame: 5,
      label: 'several batches coalesced into each frame',
      opacity: 0.6,
      size: 220,
      thinning: 0.5,
    },
    { batchSize: 1, label: 'one sample per batch, semi-transparent', opacity: 0.4, size: 220, thinning: 0 },
    { batchSize: 3, label: 'coalesced batches with pressure thinning', opacity: 0.75, size: 400, thinning: 0.5 },
    { batchSize: 7, label: 'large brush, coarse batches', opacity: 1, size: 900, thinning: 0 },
  ];

  it.each(cases)('$label', ({ batchSize, batchesPerFrame, opacity, size, thinning }) => {
    const path = sweep();
    const incremental = paint(path, batchSize, { batchesPerFrame, opacity, size, thinning });
    const wholesale = paint(path, path.length, { opacity, size, thinning });

    expect(incremental.rect).toEqual(wholesale.rect);
    expect(incremental.pixels.length).toBe(wholesale.pixels.length);

    // Compare essentially opaque interior pixels to exclude antialiasing differences.
    let interiorDiffering = 0;
    let interiorWorst = 0;
    let edgeDiffering = 0;
    for (let i = 0; i < incremental.pixels.length; i += 4) {
      const alphaA = incremental.pixels[i + 3]!;
      const alphaB = wholesale.pixels[i + 3]!;
      let delta = 0;
      for (let channel = 0; channel < 4; channel++) {
        delta = Math.max(delta, Math.abs(incremental.pixels[i + channel]! - wholesale.pixels[i + channel]!));
      }
      if (delta === 0) {
        continue;
      }
      if (alphaA > 250 && alphaB > 250) {
        interiorDiffering++;
        interiorWorst = Math.max(interiorWorst, delta);
      } else {
        edgeDiffering++;
      }
    }
    const total = incremental.pixels.length / 4;
    // Allow rounding-scale interior drift, not gaps or double-composite errors hundreds of channel steps larger.
    expect(interiorWorst).toBeLessThanOrEqual(ROUNDING_TOLERANCE);
    expect(interiorDiffering / total).toBeLessThan(0.001);
    // The antialiased boundary may land a subpixel differently, but only there.
    expect(edgeDiffering / total).toBeLessThan(0.005);
  });
});

describe('dense stylus trace', () => {
  // A 240 Hz pen delivers about four batches per 60 Hz frame. Timings are informational; counts are the gate.
  const trace = Array.from({ length: 4000 }, (_, i) =>
    pointer(20 + i * 0.5, 300 + 120 * Math.sin(i / 60), 0.5 + 0.3 * Math.sin(i / 15))
  );
  const BATCH = 8;

  const run = (batchesPerFrame: number): { ms: number; renders: number } => {
    const backend = createDomRasterBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', 0, 0);
    const frames: (() => void)[] = [];
    const invalidate = vi.fn();
    const ctx = {
      backend,
      createPath2D: (d?: string) => new Path2D(d),
      invalidate,
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: (task: () => void) => {
        frames.push(task);
        return () => frames.splice(0);
      },
    } as unknown as ToolContext;
    const session = createStrokeSession({
      edit: ADMIT_ALL,
      clipMask: null,
      color: '#3b82f6',
      composite: 'source-over',
      ctx,
      hardness: 1,
      layerId: 'L',
      opacity: 1,
      pressureOpacity: false,
      size: 20,
      thinning: 0.5,
      tool: 'brush',
    });
    const start = performance.now();
    for (let i = 0, batch = 1; i < trace.length; i += BATCH, batch += 1) {
      session.addPoints(trace.slice(i, i + BATCH));
      if (batch % batchesPerFrame === 0) {
        frames.splice(0).forEach((task) => task());
      }
    }
    commitStroke(session);
    return { ms: Math.round((performance.now() - start) * 10) / 10, renders: invalidate.mock.calls.length };
  };

  it('renders once per frame however many batches arrive between frames', async ({ annotate }) => {
    const batches = trace.length / BATCH;
    const everyBatch = run(1);
    const perFrame = run(4);

    expect(everyBatch.renders).toBe(batches + 1);
    expect(perFrame.renders).toBe(batches / 4 + 1);
    await annotate(`dense stroke ms: every batch ${everyBatch.ms}, per frame ${perFrame.ms}`);
  });
});
