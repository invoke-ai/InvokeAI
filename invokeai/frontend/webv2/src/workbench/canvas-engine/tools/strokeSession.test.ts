import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { StubRasterBackend, StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { StrokeEdit, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';

import * as freehand from '@workbench/canvas-engine/freehand';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { ADMIT_ALL, commitStroke, createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { createStrokeSession } from '@workbench/canvas-engine/tools/strokeSession';
import { describe, expect, it, vi } from 'vitest';

const runFrameNow = (task: () => void): (() => void) => {
  task();
  return () => undefined;
};

const pointer = (x: number, y: number): PointerInput => ({
  buttons: 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

const createCapturingBackend = (): { backend: StubRasterBackend; created: StubRasterSurface[] } => {
  const inner = createTestStubRasterBackend();
  const created: StubRasterSurface[] = [];
  return {
    backend: {
      ...inner,
      createSurface: (w, h) => {
        const s = inner.createSurface(w, h);
        created.push(s);
        return s;
      },
    },
    created,
  };
};

const runStroke = (opts: { withMask: boolean }) => {
  const { backend, created } = createCapturingBackend();
  const layers = createLayerCacheStore(backend);
  const entry: LayerCacheEntry = layers.getOrCreate('L', 100, 100);
  const mask = opts.withMask ? backend.createSurface(100, 100) : null;
  const clipMask = mask ? { rect: { height: 100, width: 100, x: 0, y: 0 }, surface: mask } : null;
  const notifyLayerPainted = vi.fn();

  const ctx = {
    backend,
    createPath2D: () => {
      const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
      return path as unknown as Path2D;
    },
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted,
    scheduleFrame: runFrameNow,
  } as unknown as ToolContext;

  created.length = 0;
  const session = createStrokeSession({
    edit: ADMIT_ALL,
    clipMask,
    color: '#ff0000',
    hardness: 1,
    composite: 'source-over',
    ctx,
    layerId: 'L',
    opacity: 1,
    size: 20,
    pressureOpacity: false,
    thinning: 0,
    tool: 'brush',
  });
  session.addPoints([pointer(10, 10)]);
  session.addPoints([pointer(40, 10), pointer(40, 40)]);
  const event = commitStroke(session);

  const scratch = created[0]!;
  return { cache: entry.surface as StubRasterSurface, event, notifyLayerPainted, scratch };
};

const compositeOps = (surface: StubRasterSurface): unknown[] =>
  surface.callLog.filter((e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation').map((e) => e.args[1]);

describe('strokeSession: selection-constrained painting', () => {
  it('with a clip mask, intersects the scratch stroke with the mask (destination-in) before compositing', () => {
    const { scratch } = runStroke({ withMask: true });
    expect(compositeOps(scratch)).toContain('destination-in');
    expect(scratch.callLog.some((e) => e.op === 'drawImage')).toBe(true);
  });
});

describe('strokeSession: bbox-clipped painting', () => {
  const runClipped = (clipRect: { x: number; y: number; width: number; height: number } | null) => {
    const { backend, created } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    const entry: LayerCacheEntry = layers.getOrCreate('L', 100, 100);
    const ctx = {
      backend,
      createPath2D: () => {
        const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
        return path as unknown as Path2D;
      },
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;
    created.length = 0;

    const session = createStrokeSession({
      edit: ADMIT_ALL,
      clipRect,
      color: '#ff0000',
      hardness: 1,
      composite: 'source-over',
      ctx,
      layerId: 'L',
      opacity: 1,
      size: 20,
      pressureOpacity: false,
      thinning: 0,
      tool: 'brush',
    });
    session.addPoints([pointer(10, 10)]);
    session.addPoints([pointer(40, 10), pointer(40, 40)]);
    const event = commitStroke(session);
    return { cache: entry.surface as StubRasterSurface, event, scratch: created[0] as StubRasterSurface | undefined };
  };

  it('keeps the dirty rect inside the clip rect', () => {
    const clip = { height: 20, width: 20, x: 0, y: 0 };
    const { event } = runClipped(clip);

    expect(event).not.toBeNull();
    expect(event!.dirtyRect.x).toBeGreaterThanOrEqual(clip.x);
    expect(event!.dirtyRect.y).toBeGreaterThanOrEqual(clip.y);
    expect(event!.dirtyRect.x + event!.dirtyRect.width).toBeLessThanOrEqual(clip.x + clip.width);
    expect(event!.dirtyRect.y + event!.dirtyRect.height).toBeLessThanOrEqual(clip.y + clip.height);
  });

  it('commits nothing when the stroke falls entirely outside the clip rect', () => {
    const { event } = runClipped({ height: 5, width: 5, x: 500, y: 500 });
    expect(event).toBeNull();
  });
});

describe('strokeSession: content-sized cache growth', () => {
  const makeSession = (
    initialRect: { x: number; y: number; width: number; height: number },
    edit: Pick<StrokeEdit, 'grow'> = ADMIT_ALL
  ) => {
    const { backend } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreateRect('L', initialRect);
    entry.stale = false;
    const notifyLayerPainted = vi.fn();
    const ctx = {
      backend,
      createPath2D: () => {
        const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
        return path as unknown as Path2D;
      },
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted,
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;
    const onRefused = vi.fn();
    const session = createStrokeSession({
      edit,
      clipMask: null,
      color: '#ff0000',
      hardness: 1,
      composite: 'source-over',
      ctx,
      layerId: 'L',
      onRefused,
      opacity: 1,
      size: 20,
      pressureOpacity: false,
      thinning: 0,
      tool: 'brush',
    });
    return { entry, layers, notifyLayerPainted, onRefused, session };
  };

  it('grows an EMPTY (brand-new) paint cache to the stroke bounds on the first stroke', () => {
    const { entry, session } = makeSession({ height: 0, width: 0, x: 0, y: 0 });
    session.addPoints([pointer(50, 60)]);
    const event = commitStroke(session);

    expect(entry.rect.width).toBeGreaterThan(0);
    expect(entry.rect.height).toBeGreaterThan(0);
    expect(entry.rect.x % 64).toBe(0);
    expect(entry.rect.y % 64).toBe(0);
    expect(entry.rect.width % 64).toBe(0);
    expect(entry.rect.height % 64).toBe(0);
    expect(entry.rect.width).toBeLessThanOrEqual(128);
    expect(entry.rect.height).toBeLessThanOrEqual(128);
    expect(entry.rect.x).toBeLessThanOrEqual(50);
    expect(entry.rect.x + entry.rect.width).toBeGreaterThan(50);
    expect(entry.rect.y).toBeLessThanOrEqual(60);
    expect(entry.rect.y + entry.rect.height).toBeGreaterThan(60);
    expect(entry.surface.width).toBe(entry.rect.width);
    expect(entry.surface.height).toBe(entry.rect.height);
    expect(event!.dirtyRect).toEqual(entry.rect);
  });

  it('restores the layer and its extent once when the undo footprint cannot grow', () => {
    const first = makeSession({ height: 20, width: 20, x: 0, y: 0 }, createRecordingStrokeEdit().edit);
    first.session.addPoints([pointer(10, 10)]);
    const firstFootprint = first.entry.rect.width * first.entry.rect.height * 8;

    const { edit } = createRecordingStrokeEdit(firstFootprint);
    const { layers, onRefused, session } = makeSession({ height: 20, width: 20, x: 0, y: 0 }, edit);
    session.addPoints([pointer(10, 10)]);
    session.addPoints([pointer(400, 400)]);
    session.addPoints([pointer(410, 410)]);

    expect(onRefused).toHaveBeenCalledOnce();
    expect(layers.peek('L')!.rect).toEqual({ height: 20, width: 20, x: 0, y: 0 });
    expect(session.commit(() => true)).toBe(false);
  });

  it('returns an empty cache to empty when the stroke is cancelled', () => {
    const { layers, session } = makeSession({ height: 0, width: 0, x: 0, y: 0 });
    session.addPoints([pointer(10, 10)]);
    expect(layers.peek('L')!.rect.width).toBeGreaterThan(0);
    session.cancel();

    expect(layers.peek('L')!.rect).toMatchObject({ height: 0, width: 0 });
  });

  it('grows an existing cache to the UNION of its extent and an out-of-extent stroke (negative coords included)', () => {
    const { entry, session } = makeSession({ height: 20, width: 20, x: 0, y: 0 });
    session.addPoints([pointer(-40, -40)]);
    commitStroke(session);

    expect(entry.rect.x).toBeLessThan(-30);
    expect(entry.rect.y).toBeLessThan(-30);
    expect(entry.rect.x + entry.rect.width).toBeGreaterThanOrEqual(20);
    expect(entry.rect.y + entry.rect.height).toBeGreaterThanOrEqual(20);
    expect(entry.surface.width).toBe(entry.rect.width);
    expect(entry.surface.height).toBe(entry.rect.height);
  });

  it('reallocates the cache surface O(stroke / chunk) times — NOT once per batch — across an extending drag', () => {
    const { entry, session } = makeSession({ height: 64, width: 64, x: 64, y: 64 });
    const surface = entry.surface as StubRasterSurface;
    const resizeCount = (): number => surface.callLog.filter((e) => e.op === 'resize').length;

    const batches = 10;
    for (let i = 0; i < batches; i++) {
      session.addPoints([pointer(100 + i * 10, 100)]);
    }
    commitStroke(session);

    const resizes = resizeCount();
    expect(resizes).toBeLessThanOrEqual(2);
    expect(resizes).toBeLessThan(batches);
  });
});

describe('strokeSession: cache version bump (live adjusted-surface invalidation)', () => {
  const makeVersionSession = () => {
    const { backend } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('L', 100, 100);
    entry.stale = false;
    const ctx = {
      backend,
      createPath2D: () => {
        const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
        return path as unknown as Path2D;
      },
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;
    const session = createStrokeSession({
      edit: ADMIT_ALL,
      clipMask: null,
      color: '#ff0000',
      hardness: 1,
      composite: 'source-over',
      ctx,
      layerId: 'L',
      opacity: 1,
      size: 20,
      pressureOpacity: false,
      thinning: 0,
      tool: 'brush',
    });
    return { entry, session };
  };

  it('bumps the cache version on every mid-stroke frame (so the adjusted-surface memo recomputes live)', () => {
    const { entry, session } = makeVersionSession();
    const v0 = entry.version;
    session.addPoints([pointer(10, 10)]);
    const v1 = entry.version;
    session.addPoints([pointer(40, 40)]);
    const v2 = entry.version;
    expect(v1).toBeGreaterThan(v0);
    expect(v2).toBeGreaterThan(v1);
  });
});

describe('incremental "before" snapshot', () => {
  const makeSession = (size: number) => {
    const { backend } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', 0, 0);
    const ctx = {
      backend,
      createPath2D: () => {
        const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
        return path as unknown as Path2D;
      },
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;
    const session = createStrokeSession({
      edit: ADMIT_ALL,
      color: '#ff0000',
      hardness: 1,
      composite: 'source-over',
      ctx,
      layerId: 'L',
      opacity: 1,
      size,
      pressureOpacity: false,
      thinning: 0,
      tool: 'brush',
    });
    return { layers, session };
  };

  const drag = (session: ReturnType<typeof makeSession>['session'], to: number, step: number): void => {
    for (let x = 0; x <= to; x += step) {
      session.addPoints([pointer(x, 0)]);
    }
  };

  const surfaceOf = (layers: ReturnType<typeof createLayerCacheStore>) =>
    layers.get('L')!.surface as unknown as StubRasterSurface;

  it('reads only the strips the region gains, never the whole accumulated region', () => {
    const { layers, session } = makeSession(100);
    drag(session, 3000, 50);
    const region = layers.get('L')!.rect;
    const stripReads = surfaceOf(layers)
      .callLog.filter((entry) => entry.op === 'getImageData')
      .filter((entry) => Number(entry.args[0]) !== 0 || Number(entry.args[1]) !== 0);
    expect(region.width).toBeGreaterThan(3000);
    expect(stripReads.length).toBeGreaterThan(0);
    for (const read of stripReads) {
      expect(Number(read.args[2])).toBeLessThan(region.width / 2);
    }
  });
});

describe('strokeSession: pressure-dependent opacity', () => {
  const pointerAt = (x: number, y: number, pressure: number): PointerInput => ({
    ...pointer(x, y),
    pressure,
  });

  const runPressureStroke = (pressureOpacity: boolean, pressures: number[]) => {
    const { backend, created } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', 100, 100);
    const ctx = {
      backend,
      createPath2D: () => {
        const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
        return path as unknown as Path2D;
      },
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;

    created.length = 0;
    const session = createStrokeSession({
      edit: ADMIT_ALL,
      clipMask: null,
      color: '#ff0000',
      hardness: 1,
      composite: 'source-over',
      ctx,
      layerId: 'L',
      opacity: 1,
      pressureOpacity,
      size: 20,
      thinning: 0,
      tool: 'brush',
    });
    session.addPoints(pressures.map((pressure, index) => pointerAt(10 + index * 10, 10, pressure)));
    const scratch = created[0]!;
    scratch.callLog.length = 0;
    commitStroke(session);

    return scratch;
  };

  it('paints one punch-and-fill pair per pressure band when enabled', () => {
    const scratch = runPressureStroke(true, [1, 0.5, 0.25]);

    expect(scratch.callLog.filter((e) => e.op === 'fill')).toHaveLength(6);
    expect(compositeOps(scratch).filter((op) => op === 'destination-out')).toHaveLength(3);
  });

  it('replaces rather than blends each band, so overlaps cannot compound alpha', () => {
    const scratch = runPressureStroke(true, [1, 0.5]);
    const ops = scratch.callLog.filter(
      (e) =>
        e.op === 'fill' || (e.op === 'set' && (e.args[0] === 'globalCompositeOperation' || e.args[0] === 'globalAlpha'))
    );
    const signature = ops.map((e) => (e.op === 'fill' ? 'fill' : `${String(e.args[0])}=${String(e.args[1])}`));

    expect(signature.join(' ')).toContain(
      'globalCompositeOperation=destination-out globalAlpha=1 fill globalCompositeOperation=source-over'
    );
  });
});

describe('layer transforms', () => {
  const sessionOn = (layerTransform: ReturnType<typeof fromTRS> | null, point: { x: number; y: number }) => {
    const { backend } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    const ctx = {
      backend,
      createPath2D: () => {
        const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
        return path as unknown as Path2D;
      },
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;
    const session = createStrokeSession({
      edit: ADMIT_ALL,
      color: '#ff0000',
      hardness: 1,
      composite: 'source-over',
      ctx,
      layerId: 'L',
      layerTransform,
      opacity: 1,
      pressureOpacity: false,
      size: 8,
      thinning: 0,
      tool: 'brush',
    });
    session.addPoints([pointer(point.x, point.y)]);
    return commitStroke(session);
  };

  it('maps document points through the layer inverse, so the stroke lands under the cursor', () => {
    // Translated layer: doc (1010, 10) is layer-local (10, 10).
    const translated = sessionOn(fromTRS({ x: 1000, y: 0 }, 0, 1, 1), { x: 1010, y: 10 });
    expect(translated?.dirtyRect.x).toBe(0);
    expect(translated?.dirtyRect.y).toBe(0);

    // Scaled layer: doc (300, 300) on a 2x layer is layer-local (150, 150).
    const scaled = sessionOn(fromTRS({ x: 0, y: 0 }, 0, 2, 2), { x: 300, y: 300 });
    expect(scaled?.dirtyRect.x).toBe(128);
    expect(scaled?.dirtyRect.y).toBe(128);

    const plain = sessionOn(null, { x: 300, y: 300 });
    expect(plain?.dirtyRect.x).toBe(256);
    expect(plain?.dirtyRect.y).toBe(256);
  });
});

describe('tap collapse', () => {
  const sessionWithDrift = (drift: number) => {
    const { backend } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', 400, 400);
    const ctx = {
      backend,
      createPath2D: () =>
        ({ closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} }) as unknown as Path2D,
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;
    const session = createStrokeSession({
      edit: ADMIT_ALL,
      clipMask: null,
      color: '#f00',
      composite: 'source-over',
      ctx,
      hardness: 1,
      layerId: 'L',
      opacity: 1,
      pressureOpacity: false,
      size: 50,
      thinning: 0.5,
      tool: 'brush',
    });
    session.addPoints([pointer(100, 100)]);
    if (drift > 0) {
      session.addPoints([pointer(100 + drift / 2, 100)]);
      session.addPoints([pointer(100 + drift, 100)]);
    }
    return commitStroke(session);
  };

  it('renders a click that drifted under a quarter diameter as the round tap dot', () => {
    const clean = sessionWithDrift(0)!.dirtyRect;
    expect(sessionWithDrift(6)!.dirtyRect).toEqual(clean);
    expect(sessionWithDrift(12)!.dirtyRect).toEqual(clean);
  });
});

describe('frame-coalesced rendering', () => {
  const makeFramedSession = () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', 100, 100);
    const frames: (() => void)[] = [];
    const invalidate = vi.fn();
    const onRenderError = vi.fn();
    const ctx = {
      backend,
      createPath2D: () => {
        const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
        return path as unknown as Path2D;
      },
      invalidate,
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
      clipMask: null,
      color: '#ff0000',
      composite: 'source-over',
      ctx,
      hardness: 1,
      layerId: 'L',
      onRenderError,
      opacity: 1,
      pressureOpacity: false,
      size: 20,
      thinning: 0,
      tool: 'brush',
    });
    const renders = (): number => invalidate.mock.calls.length;
    const presentFrame = (): void => frames.splice(0).forEach((task) => task());
    return { frames, layers, onRenderError, presentFrame, renders, session };
  };

  it('renders the accumulated stroke at most once per presented frame', () => {
    const { frames, presentFrame, renders, session } = makeFramedSession();
    for (let frame = 0; frame < 4; frame++) {
      for (let batch = 0; batch < 6; batch++) {
        const x = 10 + frame * 12 + batch * 2;
        session.addPoints([pointer(x, 10), pointer(x + 1, 11)]);
      }
      expect(frames).toHaveLength(1);
      expect(renders()).toBe(frame);
      presentFrame();
      expect(renders()).toBe(frame + 1);
    }
  });

  it('commits the final stroke synchronously and withdraws the frame still scheduled', () => {
    const { frames, renders, session } = makeFramedSession();
    session.addPoints([pointer(10, 10)]);
    session.addPoints([pointer(60, 10)]);
    expect(renders()).toBe(0);

    const event = commitStroke(session);

    expect(renders()).toBe(1);
    expect(frames).toHaveLength(0);
    expect(event?.dirtyRect.width).toBeGreaterThan(50);
    session.addPoints([pointer(80, 10)]);
    expect(frames).toHaveLength(0);
  });
});

describe('strokeSession: incremental decimation', () => {
  it('never re-decimates the whole stroke while it renders frame by frame', () => {
    const decimate = vi.spyOn(freehand, 'decimateSamples');
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', 400, 100);
    const ctx = {
      backend,
      createPath2D: () =>
        ({ closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} }) as unknown as Path2D,
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
      scheduleFrame: runFrameNow,
    } as unknown as ToolContext;
    const session = createStrokeSession({
      edit: ADMIT_ALL,
      clipMask: null,
      color: '#ff0000',
      composite: 'source-over',
      ctx,
      hardness: 1,
      layerId: 'L',
      opacity: 1,
      pressureOpacity: false,
      size: 8,
      thinning: 0,
      tool: 'brush',
    });

    for (let x = 0; x < 300; x += 1) {
      session.addPoints([pointer(x, 50)]);
    }
    commitStroke(session);
    expect(decimate).not.toHaveBeenCalled();
    decimate.mockRestore();
  });
});
