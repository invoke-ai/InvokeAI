import type { StubRasterBackend, StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { eraseMaskedRegion, fillMaskedRegion } from '@workbench/canvas-engine/selection/selectionOps';
import { describe, expect, it } from 'vitest';

/** Wraps the stub backend so every scratch surface it mints is captured for assertions. */
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

const compositeOps = (surface: StubRasterSurface): unknown[] =>
  surface.callLog.filter((e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation').map((e) => e.args[1]);

describe('fillMaskedRegion', () => {
  it('masks the color to the selection (destination-in scratch) and draws it over the target', () => {
    const { backend, created } = createCapturingBackend();
    const target = backend.createSurface(100, 100);
    const mask = backend.createSurface(100, 100);
    created.length = 0; // ignore target/mask; capture only the op's scratch

    fillMaskedRegion({
      backend,
      color: '#ff0000',
      mask,
      maskOrigin: { x: 0, y: 0 },
      rect: { height: 20, width: 20, x: 10, y: 10 },
      target,
      targetOrigin: { x: 0, y: 0 },
    });

    // The scratch: fill the color, then intersect with the mask.
    const scratch = created[0]!;
    const scratchOps = scratch.callLog.map((e) => e.op);
    expect(scratchOps).toContain('fillRect');
    expect(compositeOps(scratch)).toContain('destination-in');

    // The target: the masked color is drawn source-over at the rect origin.
    const targetDraw = target.callLog.find((e) => e.op === 'drawImage');
    expect(targetDraw).toBeDefined();
    expect(compositeOps(target)).toContain('source-over');
  });

  it('uses source-atop on the target when composite is source-atop (transparency lock)', () => {
    const { backend, created } = createCapturingBackend();
    const target = backend.createSurface(100, 100);
    const mask = backend.createSurface(100, 100);
    created.length = 0;

    fillMaskedRegion({
      backend,
      color: '#ff0000',
      composite: 'source-atop',
      mask,
      maskOrigin: { x: 0, y: 0 },
      rect: { height: 20, width: 20, x: 10, y: 10 },
      target,
      targetOrigin: { x: 0, y: 0 },
    });

    // Transparency lock: the final blit is source-atop, so colour lands ONLY where
    // the target is already opaque (never fills transparent space).
    expect(compositeOps(target)).toContain('source-atop');
    expect(compositeOps(target)).not.toContain('source-over');
  });

  it('translates the draws by the target/mask origins (content-sized placement)', () => {
    const { backend, created } = createCapturingBackend();
    const target = backend.createSurface(100, 100);
    const mask = backend.createSurface(100, 100);
    created.length = 0;

    // Edit region at doc (50,50); mask surface origin (40,40); target cache
    // origin (30,30). The mask is drawn into the region-local scratch at
    // (maskOrigin - rect) = (-10,-10); the scratch is drawn onto the target at
    // (rect - targetOrigin) = (20,20).
    fillMaskedRegion({
      backend,
      color: '#ff0000',
      mask,
      maskOrigin: { x: 40, y: 40 },
      rect: { height: 20, width: 20, x: 50, y: 50 },
      target,
      targetOrigin: { x: 30, y: 30 },
    });

    const scratch = created[0]!;
    const scratchMaskDraw = scratch.callLog.find((e) => e.op === 'drawImage');
    expect(scratchMaskDraw?.args.slice(1, 3)).toEqual([-10, -10]);
    const targetDraw = target.callLog.find((e) => e.op === 'drawImage');
    expect(targetDraw?.args.slice(1, 3)).toEqual([20, 20]);
  });

  it('is a no-op for an empty rect', () => {
    const { backend } = createCapturingBackend();
    const target = backend.createSurface(10, 10);
    const mask = backend.createSurface(10, 10);
    const before = target.callLog.length;
    fillMaskedRegion({
      backend,
      color: '#000',
      mask,
      maskOrigin: { x: 0, y: 0 },
      rect: { height: 0, width: 0, x: 0, y: 0 },
      target,
      targetOrigin: { x: 0, y: 0 },
    });
    expect(target.callLog.length).toBe(before);
  });
});

describe('eraseMaskedRegion', () => {
  it('composites destination-out through the mask onto the target', () => {
    const { backend, created } = createCapturingBackend();
    const target = backend.createSurface(100, 100);
    const mask = backend.createSurface(100, 100);
    created.length = 0;

    eraseMaskedRegion({
      backend,
      mask,
      maskOrigin: { x: 0, y: 0 },
      rect: { height: 20, width: 20, x: 5, y: 5 },
      target,
      targetOrigin: { x: 0, y: 0 },
    });

    const scratch = created[0]!;
    expect(scratch.callLog.map((e) => e.op)).toContain('drawImage');
    expect(compositeOps(target)).toContain('destination-out');
    expect(target.callLog.some((e) => e.op === 'drawImage')).toBe(true);
  });
});
