import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { PlacedSurface, Rect } from '@workbench/canvas-engine/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createSelectionState, type SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import { describe, expect, it, vi } from 'vitest';

const DOC = { height: 100, width: 100 };

const rectBounds = (x: number, y: number, w: number, h: number): Rect => ({ height: h, width: w, x, y });

/** A fake Path2D carrier (Path2D is absent in node); the stub ctx just records it. */
const fakePath = (id: string): Path2D => ({ id }) as unknown as Path2D;

const imageData = (alphas: readonly number[]): ImageData => {
  const data = new Uint8ClampedArray(alphas.length * 4);
  alphas.forEach((alpha, index) => {
    data[index * 4] = index + 1;
    data[index * 4 + 1] = index + 11;
    data[index * 4 + 2] = index + 21;
    data[index * 4 + 3] = alpha;
  });
  return { colorSpace: 'srgb', data, height: 2, width: 2 } as ImageData;
};

const placedMask = (alphas: readonly number[]): { placed: PlacedSurface; pixels: ImageData } => {
  const backend = createTestStubRasterBackend();
  const surface = backend.createSurface(2, 2);
  const pixels = imageData(alphas);
  Object.defineProperty(surface.ctx, 'getImageData', { value: () => pixels });
  return { pixels, placed: { rect: rectBounds(7, -3, 2, 2), surface } };
};

const createHarness = (docSize: { width: number; height: number } | null = DOC) => {
  const backend = createTestStubRasterBackend();
  const onChange = vi.fn();
  const selection: SelectionState = createSelectionState({
    backend,
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    getDocumentSize: () => docSize,
    onChange,
  });
  return { backend, onChange, selection };
};

/** The recorded op names on the mask surface. */
const maskLog = (selection: SelectionState): { op: string; args: unknown[] }[] =>
  (selection.mask()?.surface as StubRasterSurface | undefined)?.callLog ?? [];

/** The last `globalCompositeOperation` value set before a given op, scanning the whole log. */
const compositeOpsFor = (log: { op: string; args: unknown[] }[]): unknown[] =>
  log.filter((e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation').map((e) => e.args[1]);

describe('selectionState: mask building from a path', () => {
  it('replace clears then source-over fills the path, sets bounds + hasSelection', () => {
    const { selection, onChange } = createHarness();
    selection.commit({ bounds: rectBounds(10, 10, 20, 20), op: 'replace', path: fakePath('p1') });

    expect(selection.hasSelection()).toBe(true);
    expect(selection.bounds()).toEqual(rectBounds(10, 10, 20, 20));
    expect(selection.antsPaths()).toHaveLength(1);
    expect(onChange).toHaveBeenCalledTimes(1);

    const log = maskLog(selection);
    const ops = log.map((e) => e.op);
    expect(ops).toContain('clearRect');
    expect(ops).toContain('fill');
    expect(compositeOpsFor(log)).toContain('source-over');
  });

  it('no longer clamps bounds to the document rect (infinite plane); the mask is bounded to the path', () => {
    const { selection } = createHarness();
    selection.commit({ bounds: rectBounds(-10, -10, 200, 200), op: 'replace', path: fakePath('p') });
    // Content-sized: the selection bounds are the path's own bounds, not clamped
    // to the document. The mask surface is placed at that (negative) origin.
    expect(selection.bounds()).toEqual(rectBounds(-10, -10, 200, 200));
    expect(selection.mask()?.rect).toEqual(rectBounds(-10, -10, 200, 200));
  });
});

describe('selectionState: boolean ops', () => {
  it('add unions bounds with source-over and keeps both ant paths', () => {
    const { selection } = createHarness();
    selection.commit({ bounds: rectBounds(10, 10, 10, 10), op: 'replace', path: fakePath('a') });
    selection.commit({ bounds: rectBounds(40, 40, 10, 10), op: 'add', path: fakePath('b') });

    expect(selection.hasSelection()).toBe(true);
    expect(selection.bounds()).toEqual(rectBounds(10, 10, 40, 40));
    expect(selection.antsPaths()).toHaveLength(2);
    expect(compositeOpsFor(maskLog(selection))).toContain('source-over');
  });

  it('subtract composites destination-out', () => {
    const { selection } = createHarness();
    selection.commit({ bounds: rectBounds(0, 0, 50, 50), op: 'replace', path: fakePath('a') });
    selection.commit({ bounds: rectBounds(10, 10, 10, 10), op: 'subtract', path: fakePath('b') });
    expect(selection.hasSelection()).toBe(true);
    expect(compositeOpsFor(maskLog(selection))).toContain('destination-out');
  });

  it('intersect composites destination-in and intersects the bounds', () => {
    const { selection } = createHarness();
    selection.commit({ bounds: rectBounds(0, 0, 50, 50), op: 'replace', path: fakePath('a') });
    selection.commit({ bounds: rectBounds(30, 30, 50, 50), op: 'intersect', path: fakePath('b') });
    expect(selection.hasSelection()).toBe(true);
    expect(selection.bounds()).toEqual(rectBounds(30, 30, 20, 20));
    expect(compositeOpsFor(maskLog(selection))).toContain('destination-in');
  });

  it('intersect against no existing selection clears to nothing', () => {
    const { selection } = createHarness();
    selection.commit({ bounds: rectBounds(0, 0, 10, 10), op: 'intersect', path: fakePath('a') });
    expect(selection.hasSelection()).toBe(false);
    expect(selection.bounds()).toBeNull();
    expect(selection.mask()).toBeNull();
  });
});

describe('selectionState: selectAll / invert / clear', () => {
  it('selectAll fills the whole document rect', () => {
    const { selection } = createHarness();
    selection.selectAll(rectBounds(0, 0, 100, 100));
    expect(selection.hasSelection()).toBe(true);
    expect(selection.bounds()).toEqual(rectBounds(0, 0, 100, 100));
    expect(maskLog(selection).map((e) => e.op)).toContain('fillRect');
  });

  it('invert of an empty selection selects everything', () => {
    const { selection } = createHarness();
    selection.invert(rectBounds(0, 0, 100, 100));
    expect(selection.hasSelection()).toBe(true);
    expect(selection.bounds()).toEqual(rectBounds(0, 0, 100, 100));
  });

  it('invert of a selection keeps a document-border ants path plus the former outline', () => {
    const { selection } = createHarness();
    selection.commit({ bounds: rectBounds(10, 10, 20, 20), op: 'replace', path: fakePath('a') });
    expect(selection.antsPaths()).toHaveLength(1);
    selection.invert(rectBounds(0, 0, 100, 100));
    expect(selection.hasSelection()).toBe(true);
    // The document border plus the former outline.
    expect(selection.antsPaths()).toHaveLength(2);
  });

  it('clear deselects and drops the mask + bounds', () => {
    const { selection, onChange } = createHarness();
    selection.commit({ bounds: rectBounds(10, 10, 20, 20), op: 'replace', path: fakePath('a') });
    onChange.mockClear();
    selection.clear();
    expect(selection.hasSelection()).toBe(false);
    expect(selection.bounds()).toBeNull();
    expect(selection.mask()).toBeNull();
    expect(selection.antsPaths()).toHaveLength(0);
    expect(onChange).toHaveBeenCalledTimes(1);
  });
});

describe('selectionState: replaceMask', () => {
  it('replaces prior pixels with an isolated, pixel-exact alpha mask at its world rect', () => {
    const { selection, onChange } = createHarness();
    selection.selectAll(rectBounds(0, 0, 50, 50));
    const previousSurface = selection.mask()!.surface;
    const source = placedMask([0, 64, 255, 0]);
    onChange.mockClear();

    selection.replaceMask(source.placed);

    expect(selection.hasSelection()).toBe(true);
    expect(selection.bounds()).toEqual(source.placed.rect);
    expect(selection.mask()?.rect).toEqual(source.placed.rect);
    expect(selection.mask()?.surface).not.toBe(source.placed.surface);
    expect(selection.mask()?.surface).not.toBe(previousSurface);
    expect(selection.antsPaths()).toEqual([{ d: 'M 7 -3 L 9 -3 L 9 -1 L 7 -1 Z' } as unknown as Path2D]);

    const put = maskLog(selection).find((entry) => entry.op === 'putImageData');
    const copied = put?.args[0] as ImageData | undefined;
    expect(copied).toBeDefined();
    expect(copied).not.toBe(source.pixels);
    expect([...copied!.data]).toEqual([...source.pixels.data]);

    source.pixels.data.fill(0);
    expect(copied!.data.filter((_, index) => index % 4 === 3)).toEqual(new Uint8ClampedArray([0, 64, 255, 0]));
    expect(onChange).toHaveBeenCalledOnce();
  });

  it('clears the selection when the replacement has no alpha', () => {
    const { selection, onChange } = createHarness();
    selection.selectAll(rectBounds(0, 0, 50, 50));
    onChange.mockClear();

    selection.replaceMask(placedMask([0, 0, 0, 0]).placed);

    expect(selection.hasSelection()).toBe(false);
    expect(selection.bounds()).toBeNull();
    expect(selection.mask()).toBeNull();
    expect(selection.antsPaths()).toEqual([]);
    expect(onChange).toHaveBeenCalledOnce();
  });

  it('detaches an empty replacement without touching the prior mask surface', () => {
    const { selection, onChange } = createHarness();
    selection.selectAll(rectBounds(1, 2, 30, 40));
    const before = selection.mask()!;
    const beforePath = selection.antsPaths()[0];
    const previousLogLength = (before.surface as StubRasterSurface).callLog.length;
    const clearRect = vi.fn(() => {
      throw new Error('prior selection surface must not be cleared');
    });
    Object.defineProperty(before.surface.ctx, 'clearRect', { value: clearRect });
    onChange.mockClear();

    expect(() => selection.replaceMask(placedMask([0, 0, 0, 0]).placed)).not.toThrow();

    expect(clearRect).not.toHaveBeenCalled();
    expect((before.surface as StubRasterSurface).callLog).toHaveLength(previousLogLength);
    expect(before.rect).toEqual(rectBounds(1, 2, 30, 40));
    expect(beforePath).toBeDefined();
    expect(selection.hasSelection()).toBe(false);
    expect(selection.bounds()).toBeNull();
    expect(selection.mask()).toBeNull();
    expect(selection.antsPaths()).toEqual([]);
    expect(onChange).toHaveBeenCalledOnce();
  });

  it('preserves the exact prior selection when staging replacement pixels fails', () => {
    const { backend, selection, onChange } = createHarness();
    selection.selectAll(rectBounds(1, 2, 30, 40));
    const before = selection.mask()!;
    const beforePaths = selection.antsPaths();
    const createSurface = backend.createSurface;
    vi.spyOn(backend, 'createSurface').mockImplementation((width, height) => {
      const surface = createSurface(width, height);
      Object.defineProperty(surface.ctx, 'putImageData', {
        value: () => {
          throw new Error('selection pixel staging failed');
        },
      });
      return surface;
    });
    onChange.mockClear();

    expect(() => selection.replaceMask(placedMask([0, 64, 255, 0]).placed)).toThrow('selection pixel staging failed');

    expect(selection.mask()?.surface).toBe(before.surface);
    expect(selection.mask()?.rect).toEqual(before.rect);
    expect(selection.bounds()).toEqual(before.rect);
    expect(selection.antsPaths()).toHaveLength(beforePaths.length);
    expect(selection.antsPaths()[0]).toBe(beforePaths[0]);
    expect(selection.hasSelection()).toBe(true);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('preserves the exact prior selection when staging its ants path fails', () => {
    const backend = createTestStubRasterBackend();
    const onChange = vi.fn();
    let failPath = false;
    const selection = createSelectionState({
      backend,
      createPath2D: (d) => {
        if (failPath) {
          throw new Error('selection path staging failed');
        }
        return { d } as unknown as Path2D;
      },
      getDocumentSize: () => DOC,
      onChange,
    });
    selection.selectAll(rectBounds(1, 2, 30, 40));
    const before = selection.mask()!;
    const beforePaths = selection.antsPaths();
    failPath = true;
    onChange.mockClear();

    expect(() => selection.replaceMask(placedMask([0, 64, 255, 0]).placed)).toThrow('selection path staging failed');

    expect(selection.mask()?.surface).toBe(before.surface);
    expect(selection.mask()?.rect).toEqual(before.rect);
    expect(selection.bounds()).toEqual(before.rect);
    expect(selection.antsPaths()).toHaveLength(beforePaths.length);
    expect(selection.antsPaths()[0]).toBe(beforePaths[0]);
    expect(selection.hasSelection()).toBe(true);
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('selectionState: teardown + guards', () => {
  it('commit is a no-op with no document size', () => {
    const { selection, onChange } = createHarness(null);
    selection.commit({ bounds: rectBounds(0, 0, 10, 10), op: 'replace', path: fakePath('a') });
    expect(selection.hasSelection()).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('dispose clears state (document-replace teardown path)', () => {
    const { selection } = createHarness();
    selection.commit({ bounds: rectBounds(0, 0, 10, 10), op: 'replace', path: fakePath('a') });
    selection.dispose();
    expect(selection.hasSelection()).toBe(false);
    expect(selection.mask()).toBeNull();
  });
});
