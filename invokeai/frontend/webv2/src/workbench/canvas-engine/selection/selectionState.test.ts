import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { Rect } from '@workbench/canvas-engine/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createSelectionState, type SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import { describe, expect, it, vi } from 'vitest';

const DOC = { height: 100, width: 100 };

const rectBounds = (x: number, y: number, w: number, h: number): Rect => ({ height: h, width: w, x, y });

/** A fake Path2D carrier (Path2D is absent in node); the stub ctx just records it. */
const fakePath = (id: string): Path2D => ({ id }) as unknown as Path2D;

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
