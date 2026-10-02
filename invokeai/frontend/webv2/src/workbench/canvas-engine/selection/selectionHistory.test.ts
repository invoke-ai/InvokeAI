import type { Rect } from '@workbench/canvas-engine/types';

import { createHistory } from '@workbench/canvas-engine/history/history';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createSelectionState } from '@workbench/canvas-engine/selection/selectionState';
import { describe, expect, it, vi } from 'vitest';

import { withSelectionHistory } from './selectionHistory';

const DOC = { height: 100, width: 100 };
const rectBounds = (x: number, y: number, w: number, h: number): Rect => ({ height: h, width: w, x, y });
const fakePath = (id: string): Path2D => ({ id }) as unknown as Path2D;

const createHarness = (options: Parameters<typeof createHistory>[0] = {}) => {
  const history = createHistory(options);
  const onChange = vi.fn();
  const selection = withSelectionHistory(
    createSelectionState({
      // No readable pixels: boolean ops fall back to path bookkeeping, so the
      // history is exercised through what it records rather than a traced mask.
      backend: createTestStubRasterBackend({ readbackAlpha: 0 }),
      createPath2D: (d) => ({ d }) as unknown as Path2D,
      getDocumentSize: () => DOC,
      onChange,
    }),
    history
  );
  return { history, onChange, selection };
};

describe('withSelectionHistory', () => {
  it('records each selection change as one labelled step', () => {
    const { history, selection } = createHarness();
    selection.selectAll(rectBounds(0, 0, 100, 100));
    selection.commit({ bounds: rectBounds(10, 10, 20, 20), op: 'subtract', path: fakePath('a') });
    selection.invert(rectBounds(0, 0, 100, 100));
    selection.clear();
    expect(history.entries().past).toEqual(['Select all', 'Subtract from selection', 'Invert selection', 'Deselect']);
  });

  it('undo and redo replay the captured selections without recording again', async () => {
    const { history, selection } = createHarness();
    selection.commit({ bounds: rectBounds(10, 10, 20, 20), op: 'replace', path: fakePath('a') });
    selection.commit({ bounds: rectBounds(40, 40, 10, 10), op: 'add', path: fakePath('b') });
    expect(selection.bounds()).toEqual(rectBounds(10, 10, 40, 40));

    await history.undo();
    expect(selection.bounds()).toEqual(rectBounds(10, 10, 20, 20));
    expect(selection.antsPaths()).toHaveLength(1);
    await history.undo();
    expect(selection.hasSelection()).toBe(false);
    expect(selection.antsPaths()).toEqual([]);
    await history.redo();
    await history.redo();
    expect(selection.bounds()).toEqual(rectBounds(10, 10, 40, 40));
    expect(selection.antsPaths()).toHaveLength(2);
    expect(history.entries()).toEqual({ future: [], past: ['Select', 'Add to selection'] });
  });

  it('records nothing for a call that leaves the selection unchanged', () => {
    const { history, selection } = createHarness();
    selection.clear();
    selection.commit({ bounds: rectBounds(10, 10, 20, 20), op: 'intersect', path: fakePath('a') });
    expect(history.canUndo()).toBe(false);
  });

  it('records a pixel-mask replacement as Select object and undoes it to the prior selection', async () => {
    const { history, selection } = createHarness();
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(2, 2);
    const pixels = {
      colorSpace: 'srgb',
      data: new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      height: 2,
      width: 2,
    } as ImageData;
    Object.defineProperty(surface.ctx, 'getImageData', { value: () => pixels });

    selection.replaceMask({ rect: rectBounds(7, -3, 2, 2), surface });

    expect(history.entries().past).toEqual(['Select object']);
    expect(selection.bounds()).toEqual(rectBounds(7, -3, 2, 2));
    await history.undo();
    expect(selection.hasSelection()).toBe(false);
  });

  it('labels a degenerate replace as the deselect it performs', () => {
    const { history, selection } = createHarness();
    selection.selectAll(rectBounds(0, 0, 100, 100));
    selection.commit({ bounds: rectBounds(5, 5, 0, 0), op: 'replace', path: fakePath('tiny') });
    expect(history.entries().past).toEqual(['Select all', 'Deselect']);
    expect(selection.hasSelection()).toBe(false);
  });

  it('charges a plane shared between consecutive steps once', () => {
    const { history, selection } = createHarness();
    selection.selectAll(rectBounds(0, 0, 100, 100));
    expect(history.byteSize()).toBe(100 * 100);
    selection.invert(rectBounds(0, 0, 100, 100));
    // The inverted plane is new; the select-all plane it started from was already charged.
    expect(history.byteSize()).toBe(2 * 100 * 100);
  });

  it('rolls a change back exactly when its step could never be kept', () => {
    // A 100×100 selection plane alone exceeds the whole budget.
    const { history, selection } = createHarness({ byteBudget: 100 });

    selection.selectAll(rectBounds(0, 0, 100, 100));
    expect(selection.hasSelection()).toBe(false);
    expect(history.entries().past).toEqual([]);
  });

  it('refuses a change while another step replays', async () => {
    const { history, selection } = createHarness();
    let finish = (): void => undefined;
    history.admit(0)!.publish({
      bytes: 0,
      heldAssetRefs: { images: [], videos: [] },
      label: 'Slow step',
      redo: () => undefined,
      undo: () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    });
    const replay = history.undo();

    selection.selectAll(rectBounds(0, 0, 100, 100));
    expect(selection.hasSelection()).toBe(false);
    finish();
    await replay;
    expect(history.entries()).toEqual({ future: ['Slow step'], past: [] });
  });
});
