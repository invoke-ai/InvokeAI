import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { PaintLineEndController } from './paintLineEndController';

const paint = (x: number, offsetX: number, overrides: Partial<CanvasLayerContract> = {}): CanvasLayerContract =>
  layerContract('paint', 'raster', {
    source: { bitmap: null, offset: { x: offsetX, y: 0 }, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 0 },
    ...overrides,
  } as Partial<CanvasLayerContract>);

const createHarness = (initial: CanvasLayerContract) => {
  let layer: CanvasLayerContract | null = initial;
  const ends = new PaintLineEndController({ getLayer: () => layer });
  const change = (next: CanvasLayerContract | null) => {
    layer = next;
    ends.onLayerChanged('paint');
  };
  return { change, ends };
};

describe('PaintLineEndController', () => {
  it('carries the stroke end with a layer that only moves', () => {
    const h = createHarness(paint(0, 0));
    h.ends.set('paint', { x: 60, y: 10 });

    h.change(paint(100, 0));

    expect(h.ends.get('paint')).toEqual({ x: 60, y: 10 });
  });

  it('keeps the document point when content and transform are rewritten together, as a crop does', () => {
    const h = createHarness(paint(100, 0));
    h.ends.set('paint', { x: 60, y: 10 });

    // Cropped: the pixels stay where they were, now at an identity transform.
    h.change(paint(0, 100));

    expect(h.ends.get('paint')).toEqual({ x: 160, y: 10 });
  });

  it('follows pixels redrawn from a Transform session into a new local space', () => {
    const h = createHarness(paint(0, 0));
    h.ends.set('paint', { x: 60, y: 10 });

    h.ends.rebase('paint', { rotation: 0, scaleX: 2, scaleY: 2, x: 100, y: 0 });

    expect(h.ends.get('paint')).toEqual({ x: 220, y: 20 });
  });

  it('forgets the end of a removed layer and one a degenerate transform cannot hold', () => {
    const removed = createHarness(paint(0, 0));
    removed.ends.set('paint', { x: 1, y: 1 });
    removed.change(null);
    expect(removed.ends.get('paint')).toBeUndefined();

    const flattened = createHarness(paint(0, 0));
    flattened.ends.set('paint', { x: 1, y: 1 });
    flattened.change(paint(0, 5, { transform: { rotation: 0, scaleX: 0, scaleY: 1, x: 0, y: 0 } }));
    expect(flattened.ends.get('paint')).toBeUndefined();
  });
});
