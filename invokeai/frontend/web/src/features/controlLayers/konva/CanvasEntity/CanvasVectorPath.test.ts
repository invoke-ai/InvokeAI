import { describe, expect, it, vi } from 'vitest';

vi.mock('konva', async () => {
  const { Path } = await import('konva/lib/shapes/Path');
  return { default: { Path } };
});

import { CanvasVectorPath } from './CanvasVectorPath';

describe('vector path bounds', () => {
  it('uses exact bounds for the original and cloned transform preview', () => {
    const bounds = { x: 0, y: 0, width: 1000, height: 225 };
    const node = new CanvasVectorPath({
      data: 'M 0 0 C 0 300 300 300 1000 0',
      vectorBounds: bounds,
      stroke: 'blue',
      strokeWidth: 2,
    });
    expect(node.getSelfRect()).toEqual(bounds);
    const preview = node.clone();
    expect(preview.getSelfRect()).toEqual(bounds);
    expect(preview.getClientRect()).toEqual({ x: -1, y: -1, width: 1002, height: 227 });
    node.destroy();
    preview.destroy();
  });
});
