import { describe, expect, it } from 'vitest';

import { dragRect } from './dragRect';

describe('dragRect', () => {
  const plain = { fromCenter: false, square: false };

  it('normalizes a drag in any direction', () => {
    expect(dragRect({ x: 10, y: 10 }, { x: 40, y: 30 }, plain)).toEqual({ height: 20, width: 30, x: 10, y: 10 });
    // Dragging up and to the left produces the same rect.
    expect(dragRect({ x: 40, y: 30 }, { x: 10, y: 10 }, plain)).toEqual({ height: 20, width: 30, x: 10, y: 10 });
  });

  it('constrains to a square under shift, keeping the drag direction', () => {
    expect(dragRect({ x: 0, y: 0 }, { x: 30, y: 10 }, { fromCenter: false, square: true })).toEqual({
      height: 30,
      width: 30,
      x: 0,
      y: 0,
    });
    // Up-left: the square grows away from the press point on both axes.
    expect(dragRect({ x: 50, y: 50 }, { x: 20, y: 40 }, { fromCenter: false, square: true })).toEqual({
      height: 30,
      width: 30,
      x: 20,
      y: 20,
    });
  });

  it('treats the press point as the centre under alt, doubling the extent', () => {
    expect(dragRect({ x: 50, y: 50 }, { x: 60, y: 70 }, { fromCenter: true, square: false })).toEqual({
      height: 40,
      width: 20,
      x: 40,
      y: 30,
    });
  });

  it('composes shift and alt into a square centred on the press point', () => {
    expect(dragRect({ x: 50, y: 50 }, { x: 60, y: 70 }, { fromCenter: true, square: true })).toEqual({
      height: 40,
      width: 40,
      x: 30,
      y: 30,
    });
  });

  it('rounds to whole pixels', () => {
    expect(dragRect({ x: 10.4, y: 10.6 }, { x: 20.4, y: 20.6 }, plain)).toEqual({
      height: 10,
      width: 10,
      x: 10,
      y: 11,
    });
  });
});
