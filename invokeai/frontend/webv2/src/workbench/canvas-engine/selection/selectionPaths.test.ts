import { describe, expect, it } from 'vitest';

import { ellipsePathData, rectPathData, selectionShapePathData } from './selectionPaths';

describe('rectPathData', () => {
  it('traces the four corners clockwise and closes', () => {
    expect(rectPathData({ height: 30, width: 40, x: 10, y: 20 })).toBe('M 10 20 L 50 20 L 50 50 L 10 50 Z');
  });

  it('handles a negative origin', () => {
    expect(rectPathData({ height: 4, width: 6, x: -5, y: -2 })).toBe('M -5 -2 L 1 -2 L 1 2 L -5 2 Z');
  });
});

describe('ellipsePathData', () => {
  it('traces two half-arcs between the horizontal extremes and closes', () => {
    // Inscribed in [0,0,40,20]: radii 20×10, spanning x 0→40 at y 10.
    expect(ellipsePathData({ height: 20, width: 40, x: 0, y: 0 })).toBe(
      'M 0 10 A 20 10 0 1 0 40 10 A 20 10 0 1 0 0 10 Z'
    );
  });

  it('offsets with the rect and tolerates half-pixel radii', () => {
    expect(ellipsePathData({ height: 5, width: 3, x: 10, y: 20 })).toBe(
      'M 10 22.5 A 1.5 2.5 0 1 0 13 22.5 A 1.5 2.5 0 1 0 10 22.5 Z'
    );
  });
});

describe('selectionShapePathData', () => {
  const rect = { height: 30, width: 40, x: 10, y: 20 };

  it('dispatches on the shape kind', () => {
    expect(selectionShapePathData('rect', rect)).toBe(rectPathData(rect));
    expect(selectionShapePathData('ellipse', rect)).toBe(ellipsePathData(rect));
  });
});
