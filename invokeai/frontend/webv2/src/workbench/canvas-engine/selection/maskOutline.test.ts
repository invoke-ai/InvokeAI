import { describe, expect, it } from 'vitest';

import { traceMaskOutlinePath } from './maskOutline';

/** Builds RGBA data from a row-major alpha grid. */
const alphaGrid = (
  rows: readonly (readonly number[])[]
): { data: Uint8ClampedArray; height: number; width: number } => {
  const height = rows.length;
  const width = rows[0]?.length ?? 0;
  const data = new Uint8ClampedArray(width * height * 4);
  rows.forEach((row, y) =>
    row.forEach((alpha, x) => {
      data[(y * width + x) * 4 + 3] = alpha;
    })
  );
  return { data, height, width };
};

const ORIGIN = { x: 0, y: 0 };

describe('traceMaskOutlinePath', () => {
  it('returns an empty path for an empty or sub-threshold mask', () => {
    expect(traceMaskOutlinePath(alphaGrid([]), ORIGIN)).toBe('');
    expect(
      traceMaskOutlinePath(
        alphaGrid([
          [0, 0],
          [0, 0],
        ]),
        ORIGIN
      )
    ).toBe('');
    expect(
      traceMaskOutlinePath(
        alphaGrid([
          [64, 64],
          [64, 64],
        ]),
        ORIGIN
      )
    ).toBe('');
  });

  it('honors a lower threshold for faint coverage', () => {
    const grid = alphaGrid([[64]]);
    expect(traceMaskOutlinePath(grid, ORIGIN)).toBe('');
    expect(traceMaskOutlinePath(grid, ORIGIN, 1)).toBe('M 0 0 L 1 0 L 1 1 L 0 1 Z');
  });

  it('traces a full mask as its bounding rectangle with merged collinear runs', () => {
    const grid = alphaGrid([
      [255, 255, 255],
      [255, 255, 255],
    ]);
    expect(traceMaskOutlinePath(grid, { x: 10, y: 20 })).toBe('M 10 20 L 13 20 L 13 22 L 10 22 Z');
  });

  it('traces a single interior pixel offset by the document-space origin', () => {
    const grid = alphaGrid([
      [0, 0, 0],
      [0, 255, 0],
      [0, 0, 0],
    ]);
    expect(traceMaskOutlinePath(grid, { x: 7, y: -3 })).toBe('M 8 -2 L 9 -2 L 9 -1 L 8 -1 Z');
  });

  it('emits a counter-clockwise inner loop for a hole', () => {
    const grid = alphaGrid([
      [255, 255, 255],
      [255, 0, 255],
      [255, 255, 255],
    ]);
    const path = traceMaskOutlinePath(grid, ORIGIN);
    const loops = path.split(' Z').filter((segment) => segment.trim().length > 0);
    expect(loops).toHaveLength(2);
    expect(path).toContain('M 0 0 L 3 0 L 3 3 L 0 3 Z');
    // The hole loop winds opposite to the outer loop (counter-clockwise), so
    // nonzero fill keeps the hole.
    expect(path).toContain('M 2 1 L 1 1 L 1 2 L 2 2 Z');
  });

  it('keeps diagonally touching regions as two separate loops', () => {
    const grid = alphaGrid([
      [255, 0],
      [0, 255],
    ]);
    const path = traceMaskOutlinePath(grid, ORIGIN);
    const loops = path.split(' Z').filter((segment) => segment.trim().length > 0);
    expect(loops).toHaveLength(2);
    expect(path).toContain('M 0 0 L 1 0 L 1 1 L 0 1 Z');
    expect(path).toContain('M 1 1 L 2 1 L 2 2 L 1 2 Z');
  });

  it('traces disjoint regions as independent closed loops', () => {
    const grid = alphaGrid([
      [255, 0, 255],
      [255, 0, 255],
    ]);
    const path = traceMaskOutlinePath(grid, ORIGIN);
    expect(path).toContain('M 0 0 L 1 0 L 1 2 L 0 2 Z');
    expect(path).toContain('M 2 0 L 3 0 L 3 2 L 2 2 Z');
  });
});
