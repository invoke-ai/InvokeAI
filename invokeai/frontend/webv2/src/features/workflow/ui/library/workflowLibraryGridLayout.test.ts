import { describe, expect, it } from 'vitest';

import {
  getWorkflowLibraryCardWidth,
  getWorkflowLibraryGridBounds,
  getWorkflowLibraryGridColumns,
  MAX_GRID_COLUMNS,
} from './workflowLibraryGridLayout';

describe('getWorkflowLibraryGridColumns', () => {
  it('fits as many cards as the size allows, from one to six', () => {
    expect(getWorkflowLibraryGridColumns(0, 'small')).toBe(1);
    expect(getWorkflowLibraryGridColumns(500, 'small')).toBe(3);
    expect(getWorkflowLibraryGridColumns(1000, 'small')).toBe(6);
    expect(getWorkflowLibraryGridColumns(3000, 'small')).toBe(6);
    expect(getWorkflowLibraryGridColumns(1000, 'large')).toBe(4);
    expect(getWorkflowLibraryGridColumns(3000, 'large')).toBe(6);
  });

  it.each([
    ['small', 150],
    ['large', 220],
  ] as const)('never narrows a %s card below %ipx once there is room for two', (size, minimum) => {
    for (let width = 200; width <= 2400; width += 7) {
      const columns = getWorkflowLibraryGridColumns(width, size);

      if (columns > 1) {
        expect(getWorkflowLibraryCardWidth(width, columns)).toBeGreaterThanOrEqual(minimum);
      }
      if (columns < MAX_GRID_COLUMNS) {
        // One more column would not have fitted.
        expect(getWorkflowLibraryCardWidth(width, columns + 1)).toBeLessThan(minimum);
      }
    }
  });
});

describe('getWorkflowLibraryGridBounds', () => {
  it('keeps the spanned rows within 48 enriched identities and, with overscan, 60 resident positions', () => {
    for (let columns = 1; columns <= MAX_GRID_COLUMNS; columns += 1) {
      const { maxViewportRows, overscanRows } = getWorkflowLibraryGridBounds(columns);
      // A viewport capped at R estimated rows spans at most R + 1 rows.
      const spannedRows = maxViewportRows + 1;

      expect(overscanRows).toBeGreaterThanOrEqual(1);
      expect(spannedRows * columns).toBeLessThanOrEqual(48);
      expect((spannedRows + 2 * overscanRows) * columns).toBeLessThanOrEqual(60);
    }
  });

  it('keeps fifteen rows and two overscan rows at three columns, and seven rows with one at six', () => {
    expect(getWorkflowLibraryGridBounds(3)).toEqual({ maxViewportRows: 15, overscanRows: 2 });
    expect(getWorkflowLibraryGridBounds(6)).toEqual({ maxViewportRows: 7, overscanRows: 1 });
  });
});
