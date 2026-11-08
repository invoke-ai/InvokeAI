import {
  getProjectsColumnCount,
  getProjectsGridRowHeight,
  PROJECT_CARD_META_HEIGHT_PX,
  PROJECTS_GRID_GAP_PX,
} from '@workbench/launchpad/projects/projectsMetrics';
import { describe, expect, it } from 'vitest';

const ASPECT_RATIO = 16 / 10;

describe('getProjectsColumnCount', () => {
  it('fits as many 240px cards as the width allows, up to four', () => {
    expect(getProjectsColumnCount(239)).toBe(1);
    expect(getProjectsColumnCount(480)).toBe(2);
    expect(getProjectsColumnCount(760)).toBe(3);
    expect(getProjectsColumnCount(4000)).toBe(4);
  });

  it('never returns zero for an unmeasured or collapsed container', () => {
    expect(getProjectsColumnCount(0)).toBe(1);
    expect(getProjectsColumnCount(-100)).toBe(1);
  });
});

describe('getProjectsGridRowHeight', () => {
  it('sizes a row from the column width, the cover ratio, and the meta block', () => {
    // Three 240px columns with two 16px gaps == 752px.
    const height = getProjectsGridRowHeight({ aspectRatio: ASPECT_RATIO, columnCount: 3, widthPx: 752 });

    expect(height).toBe(Math.round(240 / ASPECT_RATIO + PROJECT_CARD_META_HEIGHT_PX + PROJECTS_GRID_GAP_PX));
  });

  it('grows the row as the container widens', () => {
    const narrow = getProjectsGridRowHeight({ aspectRatio: ASPECT_RATIO, columnCount: 2, widthPx: 600 });
    const wide = getProjectsGridRowHeight({ aspectRatio: ASPECT_RATIO, columnCount: 2, widthPx: 1000 });

    expect(wide).toBeGreaterThan(narrow);
  });

  it('falls back to the meta height before the container has been measured', () => {
    expect(getProjectsGridRowHeight({ aspectRatio: ASPECT_RATIO, columnCount: 3, widthPx: 0 })).toBe(
      PROJECT_CARD_META_HEIGHT_PX + PROJECTS_GRID_GAP_PX
    );
  });

  it('treats a nonsensical column count as one column', () => {
    expect(getProjectsGridRowHeight({ aspectRatio: ASPECT_RATIO, columnCount: 0, widthPx: 400 })).toBe(
      getProjectsGridRowHeight({ aspectRatio: ASPECT_RATIO, columnCount: 1, widthPx: 400 })
    );
  });
});
