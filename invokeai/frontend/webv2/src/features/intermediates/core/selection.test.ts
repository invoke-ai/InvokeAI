import { describe, expect, it } from 'vitest';

import type { IntermediatesRow } from './types';

import {
  EMPTY_SELECTION,
  isRowSelected,
  resolveScope,
  selectAllMatching,
  summarizeSelection,
  toggleRowSelection,
  withRowSelected,
} from './selection';

const row = (projectId: string | null, overrides: Partial<IntermediatesRow> = {}): IntermediatesRow => ({
  coverImageName: null,
  images: { active: 0, recent: 0, referenced: 1, safe: 3 },
  projectId,
  projectName: projectId,
  reclaimableBytes: 100,
  referencedBytes: 10,
  unknownSizeCount: 0,
  userDisplayName: 'Alice',
  userEmail: 'alice@example.com',
  userId: 'alice',
  videos: { active: 0, recent: 0, referenced: 0, safe: 2 },
  ...overrides,
});

const page = [row('a'), row('b'), row(null)];
const totals = { reclaimableBytes: 5_000, rows: 12, safeImages: 40, safeVideos: 4, unknownSizeCount: 2 };
const noFilters = { ownerId: 'alice', projectId: null, search: '' };

describe('row selection', () => {
  it('toggles rows', () => {
    let selection = toggleRowSelection(EMPTY_SELECTION, page[0]!);
    expect(isRowSelected(selection, page[0]!)).toBe(true);

    for (const candidate of page.slice(1)) {
      selection = toggleRowSelection(selection, candidate);
    }
    expect(page.every((candidate) => isRowSelected(selection, candidate))).toBe(true);
  });

  it('keeps the unassigned row distinct from a project row of the same owner', () => {
    const selection = toggleRowSelection(EMPTY_SELECTION, page[2]!);

    expect(isRowSelected(selection, page[2]!)).toBe(true);
    expect(isRowSelected(selection, page[0]!)).toBe(false);
  });

  it('keeps picks from an earlier page when the page changes', () => {
    const firstPage = [row('a'), row('b')];
    const secondPage = [row('c'), row('d')];
    let selection = toggleRowSelection(EMPTY_SELECTION, firstPage[0]!);
    selection = toggleRowSelection(selection, secondPage[1]!);

    expect(summarizeSelection(selection, totals).rows).toBe(2);
    expect(resolveScope({ ...noFilters, selection })).toEqual({
      kind: 'selection',
      targets: [
        { projectId: 'a', userId: 'alice' },
        { projectId: 'd', userId: 'alice' },
      ],
    });
  });

  it('adds a row without disturbing an existing pick or an all-matching selection', () => {
    const picked = withRowSelected(withRowSelected(EMPTY_SELECTION, page[0]!), page[0]!);
    expect(summarizeSelection(picked, totals).rows).toBe(1);
    expect(withRowSelected(selectAllMatching(), page[1]!)).toEqual(selectAllMatching());
  });

  it('excludes a row without losing matching rows on other pages', () => {
    const selection = toggleRowSelection(selectAllMatching(), page[1]!);

    expect(selection.mode).toBe('all-matching');
    expect(isRowSelected(selection, row('off-page'))).toBe(true);
    expect(summarizeSelection(selection, totals)).toEqual({
      reclaimableBytes: 4_900,
      rows: 11,
      safeImages: 37,
      safeVideos: 2,
      unknownSizeCount: 2,
    });
    expect(toggleRowSelection(selection, page[1]!)).toEqual(selectAllMatching());
    expect(resolveScope({ ...noFilters, selection })).toEqual({
      excluded: [{ projectId: 'b', userId: 'alice' }],
      kind: 'matching',
      projectId: null,
      search: null,
      userId: 'alice',
    });
    expect(isRowSelected(selection, page[0]!)).toBe(true);
    expect(isRowSelected(selection, page[1]!)).toBe(false);
    expect(isRowSelected(selection, page[2]!)).toBe(true);
  });
});

describe('selection summary', () => {
  it('sums explicit picks from their snapshots', () => {
    const selection = toggleRowSelection(toggleRowSelection(EMPTY_SELECTION, page[0]!), page[2]!);

    expect(summarizeSelection(selection, totals)).toEqual({
      reclaimableBytes: 200,
      rows: 2,
      safeImages: 6,
      safeVideos: 4,
      unknownSizeCount: 0,
    });
  });

  it('uses the server totals for all matching rows, which may exceed the loaded page', () => {
    expect(summarizeSelection(selectAllMatching(), totals)).toEqual(totals);
  });

  it('refreshes a snapshot from the visible page before counting it', () => {
    const refreshed = row('b', { images: { active: 0, recent: 0, referenced: 1, safe: 5 }, reclaimableBytes: 1_000 });

    const excluded = toggleRowSelection(selectAllMatching(), page[1]!);
    expect(summarizeSelection(excluded, totals, [page[0]!, refreshed])).toMatchObject({
      reclaimableBytes: 4_000,
      rows: 11,
      safeImages: 35,
    });
    const picked = toggleRowSelection(EMPTY_SELECTION, page[1]!);
    expect(summarizeSelection(picked, totals, [refreshed])).toMatchObject({ reclaimableBytes: 1_000, safeImages: 5 });
  });

  it('never estimates all-matching below what is visibly selected when an excluded row has vanished', () => {
    // Totals now cover one row; the excluded row is gone off-page, but its snapshot would cancel the survivor.
    const survivor = row('c');
    const selection = toggleRowSelection(selectAllMatching(), page[0]!);
    const fewTotals = { reclaimableBytes: 100, rows: 1, safeImages: 3, safeVideos: 2, unknownSizeCount: 0 };

    expect(summarizeSelection(selection, fewTotals, [survivor])).toEqual({
      reclaimableBytes: 100,
      rows: 1,
      safeImages: 3,
      safeVideos: 2,
      unknownSizeCount: 0,
    });
    expect(summarizeSelection(selection, fewTotals, [])).toEqual({
      reclaimableBytes: 0,
      rows: 0,
      safeImages: 0,
      safeVideos: 0,
      unknownSizeCount: 0,
    });
  });
});

describe('scope resolution', () => {
  it('turns all-matching without a filter into the owner or everyone scope', () => {
    expect(resolveScope({ ...noFilters, selection: selectAllMatching() })).toEqual({
      kind: 'owner',
      userId: 'alice',
    });
    expect(resolveScope({ ...noFilters, ownerId: null, selection: selectAllMatching() })).toEqual({
      kind: 'everyone',
    });
  });

  it('turns picks into explicit targets whichever page showed them, ignoring the filters', () => {
    const picks = toggleRowSelection(EMPTY_SELECTION, page[2]!);

    expect(resolveScope({ ownerId: 'alice', projectId: 'p', search: 'port', selection: picks })).toEqual({
      kind: 'selection',
      targets: [{ projectId: null, userId: 'alice' }],
    });
  });

  it('carries the filters of an all-matching selection in a matching scope', () => {
    expect(
      resolveScope({ ownerId: 'alice', projectId: null, search: ' port ', selection: selectAllMatching() })
    ).toEqual({ excluded: [], kind: 'matching', projectId: null, search: 'port', userId: 'alice' });
    expect(resolveScope({ ownerId: null, projectId: 'p', search: '', selection: selectAllMatching() })).toEqual({
      excluded: [],
      kind: 'matching',
      projectId: 'p',
      search: null,
      userId: null,
    });
  });
});
