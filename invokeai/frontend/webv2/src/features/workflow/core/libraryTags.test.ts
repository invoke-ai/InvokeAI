import { describe, expect, it } from 'vitest';

import { mergeTagCountsByCase, parseWorkflowTags, sortTagCounts } from './libraryTags';

describe('parseWorkflowTags', () => {
  it('splits the comma-separated string, trimming each tag and dropping empties', () => {
    expect(parseWorkflowTags(' lora ,upscaling,, ,sdxl ')).toEqual(['lora', 'upscaling', 'sdxl']);
  });

  it('returns an empty list for absent or blank tag strings', () => {
    expect(parseWorkflowTags(null)).toEqual([]);
    expect(parseWorkflowTags(undefined)).toEqual([]);
    expect(parseWorkflowTags('')).toEqual([]);
    expect(parseWorkflowTags('  ,  ')).toEqual([]);
  });
});

describe('mergeTagCountsByCase', () => {
  it("takes one variant's count, not their sum, because backend counts are already case-insensitive", () => {
    // `counts_by_tag` counts each requested tag with `tags LIKE '%tag%'`, and
    // SQLite's LIKE is case-insensitive — so the row for 'sdxl' and the row for
    // 'SDXL' are both the full total over the *same* two workflows. Summing
    // them showed "sdxl 4" for a library holding two.
    expect(
      mergeTagCountsByCase([
        { count: 2, tag: 'sdxl' },
        { count: 2, tag: 'SDXL' },
      ])
    ).toEqual([{ count: 2, tag: 'sdxl' }]);
  });

  it('folds rows that differ only in casing, leaving unrelated tags alone', () => {
    expect(
      mergeTagCountsByCase([
        { count: 5, tag: 'sdxl' },
        { count: 5, tag: 'SDXL' },
        { count: 1, tag: 'lora' },
      ])
    ).toEqual([
      { count: 5, tag: 'sdxl' },
      { count: 1, tag: 'lora' },
    ]);
  });

  it('takes the largest count if the variants ever disagree', () => {
    // They should not, given the backend's LIKE semantics — but a merged chip
    // that under-reports is worse than one that is merely robust.
    expect(
      mergeTagCountsByCase([
        { count: 3, tag: 'Upscaling' },
        { count: 4, tag: 'upscaling' },
        { count: 2, tag: 'UPSCALING' },
      ])
    ).toEqual([{ count: 4, tag: 'upscaling' }]);
  });

  it('labels the merged chip with the casing of the biggest row', () => {
    expect(
      mergeTagCountsByCase([
        { count: 2, tag: 'Lora' },
        { count: 6, tag: 'LoRA' },
      ])
    ).toEqual([{ count: 6, tag: 'LoRA' }]);
  });

  it('breaks a count tie lexicographically so the chip label is stable', () => {
    // The normal case: equal counts, so this tiebreak is what actually decides.
    expect(
      mergeTagCountsByCase([
        { count: 3, tag: 'SDXL' },
        { count: 3, tag: 'sdxl' },
      ])
    ).toEqual([{ count: 3, tag: 'sdxl' }]);
    // Same rows, opposite input order — the winner must not depend on it.
    expect(
      mergeTagCountsByCase([
        { count: 3, tag: 'sdxl' },
        { count: 3, tag: 'SDXL' },
      ])
    ).toEqual([{ count: 3, tag: 'sdxl' }]);
  });

  it('leaves the input rows untouched', () => {
    const counts = [
      { count: 2, tag: 'SDXL' },
      { count: 2, tag: 'sdxl' },
    ] as const;

    mergeTagCountsByCase(counts);

    expect(counts).toEqual([
      { count: 2, tag: 'SDXL' },
      { count: 2, tag: 'sdxl' },
    ]);
  });
});

describe('sortTagCounts', () => {
  it('orders by count descending, then by tag name', () => {
    expect(
      sortTagCounts([
        { count: 2, tag: 'upscaling' },
        { count: 9, tag: 'sdxl' },
        { count: 2, tag: 'lora' },
      ])
    ).toEqual([
      { count: 9, tag: 'sdxl' },
      { count: 2, tag: 'lora' },
      { count: 2, tag: 'upscaling' },
    ]);
  });

  it('merges case-duplicate rows before ordering them, without inflating their count', () => {
    expect(
      sortTagCounts([
        { count: 2, tag: 'sdxl' },
        { count: 3, tag: 'lora' },
        { count: 2, tag: 'SDXL' },
      ])
    ).toEqual([
      { count: 3, tag: 'lora' },
      { count: 2, tag: 'sdxl' },
    ]);
  });

  it('drops tags with no matching workflows and leaves the input untouched', () => {
    const counts = [
      { count: 0, tag: 'empty' },
      { count: 1, tag: 'lora' },
    ] as const;

    expect(sortTagCounts(counts)).toEqual([{ count: 1, tag: 'lora' }]);
    expect(counts).toEqual([
      { count: 0, tag: 'empty' },
      { count: 1, tag: 'lora' },
    ]);
  });
});
