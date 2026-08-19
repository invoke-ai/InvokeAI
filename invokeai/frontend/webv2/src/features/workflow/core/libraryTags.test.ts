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
  it('folds rows that differ only in casing and sums their counts', () => {
    expect(
      mergeTagCountsByCase([
        { count: 2, tag: 'sdxl' },
        { count: 5, tag: 'SDXL' },
        { count: 1, tag: 'lora' },
      ])
    ).toEqual([
      { count: 7, tag: 'SDXL' },
      { count: 1, tag: 'lora' },
    ]);
  });

  it('displays the casing of the single biggest contributing row, not the first seen', () => {
    expect(
      mergeTagCountsByCase([
        { count: 3, tag: 'Upscaling' },
        { count: 4, tag: 'upscaling' },
        { count: 2, tag: 'UPSCALING' },
      ])
    ).toEqual([{ count: 9, tag: 'upscaling' }]);
  });

  it('breaks a count tie lexicographically so the chip label is stable', () => {
    expect(
      mergeTagCountsByCase([
        { count: 3, tag: 'SDXL' },
        { count: 3, tag: 'sdxl' },
      ])
    ).toEqual([{ count: 6, tag: 'sdxl' }]);
    // Same rows, opposite input order — the winner must not depend on it.
    expect(
      mergeTagCountsByCase([
        { count: 3, tag: 'sdxl' },
        { count: 3, tag: 'SDXL' },
      ])
    ).toEqual([{ count: 6, tag: 'sdxl' }]);
  });

  it('leaves the input rows untouched', () => {
    const counts = [
      { count: 1, tag: 'SDXL' },
      { count: 2, tag: 'sdxl' },
    ] as const;

    mergeTagCountsByCase(counts);

    expect(counts).toEqual([
      { count: 1, tag: 'SDXL' },
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

  it('merges case-duplicate rows before ordering them', () => {
    expect(
      sortTagCounts([
        { count: 2, tag: 'sdxl' },
        { count: 3, tag: 'lora' },
        { count: 2, tag: 'SDXL' },
      ])
    ).toEqual([
      { count: 4, tag: 'sdxl' },
      { count: 3, tag: 'lora' },
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
