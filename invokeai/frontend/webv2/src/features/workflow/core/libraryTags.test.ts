import { describe, expect, it } from 'vitest';

import { parseWorkflowTags, sortTagCounts } from './libraryTags';

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
