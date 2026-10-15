/** Cover every supported graph builder explicitly. */

import { describe, expect, it } from 'vitest';

import { isSupportedGenerateBase, SUPPORTED_GENERATE_BASES } from './supportedBases';

describe('SUPPORTED_GENERATE_BASES', () => {
  it('is exactly the bases with a graph builder, in order', () => {
    expect(SUPPORTED_GENERATE_BASES).toEqual([
      'sd-1',
      'sd-2',
      'sdxl',
      'sd-3',
      'flux',
      'flux2',
      'cogview4',
      'ernie-image',
      'qwen-image',
      'z-image',
      'ideogram-4',
      'krea-2',
      'wan',
      'anima',
    ]);
  });

  it('excludes architectures the backend serves but this build cannot generate with', () => {
    // Refiner/video-only capability rows are not image graph support.
    expect(isSupportedGenerateBase('sdxl-refiner')).toBe(false);
    expect(isSupportedGenerateBase('minimax-h3')).toBe(false);
  });

  it('rejects unknown and non-architecture values', () => {
    expect(isSupportedGenerateBase('unknown')).toBe(false);
    expect(isSupportedGenerateBase('made-up')).toBe(false);
    expect(isSupportedGenerateBase('external')).toBe(false);
  });
});
