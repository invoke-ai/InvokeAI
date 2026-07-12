import { describe, expect, it } from 'vitest';

import { getVideoPrefetchOptions } from './useRangeBasedImageFetching';

describe('video range prefetch', () => {
  it('does not retain an RTK Query subscription', () => {
    expect(getVideoPrefetchOptions()).toEqual({ subscribe: false });
  });
});
