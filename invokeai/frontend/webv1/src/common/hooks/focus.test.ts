import { describe, expect, it } from 'vitest';

import { getFocusedRegion, setFocusedRegion } from './focus';

describe('focus regions', () => {
  it('supports the workflows region', () => {
    setFocusedRegion('workflows');
    expect(getFocusedRegion()).toBe('workflows');

    setFocusedRegion(null);
    expect(getFocusedRegion()).toBe(null);
  });
});
