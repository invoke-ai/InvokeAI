import { describe, expect, it } from 'vitest';

import { getFocusedRegion, setFocusedRegion } from './focus';

describe('focus regions', () => {
  it('supports the workflow editor region independently of the workflows region', () => {
    setFocusedRegion('workflowEditor');
    expect(getFocusedRegion()).toBe('workflowEditor');

    setFocusedRegion('workflows');
    expect(getFocusedRegion()).toBe('workflows');

    setFocusedRegion(null);
    expect(getFocusedRegion()).toBe(null);
  });
});
