import { describe, expect, it } from 'vitest';

import { getInvocationNodeBodyComponentKey } from './getInvocationNodeBodyComponent';

describe('getInvocationNodeBodyComponentKey', () => {
  it('returns the specialized renderer for call_saved_workflows nodes', () => {
    expect(getInvocationNodeBodyComponentKey('call_saved_workflows')).toBe('call_saved_workflows');
  });

  it('falls back to the default renderer for other invocation nodes', () => {
    expect(getInvocationNodeBodyComponentKey('add')).toBe('default');
  });
});
