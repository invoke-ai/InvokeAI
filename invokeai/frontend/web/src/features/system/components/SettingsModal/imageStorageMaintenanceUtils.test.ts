import { describe, expect, it } from 'vitest';

import { isTerminalImageMoveJobState } from './imageStorageMaintenanceUtils';

describe(isTerminalImageMoveJobState.name, () => {
  it('recognizes committed and partial-success jobs as terminal', () => {
    expect(isTerminalImageMoveJobState('committed')).toBe(true);
    expect(isTerminalImageMoveJobState('error')).toBe(true);
  });

  it('does not recognize active jobs as terminal', () => {
    expect(isTerminalImageMoveJobState('planned')).toBe(false);
    expect(isTerminalImageMoveJobState('moving')).toBe(false);
    expect(isTerminalImageMoveJobState('moved')).toBe(false);
    expect(isTerminalImageMoveJobState(undefined)).toBe(false);
  });
});
