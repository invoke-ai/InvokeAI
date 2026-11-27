import { describe, expect, it } from 'vitest';

import { transitionTextSessionStatus } from './textSessionMachine';

describe('textSessionMachine', () => {
  it('transitions from idle -> pending -> editing -> committed', () => {
    let status = transitionTextSessionStatus('idle', 'BEGIN');
    expect(status).toBe('pending');
    status = transitionTextSessionStatus(status, 'EDIT');
    expect(status).toBe('editing');
    status = transitionTextSessionStatus(status, 'COMMIT');
    expect(status).toBe('committed');
  });

  it('resets to idle on cancel from any state', () => {
    expect(transitionTextSessionStatus('pending', 'CANCEL')).toBe('idle');
    expect(transitionTextSessionStatus('editing', 'CANCEL')).toBe('idle');
    expect(transitionTextSessionStatus('committed', 'CANCEL')).toBe('idle');
  });

  it('ignores invalid transitions', () => {
    expect(transitionTextSessionStatus('pending', 'BEGIN')).toBe('pending');
    expect(transitionTextSessionStatus('editing', 'EDIT')).toBe('editing');
  });
});
