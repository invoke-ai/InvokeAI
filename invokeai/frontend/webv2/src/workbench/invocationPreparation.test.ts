import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it } from 'vitest';

import { beginInvocationPreparation, endInvocationPreparation, isInvocationPreparing } from './invocationPreparation';

describe('invocation preparation', () => {
  it('does not let a stale account lease release a new preparation with the same project id', () => {
    const projectId = 'shared-project';
    const staleLease = beginInvocationPreparation(projectId);
    expect(staleLease).not.toBeNull();

    accountLifecycle.invalidate();
    const currentLease = beginInvocationPreparation(projectId);
    expect(currentLease).not.toBeNull();

    endInvocationPreparation(staleLease!);
    expect(isInvocationPreparing(projectId)).toBe(true);

    endInvocationPreparation(currentLease!);
    expect(isInvocationPreparing(projectId)).toBe(false);
  });
});
