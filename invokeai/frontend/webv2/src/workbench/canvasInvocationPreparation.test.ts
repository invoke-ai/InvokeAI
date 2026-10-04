import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it } from 'vitest';

import {
  beginCanvasInvocationPreparation,
  endCanvasInvocationPreparation,
  isCanvasInvocationPreparing,
} from './canvasInvocationPreparation';

describe('canvas invocation preparation', () => {
  it('does not let a stale account lease release a new preparation with the same project id', () => {
    const projectId = 'shared-project';
    const staleLease = beginCanvasInvocationPreparation(projectId);
    expect(staleLease).not.toBeNull();

    accountLifecycle.invalidate();
    const currentLease = beginCanvasInvocationPreparation(projectId);
    expect(currentLease).not.toBeNull();

    endCanvasInvocationPreparation(staleLease!);
    expect(isCanvasInvocationPreparing(projectId)).toBe(true);

    endCanvasInvocationPreparation(currentLease!);
    expect(isCanvasInvocationPreparing(projectId)).toBe(false);
  });
});
