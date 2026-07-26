import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it } from 'vitest';

import { requestLibraryWorkflowLoad, workflowUiStore } from './workflowUiStore';

describe('workflow UI account ownership', () => {
  it('clears pending UI state without reusing request identities across accounts', () => {
    accountLifecycle.activate('user-a');
    requestLibraryWorkflowLoad('workflow-a');
    const firstRequestId = workflowUiStore.getSnapshot().pendingLibraryWorkflowLoad?.requestId;

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    expect(workflowUiStore.getSnapshot().pendingLibraryWorkflowLoad).toBeNull();

    requestLibraryWorkflowLoad('workflow-b');
    const secondRequestId = workflowUiStore.getSnapshot().pendingLibraryWorkflowLoad?.requestId;

    expect(firstRequestId).toEqual(expect.any(Number));
    expect(secondRequestId).toEqual(expect.any(Number));
    expect(secondRequestId).toBeGreaterThan(firstRequestId ?? 0);
  });
});
