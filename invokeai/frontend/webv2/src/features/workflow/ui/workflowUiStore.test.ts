import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it } from 'vitest';

import { requestLibraryWorkflowLoad, requestWorkflowDocumentLoad, workflowUiStore } from './workflowUiStore';

describe('workflow UI account ownership', () => {
  it('clears pending UI state without reusing request identities across accounts', () => {
    accountLifecycle.activate('user-a');
    requestLibraryWorkflowLoad('workflow-a');
    const firstRequestId = workflowUiStore.getSnapshot().pendingWorkflowLoad?.requestId;

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    expect(workflowUiStore.getSnapshot().pendingWorkflowLoad).toBeNull();

    requestLibraryWorkflowLoad('workflow-b');
    const secondRequestId = workflowUiStore.getSnapshot().pendingWorkflowLoad?.requestId;

    expect(firstRequestId).toEqual(expect.any(Number));
    expect(secondRequestId).toEqual(expect.any(Number));
    expect(secondRequestId).toBeGreaterThan(firstRequestId ?? 0);
  });
});

describe('workflow load requests', () => {
  it('carries a library id or an already-fetched document as the source', () => {
    accountLifecycle.activate('user-c');
    requestLibraryWorkflowLoad('workflow-a');
    expect(workflowUiStore.getSnapshot().pendingWorkflowLoad?.source).toEqual({
      kind: 'library',
      workflowId: 'workflow-a',
    });

    requestWorkflowDocumentLoad({ nodes: [] }, 'Load workflow from image.png');
    expect(workflowUiStore.getSnapshot().pendingWorkflowLoad?.source).toEqual({
      kind: 'document',
      label: 'Load workflow from image.png',
      raw: { nodes: [] },
    });
  });
});
