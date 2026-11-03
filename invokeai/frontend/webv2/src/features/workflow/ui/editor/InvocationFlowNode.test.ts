import type { WorkflowInvocationNode } from '@features/workflow/contracts';

import { describe, expect, it } from 'vitest';

import {
  getNodeExecutionError,
  shouldShowCallSavedWorkflowLoadingHint,
  shouldShowCallSavedWorkflowNoExposedFieldsHint,
} from './InvocationFlowNode';

const node = (type: string, status?: WorkflowInvocationNode['data']['callSavedWorkflowStatus']) =>
  ({ data: { type, callSavedWorkflowStatus: status } }) as WorkflowInvocationNode;

describe('shouldShowCallSavedWorkflowLoadingHint', () => {
  it('shows the hint only while a selected child workflow is loading', () => {
    expect(shouldShowCallSavedWorkflowLoadingHint(node('call_saved_workflow', 'loading'))).toBe(true);
    expect(shouldShowCallSavedWorkflowLoadingHint(node('call_saved_workflow', 'ready'))).toBe(false);
    expect(shouldShowCallSavedWorkflowLoadingHint(node('call_saved_workflow', 'error'))).toBe(false);
    expect(shouldShowCallSavedWorkflowLoadingHint(node('call_saved_workflow'))).toBe(false);
    expect(shouldShowCallSavedWorkflowLoadingHint(node('other', 'loading'))).toBe(false);
  });
});

describe('shouldShowCallSavedWorkflowNoExposedFieldsHint', () => {
  it('shows the empty-state hint only for a ready selected child with no dynamic fields', () => {
    expect(
      shouldShowCallSavedWorkflowNoExposedFieldsHint({
        data: {
          ...node('call_saved_workflow', 'ready').data,
          inputs: { workflow_id: { label: '', name: 'workflow_id', value: 'workflow-1' } },
        },
      } as unknown as WorkflowInvocationNode)
    ).toBe(true);
    expect(shouldShowCallSavedWorkflowNoExposedFieldsHint(node('call_saved_workflow', 'loading'))).toBe(false);
    expect(
      shouldShowCallSavedWorkflowNoExposedFieldsHint({
        data: {
          ...node('call_saved_workflow', 'ready').data,
          dynamicInputTemplates: { field: {} },
          inputs: { workflow_id: { label: '', name: 'workflow_id', value: 'workflow-1' } },
        },
      } as unknown as WorkflowInvocationNode)
    ).toBe(false);
  });
});

describe('getNodeExecutionError', () => {
  const translate = (key: string, options: { error: string }) => `${key}: ${options.error}`;

  it('attributes call saved workflow failures while preserving ordinary invocation errors', () => {
    const execution = { error: 'Child node failed' };

    expect(getNodeExecutionError(node('call_saved_workflow'), execution.error, translate)).toBe(
      'nodes.childWorkflowError: Child node failed'
    );
    expect(getNodeExecutionError(node('other'), execution.error, translate)).toBe('Child node failed');
    expect(getNodeExecutionError(node('call_saved_workflow'), null, translate)).toBeNull();
  });
});
