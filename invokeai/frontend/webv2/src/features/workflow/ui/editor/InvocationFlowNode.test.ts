import type { WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState } from '@features/workflow/ui/contracts';

import { describe, expect, it } from 'vitest';

import {
  getInvocationNodeState,
  shouldShowCallSavedWorkflowLoadingHint,
  shouldShowCallSavedWorkflowNoExposedFieldsHint,
} from './InvocationFlowNode';
import { getNodeExecutionError } from './nodeStatusText';

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

describe('getInvocationNodeState', () => {
  const execution = (status: WorkflowNodeExecutionState['status']): WorkflowNodeExecutionState => ({
    error: status === 'failed' ? 'Out of memory' : null,
    latestOutput: null,
    outputImageName: null,
    progress: status === 'running' ? 0.4 : null,
    progressMessage: null,
    status,
  });
  const state = (overrides: Partial<Parameters<typeof getInvocationNodeState>[0]> = {}) =>
    getInvocationNodeState({ execution: null, hasFieldIssues: false, updateStatus: 'current', ...overrides });

  it('reports nothing for a current, valid node that has not run', () => {
    expect(state()).toEqual({
      chrome: { failed: false, invalid: false, outdated: false, running: false },
      status: null,
    });
  });

  it('shows the status that most needs attention while keeping every chrome fact', () => {
    expect(state({ execution: execution('running'), hasFieldIssues: true, updateStatus: 'updatable' })).toEqual({
      chrome: { failed: false, invalid: true, outdated: true, running: true },
      status: 'running',
    });
    expect(state({ execution: execution('failed'), hasFieldIssues: true }).status).toBe('failed');
    expect(state({ execution: execution('completed'), hasFieldIssues: true }).status).toBe('invalid');
    expect(state({ execution: execution('completed'), updateStatus: 'updatable' }).status).toBe('updatable');
    expect(state({ execution: execution('completed') }).status).toBe('completed');
  });

  it('separates an update the editor can apply from a version it cannot', () => {
    expect(state({ updateStatus: 'updatable' }).status).toBe('updatable');
    expect(state({ updateStatus: 'newer' }).status).toBe('versionMismatch');
    expect(state({ updateStatus: 'incompatible' }).status).toBe('versionMismatch');
  });

  it('marks a completed run in the status only, and a node of unknown type as invalid', () => {
    expect(state({ execution: execution('completed') }).chrome).toEqual(state().chrome);
    expect(state({ isMissingTemplate: true, updateStatus: null })).toEqual({
      chrome: { failed: false, invalid: true, outdated: false, running: false },
      status: null,
    });
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
