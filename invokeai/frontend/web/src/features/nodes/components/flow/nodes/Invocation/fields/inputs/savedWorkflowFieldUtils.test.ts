import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  buildSavedWorkflowOptions,
  getSavedWorkflowDisplayState,
  getSavedWorkflowListItemFromRecord,
  getSavedWorkflowSelectionOption,
  getSavedWorkflowSelectionState,
  MISSING_WORKFLOW_OPTION_VALUE,
} from './savedWorkflowFieldUtils';

const workflows: WorkflowRecordListItemWithThumbnailDTO[] = [
  {
    workflow_id: 'workflow-a',
    name: 'Alpha Workflow',
    created_at: '',
    updated_at: '',
    opened_at: null,
    description: '',
    tags: '',
    is_public: false,
    thumbnail_url: null,
    category: 'user',
    user_id: 'user-a',
    call_saved_workflow_compatibility: {
      is_callable: true,
      reason: 'ok',
      message: null,
    },
  },
  {
    workflow_id: 'workflow-b',
    name: 'Beta Workflow',
    created_at: '',
    updated_at: '',
    opened_at: null,
    description: '',
    tags: '',
    is_public: true,
    thumbnail_url: null,
    category: 'default',
    user_id: 'system',
    call_saved_workflow_compatibility: {
      is_callable: false,
      reason: 'missing_workflow_return',
      message: 'The workflow must contain exactly one workflow_return node.',
    },
  },
];

describe('savedWorkflowFieldUtils', () => {
  it('builds combobox options from visible workflows', () => {
    expect(buildSavedWorkflowOptions(workflows)).toEqual([
      { label: 'Alpha Workflow', value: 'workflow-a', isDisabled: false },
      { label: 'Beta Workflow', value: 'workflow-b', isDisabled: true },
    ]);
  });

  it('returns an unselected state for the default empty value', () => {
    const selectionState = getSavedWorkflowSelectionState(workflows, '');
    expect(selectionState).toEqual({ status: 'unselected' });
    expect(getSavedWorkflowSelectionOption(selectionState)).toBeNull();
  });

  it('returns a selected state for a valid workflow id', () => {
    const selectionState = getSavedWorkflowSelectionState(workflows, 'workflow-b');
    expect(selectionState).toEqual({ status: 'selected', workflow: workflows[1] });
    expect(getSavedWorkflowSelectionOption(selectionState)).toEqual({
      label: 'Beta Workflow',
      value: 'workflow-b',
    });
  });

  it('returns a missing state for a stale or inaccessible workflow id', () => {
    const selectionState = getSavedWorkflowSelectionState(workflows, 'missing-workflow');
    expect(selectionState).toEqual({ status: 'missing', workflowId: 'missing-workflow' });
    expect(getSavedWorkflowSelectionOption(selectionState)).toEqual({
      label: MISSING_WORKFLOW_OPTION_VALUE,
      value: MISSING_WORKFLOW_OPTION_VALUE,
    });
  });

  it('builds display state for an already-selected unsupported workflow', () => {
    const selectionState = getSavedWorkflowSelectionState(workflows, 'workflow-b');

    expect(getSavedWorkflowDisplayState(selectionState)).toEqual({
      selection: 'selected',
      statusLabelKey: null,
      badges: ['unsupported', 'default'],
      compatibilityMessage: 'The workflow must contain exactly one workflow_return node.',
    });
  });

  it('builds display state for missing and unselected workflows', () => {
    expect(getSavedWorkflowDisplayState({ status: 'unselected' })).toEqual({
      selection: 'unselected',
      statusLabelKey: 'nodes.savedWorkflowChoose',
      badges: [],
      compatibilityMessage: null,
    });

    expect(getSavedWorkflowDisplayState({ status: 'missing', workflowId: 'missing-workflow' })).toEqual({
      selection: 'missing',
      statusLabelKey: 'nodes.savedWorkflowMissing',
      badges: [],
      compatibilityMessage: null,
    });
  });

  it('uses a fetched selected workflow when it is not present in the first list page', () => {
    const selectedWorkflow = getSavedWorkflowListItemFromRecord({
      workflow_id: 'workflow-z',
      name: 'Zeta Workflow',
      created_at: '',
      updated_at: '',
      opened_at: null,
      user_id: 'user-z',
      is_public: true,
      thumbnail_url: null,
      call_saved_workflow_compatibility: {
        is_callable: true,
        reason: 'ok',
        message: null,
      },
      workflow: {
        id: 'workflow-z',
        name: 'Zeta Workflow',
        author: '',
        description: 'A workflow outside the first page',
        version: '',
        contact: '',
        tags: 'paged',
        notes: '',
        exposedFields: [],
        meta: {
          category: 'user',
          version: '1.0.0',
        },
        nodes: [],
        edges: [],
        form: null,
      },
    });

    const selectionState = getSavedWorkflowSelectionState(workflows, 'workflow-z', selectedWorkflow);

    expect(selectionState).toEqual({ status: 'selected', workflow: selectedWorkflow });
    expect(getSavedWorkflowDisplayState(selectionState)).toEqual({
      selection: 'selected',
      statusLabelKey: null,
      badges: ['shared'],
      compatibilityMessage: null,
    });
  });
});
