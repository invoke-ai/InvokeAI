import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  buildSavedWorkflowOptions,
  getSavedWorkflowSelectionState,
  getSelectedWorkflow,
  getSelectedWorkflowOption,
  MISSING_WORKFLOW_OPTION_VALUE,
} from './callSavedWorkflowsNodeUtils';

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
  },
];

describe('callSavedWorkflowsNodeUtils', () => {
  it('builds combobox options from visible workflows', () => {
    expect(buildSavedWorkflowOptions(workflows)).toEqual([
      { label: 'Alpha Workflow', value: 'workflow-a' },
      { label: 'Beta Workflow', value: 'workflow-b' },
    ]);
  });

  it('resolves the selected workflow for a stored workflow id', () => {
    expect(getSelectedWorkflow(workflows, 'workflow-b')).toEqual(workflows[1]);
  });

  it('returns null when no workflow is selected', () => {
    expect(getSavedWorkflowSelectionState(workflows, '')).toEqual({ status: 'unselected' });
    expect(getSelectedWorkflow(workflows, '')).toBeNull();
    expect(getSelectedWorkflowOption(workflows, '', 'Missing workflow')).toBeNull();
  });

  it('returns a selected state when the workflow id resolves', () => {
    expect(getSavedWorkflowSelectionState(workflows, 'workflow-a')).toEqual({
      status: 'selected',
      workflow: workflows[0],
    });
  });

  it('returns a synthetic missing option when the stored workflow id is stale', () => {
    expect(getSavedWorkflowSelectionState(workflows, 'missing-workflow')).toEqual({
      status: 'missing',
      workflowId: 'missing-workflow',
    });
    expect(getSelectedWorkflowOption(workflows, 'missing-workflow', 'Missing workflow')).toEqual({
      label: 'Missing workflow',
      value: MISSING_WORKFLOW_OPTION_VALUE,
    });
  });
});
