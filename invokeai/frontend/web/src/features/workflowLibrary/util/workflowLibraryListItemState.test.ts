import { describe, expect, it } from 'vitest';

import { getWorkflowLibraryListItemState } from './workflowLibraryListItemState';

describe('workflowLibraryListItemState', () => {
  it('does not mark ordinary workflows unsupported when they are not callable as sub-workflows', () => {
    expect(
      getWorkflowLibraryListItemState({
        category: 'user',
        is_public: false,
        call_saved_workflow_compatibility: {
          is_callable: false,
          reason: 'missing_workflow_return',
          message: 'The workflow must contain exactly one workflow_return node.',
        },
      })
    ).toEqual({
      showCallableBadge: false,
      showSharedBadge: false,
      showDefaultIcon: false,
    });
  });

  it('marks callable workflows with a positive callable badge', () => {
    expect(
      getWorkflowLibraryListItemState({
        category: 'user',
        is_public: false,
        call_saved_workflow_compatibility: {
          is_callable: true,
          reason: 'ok',
          message: null,
        },
      })
    ).toEqual({
      showCallableBadge: true,
      showSharedBadge: false,
      showDefaultIcon: false,
    });
  });

  it('marks shared non-default workflows and leaves callable workflows supported', () => {
    expect(
      getWorkflowLibraryListItemState({
        category: 'user',
        is_public: true,
        call_saved_workflow_compatibility: {
          is_callable: true,
          reason: 'ok',
          message: null,
        },
      })
    ).toEqual({
      showCallableBadge: true,
      showSharedBadge: true,
      showDefaultIcon: false,
    });
  });

  it('marks default workflows with the default icon instead of the shared badge', () => {
    expect(
      getWorkflowLibraryListItemState({
        category: 'default',
        is_public: true,
        call_saved_workflow_compatibility: {
          is_callable: true,
          reason: 'ok',
          message: null,
        },
      })
    ).toEqual({
      showCallableBadge: true,
      showSharedBadge: false,
      showDefaultIcon: true,
    });
  });
});
