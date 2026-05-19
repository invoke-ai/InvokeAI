import { describe, expect, it } from 'vitest';

import { getWorkflowCallCompatibilityState } from './workflowCallCompatibility';

describe('workflowCallCompatibility', () => {
  it('treats missing compatibility metadata as supported', () => {
    expect(
      getWorkflowCallCompatibilityState({
        call_saved_workflow_compatibility: null,
      })
    ).toEqual({
      isUnsupported: false,
      message: null,
    });
  });

  it('treats callable workflows as supported', () => {
    expect(
      getWorkflowCallCompatibilityState({
        call_saved_workflow_compatibility: {
          is_callable: true,
          reason: 'ok',
          message: null,
        },
      })
    ).toEqual({
      isUnsupported: false,
      message: null,
    });
  });

  it('returns the backend compatibility message for unsupported workflows', () => {
    expect(
      getWorkflowCallCompatibilityState({
        call_saved_workflow_compatibility: {
          is_callable: false,
          reason: 'missing_workflow_return',
          message: 'The workflow must contain exactly one workflow_return node.',
        },
      })
    ).toEqual({
      isUnsupported: true,
      message: 'The workflow must contain exactly one workflow_return node.',
    });
  });
});
