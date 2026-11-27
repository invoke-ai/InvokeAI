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
      messageKey: null,
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
      messageKey: null,
    });
  });

  it('maps structured compatibility reasons to localized message keys', () => {
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
      messageKey: 'workflows.savedWorkflowCompatibility.missingWorkflowReturn',
    });
  });

  it('uses a localized fallback for unknown compatibility reasons', () => {
    expect(
      getWorkflowCallCompatibilityState({
        call_saved_workflow_compatibility: {
          is_callable: false,
          reason: 'unknown',
          message: 'Backend-only diagnostic text',
        },
      })
    ).toEqual({
      isUnsupported: true,
      messageKey: 'workflows.savedWorkflowCompatibility.unknown',
    });
  });

  it('maps queue capacity failures to a localized message key', () => {
    expect(
      getWorkflowCallCompatibilityState({
        call_saved_workflow_compatibility: {
          is_callable: false,
          reason: 'exceeds_capacity',
          message: 'call_saved_workflow exceeds remaining queue capacity for child workflow executions',
        },
      })
    ).toEqual({
      isUnsupported: true,
      messageKey: 'workflows.savedWorkflowCompatibility.exceedsCapacity',
    });
  });
});
