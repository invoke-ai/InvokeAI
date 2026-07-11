import { describe, expect, it } from 'vitest';

import { getWorkflowActionEligibility } from './WorkflowOptions';

describe('getWorkflowActionEligibility', () => {
  it('allows a ready, fully-bound workflow to run', () => {
    expect(getWorkflowActionEligibility({ hasInput: true, hasOutput: true, isRunning: false })).toEqual({
      canCancel: true,
      canEdit: true,
      canRun: true,
    });
  });

  it('blocks edits and duplicate runs while the workflow is running', () => {
    expect(getWorkflowActionEligibility({ hasInput: true, hasOutput: true, isRunning: true })).toEqual({
      canCancel: true,
      canEdit: false,
      canRun: false,
    });
  });

  it('requires both an input and an output', () => {
    expect(getWorkflowActionEligibility({ hasInput: false, hasOutput: true, isRunning: false }).canRun).toBe(false);
    expect(getWorkflowActionEligibility({ hasInput: true, hasOutput: false, isRunning: false }).canRun).toBe(false);
  });
});
