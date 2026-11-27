import { describe, expect, it } from 'vitest';

import { getSyncedWorkflowLibraryView } from './WorkflowLibraryModal.utils';

describe('getSyncedWorkflowLibraryView', () => {
  it('switches recent to yours when there are no recent workflows and yours has workflows', () => {
    expect(
      getSyncedWorkflowLibraryView({
        view: 'recent',
        recentWorkflowsCount: 0,
        yourWorkflowsCount: 3,
      })
    ).toBe('yours');
  });

  it('switches recent to defaults when there are no recent or yours workflows', () => {
    expect(
      getSyncedWorkflowLibraryView({
        view: 'recent',
        recentWorkflowsCount: 0,
        yourWorkflowsCount: 0,
      })
    ).toBe('defaults');
  });

  it('switches yours to recent when there are no yours workflows and recent has workflows', () => {
    expect(
      getSyncedWorkflowLibraryView({
        view: 'yours',
        recentWorkflowsCount: 2,
        yourWorkflowsCount: 0,
      })
    ).toBe('recent');
  });

  it('keeps a valid selected view unchanged', () => {
    expect(
      getSyncedWorkflowLibraryView({
        view: 'recent',
        recentWorkflowsCount: 1,
        yourWorkflowsCount: 0,
      })
    ).toBe('recent');
  });
});
