import { createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { planLibraryWorkflowOpen } from './useOpenLibraryWorkflow';

const entry = (id: string, libraryWorkflowId?: string) => ({
  document: createProjectGraph(id),
  ...(libraryWorkflowId ? { source: { libraryWorkflowId, revision: 1 } } : {}),
});

describe('planLibraryWorkflowOpen', () => {
  it('adds a copy when the project has none, resumes a single copy, and asks when there are several', () => {
    expect(planLibraryWorkflowOpen([entry('a'), entry('b', 'other')], 'lib')).toEqual({ kind: 'add' });
    expect(planLibraryWorkflowOpen([entry('a', 'lib'), entry('b')], 'lib')).toEqual({
      kind: 'resume',
      workflowId: 'a',
    });

    const copies = [entry('a', 'lib'), entry('b', 'lib')];

    expect(planLibraryWorkflowOpen([...copies, entry('c')], 'lib')).toEqual({ copies, kind: 'choose' });
  });
});
