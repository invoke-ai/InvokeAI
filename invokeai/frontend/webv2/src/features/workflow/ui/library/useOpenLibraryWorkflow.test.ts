import { createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { planLibraryWorkflowOpen } from './useOpenLibraryWorkflow';

const entry = (id: string, libraryWorkflowId?: string) => ({
  document: createProjectGraph(id),
  ...(libraryWorkflowId ? { source: { libraryWorkflowId, revision: 1 } } : {}),
});

describe('planLibraryWorkflowOpen', () => {
  it('adds the first copy and asks once the project holds any', () => {
    expect(planLibraryWorkflowOpen([entry('a'), entry('b', 'other')], 'lib')).toEqual({ kind: 'add' });

    const single = [entry('a', 'lib')];

    expect(planLibraryWorkflowOpen([...single, entry('b')], 'lib')).toEqual({ copies: single, kind: 'choose' });

    const copies = [entry('a', 'lib'), entry('b', 'lib')];

    expect(planLibraryWorkflowOpen([...copies, entry('c')], 'lib')).toEqual({ copies, kind: 'choose' });
  });
});
