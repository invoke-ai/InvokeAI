import { createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { getFormSectionOpenState, updateFormSectionOpenState } from './formSections';
import { getInvalidFormFieldIds } from './formSectionStatus';

describe('linear workflow form sections', () => {
  it('keeps expansion by stable workflow and collapsible element ids without changing the authored form', () => {
    const values = { formSectionsOpen: { workflowA: { first: false }, workflowB: { other: false } } };
    const updated = updateFormSectionOpenState(values, 'workflowA', 'second', true);
    expect(getFormSectionOpenState(updated, 'workflowA')).toEqual({ first: false, second: true });
    expect(getFormSectionOpenState(updated, 'workflowB')).toEqual({ other: false });
    expect(values.formSectionsOpen.workflowA).toEqual({ first: false });
  });

  it('ignores malformed stored states without hiding fields', () => {
    expect(
      getFormSectionOpenState({ formSectionsOpen: { workflowA: { first: 'false', second: true } } }, 'workflowA')
    ).toEqual({
      second: true,
    });
    expect(getFormSectionOpenState({ formSectionsOpen: [] }, 'workflowA')).toEqual({});
  });

  it('prunes deleted section ids when updating expansion state and skips invalid flags while templates load', () => {
    const values = { formSectionsOpen: { workflowA: { keep: false, stale: false } } };
    const updated = updateFormSectionOpenState(values, 'workflowA', 'next', false, new Set(['keep', 'next']));
    expect(getFormSectionOpenState(updated, 'workflowA')).toEqual({ keep: false, next: false });

    const graph = createProjectGraph('wf-loading');
    graph.form.elements.missing = {
      data: {
        fieldIdentifier: { fieldName: 'value', nodeId: 'missing-node' },
        showDescription: false,
        showShuffle: false,
      },
      id: 'missing',
      parentId: graph.form.rootElementId,
      type: 'node-field',
    };
    expect(getInvalidFormFieldIds(graph, {}, 'loading').size).toBe(0);
    expect(getInvalidFormFieldIds(graph, {}, 'loaded').has('missing')).toBe(true);
  });
});
