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

  it('bounds workflow histories and retains the most recently edited workflow preferences without mutating input', () => {
    let values: Record<string, unknown> = {};
    for (let index = 0; index < 64; index += 1) {
      values = updateFormSectionOpenState(values, `workflow-${index}`, 'section', false);
    }
    const original = structuredClone(values);
    const revisited = updateFormSectionOpenState(values, 'workflow-0', 'section', true);
    const updated = updateFormSectionOpenState(revisited, 'workflow-64', 'section', false);
    expect(Object.keys(updated.formSectionsOpen as object)).toHaveLength(64);
    expect(getFormSectionOpenState(updated, 'workflow-0')).toEqual({ section: true });
    expect(getFormSectionOpenState(updated, 'workflow-1')).toEqual({});
    expect(getFormSectionOpenState(updated, 'workflow-2')).toEqual({ section: false });
    expect(getFormSectionOpenState(updated, 'workflow-64')).toEqual({ section: false });
    expect(values).toEqual(original);
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
