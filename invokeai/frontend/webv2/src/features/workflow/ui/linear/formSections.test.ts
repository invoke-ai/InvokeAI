import { describe, expect, it } from 'vitest';

import { getFormSectionOpenState, updateFormSectionOpenState } from './formSections';

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
});
