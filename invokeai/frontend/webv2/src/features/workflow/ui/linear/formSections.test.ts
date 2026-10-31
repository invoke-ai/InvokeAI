import type { WorkflowFormElement } from '@features/workflow/contracts';

import { describe, expect, it } from 'vitest';

import { getFormSectionOpenState, groupFormSections, updateFormSectionOpenState } from './formSections';

const heading = (id: string): WorkflowFormElement => ({ id, type: 'heading', data: { content: id } });
const text = (id: string): WorkflowFormElement => ({ id, type: 'text', data: { content: id } });
const divider = (id: string): WorkflowFormElement => ({ id, type: 'divider' });

describe('linear workflow form sections', () => {
  it('keeps content before the first heading visible and preserves divider boundaries', () => {
    const sections = groupFormSections([
      text('intro'),
      heading('input'),
      text('field-a'),
      divider('separator'),
      text('between'),
      heading('output'),
      text('field-b'),
    ]);

    expect(sections).toMatchObject([
      { kind: 'element', element: { id: 'intro' } },
      { kind: 'section', heading: { id: 'input' }, body: [{ id: 'field-a' }] },
      { kind: 'element', element: { id: 'separator' } },
      { kind: 'element', element: { id: 'between' } },
      { kind: 'section', heading: { id: 'output' }, body: [{ id: 'field-b' }] },
    ]);
  });

  it('retains adjacent headings as separate empty and nonempty sections', () => {
    expect(groupFormSections([heading('first'), heading('second'), text('field')])).toMatchObject([
      { kind: 'section', heading: { id: 'first' }, body: [] },
      { kind: 'section', heading: { id: 'second' }, body: [{ id: 'field' }] },
    ]);
  });

  it('uses spacing instead of drawing a divider only between two adjacent accordion sections', () => {
    expect(
      groupFormSections([heading('first'), text('one'), divider('between'), heading('second'), text('two')])
    ).toMatchObject([
      { kind: 'section', heading: { id: 'first' }, body: [{ id: 'one' }] },
      { kind: 'section', heading: { id: 'second' }, body: [{ id: 'two' }] },
    ]);
    expect(groupFormSections([heading('first'), divider('before-text'), text('unheaded')])).toMatchObject([
      { kind: 'section', heading: { id: 'first' }, body: [] },
      { kind: 'element', element: { id: 'before-text' } },
      { kind: 'element', element: { id: 'unheaded' } },
    ]);
  });

  it('keeps expansion by stable workflow and heading ids without changing the authored form', () => {
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
