import { describe, expect, it } from 'vitest';

import { getWorkflowNodeAppearance } from './nodeChrome';

describe('getWorkflowNodeAppearance', () => {
  it('draws a failed node exactly like an invalid one, selected or not', () => {
    for (const selected of [false, true]) {
      expect(getWorkflowNodeAppearance({ failed: true, selected })).toEqual(
        getWorkflowNodeAppearance({ invalid: true, selected })
      );
    }
    expect(getWorkflowNodeAppearance({ failed: true, selected: true })).toMatchObject({
      border: 'border.error',
      selectionRing: 'border.error',
      tone: 'error',
    });
  });

  it('keeps the outline neutral for a running node and glows instead', () => {
    expect(getWorkflowNodeAppearance({ running: true, selected: false })).toEqual({
      ...getWorkflowNodeAppearance({ selected: false }),
      glows: true,
    });
  });

  it('composes states instead of letting one overwrite another', () => {
    const invalidOutdatedRunning = getWorkflowNodeAppearance({
      invalid: true,
      outdated: true,
      running: true,
      selected: true,
    });

    expect(invalidOutdatedRunning).toEqual({
      border: 'border.error',
      glows: true,
      hoverBorder: expect.stringContaining('border.error'),
      selectionRing: 'border.error',
      tone: 'error',
    });
    // An outdated node keeps its amber in its selection ring and takes a stronger amber under the pointer.
    expect(getWorkflowNodeAppearance({ outdated: true, selected: true })).toEqual({
      border: 'fg.warning',
      glows: false,
      hoverBorder: expect.stringContaining('fg.warning'),
      selectionRing: 'fg.warning',
      tone: 'warning',
    });
  });

  it('turns only a neutral outline accent on hover, and rings only a selected node', () => {
    expect(getWorkflowNodeAppearance({ selected: false })).toEqual({
      border: 'border.emphasized',
      glows: false,
      hoverBorder: 'accent.solid',
      selectionRing: null,
      tone: 'neutral',
    });
    expect(getWorkflowNodeAppearance({ selected: true }).selectionRing).toBe('accent.solid');
  });
});
