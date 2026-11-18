import { describe, expect, it } from 'vitest';

import { flowThemeCss } from './graph-preview/flowTheme';
import { getWorkflowNodeAppearance, getWorkflowNodeHandleStyle, WORKFLOW_NODE_DIAMOND_ROTATION } from './nodeChrome';

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

  it('gives every tone a hover outline of its own family that differs from its resting outline', () => {
    for (const state of [{ invalid: true }, { failed: true }, { outdated: true }]) {
      const { border, hoverBorder } = getWorkflowNodeAppearance({ ...state, selected: false });

      expect(hoverBorder).not.toBe(border);
      expect(hoverBorder).toContain(border);
      expect(hoverBorder).not.toContain('accent');
    }
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

describe('batch port grab area', () => {
  it('turns back the rotation a batch handle is drawn with, so its grab area stays upright', () => {
    const transform = getWorkflowNodeHandleStyle(
      { batch: true, cardinality: 'COLLECTION', name: 'IntegerField' },
      'left'
    ).transform;
    const counterRotation = Object.keys(flowThemeCss).find(
      (selector) => selector.includes('::before') && selector.includes('style*=')
    );

    expect(transform).toContain(WORKFLOW_NODE_DIAMOND_ROTATION);
    expect(counterRotation).toBe(`& .react-flow__handle[style*="${WORKFLOW_NODE_DIAMOND_ROTATION}"]::before`);
  });
});
