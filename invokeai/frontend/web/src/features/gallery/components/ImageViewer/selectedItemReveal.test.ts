import { describe, expect, it } from 'vitest';

import type { SelectedItemRevealInputs } from './selectedItemReveal';
import { getSelectedItemRevealDecision } from './selectedItemReveal';

// A mid-generation user click on b.png, with the previous item still remembered: the case the
// reveal exists for. Each test below perturbs exactly one input away from it.
const userClickMidGeneration: SelectedItemRevealInputs = {
  shouldShowProgressInViewer: true,
  hasProgressImage: true,
  isProgressImageResolving: false,
  renderedItemName: 'b.png',
  selectedItemName: 'b.png',
  previousRenderedItemName: 'a.png',
  wasAutoSwitchedTo: false,
};

describe('getSelectedItemRevealDecision', () => {
  it('reveals a mid-generation user click so it visibly lands', () => {
    expect(getSelectedItemRevealDecision(userClickMidGeneration)).toBe('reveal');
  });

  it('does not reveal an auto-switch to a just-finished item', () => {
    // Without this the finished image flashes over the next generation's live preview for 2s.
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, wasAutoSwitchedTo: true })).toBe('hide');
  });

  it('hides when no progress preview is covering the viewer', () => {
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, hasProgressImage: false })).toBe('hide');
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, shouldShowProgressInViewer: false })).toBe(
      'hide'
    );
  });

  it('hides while a finished generation is resolving into its final image', () => {
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, isProgressImageResolving: true })).toBe('hide');
  });

  it('hides while the render still lags the selection', () => {
    // The preload has not settled, so the item on screen is not the one that was clicked.
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, renderedItemName: 'a.png' })).toBe('hide');
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, renderedItemName: null })).toBe('hide');
  });

  it('hides when the displayed item did not change', () => {
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, previousRenderedItemName: 'b.png' })).toBe(
      'hide'
    );
  });

  it('hides on the first render, which is not a click', () => {
    expect(getSelectedItemRevealDecision({ ...userClickMidGeneration, previousRenderedItemName: null })).toBe('hide');
  });

  it('never answers anything but reveal or hide', () => {
    // The caller has already cleared the running reveal's timer by the time it asks, so a third
    // "leave it alone" outcome would wedge the overlay off for the rest of the render.
    const inputs: SelectedItemRevealInputs[] = [
      userClickMidGeneration,
      { ...userClickMidGeneration, wasAutoSwitchedTo: true },
      { ...userClickMidGeneration, hasProgressImage: false },
      { ...userClickMidGeneration, renderedItemName: null },
      { ...userClickMidGeneration, previousRenderedItemName: null },
    ];
    for (const input of inputs) {
      expect(['reveal', 'hide']).toContain(getSelectedItemRevealDecision(input));
    }
  });
});
