import { describe, expect, it } from 'vitest';

import { createAutoSwitchedSelectionMarker } from './autoSwitchedImages';

describe('createAutoSwitchedSelectionMarker', () => {
  it('reports the auto-switched item on its first render, once', () => {
    const marker = createAutoSwitchedSelectionMarker();
    marker.record('a.png');
    marker.settle('a.png');
    expect(marker.consume('a.png')).toBe(true);
    expect(marker.consume('a.png')).toBe(false);
  });

  it('reports nothing for an item that was never auto-switched to', () => {
    const marker = createAutoSwitchedSelectionMarker();
    expect(marker.consume('a.png')).toBe(false);
  });

  it('drops a marker whose selection was superseded before it rendered', () => {
    // The auto-switch to A is dispatched, then the user clicks B before A's preload settles. A can
    // never render as that auto-switch, and the user's later click on A must reveal it.
    const marker = createAutoSwitchedSelectionMarker();
    marker.record('a.png');
    marker.settle('a.png');
    marker.settle('b.png');
    marker.settle('a.png');
    expect(marker.consume('a.png')).toBe(false);
  });

  it('keeps only the last of several auto-switches settled in one batch', () => {
    // Two sessions completing together record two names, but only the last selection stands.
    const marker = createAutoSwitchedSelectionMarker();
    marker.record('a.png');
    marker.settle('a.png');
    marker.record('b.png');
    marker.settle('b.png');
    expect(marker.consume('b.png')).toBe(true);
    expect(marker.consume('a.png')).toBe(false);
  });

  it('survives settles that do not move the selection', () => {
    // Re-selecting the same item (or any other action that leaves the selection alone) must not
    // discard a marker whose item has not rendered yet.
    const marker = createAutoSwitchedSelectionMarker();
    marker.record('a.png');
    marker.settle('a.png');
    marker.settle('a.png');
    expect(marker.consume('a.png')).toBe(true);
  });

  it('drops the marker when the selection is cleared', () => {
    const marker = createAutoSwitchedSelectionMarker();
    marker.record('a.png');
    marker.settle('a.png');
    marker.settle(null);
    expect(marker.consume('a.png')).toBe(false);
  });

  it('does not report a second render of an item after its marker was consumed', () => {
    // The user navigates away and back: that second render is a user selection and must reveal.
    const marker = createAutoSwitchedSelectionMarker();
    marker.record('a.png');
    marker.settle('a.png');
    expect(marker.consume('a.png')).toBe(true);
    marker.settle('b.png');
    marker.settle('a.png');
    expect(marker.consume('a.png')).toBe(false);
  });
});
