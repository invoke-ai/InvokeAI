import { beforeEach, describe, expect, it } from 'vitest';

import {
  $gallerySelection,
  markNextSelectionAutoSwitched,
  recordGallerySelection,
  resetGallerySelectionSource,
} from './gallerySelectionSource';

describe('gallery selection source', () => {
  beforeEach(() => {
    resetGallerySelectionSource();
  });

  it('advances the generation on every selection, including a re-selection of the same item', () => {
    recordGallerySelection('a.png');
    const first = $gallerySelection.get();
    recordGallerySelection('a.png');
    const second = $gallerySelection.get();

    expect(second.name).toBe('a.png');
    expect(second.generation, 'picking the item already on screen is a new selection').toBeGreaterThan(
      first.generation
    );
  });

  it('attributes only the selection the mark was set for', () => {
    markNextSelectionAutoSwitched();
    recordGallerySelection('auto.png');
    expect($gallerySelection.get().isAutoSwitch).toBe(true);

    recordGallerySelection('clicked.png');
    expect($gallerySelection.get().isAutoSwitch, 'the mark is spent, not sticky').toBe(false);
  });

  it('does not carry an auto-switch mark across an intervening user selection', () => {
    // The auto-switch dispatch follows its mark synchronously, so anything landing in between is
    // the user's — and inheriting the mark would make their click read as a handoff and go
    // unrevealed, the dead click this whole mechanism exists to prevent.
    markNextSelectionAutoSwitched();
    recordGallerySelection('clicked.png');
    recordGallerySelection('auto.png');

    expect($gallerySelection.get().isAutoSwitch).toBe(false);
  });

  it('records an empty selection', () => {
    recordGallerySelection('a.png');
    recordGallerySelection(null);
    expect($gallerySelection.get().name).toBeNull();
  });
});
