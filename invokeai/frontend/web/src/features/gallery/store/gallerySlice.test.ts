import { describe, expect, it } from 'vitest';

import { gallerySliceConfig, imageSelected, imageToCompareChanged, selectionChanged } from './gallerySlice';

const reducer = gallerySliceConfig.slice.reducer;

describe('gallery comparison state', () => {
  it('keeps comparison active when another image is selected', () => {
    let state = reducer(undefined, imageToCompareChanged('comparison.png'));

    state = reducer(state, imageSelected('selected.png'));

    expect(state.imageToCompare).toBe('comparison.png');
  });

  it('clears comparison when a video is selected', () => {
    let state = reducer(undefined, imageToCompareChanged('comparison.png'));

    state = reducer(state, imageSelected('selected.mp4'));

    expect(state.imageToCompare).toBeNull();
  });

  it('clears comparison when a multi-selection contains a video', () => {
    let state = reducer(undefined, imageToCompareChanged('comparison.png'));

    state = reducer(state, selectionChanged(['selected.png', 'selected.mp4']));

    expect(state.imageToCompare).toBeNull();
  });

  it('rejects a video as the comparison item', () => {
    const state = reducer(undefined, imageToCompareChanged('comparison.mp4'));

    expect(state.imageToCompare).toBeNull();
  });
});
