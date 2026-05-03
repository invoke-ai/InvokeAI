import { describe, expect, it } from 'vitest';

import { buildCommittedTextImageState, getCommittedTextImageDimensions } from './textCommit';

describe('text commit helpers', () => {
  it('preserves logical dimensions when committing a hidpi render', () => {
    const imageDTO = {
      image_name: 'canvas-text.png',
      width: 300,
      height: 150,
    };

    const imageState = buildCommittedTextImageState(imageDTO as never, 200, 100);

    expect(imageState.image).toEqual({
      image_name: 'canvas-text.png',
      width: 200,
      height: 100,
    });
  });

  it('rounds up fractional logical dimensions and clamps to at least one pixel', () => {
    expect(getCommittedTextImageDimensions(99.1, 0.2)).toEqual({
      width: 100,
      height: 1,
    });
  });
});
