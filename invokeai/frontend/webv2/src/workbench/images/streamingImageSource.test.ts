import { describe, expect, it } from 'vitest';

import {
  mergeDenoiseProgressImage,
  progressImageToStreamingSource,
  resolveStreamingImageSource,
  type DenoiseProgressImage,
  type StreamingImageSource,
} from './streamingImageSource';

const liveImage: StreamingImageSource = {
  alt: 'Live preview',
  height: 32,
  kind: 'live',
  src: 'data:image/png;base64,live',
  width: 64,
};
const heldLiveImage: StreamingImageSource = {
  alt: 'Held preview',
  height: 32,
  kind: 'live',
  src: 'data:image/png;base64,held',
  width: 64,
};
const finalImage: StreamingImageSource = {
  alt: 'Final image',
  height: 512,
  kind: 'final',
  src: '/api/final.png',
  width: 512,
};
const fallbackImage: StreamingImageSource = {
  alt: 'Fallback image',
  height: 512,
  kind: 'fallback',
  src: '/api/fallback.png',
  width: 512,
};

describe('resolveStreamingImageSource', () => {
  it('prioritizes final images over live and fallback images', () => {
    expect(resolveStreamingImageSource({ fallbackImage, finalImage, heldLiveImage, liveImage })).toBe(finalImage);
  });

  it('uses the current live image before a held live image', () => {
    expect(resolveStreamingImageSource({ fallbackImage, heldLiveImage, liveImage })).toBe(liveImage);
  });

  it('keeps the held live image while waiting for final image data', () => {
    expect(resolveStreamingImageSource({ fallbackImage, heldLiveImage })).toBe(heldLiveImage);
  });

  it('falls back when no final or live image is available', () => {
    expect(resolveStreamingImageSource({ fallbackImage })).toBe(fallbackImage);
  });
});

describe('mergeDenoiseProgressImage', () => {
  it('keeps the previous denoise image when the next update is omitted', () => {
    const previous: DenoiseProgressImage = { dataUrl: 'data:image/png;base64,old', height: 32, width: 64 };

    expect(mergeDenoiseProgressImage(previous, undefined)).toBe(previous);
  });

  it('replaces the previous denoise image when a next update is provided', () => {
    const previous: DenoiseProgressImage = { dataUrl: 'data:image/png;base64,old', height: 32, width: 64 };
    const next: DenoiseProgressImage = { dataUrl: 'data:image/png;base64,next', height: 48, width: 96 };

    expect(mergeDenoiseProgressImage(previous, next)).toBe(next);
  });

  it('clears the previous denoise image when the next update is null', () => {
    const previous: DenoiseProgressImage = { dataUrl: 'data:image/png;base64,old', height: 32, width: 64 };

    expect(mergeDenoiseProgressImage(previous, null)).toBeNull();
  });
});

describe('progressImageToStreamingSource', () => {
  it('maps a denoise progress image to a live streaming source', () => {
    expect(
      progressImageToStreamingSource({ dataUrl: 'data:image/png;base64,abc', height: 32, width: 64 }, 'Preview')
    ).toEqual({
      alt: 'Preview',
      height: 32,
      kind: 'live',
      src: 'data:image/png;base64,abc',
      width: 64,
    });
  });
});
