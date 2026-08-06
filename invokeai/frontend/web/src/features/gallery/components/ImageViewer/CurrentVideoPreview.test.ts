import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

describe('CurrentVideoPreview playback errors', () => {
  it('shows a persistent user-facing error instead of only restoring the retry button', () => {
    const source = readFileSync(fileURLToPath(new URL('./CurrentVideoPreview.tsx', import.meta.url)), 'utf8');

    expect(source).toContain("t('toast.videoPlaybackFailed')");
    expect(source).toContain('onError={handleVideoError}');
  });
});

describe('CurrentVideoPreview progress overlay', () => {
  const source = readFileSync(fileURLToPath(new URL('./CurrentVideoPreview.tsx', import.meta.url)), 'utf8');

  it('lifts the overlay during the temporary reveal so mid-render thumbnail clicks visibly land', () => {
    // The overlay must consult the shared reveal atom (and never re-cover an actively-playing
    // video) — an unconditional overlay swallows every gallery click for the whole render.
    expect(source).toMatch(
      /withProgress =\s+shouldShowProgressInViewer && hasProgressImage && !isTemporarilyShowingSelectedImage && !isPlaying/
    );
    expect(source).toContain('SELECTED_ITEM_REVEAL_DURATION_MS');
    // The previous-item ref must be the shared one from the viewer context, so image -> video
    // clicks still read as a selection change after the preview component swaps.
    expect(source).toContain('lastRenderedItemNameRef.current = videoName');
  });

  it('clears a pending post-render overlay when the video element errors', () => {
    // onLoadedMetadata normally clears the resolve state; an errored element never fires it.
    const errorHandler = source.slice(source.indexOf('const handleVideoError'), source.indexOf('const handlePlay'));
    expect(errorHandler).toContain('onLoadImage()');
  });
});
