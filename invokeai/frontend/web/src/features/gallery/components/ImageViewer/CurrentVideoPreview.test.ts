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

  it('tiles concurrent sessions instead of letting them overwrite each other (multi-GPU)', () => {
    // CurrentImagePreview tiles per-session previews when several renders run at once; the video
    // overlay must do the same or the sessions fight over the single full-size preview slot.
    expect(source).toMatch(/withTiledProgress = withProgress && activeProgressData\.length > 1/);
    expect(source).toContain('<ProgressImageTiles data={activeProgressData} />');
  });

  it('does not treat an auto-switch to the finished video as a user reveal', () => {
    // The auto-switch selection lands after onInvocationComplete's DTO fetch, so a quickly-started
    // next render's first progress event can reset $isProgressImageResolving ahead of it. Without
    // identity-based suppression the handoff reads as a gallery click and hides the new render's
    // live preview for 2 seconds.
    expect(source).toContain('autoSwitchedImages.consume(videoName)');
    // Consumption must happen on every change of the rendered video — an entry left behind would
    // swallow a genuine later click on the same video — so it cannot sit behind the reveal guards.
    const revealEffect = source.slice(
      source.indexOf('const previousRenderedItemName = lastRenderedItemNameRef.current;'),
      source.indexOf('SELECTED_ITEM_REVEAL_DURATION_MS)')
    );
    expect(revealEffect.indexOf('autoSwitchedImages.consume')).toBeLessThan(
      revealEffect.indexOf('if (!shouldShowProgressInViewer')
    );
    // The suppression branch must lower the atom: the effect has already cleared the running
    // reveal's timer, so returning with it still true would wedge the reveal on.
    expect(revealEffect).toMatch(
      /if \(wasAutoSwitchedTo\) \{\s+\$isTemporarilyShowingSelectedImage\.set\(false\);\s+return;/
    );
  });

  it('ends a pending post-render resolve when the video element errors', () => {
    // onLoadedMetadata normally ends the resolve illusion; an errored element never fires it. The
    // call must carry this video's session id so the lifecycle can tell it apart from a late load
    // belonging to another concurrently-completed session.
    const errorHandler = source.slice(source.indexOf('const handleVideoError'), source.indexOf('const handlePlay'));
    expect(errorHandler).toContain('onLoadImage(videoDTO?.session_id ?? null)');
  });
});
