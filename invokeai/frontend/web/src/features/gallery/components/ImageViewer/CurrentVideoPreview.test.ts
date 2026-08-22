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
  });

  it('tiles concurrent sessions instead of letting them overwrite each other (multi-GPU)', () => {
    // CurrentImagePreview tiles per-session previews when several renders run at once; the video
    // overlay must do the same or the sessions fight over the single full-size preview slot.
    expect(source).toMatch(/withTiledProgress = withProgress && activeProgressData\.length > 1/);
    expect(source).toContain('<ProgressImageTiles data={activeProgressData} />');
  });

  it('routes the reveal through the mounted-tested hook, fed by the painted-name readiness', () => {
    // The wiring itself — effect order, cleanup, unmount, the component swap — is behaviorally
    // covered in useSelectedItemReveal.test.tsx. What only this file can see is that this
    // component actually uses that hook, with readiness from the real element's onLoadedData
    // rather than from mount.
    expect(source).toContain('useSelectedItemReveal({');
    expect(source).toContain('renderedItemName: videoName,');
    expect(source).toMatch(/const \{ isMediaReady, onPainted \} = usePaintedItemName\(videoName\);/);
    expect(source).toContain('onLoadedData={onPainted}');
  });

  it('does not cover playback or a temporary reveal with the metadata panel', () => {
    // Playing and revealing both turn withProgress off, so gating the full-screen metadata panel
    // on !withProgress alone drops it exactly on top of the native controls / the just-revealed
    // video whenever item details are enabled.
    expect(source).toMatch(
      /shouldShowItemDetails && !isPlaying && !isTemporarilyShowingSelectedImage && !withProgress &&/
    );
  });

  it('restores the overlay when playback ends on its own, not only when the player is closed', () => {
    // isPlaying suppresses the overlay; without onEnded it never falls back, so the live preview
    // stays hidden for the rest of the generation after a short video plays out.
    expect(source).toContain('onEnded={handleClose}');
  });

  it('does not end a pending resolve when play() is rejected', () => {
    // A rejected play() is not a load failure — the element is intact and its metadata has usually
    // already loaded — so it must not clear an overlay belonging to some other session's render.
    const playHandler = source.slice(source.indexOf('const handlePlay'), source.indexOf('const handleClose'));
    expect(playHandler).toContain('reportPlaybackFailure()');
    expect(playHandler).not.toContain('onLoadImage');
  });

  it('ends a pending post-render resolve when the video element errors', () => {
    // onLoadedMetadata normally ends the resolve illusion; an errored element never fires it. The
    // call must carry this video's session id so the lifecycle can tell it apart from a late load
    // belonging to another concurrently-completed session.
    const errorHandler = source.slice(source.indexOf('const handleVideoError'), source.indexOf('const handlePlay'));
    expect(errorHandler).toContain('onLoadImage(videoDTO?.session_id ?? null)');
  });
});
