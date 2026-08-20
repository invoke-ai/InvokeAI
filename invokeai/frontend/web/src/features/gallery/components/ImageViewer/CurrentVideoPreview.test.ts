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
    // The machine is the one from the viewer context, shared with CurrentImagePreview, so a click
    // that switches media type is just another selection rather than a component swap to reason
    // about.
    expect(source).toContain('revealMachine,');
  });

  it('tiles concurrent sessions instead of letting them overwrite each other (multi-GPU)', () => {
    // CurrentImagePreview tiles per-session previews when several renders run at once; the video
    // overlay must do the same or the sessions fight over the single full-size preview slot.
    expect(source).toMatch(/withTiledProgress = withProgress && activeProgressData\.length > 1/);
    expect(source).toContain('<ProgressImageTiles data={activeProgressData} />');
  });

  it('drives the shared reveal machine, and only reports the video visible once it has painted', () => {
    // The auto-switch selection lands after onInvocationComplete's DTO fetch, so a quickly-started
    // next render's first progress event can reset $isProgressImageResolving ahead of it. Timing
    // cannot tell that handoff from a gallery click; the selection descriptor can, and the machine
    // owns the sequencing (see selectedItemReveal.test.ts for the behavior).
    expect(source).toContain('revealMachine.sync({');
    // The machine must be told about paint, not about mount: a <video> is black until it decodes.
    expect(source).toContain('onLoadedData={handleLoadedData}');
    // Readiness must be reported as *which* video has painted, compared against the one being
    // rendered. A boolean is reset from a different effect than the one that reads it, and a
    // passive effect's setState does not reach the next effect's closure in the same commit — so
    // the sync would see the new video's name with the previous video's readiness and lift the
    // overlay onto a black element.
    expect(source).toContain('isMediaReady: paintedVideoName === videoName,');
    expect(source).toContain('setPaintedVideoName(videoName);');
    expect(source).toMatch(/renderedItemName: videoName,/);
    // Nothing else pins the machine to the atom the overlay actually reads: selectedItemReveal's
    // own tests substitute their own setRevealed, and every assertion here and in
    // CurrentImagePreview.test.ts is on the read side. Replacing that wiring with a no-op left
    // the whole suite green with the reveal dead (found by an adversarial review of the #9434
    // merge). The machine is constructed once for both previews, so it is checked once.
    const context = readFileSync(fileURLToPath(new URL('./context.tsx', import.meta.url)), 'utf8');
    expect(context).toMatch(/setRevealed: \(revealed\) => \$isTemporarilyShowingSelectedImage\.set\(revealed\)/);
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
